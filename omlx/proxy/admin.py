# SPDX-License-Identifier: Apache-2.0
"""Reduced admin UI compatibility layer for the MLX-free proxy."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from omlx._version import __version__

from .backend import OpenAIBackend
from .config import BACKEND_URL_DEFAULTS, SIDECAR_BACKEND_TYPES, ProxyConfig
from .docker_control import (
    DockerControlError,
    DockerUnavailableError,
    docker_socket_path,
    restart_compose_service,
)
from .metrics import collect_backend_metrics_cached
from .sidecar_compose import (
    backend_spec,
    env_from_compose,
    load_env_file,
    render_env_file,
    write_env_file,
)

ADMIN_DIR = Path(__file__).resolve().parents[1] / "admin"
TEMPLATES_DIR = ADMIN_DIR / "templates"
STATIC_DIR = ADMIN_DIR / "static"
I18N_DIR = ADMIN_DIR / "i18n"


@dataclass
class ProxyAdminState:
    started_at: float = field(default_factory=time.time)
    model_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    global_overrides: dict[str, Any] = field(default_factory=dict)
    backend_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    state_path: Path | None = None

    def log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"{timestamp} proxy {message}")
        self.logs = self.logs[-1000:]

    @classmethod
    def load(cls, path: Path | None) -> "ProxyAdminState":
        state = cls(state_path=path)
        if path is None or not path.exists():
            return state
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            state.log(f"failed to read state file {path}")
            return state
        if isinstance(data.get("model_settings"), dict):
            state.model_settings = data["model_settings"]
        if isinstance(data.get("global_overrides"), dict):
            state.global_overrides = data["global_overrides"]
        if isinstance(data.get("backend_profiles"), dict):
            state.backend_profiles = {
                key: dict(value)
                for key, value in data["backend_profiles"].items()
                if isinstance(value, dict)
            }
        saved_type = str(state.global_overrides.get("proxy_backend_type") or "")
        if saved_type.strip().lower() == "ollama":
            state.global_overrides["proxy_backend_type"] = "openai-compatible"
            state.log("migrated saved backend type ollama -> openai-compatible")
        state.log(f"loaded state from {path}")
        return state

    def save(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_settings": self.model_settings,
            "global_overrides": self.global_overrides,
            "backend_profiles": self.backend_profiles,
        }
        self.state_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def configure_admin(app, backend: OpenAIBackend, config: ProxyConfig) -> None:
    """Attach admin UI routes and static assets to a proxy FastAPI app."""
    state_path = Path(os.getenv("OMLX_PROXY_STATE_PATH", "/data/proxy-state.json"))
    state = ProxyAdminState.load(state_path)
    _apply_env_sampling_overrides(state)
    app.state.proxy_admin_state = state
    _apply_proxy_backend_overrides(backend, config, state)
    templates = _templates()
    router = APIRouter(prefix="/admin", tags=["proxy-admin"])

    app.mount("/admin/static", StaticFiles(directory=STATIC_DIR), name="admin-static")

    @router.get("")
    @router.get("/")
    async def admin_root():
        return RedirectResponse(url="/admin/dashboard", status_code=302)

    @router.get("/dashboard")
    async def dashboard_page(request: Request):
        return templates.TemplateResponse(request, "dashboard.html", {})

    @router.get("/chat")
    async def chat_page(request: Request):
        api_key = config.proxy_api_key or ""
        return templates.TemplateResponse(
            request,
            "chat.html",
            {"api_key": api_key, "api_key_configured": bool(api_key)},
        )

    @router.get("/api/update-check")
    async def update_check():
        return {
            "update_available": False,
            "latest_version": __version__,
            "current_version": __version__,
            "release_url": "https://github.com/jundot/omlx",
        }

    @router.get("/api/server-info")
    async def server_info():
        return {
            "version": __version__,
            "mode": "proxy",
            "host": backend.config.host,
            "port": backend.config.port,
            "backend_url": backend.config.normalized_backend_url,
            "aliases": ["::1", "localhost", "127.0.0.1"],
            "capabilities": _capabilities(),
        }

    @router.get("/api/proxy/status")
    async def proxy_status():
        started = time.monotonic()
        try:
            data = await backend.get_models(None)
            models = data.get("data") or []
            reachable = True
            error = None
        except Exception as exc:
            models = []
            reachable = False
            error = str(exc)
        return {
            "mode": "proxy",
            "backend_url": backend.config.normalized_backend_url,
            "backend_type": _proxy_backend_type(state),
            "backend_reachable": reachable,
            "backend_error": error,
            "model_count": len(models),
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "state_path": str(state.state_path) if state.state_path else None,
            "capabilities": _capabilities(),
        }

    @router.get("/api/proxy/config")
    async def proxy_config():
        return {
            "backend_url": backend.config.normalized_backend_url,
            "backend_type": _proxy_backend_type(state),
            "backend_api_key": backend.config.backend_api_key or "",
            "backend_api_key_set": bool(backend.config.backend_api_key),
            "proxy_host": backend.config.host,
            "proxy_port": backend.config.port,
            "context_scaling_enabled": state.global_overrides.get(
                "context_scaling_enabled",
                backend.config.context_scaling_enabled,
            ),
            "target_context_size": state.global_overrides.get(
                "target_context_size",
                backend.config.target_context_size,
            ),
            "actual_context_size": backend.config.actual_context_size,
            "sse_keepalive_mode": backend.config.sse_keepalive_mode,
            "state_path": str(state.state_path) if state.state_path else None,
            "hot_reloadable": ["backend_url", "backend_api_key", "backend_type"],
        }

    @router.get("/api/proxy/metrics")
    async def proxy_metrics():
        started = time.monotonic()
        metrics = await collect_backend_metrics_cached(backend)
        metrics["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
        return metrics

    @router.get("/api/proxy/sidecar-compose")
    async def proxy_sidecar_compose():
        backend_name = _sidecar_backend(state)
        spec = backend_spec(backend_name)
        settings = _sidecar_settings_from_files(state)
        compose_path = _compose_output_path()
        env_path = _env_output_path()
        env_values = spec.environment(settings)
        return {
            "backend": backend_name,
            "settings": settings.__dict__,
            "content": spec.render_compose(settings),
            "env_content": render_env_file(env_values, spec.env_keys),
            "output_path": str(compose_path) if compose_path else None,
            "env_output_path": str(env_path) if env_path else None,
            "writable": bool(compose_path or env_path),
        }

    @router.post("/api/proxy/sidecar-compose/regenerate")
    async def regenerate_proxy_sidecar_compose():
        result = _write_sidecar_compose_if_configured(state)
        status = 200 if result.get("written") or not result.get("error") else 500
        return JSONResponse(result, status_code=status)

    @router.get("/api/global-settings")
    async def get_global_settings():
        return _global_settings_payload(backend.config, state)

    @router.post("/api/global-settings")
    async def update_global_settings(request: Request):
        payload = await request.json()
        try:
            proxy_updates = _extract_proxy_backend_updates(payload)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        old_type = _proxy_backend_type(state)
        new_type = proxy_updates.get("proxy_backend_type", old_type)
        type_changed = new_type != old_type
        if type_changed:
            _archive_backend_profile(state, old_type)
            state.global_overrides.update(_backend_profile_seed(state, new_type))
        state.global_overrides.update(payload)
        if proxy_updates:
            state.global_overrides.update(proxy_updates)
        if new_type in SIDECAR_BACKEND_TYPES:
            # Sidecar URLs are managed by the compose stack; the field is
            # readonly in the UI and enforced here for any other caller.
            state.global_overrides["proxy_backend_url"] = _default_backend_url(new_type)
        _archive_backend_profile(state, new_type)
        if proxy_updates or type_changed:
            backend.config = _config_with_proxy_overrides(backend.config, state)
            app.state.proxy_config = backend.config
        if "claude_code_context_scaling_enabled" in payload:
            state.global_overrides["context_scaling_enabled"] = payload[
                "claude_code_context_scaling_enabled"
            ]
        if "claude_code_target_context_size" in payload:
            state.global_overrides["target_context_size"] = payload[
                "claude_code_target_context_size"
            ]
        sidecar_updates = (
            _extract_sidecar_compose_updates(payload, _sidecar_backend(state))
            if new_type in SIDECAR_BACKEND_TYPES
            else {}
        )
        if sidecar_updates:
            state.global_overrides.update(sidecar_updates)
        state.log("updated proxy admin settings")
        state.save()
        compose_settings = _sidecar_settings_from_files(state, sidecar_updates)
        compose_result = _write_sidecar_compose_if_configured(state, compose_settings)
        runtime_applied = ["proxy_admin_settings"]
        if proxy_updates:
            runtime_applied.append("proxy_backend_config")
        if compose_result.get("env_written"):
            runtime_applied.append("sidecar_env_file")
        if compose_result.get("compose_written"):
            runtime_applied.append("sidecar_compose_file")
        return {
            "status": "ok",
            "message": (
                "Proxy backend settings saved and applied"
                if proxy_updates
                else "Proxy settings saved for this process"
            ),
            "runtime_applied": runtime_applied,
            "compose": compose_result,
            "requires_restart": bool(sidecar_updates)
            or (type_changed and new_type in SIDECAR_BACKEND_TYPES),
            "restart_required_settings": ["proxy_host", "proxy_port", "sidecar"],
        }

    @router.get("/api/models")
    async def admin_models():
        return {"models": await _admin_models(backend, state)}

    @router.post("/api/reload")
    async def reload_models():
        state.log("model list refreshed from backend")
        return {"status": "ok", "message": "Model list refreshed from backend"}

    @router.put("/api/models/{model_id}/settings")
    async def update_model_settings(model_id: str, request: Request):
        payload = await request.json()
        settings = state.model_settings.setdefault(model_id, {})
        settings.update(payload)
        if payload.get("is_default"):
            for mid, mid_settings in state.model_settings.items():
                if mid != model_id:
                    mid_settings["is_default"] = False
        state.log(f"updated settings for model {model_id}")
        state.save()
        return {
            "status": "ok",
            "model_id": model_id,
            "requires_reload": False,
            "auto_reloaded": False,
        }

    @router.post("/api/models/{model_id}/load")
    async def load_model(model_id: str):
        return {
            "status": "ok",
            "model_id": model_id,
            "message": "Remote backend manages model loading",
        }

    @router.post("/api/models/{model_id}/unload")
    async def unload_model(model_id: str):
        return {
            "status": "ok",
            "model_id": model_id,
            "message": "Remote backend manages model unloading",
        }

    @router.get("/api/models/{model_id}/profiles")
    async def model_profiles(model_id: str):
        return {"profiles": [], "active_profile_name": None}

    @router.post("/api/models/{model_id}/profiles")
    async def create_model_profile(model_id: str):
        return JSONResponse(
            {"detail": "Profiles are not implemented in proxy mode"},
            status_code=501,
        )

    @router.get("/api/profile-templates")
    async def profile_templates():
        return {"templates": []}

    @router.get("/api/profile-fields")
    async def profile_fields():
        return {
            "universal": [
                "model_alias",
                "max_context_window",
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "presence_penalty",
                "repetition_penalty",
                "enable_thinking",
                "thinking_budget_tokens",
                "max_tool_result_tokens",
                "ttl_seconds",
                "is_default",
            ],
            "model_specific": [],
        }

    @router.get("/api/grammar/parsers")
    async def grammar_parsers():
        return {"parsers": []}

    @router.get("/api/models/{model_id}/generation_config")
    async def generation_config(model_id: str):
        return {"config": {}, "source": "proxy"}

    @router.get("/api/stats")
    async def stats(model: str = "", scope: str = "session"):
        models = await _admin_models(backend, state)
        metrics_obj = getattr(app.state, "server_metrics", None)
        snapshot = (
            metrics_obj.get_snapshot(model_id=model, scope=scope)
            if metrics_obj is not None
            else None
        )
        backend_metrics = await collect_backend_metrics_cached(backend)
        return _stats_payload(
            backend.config,
            state,
            models,
            metrics=backend_metrics,
            snapshot=snapshot,
        )

    @router.post("/api/stats/clear")
    async def clear_stats():
        metrics_obj = getattr(app.state, "server_metrics", None)
        if metrics_obj is not None:
            metrics_obj.clear_metrics()
        state.log("cleared proxy session stats")
        return {"status": "ok"}

    @router.post("/api/stats/clear-alltime")
    async def clear_alltime_stats():
        metrics_obj = getattr(app.state, "server_metrics", None)
        if metrics_obj is not None:
            metrics_obj.clear_alltime_metrics()
        state.log("cleared proxy all-time stats")
        return {"status": "ok"}

    @router.post("/api/ssd-cache/clear")
    @router.post("/api/hot-cache/clear")
    async def clear_cache():
        return {
            "status": "ok",
            "message": "Remote backend cache controls are not managed by proxy mode",
        }

    @router.get("/api/logs")
    async def logs(lines: int = 200, file: str = "server.log"):
        content = "\n".join(state.logs[-max(1, min(lines, 1000)) :])
        return {
            "logs": content,
            "total_lines": len(state.logs),
            "available_files": ["server.log"],
            "file": file,
        }

    @router.get("/api/device-info")
    async def device_info():
        return {
            "device": "remote-backend",
            "backend_url": backend.config.normalized_backend_url,
            "memory_total": 0,
            "memory_available": 0,
        }

    @router.get("/api/bench/active")
    async def bench_active():
        return {"active": False, "run": None}

    @router.post("/api/bench/start")
    async def bench_start():
        return JSONResponse(
            {"detail": "Benchmarks are not implemented in proxy mode"},
            status_code=501,
        )

    @router.get("/api/bench/accuracy/results")
    async def accuracy_results():
        return {"results": []}

    @router.get("/api/bench/accuracy/queue/status")
    async def accuracy_queue():
        return {"queue": [], "active": None}

    @router.get("/api/hf/tasks")
    @router.get("/api/ms/tasks")
    @router.get("/api/oq/tasks")
    @router.get("/api/upload/tasks")
    async def empty_tasks():
        return {"tasks": []}

    @router.get("/api/hf/models")
    async def hf_models():
        return {"models": []}

    @router.get("/api/hf/recommended")
    @router.get("/api/ms/recommended")
    async def recommended_models():
        return {"trending": [], "popular": []}

    @router.get("/api/hf/search")
    @router.get("/api/ms/search")
    async def search_models():
        return {"models": [], "total": 0}

    @router.get("/api/hf/model-info")
    @router.get("/api/ms/model-info")
    async def model_info():
        return {"model": None}

    @router.get("/api/ms/status")
    async def ms_status():
        return {"available": False}

    @router.get("/api/oq/models")
    async def oq_models():
        return {"models": [], "all_models": []}

    @router.get("/api/upload/oq-models")
    async def upload_oq_models():
        return {"oq_models": [], "all_models": []}

    @router.post("/api/logout")
    async def logout():
        return {"status": "ok"}

    @router.post("/api/server/restart")
    async def restart():
        return JSONResponse(
            {"detail": "Restart is managed by Docker/Compose in proxy mode"},
            status_code=501,
        )

    @router.post("/api/sidecar/restart")
    async def restart_sidecar():
        backend_type = _proxy_backend_type(state)
        if backend_type not in SIDECAR_BACKEND_TYPES:
            return JSONResponse(
                {
                    "detail": (
                        "No managed sidecar to restart: the proxy routes to a "
                        "remote OpenAI-compatible backend"
                    )
                },
                status_code=409,
            )
        service = _sidecar_backend(state)
        try:
            container_id = await restart_compose_service(service)
        except DockerUnavailableError as exc:
            return JSONResponse(
                {
                    "detail": (
                        f"{exc} Mount /var/run/docker.sock into the proxy "
                        "container to enable backend restarts."
                    )
                },
                status_code=501,
            )
        except DockerControlError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=502)
        state.log(f"restarted {service} sidecar container {container_id[:12]}")
        return JSONResponse(
            {
                "status": "restarting",
                "service": service,
                "container_id": container_id,
                "message": (
                    f"Restarting {service} container. Image and port changes "
                    "still require recreating the Compose stack on the host."
                ),
            },
            status_code=202,
        )

    @router.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def unsupported_admin_api(path: str):
        return {
            "status": "unsupported",
            "detail": f"/admin/api/{path} is not implemented in proxy mode",
        }

    app.include_router(router)


def _templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
    locale = _load_locale("en")

    def static(path: str) -> str:
        file_path = STATIC_DIR / path
        if file_path.is_file():
            return f"/admin/static/{path}?v={int(file_path.stat().st_mtime)}"
        return f"/admin/static/{path}"

    templates.env.globals["static"] = static
    templates.env.globals["version"] = __version__
    templates.env.globals["t"] = lambda key: locale.get(key, key)
    templates.env.globals["locale_json"] = json.dumps(locale, ensure_ascii=False)
    templates.env.globals["current_lang"] = "en"
    return templates


def _load_locale(language: str) -> dict[str, str]:
    try:
        return json.loads((I18N_DIR / f"{language}.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _apply_proxy_backend_overrides(
    backend: OpenAIBackend,
    base_config: ProxyConfig,
    state: ProxyAdminState,
) -> None:
    try:
        backend.config = _config_with_proxy_overrides(base_config, state)
    except ValueError as exc:
        state.log(f"ignored invalid saved proxy backend config: {exc}")
        backend.config = base_config


def _apply_env_sampling_overrides(state: ProxyAdminState) -> None:
    env_map = {
        "OMLX_SAMPLING_MAX_TOKENS": "sampling_max_tokens",
        "OMLX_SAMPLING_TEMPERATURE": "sampling_temperature",
        "OMLX_SAMPLING_TOP_P": "sampling_top_p",
        "OMLX_SAMPLING_TOP_K": "sampling_top_k",
        "OMLX_SAMPLING_REPETITION_PENALTY": "sampling_repetition_penalty",
    }
    for env_key, override_key in env_map.items():
        value = os.getenv(env_key)
        if value is None or value == "" or override_key in state.global_overrides:
            continue
        state.global_overrides[override_key] = _coerce_sampling_env_value(
            override_key,
            value,
        )


def _coerce_sampling_env_value(key: str, value: str) -> int | float | str:
    try:
        if key in {"sampling_max_tokens", "sampling_top_k"}:
            return int(value)
        return float(value)
    except ValueError:
        return value


def _config_with_proxy_overrides(
    config: ProxyConfig,
    state: ProxyAdminState,
) -> ProxyConfig:
    overrides = state.global_overrides
    backend_url = str(
        overrides.get("proxy_backend_url") or config.normalized_backend_url
    ).strip()
    backend_api_key = overrides.get("proxy_backend_api_key", config.backend_api_key)
    if backend_api_key == "":
        backend_api_key = None
    return replace(
        config,
        backend_url=_normalize_backend_url(backend_url),
        backend_api_key=backend_api_key,
    )


def _extract_proxy_backend_updates(payload: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if "proxy_backend_url" in payload:
        updates["proxy_backend_url"] = _normalize_backend_url(
            str(payload.get("proxy_backend_url") or "")
        )
    if "proxy_backend_api_key" in payload:
        value = payload.get("proxy_backend_api_key")
        updates["proxy_backend_api_key"] = str(value).strip() if value else ""
    if "proxy_backend_type" in payload:
        updates["proxy_backend_type"] = _normalize_backend_type(
            str(payload.get("proxy_backend_type") or "")
        )
    return updates


_COMMON_UPDATE_FIELDS = (
    "omni_model",
    "omni_served_model_name",
    "omni_context_length",
    "omni_max_parallel",
    "omni_backend_port",
    "omni_hf_home",
    "network_http_proxy",
    "network_https_proxy",
    "network_no_proxy",
    "network_ca_bundle",
    "huggingface_endpoint",
    "sampling_max_tokens",
    "sampling_temperature",
    "sampling_top_p",
    "sampling_top_k",
    "sampling_repetition_penalty",
)

_VLLM_UPDATE_FIELDS = (
    "vllm_image",
    "vllm_gpu_memory_utilization",
    "vllm_generation_config",
    "vllm_default_chat_template_kwargs",
    "vllm_trust_remote_code",
    "vllm_enforce_eager",
    "vllm_enable_auto_tool_choice",
    "vllm_tool_call_parser",
    "vllm_reasoning_parser",
    "vllm_dtype",
    "vllm_tokenizer",
    "vllm_tokenizer_mode",
    "vllm_revision",
    "vllm_load_format",
    "vllm_quantization",
    "vllm_download_dir",
    "vllm_max_num_batched_tokens",
    "vllm_enable_chunked_prefill",
    "vllm_enable_prefix_caching",
    "vllm_kv_cache_dtype",
    "vllm_cpu_offload_gb",
    "vllm_swap_space",
    "vllm_tensor_parallel_size",
    "vllm_pipeline_parallel_size",
    "vllm_uvicorn_log_level",
    "vllm_disable_log_stats",
    "vllm_extra_args_json",
)

_LLAMACPP_UPDATE_FIELDS = (
    "llamacpp_image",
    "llamacpp_n_gpu_layers",
    "llamacpp_flash_attn",
    "llamacpp_cache_type_k",
    "llamacpp_cache_type_v",
    "llamacpp_threads",
    "llamacpp_batch_size",
    "llamacpp_ubatch_size",
    "llamacpp_jinja",
    "llamacpp_reasoning_format",
    "llamacpp_cache_dir",
    "llamacpp_model_dir",
    "llamacpp_extra_args",
)


def _extract_sidecar_compose_updates(
    payload: dict[str, Any],
    backend: str,
) -> dict[str, Any]:
    fields = set(_COMMON_UPDATE_FIELDS)
    if backend == "llamacpp":
        fields.update(_LLAMACPP_UPDATE_FIELDS)
    else:
        fields.update(_VLLM_UPDATE_FIELDS)
    updates = {key: payload[key] for key in fields if key in payload}
    if not updates:
        return {}
    if (
        "sampling_max_context_window" in payload
        and "omni_context_length" not in updates
    ):
        updates["omni_context_length"] = payload["sampling_max_context_window"]
    if "max_concurrent_requests" in payload and "omni_max_parallel" not in updates:
        updates["omni_max_parallel"] = payload["max_concurrent_requests"]
    if (
        backend != "llamacpp"
        and "chunked_prefill" in payload
        and "vllm_enable_chunked_prefill" not in updates
    ):
        updates["vllm_enable_chunked_prefill"] = payload["chunked_prefill"]
    return updates


def _sidecar_settings_payload(state: ProxyAdminState) -> dict[str, Any]:
    settings = _sidecar_settings_from_files(state)
    compose_path = _compose_output_path()
    env_path = _env_output_path()
    payload = settings.__dict__.copy()
    payload["compose_output_path"] = str(compose_path) if compose_path else None
    payload["env_output_path"] = str(env_path) if env_path else None
    return payload


def _sidecar_settings_from_files(
    state: ProxyAdminState,
    updates: dict[str, Any] | None = None,
):
    backend = _sidecar_backend(state)
    spec = backend_spec(backend)
    values = spec.environment(spec.settings_from_overrides(state.global_overrides))
    compose_path = _compose_output_path()
    if compose_path is not None:
        values.update(env_from_compose(compose_path, spec.env_keys))
    env_path = _env_output_path()
    if env_path is not None:
        values.update(load_env_file(env_path))
    if updates:
        values.update(_sidecar_env_from_admin_updates(updates, backend))
    return spec.settings_from_env(values)


_COMMON_UPDATE_ENV_MAP = {
    "omni_model": "OMNI_MODEL",
    "omni_served_model_name": "OMNI_SERVED_MODEL_NAME",
    "omni_context_length": "OMNI_CONTEXT_LENGTH",
    "omni_max_parallel": "OMNI_MAX_PARALLEL",
    "omni_backend_port": "OMNI_BACKEND_PORT",
    "omni_hf_home": "OMNI_HF_HOME",
    "network_http_proxy": "OMNI_HTTP_PROXY",
    "network_https_proxy": "OMNI_HTTPS_PROXY",
    "network_no_proxy": "OMNI_NO_PROXY",
    "network_ca_bundle": "OMNI_CA_BUNDLE",
    "huggingface_endpoint": "OMNI_HF_ENDPOINT",
    "sampling_max_tokens": "OMLX_SAMPLING_MAX_TOKENS",
    "sampling_temperature": "OMLX_SAMPLING_TEMPERATURE",
    "sampling_top_p": "OMLX_SAMPLING_TOP_P",
    "sampling_top_k": "OMLX_SAMPLING_TOP_K",
    "sampling_repetition_penalty": "OMLX_SAMPLING_REPETITION_PENALTY",
}

_VLLM_UPDATE_ENV_MAP = {
    "vllm_image": "VLLM_IMAGE",
    "vllm_gpu_memory_utilization": "VLLM_GPU_MEMORY_UTILIZATION",
    "vllm_generation_config": "VLLM_GENERATION_CONFIG",
    "vllm_default_chat_template_kwargs": "VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS",
    "vllm_trust_remote_code": "VLLM_TRUST_REMOTE_CODE",
    "vllm_enforce_eager": "VLLM_ENFORCE_EAGER",
    "vllm_enable_auto_tool_choice": "VLLM_ENABLE_AUTO_TOOL_CHOICE",
    "vllm_tool_call_parser": "VLLM_TOOL_CALL_PARSER",
    "vllm_reasoning_parser": "VLLM_REASONING_PARSER",
    "vllm_dtype": "VLLM_DTYPE",
    "vllm_tokenizer": "VLLM_TOKENIZER",
    "vllm_tokenizer_mode": "VLLM_TOKENIZER_MODE",
    "vllm_revision": "VLLM_REVISION",
    "vllm_load_format": "VLLM_LOAD_FORMAT",
    "vllm_quantization": "VLLM_QUANTIZATION",
    "vllm_download_dir": "VLLM_DOWNLOAD_DIR",
    "vllm_max_num_batched_tokens": "VLLM_MAX_NUM_BATCHED_TOKENS",
    "vllm_enable_chunked_prefill": "VLLM_ENABLE_CHUNKED_PREFILL",
    "vllm_enable_prefix_caching": "VLLM_ENABLE_PREFIX_CACHING",
    "vllm_kv_cache_dtype": "VLLM_KV_CACHE_DTYPE",
    "vllm_cpu_offload_gb": "VLLM_CPU_OFFLOAD_GB",
    "vllm_swap_space": "VLLM_SWAP_SPACE",
    "vllm_tensor_parallel_size": "VLLM_TENSOR_PARALLEL_SIZE",
    "vllm_pipeline_parallel_size": "VLLM_PIPELINE_PARALLEL_SIZE",
    "vllm_uvicorn_log_level": "VLLM_UVICORN_LOG_LEVEL",
    "vllm_disable_log_stats": "VLLM_DISABLE_LOG_STATS",
    "vllm_extra_args_json": "VLLM_EXTRA_ARGS_JSON",
}

_LLAMACPP_UPDATE_ENV_MAP = {
    "llamacpp_image": "LLAMACPP_IMAGE",
    "llamacpp_n_gpu_layers": "LLAMACPP_N_GPU_LAYERS",
    "llamacpp_flash_attn": "LLAMACPP_FLASH_ATTN",
    "llamacpp_cache_type_k": "LLAMACPP_CACHE_TYPE_K",
    "llamacpp_cache_type_v": "LLAMACPP_CACHE_TYPE_V",
    "llamacpp_threads": "LLAMACPP_THREADS",
    "llamacpp_batch_size": "LLAMACPP_BATCH_SIZE",
    "llamacpp_ubatch_size": "LLAMACPP_UBATCH_SIZE",
    "llamacpp_jinja": "LLAMACPP_JINJA",
    "llamacpp_reasoning_format": "LLAMACPP_REASONING_FORMAT",
    "llamacpp_cache_dir": "LLAMACPP_CACHE_DIR",
    "llamacpp_model_dir": "LLAMACPP_MODEL_DIR",
    "llamacpp_extra_args": "LLAMACPP_EXTRA_ARGS",
}

_BOOL_UPDATE_FIELDS = {
    "vllm_trust_remote_code",
    "vllm_enforce_eager",
    "vllm_enable_auto_tool_choice",
    "vllm_enable_chunked_prefill",
    "vllm_enable_prefix_caching",
    "vllm_disable_log_stats",
    "llamacpp_jinja",
}

_OPTIONAL_BOOL_UPDATE_FIELDS = {
    "vllm_enable_chunked_prefill",
    "vllm_enable_prefix_caching",
}


def _sidecar_env_from_admin_updates(
    updates: dict[str, Any],
    backend: str,
) -> dict[str, str]:
    field_map = dict(_COMMON_UPDATE_ENV_MAP)
    if backend == "llamacpp":
        field_map.update(_LLAMACPP_UPDATE_ENV_MAP)
    else:
        field_map.update(_VLLM_UPDATE_ENV_MAP)
    env = {}
    for field, env_key in field_map.items():
        if field not in updates:
            continue
        value = updates[field]
        if field in _BOOL_UPDATE_FIELDS:
            if field in _OPTIONAL_BOOL_UPDATE_FIELDS and (value is None or value == ""):
                env[env_key] = ""
            else:
                env[env_key] = "true" if _truthy(value) else "false"
        else:
            env[env_key] = str(value)
    return env


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _sidecar_backend(state: ProxyAdminState) -> str:
    value = os.getenv("OMLX_SIDECAR_BACKEND", "").strip().lower()
    if value in ("vllm", "llamacpp"):
        return value
    backend_type = _proxy_backend_type(state)
    if backend_type == "llama.cpp":
        return "llamacpp"
    return "vllm"


def _compose_output_path() -> Path | None:
    value = os.getenv("OMLX_COMPOSE_OUTPUT_PATH", "").strip()
    if not value:
        return None
    return Path(value)


def _env_output_path() -> Path | None:
    value = os.getenv("OMLX_ENV_OUTPUT_PATH", "").strip()
    if not value:
        return None
    return Path(value)


def _write_sidecar_compose_if_configured(
    state: ProxyAdminState,
    settings=None,
) -> dict[str, Any]:
    backend = _sidecar_backend(state)
    spec = backend_spec(backend)
    compose_path = _compose_output_path()
    env_path = _env_output_path()
    result: dict[str, Any] = {
        "backend": backend,
        "written": False,
        "compose_written": False,
        "env_written": False,
        "output_path": str(compose_path) if compose_path else None,
        "env_output_path": str(env_path) if env_path else None,
    }
    if compose_path is None and env_path is None:
        return result

    settings = settings or _sidecar_settings_from_files(state)
    errors = []
    if env_path is not None:
        try:
            written_env = write_env_file(
                env_path, spec.environment(settings), spec.env_keys
            )
        except Exception as exc:
            state.log(f"failed to write {backend} env file: {exc}")
            errors.append(str(exc))
        else:
            state.log(f"wrote {backend} env file {written_env}")
            result["env_written"] = True
            result["env_output_path"] = str(written_env)

    if compose_path is not None:
        try:
            written_compose = spec.write_compose(compose_path, settings)
        except Exception as exc:
            state.log(f"failed to write {backend} compose file: {exc}")
            errors.append(str(exc))
        else:
            state.log(f"wrote {backend} compose file {written_compose}")
            result["compose_written"] = True
            result["output_path"] = str(written_compose)

    result["written"] = bool(result["compose_written"] or result["env_written"])
    if errors:
        result["error"] = "; ".join(errors)
    return result


def _normalize_backend_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        raise ValueError("proxy_backend_url is required")
    if not value.startswith(("http://", "https://")):
        raise ValueError("proxy_backend_url must start with http:// or https://")
    return value


_BACKEND_TYPE_ALIASES = {"ollama": "openai-compatible"}


def _normalize_backend_type(value: str) -> str:
    allowed = {"openai-compatible", "vllm", "llama.cpp"}
    value = value.strip().lower() or "openai-compatible"
    value = _BACKEND_TYPE_ALIASES.get(value, value)
    if value not in allowed:
        return "openai-compatible"
    return value


def _proxy_backend_type(state: ProxyAdminState) -> str:
    return _normalize_backend_type(
        str(state.global_overrides.get("proxy_backend_type") or "openai-compatible")
    )


# Settings that persist per backend type so switching back restores them.
# vllm_*/llamacpp_* keys stay in the flat overrides; their prefixes already
# key them by backend.
_PROFILE_KEYS = (
    "proxy_backend_url",
    "proxy_backend_api_key",
    "omni_model",
    "omni_served_model_name",
)


def _default_backend_url(backend_type: str) -> str:
    return BACKEND_URL_DEFAULTS.get(
        backend_type, BACKEND_URL_DEFAULTS["openai-compatible"]
    )


def _archive_backend_profile(state: ProxyAdminState, backend_type: str) -> None:
    profile = state.backend_profiles.setdefault(backend_type, {})
    for key in _PROFILE_KEYS:
        if key in state.global_overrides:
            profile[key] = state.global_overrides[key]


def _backend_profile_seed(
    state: ProxyAdminState,
    backend_type: str,
) -> dict[str, Any]:
    stored = state.backend_profiles.get(backend_type, {})
    seed = {key: stored[key] for key in _PROFILE_KEYS if key in stored}
    url = str(seed.get("proxy_backend_url") or "").strip()
    if not url or backend_type in SIDECAR_BACKEND_TYPES:
        seed["proxy_backend_url"] = _default_backend_url(backend_type)
    seed.setdefault("proxy_backend_api_key", "")
    return seed


def _backend_profiles_payload(state: ProxyAdminState) -> dict[str, dict[str, Any]]:
    active_type = _proxy_backend_type(state)
    payload: dict[str, dict[str, Any]] = {}
    for backend_type in BACKEND_URL_DEFAULTS:
        values = dict(state.backend_profiles.get(backend_type, {}))
        if backend_type == active_type:
            for key in _PROFILE_KEYS:
                if key in state.global_overrides:
                    values[key] = state.global_overrides[key]
        url = str(values.get("proxy_backend_url") or "").strip()
        if not url or backend_type in SIDECAR_BACKEND_TYPES:
            url = _default_backend_url(backend_type)
        payload[backend_type] = {
            "backend_url": url,
            "backend_api_key": str(values.get("proxy_backend_api_key") or ""),
            "model": values.get("omni_model"),
            "served_model_name": values.get("omni_served_model_name"),
        }
    return payload


async def _backend_model_data(backend: OpenAIBackend) -> list[dict[str, Any]]:
    try:
        data = await backend.get_models(None)
    except Exception:
        return []
    models = data.get("data") or []
    return [m for m in models if isinstance(m, dict)]


async def _admin_models(
    backend: OpenAIBackend,
    state: ProxyAdminState,
) -> list[dict[str, Any]]:
    backend_models = await _backend_model_data(backend)
    default_id = backend_models[0].get("id") if backend_models else None
    configured_default = next(
        (
            mid
            for mid, settings in state.model_settings.items()
            if settings.get("is_default")
        ),
        None,
    )
    if configured_default:
        default_id = configured_default

    result = []
    for item in backend_models:
        model_id = item.get("id")
        if not model_id:
            continue
        settings = state.model_settings.setdefault(model_id, {})
        result.append(
            {
                "id": model_id,
                "name": model_id,
                "path": model_id,
                "model_type": settings.get("model_type_override") or "llm",
                "engine_type": "remote",
                "loaded": True,
                "is_loading": False,
                "pinned": bool(settings.get("is_pinned", False)),
                "is_pinned": bool(settings.get("is_pinned", False)),
                "is_default": model_id == default_id,
                "estimated_size": 0,
                "estimated_size_formatted": "remote",
                "actual_size": None,
                "settings": {
                    "max_context_window": item.get("max_model_len"),
                    **settings,
                },
                "model_alias": settings.get("model_alias"),
                "max_context_window": item.get("max_model_len"),
                "max_tokens": settings.get("max_tokens"),
                "dflash_compatible": False,
                "dflash_ssd_cache_available": False,
                "mtp_compatible": False,
            }
        )
    return result


def _global_settings_payload(
    config: ProxyConfig,
    state: ProxyAdminState,
) -> dict[str, Any]:
    overrides = state.global_overrides
    sidecar_settings = _sidecar_settings_from_files(state)
    return {
        "base_path": "",
        "server": {
            "host": config.host,
            "port": config.port,
            "log_level": overrides.get("log_level", "info"),
            "server_aliases": ["::1", "localhost", "127.0.0.1"],
            "sse_keepalive_mode": config.sse_keepalive_mode,
        },
        "proxy": {
            "mode": "proxy",
            "backend_url": config.normalized_backend_url,
            "backend_type": _proxy_backend_type(state),
            "backend_api_key": config.backend_api_key or "",
            "backend_api_key_set": bool(config.backend_api_key),
            "state_path": str(state.state_path) if state.state_path else None,
            "capabilities": _capabilities(),
            "sidecar_backend": _sidecar_backend(state),
            "sidecar": _sidecar_settings_payload(state),
            "backend_url_defaults": dict(BACKEND_URL_DEFAULTS),
            "backend_url_locked": list(SIDECAR_BACKEND_TYPES),
            "backend_profiles": _backend_profiles_payload(state),
            "docker_socket_available": Path(docker_socket_path()).exists(),
        },
        "model": {
            "model_dirs": [config.normalized_backend_url],
            "model_dir": config.normalized_backend_url,
            "model_fallback": False,
        },
        "memory": {
            "prefill_memory_guard": False,
            "memory_guard_tier": "balanced",
            "memory_guard_custom_ceiling_gb": 0,
        },
        "scheduler": {
            "max_concurrent_requests": overrides.get(
                "max_concurrent_requests",
                sidecar_settings.max_parallel,
            ),
            "embedding_batch_size": 0,
            "chunked_prefill": overrides.get(
                "chunked_prefill",
                bool(getattr(sidecar_settings, "enable_chunked_prefill", False)),
            ),
        },
        "cache": {
            "enabled": False,
            "ssd_cache_dir": "",
            "ssd_cache_max_size": "0",
            "hot_cache_only": False,
            "hot_cache_max_size": "0",
            "initial_cache_blocks": 0,
        },
        "mcp": {"config_path": ""},
        "huggingface": {"endpoint": sidecar_settings.hf_endpoint},
        "modelscope": {"endpoint": ""},
        "network": {
            "http_proxy": sidecar_settings.http_proxy,
            "https_proxy": sidecar_settings.https_proxy,
            "no_proxy": sidecar_settings.no_proxy,
            "ca_bundle": sidecar_settings.ca_bundle,
        },
        "sampling": {
            "max_context_window": overrides.get(
                "sampling_max_context_window",
                sidecar_settings.context_length,
            ),
            "max_tokens": overrides.get(
                "sampling_max_tokens", config.actual_context_size
            ),
            "temperature": overrides.get("sampling_temperature", 1.0),
            "top_p": overrides.get("sampling_top_p", 1.0),
            "top_k": overrides.get("sampling_top_k", 0),
            "repetition_penalty": overrides.get("sampling_repetition_penalty", 1.0),
        },
        "auth": {
            "api_key_set": bool(config.proxy_api_key),
            "api_key": config.proxy_api_key or "",
            "skip_api_key_verification": not bool(config.proxy_api_key),
            "sub_keys": [],
        },
        "claude_code": {
            "context_scaling_enabled": overrides.get(
                "context_scaling_enabled",
                config.context_scaling_enabled,
            ),
            "target_context_size": overrides.get(
                "target_context_size",
                config.target_context_size,
            ),
            "mode": "local",
            "opus_model": None,
            "sonnet_model": None,
            "haiku_model": None,
        },
        "integrations": {
            "codex_model": None,
            "opencode_model": None,
            "openclaw_model": None,
            "hermes_model": None,
            "pi_model": None,
            "copilot_model": None,
            "openclaw_tools_profile": "full",
        },
        "system": {
            "total_memory_bytes": 0,
            "total_memory": "remote",
            "auto_model_memory": "remote",
            "available_memory_bytes": 0,
            "omlx_phys_footprint_bytes": 0,
            "free_memory_bytes": 0,
            "inactive_memory_bytes": 0,
            "active_memory_bytes": 0,
            "iogpu_wired_limit_bytes": 0,
            "omlx_wired_limit_request_bytes": 0,
            "ssd_total_bytes": 0,
            "ssd_total": "remote",
        },
        "ui": {"language": "en"},
        "idle_timeout": {"idle_timeout_seconds": None},
    }


def _format_size_bytes(num: float) -> str:
    if num >= 1024**3:
        return f"{num / 1024**3:.2f} GB"
    if num >= 1024**2:
        return f"{num / 1024**2:.1f} MB"
    if num > 0:
        return f"{num / 1024:.0f} KB"
    return "0 B"


def _ttl_remaining_seconds(expires_at: Any) -> float | None:
    """Seconds until an Ollama ``expires_at`` timestamp, clamped to >= 0."""
    if not expires_at or not isinstance(expires_at, str):
        return None
    text = expires_at.strip()
    # Ollama reports nanosecond precision; fromisoformat wants <= 6 digits.
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed.year <= 1:  # Ollama's zero-value for "not expiring"
        return None
    return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


def _model_row(model_id: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": model_id,
        "estimated_size": 0,
        "estimated_size_formatted": "remote",
        "actual_size": None,
        "actual_size_formatted": None,
        "pinned": False,
        "is_loading": False,
        "active_requests": 0,
        "waiting_requests": 0,
        "waiting": [],
        "activities": [],
        "prefilling": [],
        "generating": [],
        "idle_seconds": None,
        "ttl_remaining_seconds": None,
    }
    row.update(overrides)
    return row


def _active_models_payload(
    models: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build the Active Models panel data from backend-reported state."""
    summary = metrics.get("summary") or {}
    running = int(summary.get("running_requests") or 0)
    waiting = int(summary.get("waiting_requests") or 0)
    ollama = metrics.get("ollama") or {}

    rows: list[dict[str, Any]]
    memory_used = 0
    if ollama.get("available") and ollama.get("loaded_models"):
        rows = []
        for item in ollama["loaded_models"]:
            model_id = str(item.get("name") or item.get("model") or "")
            if not model_id:
                continue
            size = int(item.get("size") or 0)
            memory_used += size
            rows.append(
                _model_row(
                    model_id,
                    estimated_size=size,
                    estimated_size_formatted=_format_size_bytes(size),
                    ttl_remaining_seconds=_ttl_remaining_seconds(
                        item.get("expires_at")
                    ),
                )
            )
    else:
        rows = [
            _model_row(str(m["id"]), pinned=bool(m.get("pinned", False)))
            for m in models
            if m.get("id")
        ]
        # vLLM/llama.cpp serve one model; attribute queue depth to it.
        if len(rows) == 1:
            rows[0]["active_requests"] = running
            rows[0]["waiting_requests"] = waiting

    return {
        "models": rows,
        "model_memory_used": memory_used,
        "model_memory_max": 0,
        "memory_pressure": {
            "enabled": False,
            "current_bytes": 0,
            "soft_bytes": 0,
            "hard_bytes": 0,
            "current_formatted": "remote",
            "soft_formatted": "remote",
            "hard_formatted": "remote",
            "pressure_level": "ok",
        },
        "total_active_requests": running,
        "total_waiting_requests": waiting,
    }


def _stats_payload(
    config: ProxyConfig,
    state: ProxyAdminState,
    models: list[dict[str, Any]],
    metrics: dict[str, Any] | None = None,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    uptime = max(0.0, time.time() - state.started_at)
    metrics = metrics or {}
    summary = metrics.get("summary") or {}
    running = int(summary.get("running_requests") or 0)
    waiting = int(summary.get("waiting_requests") or 0)
    active_models_data = _active_models_payload(models, metrics)
    snapshot = snapshot or {}
    return {
        "uptime_seconds": uptime,
        "host": config.host,
        "port": config.port,
        "api_key": config.proxy_api_key or "",
        "cli_prefix": "omni",
        "total_requests": int(snapshot.get("total_requests") or 0),
        "active_requests": running,
        "waiting_requests": waiting,
        "total_prompt_tokens": int(snapshot.get("total_prompt_tokens") or 0),
        "total_completion_tokens": int(snapshot.get("total_completion_tokens") or 0),
        "total_cached_tokens": int(snapshot.get("total_cached_tokens") or 0),
        "total_tokens_served": int(snapshot.get("total_tokens_served") or 0),
        "cache_efficiency": float(snapshot.get("cache_efficiency") or 0.0),
        "avg_prefill_tps": float(snapshot.get("avg_prefill_tps") or 0.0),
        "avg_generation_tps": float(snapshot.get("avg_generation_tps") or 0.0),
        "claude_code_context_scaling_enabled": config.context_scaling_enabled,
        "claude_code_target_context_size": config.target_context_size,
        "engines": {"mode": "proxy", "backend_url": config.normalized_backend_url},
        "active_models": active_models_data,
        "runtime_cache": {
            "base_path": "",
            "ssd_cache_dir": "",
            "response_state_dir": "",
            "models": [],
            "total_num_files": 0,
            "total_size_bytes": 0,
            "effective_block_sizes": [],
            "disk_max_bytes": 0,
            "hot_cache_max_bytes": 0,
            "hot_cache_size_bytes": 0,
            "hot_cache_entries": 0,
        },
        "proxy": {
            "mode": "proxy",
            "backend_url": config.normalized_backend_url,
            "capabilities": _capabilities(),
            "metrics": metrics,
        },
    }


def _capabilities() -> dict[str, bool]:
    return {
        "proxy_mode": True,
        "chat": True,
        "model_aliases": True,
        "model_load_unload": False,
        "local_model_files": False,
        "hf_downloader": False,
        "modelscope_downloader": False,
        "quantizer": False,
        "uploader": False,
        "benchmarks": False,
        "cache_controls": False,
        "native_memory_guard": False,
        "backend_metrics": True,
    }
