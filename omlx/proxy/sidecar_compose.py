# SPDX-License-Identifier: Apache-2.0
"""Shared settings and helpers for managed sidecar backends (vLLM, llama.cpp).

Backend-portable settings use the OMNI_* env prefix and live on
``CommonSidecarSettings``; backend-specific knobs keep VLLM_*/LLAMACPP_*
prefixes on the per-backend subclasses.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommonSidecarSettings:
    model: str = ""
    served_model_name: str = "qwen"
    context_length: int = 8192
    max_parallel: int = 4
    backend_port: int = 8000
    hf_home: str = "${HOME}/.cache/huggingface"
    hf_endpoint: str = ""
    http_proxy: str = ""
    https_proxy: str = ""
    no_proxy: str = ""
    ca_bundle: str = ""
    proxy_port: int = 8080
    proxy_api_key: str = ""
    backend_api_key: str = ""
    context_scaling: bool = False
    target_context_size: int = 200000
    sse_keepalive_mode: str = "ping"
    # Opt-in local model scan (omni serve --scan-models [--model-dir DIR]).
    scan_models: bool = False
    model_scan_host_dir: str = ""
    # 0 = no default output cap injected; the backend applies its own
    # limit (vLLM caps to the remaining context when max_tokens is unset).
    sampling_max_tokens: int = 0
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
    sampling_top_k: int = 0
    sampling_repetition_penalty: float = 1.0


OMNI_ENV_KEYS = (
    "OMNI_MODEL",
    "OMNI_SERVED_MODEL_NAME",
    "OMNI_CONTEXT_LENGTH",
    "OMNI_MAX_PARALLEL",
    "OMNI_BACKEND_PORT",
    "OMNI_HF_HOME",
    "OMNI_HF_ENDPOINT",
    "OMNI_HTTP_PROXY",
    "OMNI_HTTPS_PROXY",
    "OMNI_NO_PROXY",
    "OMNI_CA_BUNDLE",
)

OMLX_PROXY_SIDECAR_KEYS = (
    "OMLX_PROXY_PORT",
    "OMLX_PROXY_API_KEY",
    "OMLX_BACKEND_API_KEY",
    "OMLX_CONTEXT_SCALING",
    "OMLX_TARGET_CONTEXT_SIZE",
    "OMLX_SSE_KEEPALIVE_MODE",
    "OMLX_MODEL_SCAN",
    "OMLX_MODEL_SCAN_HOST_DIR",
    "OMLX_SAMPLING_MAX_TOKENS",
    "OMLX_SAMPLING_TEMPERATURE",
    "OMLX_SAMPLING_TOP_P",
    "OMLX_SAMPLING_TOP_K",
    "OMLX_SAMPLING_REPETITION_PENALTY",
)


@dataclass(frozen=True)
class SidecarBackendSpec:
    name: str
    service_name: str
    env_keys: tuple[str, ...]
    settings_cls: type
    settings_from_env: Callable[[Mapping[str, str]], Any]
    settings_from_overrides: Callable[[dict[str, Any]], Any]
    environment: Callable[[Any], dict[str, str]]
    default_environment: Callable[..., dict[str, str]]
    render_compose: Callable[..., str]
    render_compose_for_path: Callable[..., str]
    write_compose: Callable[..., Path]
    write_compose_for_path: Callable[..., Path]
    default_compose_name: str
    default_env_name: str


BACKEND_SPECS: dict[str, SidecarBackendSpec] = {}


def register_backend(spec: SidecarBackendSpec) -> None:
    BACKEND_SPECS[spec.name] = spec


def backend_spec(name: str) -> SidecarBackendSpec:
    # Import for the side effect of registering each backend's spec.
    from . import llamacpp_compose, vllm_compose  # noqa: F401

    return BACKEND_SPECS[name]


def common_environment(settings: CommonSidecarSettings) -> dict[str, str]:
    return {
        "OMNI_MODEL": settings.model,
        "OMNI_SERVED_MODEL_NAME": settings.served_model_name,
        "OMNI_CONTEXT_LENGTH": str(settings.context_length),
        "OMNI_MAX_PARALLEL": str(settings.max_parallel),
        "OMNI_BACKEND_PORT": str(settings.backend_port),
        "OMNI_HF_HOME": settings.hf_home,
        "OMNI_HF_ENDPOINT": settings.hf_endpoint,
        "OMNI_HTTP_PROXY": settings.http_proxy,
        "OMNI_HTTPS_PROXY": settings.https_proxy,
        "OMNI_NO_PROXY": settings.no_proxy,
        "OMNI_CA_BUNDLE": settings.ca_bundle,
    }


def proxy_sidecar_environment(settings: CommonSidecarSettings) -> dict[str, str]:
    return {
        "OMLX_PROXY_PORT": str(settings.proxy_port),
        "OMLX_PROXY_API_KEY": settings.proxy_api_key,
        "OMLX_BACKEND_API_KEY": settings.backend_api_key,
        "OMLX_CONTEXT_SCALING": _bool_str(settings.context_scaling),
        "OMLX_TARGET_CONTEXT_SIZE": str(settings.target_context_size),
        "OMLX_SSE_KEEPALIVE_MODE": settings.sse_keepalive_mode,
        "OMLX_MODEL_SCAN": _bool_str(settings.scan_models),
        "OMLX_MODEL_SCAN_HOST_DIR": settings.model_scan_host_dir,
        "OMLX_SAMPLING_MAX_TOKENS": str(settings.sampling_max_tokens),
        "OMLX_SAMPLING_TEMPERATURE": str(settings.sampling_temperature),
        "OMLX_SAMPLING_TOP_P": str(settings.sampling_top_p),
        "OMLX_SAMPLING_TOP_K": str(settings.sampling_top_k),
        "OMLX_SAMPLING_REPETITION_PENALTY": str(settings.sampling_repetition_penalty),
    }


def common_settings_kwargs_from_env(
    values: Mapping[str, str],
    defaults: CommonSidecarSettings,
) -> dict[str, Any]:
    return {
        "model": values.get("OMNI_MODEL", defaults.model),
        "served_model_name": values.get(
            "OMNI_SERVED_MODEL_NAME", defaults.served_model_name
        ),
        "context_length": _int_value(
            values.get("OMNI_CONTEXT_LENGTH"), defaults.context_length
        ),
        "max_parallel": _int_value(
            values.get("OMNI_MAX_PARALLEL"), defaults.max_parallel
        ),
        "backend_port": _int_value(
            values.get("OMNI_BACKEND_PORT"), defaults.backend_port
        ),
        "hf_home": values.get("OMNI_HF_HOME", defaults.hf_home),
        "hf_endpoint": values.get("OMNI_HF_ENDPOINT", defaults.hf_endpoint),
        "http_proxy": values.get("OMNI_HTTP_PROXY", defaults.http_proxy),
        "https_proxy": values.get("OMNI_HTTPS_PROXY", defaults.https_proxy),
        "no_proxy": values.get("OMNI_NO_PROXY", defaults.no_proxy),
        "ca_bundle": values.get("OMNI_CA_BUNDLE", defaults.ca_bundle),
        "proxy_port": _int_value(values.get("OMLX_PROXY_PORT"), defaults.proxy_port),
        "proxy_api_key": values.get("OMLX_PROXY_API_KEY", ""),
        "backend_api_key": values.get("OMLX_BACKEND_API_KEY", ""),
        "context_scaling": _bool_value(
            values.get("OMLX_CONTEXT_SCALING"), defaults.context_scaling
        ),
        "target_context_size": _int_value(
            values.get("OMLX_TARGET_CONTEXT_SIZE"), defaults.target_context_size
        ),
        "sse_keepalive_mode": values.get(
            "OMLX_SSE_KEEPALIVE_MODE", defaults.sse_keepalive_mode
        ),
        "scan_models": _bool_value(values.get("OMLX_MODEL_SCAN"), defaults.scan_models),
        "model_scan_host_dir": values.get(
            "OMLX_MODEL_SCAN_HOST_DIR", defaults.model_scan_host_dir
        ),
        "sampling_max_tokens": _int_value(
            values.get("OMLX_SAMPLING_MAX_TOKENS"), defaults.sampling_max_tokens
        ),
        "sampling_temperature": _float_value(
            values.get("OMLX_SAMPLING_TEMPERATURE"), defaults.sampling_temperature
        ),
        "sampling_top_p": _float_value(
            values.get("OMLX_SAMPLING_TOP_P"), defaults.sampling_top_p
        ),
        "sampling_top_k": _int_or_zero_value(
            values.get("OMLX_SAMPLING_TOP_K"), defaults.sampling_top_k
        ),
        "sampling_repetition_penalty": _float_value(
            values.get("OMLX_SAMPLING_REPETITION_PENALTY"),
            defaults.sampling_repetition_penalty,
        ),
    }


def common_settings_kwargs_from_overrides(
    overrides: dict[str, Any],
    defaults: CommonSidecarSettings,
) -> dict[str, Any]:
    def pick(name: str, default: Any) -> Any:
        return overrides.get(name, default)

    return {
        "model": str(pick("omni_model", defaults.model)).strip() or defaults.model,
        "served_model_name": str(
            pick("omni_served_model_name", defaults.served_model_name)
        ).strip()
        or defaults.served_model_name,
        "context_length": _positive_int(
            pick("omni_context_length", defaults.context_length),
            defaults.context_length,
        ),
        "max_parallel": _positive_int(
            pick("omni_max_parallel", defaults.max_parallel),
            defaults.max_parallel,
        ),
        "backend_port": _positive_int(
            pick("omni_backend_port", defaults.backend_port),
            defaults.backend_port,
        ),
        "hf_home": str(pick("omni_hf_home", defaults.hf_home)).strip()
        or defaults.hf_home,
        "hf_endpoint": str(pick("huggingface_endpoint", defaults.hf_endpoint)).strip(),
        "http_proxy": str(pick("network_http_proxy", defaults.http_proxy)).strip(),
        "https_proxy": str(pick("network_https_proxy", defaults.https_proxy)).strip(),
        "no_proxy": str(pick("network_no_proxy", defaults.no_proxy)).strip(),
        "ca_bundle": str(pick("network_ca_bundle", defaults.ca_bundle)).strip(),
        "proxy_port": _positive_int(
            pick("omlx_proxy_port", defaults.proxy_port),
            defaults.proxy_port,
        ),
        "proxy_api_key": str(pick("omlx_proxy_api_key", "")).strip(),
        "backend_api_key": str(pick("omlx_backend_api_key", "")).strip(),
        "context_scaling": _bool(
            pick("context_scaling_enabled", defaults.context_scaling),
            defaults.context_scaling,
        ),
        "target_context_size": _positive_int(
            pick("target_context_size", defaults.target_context_size),
            defaults.target_context_size,
        ),
        "sse_keepalive_mode": str(
            pick("omlx_sse_keepalive_mode", defaults.sse_keepalive_mode)
        ).strip()
        or defaults.sse_keepalive_mode,
        "sampling_max_tokens": _positive_int(
            pick("sampling_max_tokens", defaults.sampling_max_tokens),
            defaults.sampling_max_tokens,
        ),
        "sampling_temperature": _float(
            pick("sampling_temperature", defaults.sampling_temperature),
            defaults.sampling_temperature,
        ),
        "sampling_top_p": _float(
            pick("sampling_top_p", defaults.sampling_top_p),
            defaults.sampling_top_p,
        ),
        "sampling_top_k": _int(
            pick("sampling_top_k", defaults.sampling_top_k),
            defaults.sampling_top_k,
        ),
        "sampling_repetition_penalty": _float(
            pick("sampling_repetition_penalty", defaults.sampling_repetition_penalty),
            defaults.sampling_repetition_penalty,
        ),
    }


def load_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _unquote_env_value(value.strip())
    return values


def render_env_file(
    values: Mapping[str, str],
    keys: tuple[str, ...],
    *,
    include_header: bool = False,
) -> str:
    lines = []
    if include_header:
        lines.append(
            "# Generated by omni/admin. Edit with `omni serve` flags or the admin UI."
        )
    for key in keys:
        value = str(values.get(key, ""))
        if "\n" in value or "\r" in value:
            raise ValueError(f"Environment value for {key} cannot contain newlines")
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def write_env_file(
    path: str | os.PathLike[str],
    values: Mapping[str, str],
    keys: tuple[str, ...],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_env_file(values, keys, include_header=True), encoding="utf-8"
    )
    return output


def env_from_compose(
    path: str | os.PathLike[str],
    keys: tuple[str, ...],
) -> dict[str, str]:
    compose_path = Path(path)
    if not compose_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in compose_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        for key in keys:
            prefix = f"{key}:"
            if not stripped.startswith(prefix):
                continue
            raw_value = stripped[len(prefix) :].strip()
            parsed = _compose_default_value(key, raw_value)
            if parsed is not None:
                values[key] = parsed
    return values


def known_env(values: Mapping[str, str], keys: tuple[str, ...]) -> dict[str, str]:
    return {key: str(values[key]) for key in keys if key in values}


def render_proxy_service(
    settings: CommonSidecarSettings,
    *,
    backend_name: str,
    backend_service: str,
    compose_name: str,
    env_name: str,
    project_context: str,
    compose_output_dir: str,
    extra_scan_volumes: str = "",
) -> str:
    hf_default = _compose_default_expr("OMNI_HF_HOME", settings.hf_home)
    scan_source = "${OMLX_MODEL_SCAN_HOST_DIR:-" + hf_default + "}"
    return f"""  omlx-proxy:
    build:
      context: {_yaml_quote(project_context)}
      dockerfile: docker/Dockerfile.proxy
    ports:
      - "{_compose_default_expr('OMLX_PROXY_PORT', str(settings.proxy_port))}:8080"
    environment:
      OMLX_BACKEND_URL: "http://{backend_service}:8000/v1"
      OMLX_SIDECAR_BACKEND: "{backend_name}"
      OMLX_BACKEND_API_KEY: {_yaml_quote(_compose_default_expr('OMLX_BACKEND_API_KEY', settings.backend_api_key))}
      OMLX_PROXY_API_KEY: {_yaml_quote(_compose_default_expr('OMLX_PROXY_API_KEY', settings.proxy_api_key))}
      OMLX_PROXY_HOST: "0.0.0.0"
      OMLX_PROXY_PORT: "8080"
      OMLX_CONTEXT_SCALING: {_yaml_quote(_compose_default_expr('OMLX_CONTEXT_SCALING', _bool_str(settings.context_scaling)))}
      OMLX_TARGET_CONTEXT_SIZE: {_yaml_quote(_compose_default_expr('OMLX_TARGET_CONTEXT_SIZE', str(settings.target_context_size)))}
      OMLX_ACTUAL_CONTEXT_SIZE: {_yaml_quote(_compose_default_expr('OMNI_CONTEXT_LENGTH', str(settings.context_length)))}
      OMLX_SSE_KEEPALIVE_MODE: {_yaml_quote(_compose_default_expr('OMLX_SSE_KEEPALIVE_MODE', settings.sse_keepalive_mode))}
      OMLX_MODEL_SCAN: {_yaml_quote(_compose_default_expr('OMLX_MODEL_SCAN', _bool_str(settings.scan_models)))}
      OMLX_MODEL_SCAN_DIR: "/models-scan"
      OMLX_SAMPLING_MAX_TOKENS: {_yaml_quote(_compose_default_expr('OMLX_SAMPLING_MAX_TOKENS', str(settings.sampling_max_tokens)))}
      OMLX_SAMPLING_TEMPERATURE: {_yaml_quote(_compose_default_expr('OMLX_SAMPLING_TEMPERATURE', str(settings.sampling_temperature)))}
      OMLX_SAMPLING_TOP_P: {_yaml_quote(_compose_default_expr('OMLX_SAMPLING_TOP_P', str(settings.sampling_top_p)))}
      OMLX_SAMPLING_TOP_K: {_yaml_quote(_compose_default_expr('OMLX_SAMPLING_TOP_K', str(settings.sampling_top_k)))}
      OMLX_SAMPLING_REPETITION_PENALTY: {_yaml_quote(_compose_default_expr('OMLX_SAMPLING_REPETITION_PENALTY', str(settings.sampling_repetition_penalty)))}
      OMLX_PROXY_STATE_PATH: "/data/proxy-state.json"
      OMLX_COMPOSE_OUTPUT_PATH: "/compose-output/{compose_name}"
      OMLX_ENV_OUTPUT_PATH: "/compose-output/{env_name}"
      HTTP_PROXY: {_yaml_quote(_compose_default_expr('OMNI_HTTP_PROXY', settings.http_proxy))}
      HTTPS_PROXY: {_yaml_quote(_compose_default_expr('OMNI_HTTPS_PROXY', settings.https_proxy))}
      NO_PROXY: {_yaml_quote(_compose_default_expr('OMNI_NO_PROXY', settings.no_proxy))}
      REQUESTS_CA_BUNDLE: {_yaml_quote(_compose_default_expr('OMNI_CA_BUNDLE', settings.ca_bundle))}
      SSL_CERT_FILE: {_yaml_quote(_compose_default_expr('OMNI_CA_BUNDLE', settings.ca_bundle))}
    volumes:
      - proxy-state:/data
      - {_yaml_quote(f'{compose_output_dir}:/compose-output')}
      # Read-only host model caches for the opt-in local model scan
      # (omni serve --scan-models); inert while the feature is off.
      - {_yaml_quote(f'{scan_source}:/models-scan/hf:ro')}
{extra_scan_volumes}      # Grants the admin UI control of the sidecar container (restart
      # backend). Remove if you don't want the proxy to reach the Docker
      # daemon; the restart button then reports it is unavailable.
      - /var/run/docker.sock:/var/run/docker.sock
    depends_on:
      - {backend_service}
"""


def render_env_reload_snippet(env_name: str) -> str:
    """Shell lines that re-export the regenerated env file at container start.

    Compose bakes the env values in when the container is created; sourcing
    the bind-mounted env file on every start lets a plain container restart
    pick up settings saved from the admin UI.
    """
    return f"""        if [ -f "/compose-output/{env_name}" ]; then
          while IFS= read -r omni_env_line; do
            case "$$omni_env_line" in ''|'#'*) continue;; esac
            export "$$omni_env_line"
          done < "/compose-output/{env_name}"
        fi"""


def compose_paths_for_render(
    path: str | os.PathLike[str],
    repo_root: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    """Project context and compose-output dir, relative to a compose file path."""
    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    output_dir = Path(path).parent.resolve()
    project_context = _relative_path(root.resolve(), output_dir)
    compose_output_dir = _relative_path((root / "docker").resolve(), output_dir)
    return project_context, compose_output_dir


def _compose_default_value(key: str, value: str) -> str | None:
    value = _unquote_env_value(value)
    prefix = "${" + key + ":-"
    if not value.startswith(prefix) or not value.endswith("}"):
        return None
    parsed = value[len(prefix) : -1]
    return parsed.replace('\\"', '"')


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _int_value(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _float_value(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int_or_zero_value(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _bool_value(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _host_path(value: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(value)))


def _relative_path(target: Path, start: Path) -> str:
    rel = os.path.relpath(target, start)
    return "." if rel == "." else rel


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _positive_int(value: Any, default: int) -> int:
    parsed = _int(value, default)
    return parsed if parsed > 0 else default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    return _bool(value)


def _optional_bool_value(value: str | None, default: bool | None) -> bool | None:
    if value is None or value == "":
        return default
    return _bool_value(value, False)


def _optional_bool_str(value: bool | None) -> str:
    if value is None:
        return ""
    return _bool_str(value)


def _nonnegative_float(value: Any, default: float) -> float:
    parsed = _float(value, default)
    return parsed if parsed >= 0 else default


def _nonnegative_float_value(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _float_env_str(value: float) -> str:
    return "" if value == 0 else str(value)


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _compose_default_expr(name: str, value: Any) -> str:
    text = str(value)
    if not text:
        return "${" + name + ":-}"
    return "${" + name + ":-" + text + "}"


def _yaml_quote(value: Any) -> str:
    text = str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
