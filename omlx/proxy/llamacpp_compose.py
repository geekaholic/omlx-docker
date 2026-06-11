# SPDX-License-Identifier: Apache-2.0
'''Render Docker Compose configuration for the llama.cpp proxy sidecar.'''

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sidecar_compose import (
    OMLX_PROXY_SIDECAR_KEYS,
    OMNI_ENV_KEYS,
    CommonSidecarSettings,
    SidecarBackendSpec,
    _bool,
    _bool_str,
    _bool_value,
    _compose_default_expr,
    _host_path,
    _int,
    _int_or_zero_value,
    _yaml_quote,
    common_environment,
    common_settings_kwargs_from_env,
    common_settings_kwargs_from_overrides,
    compose_paths_for_render,
    env_from_compose,
    known_env,
    load_env_file,
    proxy_sidecar_environment,
    register_backend,
    render_env_file,
    render_env_reload_snippet,
    render_proxy_service,
    write_env_file,
)

DEFAULT_LLAMACPP_COMPOSE_NAME = "docker-compose.llamacpp.yml"
DEFAULT_LLAMACPP_ENV_NAME = "docker-compose.llamacpp.env"


@dataclass(frozen=True)
class LlamacppComposeSettings(CommonSidecarSettings):
    # OMNI_MODEL accepts a HF GGUF repo (owner/repo[:quantTag], passed as -hf)
    # or a *.gguf path (passed as -m; relative paths resolve in /models).
    model: str = "ggml-org/Qwen3-1.7B-GGUF:Q8_0"
    image: str = "ghcr.io/ggml-org/llama.cpp:server-cuda"
    n_gpu_layers: int = 999
    flash_attn: str = ""
    cache_type_k: str = ""
    cache_type_v: str = ""
    threads: str = ""
    batch_size: str = ""
    ubatch_size: str = ""
    jinja: bool = True
    reasoning_format: str = ""
    cache_dir: str = "${HOME}/.cache/llama.cpp"
    model_dir: str = ""
    extra_args: str = ""


LLAMACPP_SPECIFIC_KEYS = (
    "LLAMACPP_IMAGE",
    "LLAMACPP_N_GPU_LAYERS",
    "LLAMACPP_FLASH_ATTN",
    "LLAMACPP_CACHE_TYPE_K",
    "LLAMACPP_CACHE_TYPE_V",
    "LLAMACPP_THREADS",
    "LLAMACPP_BATCH_SIZE",
    "LLAMACPP_UBATCH_SIZE",
    "LLAMACPP_JINJA",
    "LLAMACPP_REASONING_FORMAT",
    "LLAMACPP_CACHE_DIR",
    "LLAMACPP_MODEL_DIR",
    "LLAMACPP_EXTRA_ARGS",
)

LLAMACPP_ENV_KEYS = OMNI_ENV_KEYS + LLAMACPP_SPECIFIC_KEYS + OMLX_PROXY_SIDECAR_KEYS


def llamacpp_settings_from_overrides(overrides: dict[str, Any]) -> LlamacppComposeSettings:
    defaults = LlamacppComposeSettings()

    def pick(name: str, default: Any) -> Any:
        return overrides.get(name, default)

    kwargs = common_settings_kwargs_from_overrides(overrides, defaults)
    kwargs.update(
        image=str(pick("llamacpp_image", defaults.image)).strip() or defaults.image,
        n_gpu_layers=_nonnegative_int(
            pick("llamacpp_n_gpu_layers", defaults.n_gpu_layers),
            defaults.n_gpu_layers,
        ),
        flash_attn=str(pick("llamacpp_flash_attn", defaults.flash_attn)).strip(),
        cache_type_k=str(pick("llamacpp_cache_type_k", defaults.cache_type_k)).strip(),
        cache_type_v=str(pick("llamacpp_cache_type_v", defaults.cache_type_v)).strip(),
        threads=str(pick("llamacpp_threads", defaults.threads)).strip(),
        batch_size=str(pick("llamacpp_batch_size", defaults.batch_size)).strip(),
        ubatch_size=str(pick("llamacpp_ubatch_size", defaults.ubatch_size)).strip(),
        jinja=_bool(pick("llamacpp_jinja", defaults.jinja), defaults.jinja),
        reasoning_format=str(
            pick("llamacpp_reasoning_format", defaults.reasoning_format)
        ).strip(),
        cache_dir=str(pick("llamacpp_cache_dir", defaults.cache_dir)).strip()
        or defaults.cache_dir,
        model_dir=str(pick("llamacpp_model_dir", defaults.model_dir)).strip(),
        extra_args=str(pick("llamacpp_extra_args", defaults.extra_args)).strip(),
    )
    return LlamacppComposeSettings(**kwargs)


def llamacpp_environment(settings: LlamacppComposeSettings) -> dict[str, str]:
    return {
        **common_environment(settings),
        "LLAMACPP_IMAGE": settings.image,
        "LLAMACPP_N_GPU_LAYERS": str(settings.n_gpu_layers),
        "LLAMACPP_FLASH_ATTN": settings.flash_attn,
        "LLAMACPP_CACHE_TYPE_K": settings.cache_type_k,
        "LLAMACPP_CACHE_TYPE_V": settings.cache_type_v,
        "LLAMACPP_THREADS": settings.threads,
        "LLAMACPP_BATCH_SIZE": settings.batch_size,
        "LLAMACPP_UBATCH_SIZE": settings.ubatch_size,
        "LLAMACPP_JINJA": _bool_str(settings.jinja),
        "LLAMACPP_REASONING_FORMAT": settings.reasoning_format,
        "LLAMACPP_CACHE_DIR": settings.cache_dir,
        "LLAMACPP_MODEL_DIR": settings.model_dir,
        "LLAMACPP_EXTRA_ARGS": settings.extra_args,
        **proxy_sidecar_environment(settings),
    }


def default_llamacpp_environment(*, expand_hf_home: bool = False) -> dict[str, str]:
    defaults = LlamacppComposeSettings()
    settings = defaults
    if expand_hf_home:
        settings = LlamacppComposeSettings(
            hf_home=_host_path(defaults.hf_home),
            cache_dir=_host_path(defaults.cache_dir),
        )
    return llamacpp_environment(settings)


def llamacpp_settings_from_env(values: Mapping[str, str]) -> LlamacppComposeSettings:
    defaults = LlamacppComposeSettings()
    kwargs = common_settings_kwargs_from_env(values, defaults)
    kwargs.update(
        image=values.get("LLAMACPP_IMAGE", defaults.image),
        n_gpu_layers=_int_or_zero_value(
            values.get("LLAMACPP_N_GPU_LAYERS"),
            defaults.n_gpu_layers,
        ),
        flash_attn=values.get("LLAMACPP_FLASH_ATTN", defaults.flash_attn),
        cache_type_k=values.get("LLAMACPP_CACHE_TYPE_K", defaults.cache_type_k),
        cache_type_v=values.get("LLAMACPP_CACHE_TYPE_V", defaults.cache_type_v),
        threads=values.get("LLAMACPP_THREADS", defaults.threads),
        batch_size=values.get("LLAMACPP_BATCH_SIZE", defaults.batch_size),
        ubatch_size=values.get("LLAMACPP_UBATCH_SIZE", defaults.ubatch_size),
        jinja=_bool_value(values.get("LLAMACPP_JINJA"), defaults.jinja),
        reasoning_format=values.get(
            "LLAMACPP_REASONING_FORMAT", defaults.reasoning_format
        ),
        cache_dir=values.get("LLAMACPP_CACHE_DIR", defaults.cache_dir),
        model_dir=values.get("LLAMACPP_MODEL_DIR", defaults.model_dir),
        extra_args=values.get("LLAMACPP_EXTRA_ARGS", defaults.extra_args),
    )
    return LlamacppComposeSettings(**kwargs)


def load_llamacpp_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    return load_env_file(path)


def write_llamacpp_env_file(
    path: str | os.PathLike[str],
    values: Mapping[str, str],
) -> Path:
    return write_env_file(path, values, LLAMACPP_ENV_KEYS)


def render_llamacpp_env_file(
    values: Mapping[str, str],
    *,
    include_header: bool = False,
) -> str:
    return render_env_file(values, LLAMACPP_ENV_KEYS, include_header=include_header)


def llamacpp_env_from_compose(path: str | os.PathLike[str]) -> dict[str, str]:
    return env_from_compose(path, LLAMACPP_ENV_KEYS)


def known_llamacpp_env(values: Mapping[str, str]) -> dict[str, str]:
    return known_env(values, LLAMACPP_ENV_KEYS)


def render_llamacpp_compose(
    settings: LlamacppComposeSettings,
    *,
    project_context: str = "..",
    compose_output_dir: str = "../docker",
) -> str:
    env = llamacpp_environment(settings)
    env_lines = "\n".join(
        f"      {key}: {_yaml_quote(_compose_default_expr(key, value))}"
        for key, value in env.items()
    )
    proxy_service = render_proxy_service(
        settings,
        backend_name="llamacpp",
        backend_service="llamacpp",
        compose_name=DEFAULT_LLAMACPP_COMPOSE_NAME,
        env_name=DEFAULT_LLAMACPP_ENV_NAME,
        project_context=project_context,
        compose_output_dir=compose_output_dir,
    )
    env_reload = render_env_reload_snippet(DEFAULT_LLAMACPP_ENV_NAME)
    cache_dir_expr = _compose_default_expr("LLAMACPP_CACHE_DIR", settings.cache_dir)
    model_dir_expr = _compose_default_expr(
        "LLAMACPP_MODEL_DIR",
        settings.model_dir or cache_dir_expr,
    )
    return f'''# Generated by oMNI proxy admin. Edit docker-compose.llamacpp.template.yml or
# admin proxy settings, then regenerate instead of hand-editing this file.
services:
{proxy_service}
  llamacpp:
    image: {_yaml_quote(_compose_default_expr('LLAMACPP_IMAGE', settings.image))}
    ports:
      - "{_compose_default_expr('OMNI_BACKEND_PORT', str(settings.backend_port))}:8000"
    gpus: all
    environment:
{env_lines}
      HF_TOKEN: "${{HF_TOKEN:-}}"
      LLAMA_CACHE: "/root/.cache/llama.cpp"
      HF_HOME: "/root/.cache/huggingface"
    volumes:
      - "{_compose_default_expr('OMNI_HF_HOME', settings.hf_home)}:/root/.cache/huggingface"
      - "{cache_dir_expr}:/root/.cache/llama.cpp"
      - "{model_dir_expr}:/models:ro"
      - {_yaml_quote(f'{compose_output_dir}:/compose-output:ro')}
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://127.0.0.1:8000/health"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 300s
    entrypoint: ["/bin/sh", "-lc"]
    command:
      - |
        set -eu
{env_reload}
        if [ -n "$${{OMNI_HF_ENDPOINT:-}}" ]; then
          export HF_ENDPOINT="$${{OMNI_HF_ENDPOINT}}"
          export MODEL_ENDPOINT="$${{OMNI_HF_ENDPOINT}}"
        else
          unset HF_ENDPOINT
          unset MODEL_ENDPOINT
        fi
        if [ -n "$${{OMNI_HTTP_PROXY:-}}" ]; then
          export HTTP_PROXY="$${{OMNI_HTTP_PROXY}}"
          export http_proxy="$${{OMNI_HTTP_PROXY}}"
        else
          unset HTTP_PROXY http_proxy
        fi
        if [ -n "$${{OMNI_HTTPS_PROXY:-}}" ]; then
          export HTTPS_PROXY="$${{OMNI_HTTPS_PROXY}}"
          export https_proxy="$${{OMNI_HTTPS_PROXY}}"
        else
          unset HTTPS_PROXY https_proxy
        fi
        if [ -n "$${{OMNI_NO_PROXY:-}}" ]; then
          export NO_PROXY="$${{OMNI_NO_PROXY}}"
          export no_proxy="$${{OMNI_NO_PROXY}}"
        else
          unset NO_PROXY no_proxy
        fi
        if [ -n "$${{OMNI_CA_BUNDLE:-}}" ]; then
          export CURL_CA_BUNDLE="$${{OMNI_CA_BUNDLE}}"
          export SSL_CERT_FILE="$${{OMNI_CA_BUNDLE}}"
        else
          unset CURL_CA_BUNDLE
          unset SSL_CERT_FILE
        fi
        model="$${{OMNI_MODEL:?Set OMNI_MODEL to a GGUF repo (owner/repo[:quantTag]) or a .gguf path}}"
        case "$$model" in
          *.gguf)
            case "$$model" in
              /*) set -- -m "$$model" ;;
              *) set -- -m "/models/$$model" ;;
            esac
            ;;
          *)
            set -- -hf "$$model"
            ;;
        esac
        set -- "$${{@}}" --host 0.0.0.0
        set -- "$${{@}}" --port 8000
        set -- "$${{@}}" --alias "$${{OMNI_SERVED_MODEL_NAME:-$$model}}"
        set -- "$${{@}}" --ctx-size "$${{OMNI_CONTEXT_LENGTH:-8192}}"
        set -- "$${{@}}" --parallel "$${{OMNI_MAX_PARALLEL:-4}}"
        set -- "$${{@}}" --n-gpu-layers "$${{LLAMACPP_N_GPU_LAYERS:-999}}"
        if [ "$${{LLAMACPP_JINJA:-true}}" = "true" ]; then
          set -- "$${{@}}" --jinja
        fi
        if [ -n "$${{LLAMACPP_FLASH_ATTN:-}}" ]; then
          set -- "$${{@}}" --flash-attn "$${{LLAMACPP_FLASH_ATTN}}"
        fi
        if [ -n "$${{LLAMACPP_CACHE_TYPE_K:-}}" ]; then
          set -- "$${{@}}" --cache-type-k "$${{LLAMACPP_CACHE_TYPE_K}}"
        fi
        if [ -n "$${{LLAMACPP_CACHE_TYPE_V:-}}" ]; then
          set -- "$${{@}}" --cache-type-v "$${{LLAMACPP_CACHE_TYPE_V}}"
        fi
        if [ -n "$${{LLAMACPP_THREADS:-}}" ]; then
          set -- "$${{@}}" --threads "$${{LLAMACPP_THREADS}}"
        fi
        if [ -n "$${{LLAMACPP_BATCH_SIZE:-}}" ]; then
          set -- "$${{@}}" --batch-size "$${{LLAMACPP_BATCH_SIZE}}"
        fi
        if [ -n "$${{LLAMACPP_UBATCH_SIZE:-}}" ]; then
          set -- "$${{@}}" --ubatch-size "$${{LLAMACPP_UBATCH_SIZE}}"
        fi
        if [ -n "$${{LLAMACPP_REASONING_FORMAT:-}}" ]; then
          set -- "$${{@}}" --reasoning-format "$${{LLAMACPP_REASONING_FORMAT}}"
        fi
        if [ -n "$${{OMLX_BACKEND_API_KEY:-}}" ]; then
          set -- "$${{@}}" --api-key "$${{OMLX_BACKEND_API_KEY}}"
        fi
        if [ -n "$${{LLAMACPP_EXTRA_ARGS:-}}" ]; then
          set -- "$${{@}}" $$LLAMACPP_EXTRA_ARGS
        fi
        unset OMNI_MODEL OMNI_SERVED_MODEL_NAME OMNI_CONTEXT_LENGTH OMNI_MAX_PARALLEL OMNI_BACKEND_PORT OMNI_HF_HOME OMNI_HF_ENDPOINT OMNI_HTTP_PROXY OMNI_HTTPS_PROXY OMNI_NO_PROXY OMNI_CA_BUNDLE
        unset LLAMACPP_IMAGE LLAMACPP_N_GPU_LAYERS LLAMACPP_FLASH_ATTN LLAMACPP_CACHE_TYPE_K LLAMACPP_CACHE_TYPE_V LLAMACPP_THREADS LLAMACPP_BATCH_SIZE LLAMACPP_UBATCH_SIZE LLAMACPP_JINJA LLAMACPP_REASONING_FORMAT LLAMACPP_CACHE_DIR LLAMACPP_MODEL_DIR LLAMACPP_EXTRA_ARGS
        unset OMLX_PROXY_PORT OMLX_PROXY_API_KEY OMLX_BACKEND_API_KEY OMLX_CONTEXT_SCALING OMLX_TARGET_CONTEXT_SIZE OMLX_SSE_KEEPALIVE_MODE OMLX_SAMPLING_MAX_TOKENS OMLX_SAMPLING_TEMPERATURE OMLX_SAMPLING_TOP_P OMLX_SAMPLING_TOP_K OMLX_SAMPLING_REPETITION_PENALTY
        if command -v llama-server >/dev/null 2>&1; then
          exec llama-server "$${{@}}"
        fi
        exec /app/llama-server "$${{@}}"

volumes:
  proxy-state:
'''


def render_llamacpp_compose_for_path(
    path: str | os.PathLike[str],
    settings: LlamacppComposeSettings,
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> str:
    project_context, compose_output_dir = compose_paths_for_render(path, repo_root)
    return render_llamacpp_compose(
        settings,
        project_context=project_context,
        compose_output_dir=compose_output_dir,
    )


def write_llamacpp_compose(
    path: str | os.PathLike[str],
    settings: LlamacppComposeSettings,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_llamacpp_compose(settings), encoding="utf-8")
    return output


def write_llamacpp_compose_for_path(
    path: str | os.PathLike[str],
    settings: LlamacppComposeSettings,
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_llamacpp_compose_for_path(output, settings, repo_root=repo_root),
        encoding="utf-8",
    )
    return output


def _nonnegative_int(value: Any, default: int) -> int:
    parsed = _int(value, default)
    return parsed if parsed >= 0 else default


register_backend(
    SidecarBackendSpec(
        name="llamacpp",
        service_name="llamacpp",
        env_keys=LLAMACPP_ENV_KEYS,
        settings_cls=LlamacppComposeSettings,
        settings_from_env=llamacpp_settings_from_env,
        settings_from_overrides=llamacpp_settings_from_overrides,
        environment=llamacpp_environment,
        default_environment=default_llamacpp_environment,
        render_compose=render_llamacpp_compose,
        render_compose_for_path=render_llamacpp_compose_for_path,
        write_compose=write_llamacpp_compose,
        write_compose_for_path=write_llamacpp_compose_for_path,
        default_compose_name=DEFAULT_LLAMACPP_COMPOSE_NAME,
        default_env_name=DEFAULT_LLAMACPP_ENV_NAME,
    )
)
