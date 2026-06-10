# SPDX-License-Identifier: Apache-2.0
'''Render Docker Compose configuration for the vLLM proxy sidecar.'''

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .sidecar_compose import (
    OMLX_PROXY_SIDECAR_KEYS,
    OMNI_ENV_KEYS,
    CommonSidecarSettings,
    SidecarBackendSpec,
    _bool,
    _bool_str,
    _bool_value,
    _compose_default_expr,
    _float,
    _float_env_str,
    _float_value,
    _host_path,
    _int_value,
    _nonnegative_float,
    _nonnegative_float_value,
    _optional_bool,
    _optional_bool_str,
    _optional_bool_value,
    _positive_int,
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
    render_proxy_service,
    write_env_file,
)

DEFAULT_VLLM_COMPOSE_NAME = "docker-compose.vllm.yml"
DEFAULT_VLLM_ENV_NAME = "docker-compose.vllm.env"


@dataclass(frozen=True)
class VllmComposeSettings(CommonSidecarSettings):
    model: str = "Qwen/Qwen3-1.7B"
    image: str = "vllm/vllm-openai:latest"
    gpu_memory_utilization: float = 0.80
    generation_config: str = "vllm"
    default_chat_template_kwargs: str = '{"enable_thinking":false}'
    trust_remote_code: bool = True
    enforce_eager: bool = False
    enable_auto_tool_choice: bool = False
    tool_call_parser: str = "hermes"
    reasoning_parser: str = ""
    dtype: str = ""
    tokenizer: str = ""
    tokenizer_mode: str = ""
    revision: str = ""
    load_format: str = ""
    quantization: str = ""
    download_dir: str = ""
    max_num_batched_tokens: str = ""
    enable_chunked_prefill: bool | None = None
    enable_prefix_caching: bool | None = None
    kv_cache_dtype: str = ""
    cpu_offload_gb: float = 0.0
    swap_space: float = 0.0
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    uvicorn_log_level: str = ""
    disable_log_stats: bool = False
    extra_args_json: str = "[]"


VLLM_SPECIFIC_KEYS = (
    "VLLM_IMAGE",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_GENERATION_CONFIG",
    "VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS",
    "VLLM_TRUST_REMOTE_CODE",
    "VLLM_ENFORCE_EAGER",
    "VLLM_ENABLE_AUTO_TOOL_CHOICE",
    "VLLM_TOOL_CALL_PARSER",
    "VLLM_REASONING_PARSER",
    "VLLM_DTYPE",
    "VLLM_TOKENIZER",
    "VLLM_TOKENIZER_MODE",
    "VLLM_REVISION",
    "VLLM_LOAD_FORMAT",
    "VLLM_QUANTIZATION",
    "VLLM_DOWNLOAD_DIR",
    "VLLM_MAX_NUM_BATCHED_TOKENS",
    "VLLM_ENABLE_CHUNKED_PREFILL",
    "VLLM_ENABLE_PREFIX_CACHING",
    "VLLM_KV_CACHE_DTYPE",
    "VLLM_CPU_OFFLOAD_GB",
    "VLLM_SWAP_SPACE",
    "VLLM_TENSOR_PARALLEL_SIZE",
    "VLLM_PIPELINE_PARALLEL_SIZE",
    "VLLM_UVICORN_LOG_LEVEL",
    "VLLM_DISABLE_LOG_STATS",
    "VLLM_EXTRA_ARGS_JSON",
)

VLLM_ENV_KEYS = OMNI_ENV_KEYS + VLLM_SPECIFIC_KEYS + OMLX_PROXY_SIDECAR_KEYS


def settings_from_overrides(overrides: dict[str, Any]) -> VllmComposeSettings:
    defaults = VllmComposeSettings()

    def pick(name: str, default: Any) -> Any:
        return overrides.get(name, default)

    kwargs = common_settings_kwargs_from_overrides(overrides, defaults)
    kwargs.update(
        image=str(pick("vllm_image", defaults.image)).strip() or defaults.image,
        gpu_memory_utilization=_float(
            pick("vllm_gpu_memory_utilization", defaults.gpu_memory_utilization),
            defaults.gpu_memory_utilization,
        ),
        generation_config=str(
            pick("vllm_generation_config", defaults.generation_config)
        ).strip()
        or defaults.generation_config,
        default_chat_template_kwargs=str(
            pick(
                "vllm_default_chat_template_kwargs",
                defaults.default_chat_template_kwargs,
            )
        ).strip(),
        trust_remote_code=_bool(
            pick("vllm_trust_remote_code", defaults.trust_remote_code),
            defaults.trust_remote_code,
        ),
        enforce_eager=_bool(
            pick("vllm_enforce_eager", defaults.enforce_eager),
            defaults.enforce_eager,
        ),
        enable_auto_tool_choice=_bool(
            pick("vllm_enable_auto_tool_choice", defaults.enable_auto_tool_choice),
            defaults.enable_auto_tool_choice,
        ),
        tool_call_parser=str(
            pick("vllm_tool_call_parser", defaults.tool_call_parser)
        ).strip(),
        reasoning_parser=str(
            pick("vllm_reasoning_parser", defaults.reasoning_parser)
        ).strip(),
        dtype=str(pick("vllm_dtype", defaults.dtype)).strip(),
        tokenizer=str(pick("vllm_tokenizer", defaults.tokenizer)).strip(),
        tokenizer_mode=str(pick("vllm_tokenizer_mode", defaults.tokenizer_mode)).strip(),
        revision=str(pick("vllm_revision", defaults.revision)).strip(),
        load_format=str(pick("vllm_load_format", defaults.load_format)).strip(),
        quantization=str(pick("vllm_quantization", defaults.quantization)).strip(),
        download_dir=str(pick("vllm_download_dir", defaults.download_dir)).strip(),
        max_num_batched_tokens=str(
            pick("vllm_max_num_batched_tokens", defaults.max_num_batched_tokens)
        ).strip(),
        enable_chunked_prefill=_optional_bool(
            pick("vllm_enable_chunked_prefill", defaults.enable_chunked_prefill)
        ),
        enable_prefix_caching=_optional_bool(
            pick("vllm_enable_prefix_caching", defaults.enable_prefix_caching)
        ),
        kv_cache_dtype=str(pick("vllm_kv_cache_dtype", defaults.kv_cache_dtype)).strip(),
        cpu_offload_gb=_nonnegative_float(
            pick("vllm_cpu_offload_gb", defaults.cpu_offload_gb),
            defaults.cpu_offload_gb,
        ),
        swap_space=_nonnegative_float(
            pick("vllm_swap_space", defaults.swap_space),
            defaults.swap_space,
        ),
        tensor_parallel_size=_positive_int(
            pick("vllm_tensor_parallel_size", defaults.tensor_parallel_size),
            defaults.tensor_parallel_size,
        ),
        pipeline_parallel_size=_positive_int(
            pick("vllm_pipeline_parallel_size", defaults.pipeline_parallel_size),
            defaults.pipeline_parallel_size,
        ),
        uvicorn_log_level=str(
            pick("vllm_uvicorn_log_level", defaults.uvicorn_log_level)
        ).strip(),
        disable_log_stats=_bool(
            pick("vllm_disable_log_stats", defaults.disable_log_stats),
            defaults.disable_log_stats,
        ),
        extra_args_json=str(
            pick("vllm_extra_args_json", defaults.extra_args_json)
        ).strip()
        or defaults.extra_args_json,
    )
    return VllmComposeSettings(**kwargs)


def vllm_environment(settings: VllmComposeSettings) -> dict[str, str]:
    return {
        **common_environment(settings),
        "VLLM_IMAGE": settings.image,
        "VLLM_GPU_MEMORY_UTILIZATION": str(settings.gpu_memory_utilization),
        "VLLM_GENERATION_CONFIG": settings.generation_config,
        "VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS": settings.default_chat_template_kwargs,
        "VLLM_TRUST_REMOTE_CODE": _bool_str(settings.trust_remote_code),
        "VLLM_ENFORCE_EAGER": _bool_str(settings.enforce_eager),
        "VLLM_ENABLE_AUTO_TOOL_CHOICE": _bool_str(settings.enable_auto_tool_choice),
        "VLLM_TOOL_CALL_PARSER": settings.tool_call_parser,
        "VLLM_REASONING_PARSER": settings.reasoning_parser,
        "VLLM_DTYPE": settings.dtype,
        "VLLM_TOKENIZER": settings.tokenizer,
        "VLLM_TOKENIZER_MODE": settings.tokenizer_mode,
        "VLLM_REVISION": settings.revision,
        "VLLM_LOAD_FORMAT": settings.load_format,
        "VLLM_QUANTIZATION": settings.quantization,
        "VLLM_DOWNLOAD_DIR": settings.download_dir,
        "VLLM_MAX_NUM_BATCHED_TOKENS": settings.max_num_batched_tokens,
        "VLLM_ENABLE_CHUNKED_PREFILL": _optional_bool_str(settings.enable_chunked_prefill),
        "VLLM_ENABLE_PREFIX_CACHING": _optional_bool_str(settings.enable_prefix_caching),
        "VLLM_KV_CACHE_DTYPE": settings.kv_cache_dtype,
        "VLLM_CPU_OFFLOAD_GB": _float_env_str(settings.cpu_offload_gb),
        "VLLM_SWAP_SPACE": _float_env_str(settings.swap_space),
        "VLLM_TENSOR_PARALLEL_SIZE": str(settings.tensor_parallel_size),
        "VLLM_PIPELINE_PARALLEL_SIZE": str(settings.pipeline_parallel_size),
        "VLLM_UVICORN_LOG_LEVEL": settings.uvicorn_log_level,
        "VLLM_DISABLE_LOG_STATS": _bool_str(settings.disable_log_stats),
        "VLLM_EXTRA_ARGS_JSON": settings.extra_args_json,
        **proxy_sidecar_environment(settings),
    }


def default_vllm_environment(*, expand_hf_home: bool = False) -> dict[str, str]:
    defaults = VllmComposeSettings()
    settings = defaults
    if expand_hf_home:
        settings = VllmComposeSettings(hf_home=_host_path(defaults.hf_home))
    return vllm_environment(settings)


def vllm_settings_from_env(values: Mapping[str, str]) -> VllmComposeSettings:
    defaults = VllmComposeSettings()
    kwargs = common_settings_kwargs_from_env(values, defaults)
    kwargs.update(
        image=values.get("VLLM_IMAGE", defaults.image),
        gpu_memory_utilization=_float_value(
            values.get("VLLM_GPU_MEMORY_UTILIZATION"),
            defaults.gpu_memory_utilization,
        ),
        generation_config=values.get("VLLM_GENERATION_CONFIG", defaults.generation_config),
        default_chat_template_kwargs=values.get(
            "VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS",
            defaults.default_chat_template_kwargs,
        ),
        trust_remote_code=_bool_value(
            values.get("VLLM_TRUST_REMOTE_CODE"),
            defaults.trust_remote_code,
        ),
        enforce_eager=_bool_value(
            values.get("VLLM_ENFORCE_EAGER"),
            defaults.enforce_eager,
        ),
        enable_auto_tool_choice=_bool_value(
            values.get("VLLM_ENABLE_AUTO_TOOL_CHOICE"),
            defaults.enable_auto_tool_choice,
        ),
        tool_call_parser=values.get("VLLM_TOOL_CALL_PARSER", defaults.tool_call_parser),
        reasoning_parser=values.get("VLLM_REASONING_PARSER", defaults.reasoning_parser),
        dtype=values.get("VLLM_DTYPE", defaults.dtype),
        tokenizer=values.get("VLLM_TOKENIZER", defaults.tokenizer),
        tokenizer_mode=values.get("VLLM_TOKENIZER_MODE", defaults.tokenizer_mode),
        revision=values.get("VLLM_REVISION", defaults.revision),
        load_format=values.get("VLLM_LOAD_FORMAT", defaults.load_format),
        quantization=values.get("VLLM_QUANTIZATION", defaults.quantization),
        download_dir=values.get("VLLM_DOWNLOAD_DIR", defaults.download_dir),
        max_num_batched_tokens=values.get(
            "VLLM_MAX_NUM_BATCHED_TOKENS",
            defaults.max_num_batched_tokens,
        ),
        enable_chunked_prefill=_optional_bool_value(
            values.get("VLLM_ENABLE_CHUNKED_PREFILL"),
            defaults.enable_chunked_prefill,
        ),
        enable_prefix_caching=_optional_bool_value(
            values.get("VLLM_ENABLE_PREFIX_CACHING"),
            defaults.enable_prefix_caching,
        ),
        kv_cache_dtype=values.get("VLLM_KV_CACHE_DTYPE", defaults.kv_cache_dtype),
        cpu_offload_gb=_nonnegative_float_value(
            values.get("VLLM_CPU_OFFLOAD_GB"),
            defaults.cpu_offload_gb,
        ),
        swap_space=_nonnegative_float_value(
            values.get("VLLM_SWAP_SPACE"),
            defaults.swap_space,
        ),
        tensor_parallel_size=_int_value(
            values.get("VLLM_TENSOR_PARALLEL_SIZE"),
            defaults.tensor_parallel_size,
        ),
        pipeline_parallel_size=_int_value(
            values.get("VLLM_PIPELINE_PARALLEL_SIZE"),
            defaults.pipeline_parallel_size,
        ),
        uvicorn_log_level=values.get("VLLM_UVICORN_LOG_LEVEL", defaults.uvicorn_log_level),
        disable_log_stats=_bool_value(
            values.get("VLLM_DISABLE_LOG_STATS"),
            defaults.disable_log_stats,
        ),
        extra_args_json=values.get("VLLM_EXTRA_ARGS_JSON", defaults.extra_args_json),
    )
    return VllmComposeSettings(**kwargs)


def load_vllm_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    return load_env_file(path)


def write_vllm_env_file(path: str | os.PathLike[str], values: Mapping[str, str]) -> Path:
    return write_env_file(path, values, VLLM_ENV_KEYS)


def render_vllm_env_file(
    values: Mapping[str, str],
    *,
    include_header: bool = False,
) -> str:
    return render_env_file(values, VLLM_ENV_KEYS, include_header=include_header)


def vllm_env_from_compose(path: str | os.PathLike[str]) -> dict[str, str]:
    return env_from_compose(path, VLLM_ENV_KEYS)


def known_vllm_env(values: Mapping[str, str]) -> dict[str, str]:
    return known_env(values, VLLM_ENV_KEYS)


def render_vllm_compose(
    settings: VllmComposeSettings,
    *,
    project_context: str = "..",
    compose_output_dir: str = "../docker",
) -> str:
    env = vllm_environment(settings)
    env_lines = "\n".join(
        f"      {key}: {_yaml_quote(_compose_default_expr(key, value))}"
        for key, value in env.items()
    )
    proxy_service = render_proxy_service(
        settings,
        backend_name="vllm",
        backend_service="vllm",
        compose_name=DEFAULT_VLLM_COMPOSE_NAME,
        env_name=DEFAULT_VLLM_ENV_NAME,
        project_context=project_context,
        compose_output_dir=compose_output_dir,
    )
    return f'''# Generated by oMNI proxy admin. Edit docker-compose.vllm.template.yml or
# admin proxy settings, then regenerate instead of hand-editing this file.
services:
{proxy_service}
  vllm:
    image: {_yaml_quote(_compose_default_expr('VLLM_IMAGE', settings.image))}
    ports:
      - "{_compose_default_expr('OMNI_BACKEND_PORT', str(settings.backend_port))}:8000"
    ipc: host
    gpus: all
    environment:
{env_lines}
      HF_TOKEN: "${{HF_TOKEN:-}}"
      HUGGING_FACE_HUB_TOKEN: "${{HUGGING_FACE_HUB_TOKEN:-${{HF_TOKEN:-}}}}"
      HF_HOME: "/root/.cache/huggingface"
    volumes:
      - "{_compose_default_expr('OMNI_HF_HOME', settings.hf_home)}:/root/.cache/huggingface"
    entrypoint: ["/bin/sh", "-lc"]
    command:
      - |
        set -eu
        if [ -n "$${{OMNI_HF_ENDPOINT:-}}" ]; then
          export HF_ENDPOINT="$${{OMNI_HF_ENDPOINT}}"
        else
          unset HF_ENDPOINT
        fi
        if [ -n "$${{OMNI_HTTP_PROXY:-}}" ]; then
          export HTTP_PROXY="$${{OMNI_HTTP_PROXY}}"
        else
          unset HTTP_PROXY
        fi
        if [ -n "$${{OMNI_HTTPS_PROXY:-}}" ]; then
          export HTTPS_PROXY="$${{OMNI_HTTPS_PROXY}}"
        else
          unset HTTPS_PROXY
        fi
        if [ -n "$${{OMNI_NO_PROXY:-}}" ]; then
          export NO_PROXY="$${{OMNI_NO_PROXY}}"
        else
          unset NO_PROXY
        fi
        if [ -n "$${{OMNI_CA_BUNDLE:-}}" ]; then
          export REQUESTS_CA_BUNDLE="$${{OMNI_CA_BUNDLE}}"
          export SSL_CERT_FILE="$${{OMNI_CA_BUNDLE}}"
        else
          unset REQUESTS_CA_BUNDLE
          unset SSL_CERT_FILE
        fi
        set -- "$${{OMNI_MODEL:?Set OMNI_MODEL to a Hugging Face model id or local container path}}"
        set -- "$${{@}}" --host 0.0.0.0
        set -- "$${{@}}" --port 8000
        set -- "$${{@}}" --served-model-name "$${{OMNI_SERVED_MODEL_NAME:-$${{OMNI_MODEL}}}}"
        set -- "$${{@}}" --max-model-len "$${{OMNI_CONTEXT_LENGTH:-8192}}"
        set -- "$${{@}}" --gpu-memory-utilization "$${{VLLM_GPU_MEMORY_UTILIZATION:-0.80}}"
        set -- "$${{@}}" --max-num-seqs "$${{OMNI_MAX_PARALLEL:-4}}"
        set -- "$${{@}}" --generation-config "$${{VLLM_GENERATION_CONFIG:-vllm}}"
        if [ -n "$${{VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}}" ]; then
          set -- "$${{@}}" --default-chat-template-kwargs "$${{VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS}}"
        fi
        if [ "$${{VLLM_TRUST_REMOTE_CODE:-true}}" = "true" ]; then
          set -- "$${{@}}" --trust-remote-code
        fi
        if [ "$${{VLLM_ENFORCE_EAGER:-false}}" = "true" ]; then
          set -- "$${{@}}" --enforce-eager
        fi
        if [ "$${{VLLM_ENABLE_AUTO_TOOL_CHOICE:-false}}" = "true" ]; then
          set -- "$${{@}}" --enable-auto-tool-choice
          if [ -n "$${{VLLM_TOOL_CALL_PARSER:-}}" ]; then
            set -- "$${{@}}" --tool-call-parser "$${{VLLM_TOOL_CALL_PARSER}}"
          fi
        fi
        if [ -n "$${{VLLM_REASONING_PARSER:-}}" ]; then
          set -- "$${{@}}" --reasoning-parser "$${{VLLM_REASONING_PARSER}}"
        fi
        if [ -n "$${{VLLM_DTYPE:-}}" ]; then
          set -- "$${{@}}" --dtype "$${{VLLM_DTYPE}}"
        fi
        if [ -n "$${{VLLM_TOKENIZER:-}}" ]; then
          set -- "$${{@}}" --tokenizer "$${{VLLM_TOKENIZER}}"
        fi
        if [ -n "$${{VLLM_TOKENIZER_MODE:-}}" ]; then
          set -- "$${{@}}" --tokenizer-mode "$${{VLLM_TOKENIZER_MODE}}"
        fi
        if [ -n "$${{VLLM_REVISION:-}}" ]; then
          set -- "$${{@}}" --revision "$${{VLLM_REVISION}}"
        fi
        if [ -n "$${{VLLM_LOAD_FORMAT:-}}" ]; then
          set -- "$${{@}}" --load-format "$${{VLLM_LOAD_FORMAT}}"
        fi
        if [ -n "$${{VLLM_QUANTIZATION:-}}" ]; then
          set -- "$${{@}}" --quantization "$${{VLLM_QUANTIZATION}}"
        fi
        if [ -n "$${{VLLM_DOWNLOAD_DIR:-}}" ]; then
          set -- "$${{@}}" --download-dir "$${{VLLM_DOWNLOAD_DIR}}"
        fi
        if [ -n "$${{VLLM_MAX_NUM_BATCHED_TOKENS:-}}" ]; then
          set -- "$${{@}}" --max-num-batched-tokens "$${{VLLM_MAX_NUM_BATCHED_TOKENS}}"
        fi
        if [ "$${{VLLM_ENABLE_CHUNKED_PREFILL:-}}" = "true" ]; then
          set -- "$${{@}}" --enable-chunked-prefill
        elif [ "$${{VLLM_ENABLE_CHUNKED_PREFILL:-}}" = "false" ]; then
          set -- "$${{@}}" --no-enable-chunked-prefill
        fi
        if [ "$${{VLLM_ENABLE_PREFIX_CACHING:-}}" = "true" ]; then
          set -- "$${{@}}" --enable-prefix-caching
        elif [ "$${{VLLM_ENABLE_PREFIX_CACHING:-}}" = "false" ]; then
          set -- "$${{@}}" --no-enable-prefix-caching
        fi
        if [ -n "$${{VLLM_KV_CACHE_DTYPE:-}}" ]; then
          set -- "$${{@}}" --kv-cache-dtype "$${{VLLM_KV_CACHE_DTYPE}}"
        fi
        if [ -n "$${{VLLM_CPU_OFFLOAD_GB:-}}" ]; then
          set -- "$${{@}}" --cpu-offload-gb "$${{VLLM_CPU_OFFLOAD_GB}}"
        fi
        if [ -n "$${{VLLM_SWAP_SPACE:-}}" ]; then
          set -- "$${{@}}" --swap-space "$${{VLLM_SWAP_SPACE}}"
        fi
        if [ "$${{VLLM_TENSOR_PARALLEL_SIZE:-1}}" != "1" ]; then
          set -- "$${{@}}" --tensor-parallel-size "$${{VLLM_TENSOR_PARALLEL_SIZE}}"
        fi
        if [ "$${{VLLM_PIPELINE_PARALLEL_SIZE:-1}}" != "1" ]; then
          set -- "$${{@}}" --pipeline-parallel-size "$${{VLLM_PIPELINE_PARALLEL_SIZE}}"
        fi
        if [ -n "$${{VLLM_UVICORN_LOG_LEVEL:-}}" ]; then
          set -- "$${{@}}" --uvicorn-log-level "$${{VLLM_UVICORN_LOG_LEVEL}}"
        fi
        if [ "$${{VLLM_DISABLE_LOG_STATS:-false}}" = "true" ]; then
          set -- "$${{@}}" --disable-log-stats
        fi
        if [ "$${{VLLM_EXTRA_ARGS_JSON:-[]}}" != "[]" ]; then
          extra_args=$$(python3 -c 'import json, shlex, sys; print(" ".join(shlex.quote(str(x)) for x in json.loads(sys.argv[1])))' "$${{VLLM_EXTRA_ARGS_JSON}}")
          eval "set -- \\\"\\$$@\\\" $${{extra_args}}"
        fi
        unset OMNI_MODEL OMNI_SERVED_MODEL_NAME OMNI_CONTEXT_LENGTH OMNI_MAX_PARALLEL OMNI_BACKEND_PORT OMNI_HF_HOME OMNI_HF_ENDPOINT OMNI_HTTP_PROXY OMNI_HTTPS_PROXY OMNI_NO_PROXY OMNI_CA_BUNDLE
        unset VLLM_IMAGE VLLM_GPU_MEMORY_UTILIZATION VLLM_GENERATION_CONFIG VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS VLLM_TRUST_REMOTE_CODE VLLM_ENFORCE_EAGER VLLM_ENABLE_AUTO_TOOL_CHOICE VLLM_TOOL_CALL_PARSER VLLM_REASONING_PARSER VLLM_DTYPE VLLM_TOKENIZER VLLM_TOKENIZER_MODE VLLM_REVISION VLLM_LOAD_FORMAT VLLM_QUANTIZATION VLLM_DOWNLOAD_DIR VLLM_MAX_NUM_BATCHED_TOKENS VLLM_ENABLE_CHUNKED_PREFILL VLLM_ENABLE_PREFIX_CACHING VLLM_KV_CACHE_DTYPE VLLM_CPU_OFFLOAD_GB VLLM_SWAP_SPACE VLLM_TENSOR_PARALLEL_SIZE VLLM_PIPELINE_PARALLEL_SIZE VLLM_UVICORN_LOG_LEVEL VLLM_DISABLE_LOG_STATS VLLM_EXTRA_ARGS_JSON VLLM_BUILD_COMMIT VLLM_BUILD_PIPELINE VLLM_BUILD_URL VLLM_IMAGE_TAG
        unset OMLX_PROXY_PORT OMLX_PROXY_API_KEY OMLX_BACKEND_API_KEY OMLX_CONTEXT_SCALING OMLX_TARGET_CONTEXT_SIZE OMLX_SSE_KEEPALIVE_MODE OMLX_SAMPLING_MAX_TOKENS OMLX_SAMPLING_TEMPERATURE OMLX_SAMPLING_TOP_P OMLX_SAMPLING_TOP_K OMLX_SAMPLING_REPETITION_PENALTY
        exec vllm serve "$${{@}}"

volumes:
  proxy-state:
'''


def render_vllm_compose_for_path(
    path: str | os.PathLike[str],
    settings: VllmComposeSettings,
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> str:
    project_context, compose_output_dir = compose_paths_for_render(path, repo_root)
    return render_vllm_compose(
        settings,
        project_context=project_context,
        compose_output_dir=compose_output_dir,
    )


def write_vllm_compose(path: str | os.PathLike[str], settings: VllmComposeSettings) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_vllm_compose(settings), encoding="utf-8")
    return output


def write_vllm_compose_for_path(
    path: str | os.PathLike[str],
    settings: VllmComposeSettings,
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_vllm_compose_for_path(output, settings, repo_root=repo_root),
        encoding="utf-8",
    )
    return output


register_backend(
    SidecarBackendSpec(
        name="vllm",
        service_name="vllm",
        env_keys=VLLM_ENV_KEYS,
        settings_cls=VllmComposeSettings,
        settings_from_env=vllm_settings_from_env,
        settings_from_overrides=settings_from_overrides,
        environment=vllm_environment,
        default_environment=default_vllm_environment,
        render_compose=render_vllm_compose,
        render_compose_for_path=render_vllm_compose_for_path,
        write_compose=write_vllm_compose,
        write_compose_for_path=write_vllm_compose_for_path,
        default_compose_name=DEFAULT_VLLM_COMPOSE_NAME,
        default_env_name=DEFAULT_VLLM_ENV_NAME,
    )
)
