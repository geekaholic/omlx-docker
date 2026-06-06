# SPDX-License-Identifier: Apache-2.0
'''Render Docker Compose configuration for the vLLM proxy sidecar.'''

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class VllmComposeSettings:
    image: str = "vllm/vllm-openai:latest"
    model: str = "Qwen/Qwen3-1.7B"
    served_model_name: str = "qwen"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.80
    max_num_seqs: int = 4
    port: int = 8000
    hf_home: str = "${HOME}/.cache/huggingface"
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
    http_proxy: str = ""
    https_proxy: str = ""
    no_proxy: str = ""
    ca_bundle: str = ""
    hf_endpoint: str = ""
    proxy_port: int = 8080
    proxy_api_key: str = ""
    backend_api_key: str = ""
    context_scaling: bool = False
    target_context_size: int = 200000
    sse_keepalive_mode: str = "ping"
    sampling_max_tokens: int = 32768
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
    sampling_top_k: int = 0
    sampling_repetition_penalty: float = 1.0


VLLM_ENV_KEYS = (
    "VLLM_IMAGE",
    "VLLM_MODEL",
    "VLLM_SERVED_MODEL_NAME",
    "VLLM_MAX_MODEL_LEN",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_MAX_NUM_SEQS",
    "VLLM_PORT",
    "VLLM_HF_HOME",
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
    "VLLM_HTTP_PROXY",
    "VLLM_HTTPS_PROXY",
    "VLLM_NO_PROXY",
    "VLLM_CA_BUNDLE",
    "VLLM_HF_ENDPOINT",
    "OMLX_PROXY_PORT",
    "OMLX_PROXY_API_KEY",
    "OMLX_BACKEND_API_KEY",
    "OMLX_CONTEXT_SCALING",
    "OMLX_TARGET_CONTEXT_SIZE",
    "OMLX_SSE_KEEPALIVE_MODE",
    "OMLX_SAMPLING_MAX_TOKENS",
    "OMLX_SAMPLING_TEMPERATURE",
    "OMLX_SAMPLING_TOP_P",
    "OMLX_SAMPLING_TOP_K",
    "OMLX_SAMPLING_REPETITION_PENALTY",
)


def settings_from_overrides(overrides: dict[str, Any]) -> VllmComposeSettings:
    def pick(name: str, default: Any) -> Any:
        return overrides.get(name, default)

    max_model_len = pick("vllm_max_model_len", VllmComposeSettings.max_model_len)
    max_num_seqs = pick("vllm_max_num_seqs", VllmComposeSettings.max_num_seqs)
    return VllmComposeSettings(
        image=str(pick("vllm_image", VllmComposeSettings.image)).strip()
        or VllmComposeSettings.image,
        model=str(pick("vllm_model", VllmComposeSettings.model)).strip()
        or VllmComposeSettings.model,
        served_model_name=str(
            pick("vllm_served_model_name", VllmComposeSettings.served_model_name)
        ).strip()
        or VllmComposeSettings.served_model_name,
        max_model_len=_positive_int(max_model_len, VllmComposeSettings.max_model_len),
        gpu_memory_utilization=_float(
            pick("vllm_gpu_memory_utilization", VllmComposeSettings.gpu_memory_utilization),
            VllmComposeSettings.gpu_memory_utilization,
        ),
        max_num_seqs=_positive_int(max_num_seqs, VllmComposeSettings.max_num_seqs),
        port=_positive_int(pick("vllm_port", VllmComposeSettings.port), VllmComposeSettings.port),
        hf_home=str(pick("vllm_hf_home", VllmComposeSettings.hf_home)).strip()
        or VllmComposeSettings.hf_home,
        generation_config=str(
            pick("vllm_generation_config", VllmComposeSettings.generation_config)
        ).strip()
        or VllmComposeSettings.generation_config,
        default_chat_template_kwargs=str(
            pick(
                "vllm_default_chat_template_kwargs",
                VllmComposeSettings.default_chat_template_kwargs,
            )
        ).strip(),
        trust_remote_code=_bool(
            pick("vllm_trust_remote_code", VllmComposeSettings.trust_remote_code),
            VllmComposeSettings.trust_remote_code,
        ),
        enforce_eager=_bool(
            pick("vllm_enforce_eager", VllmComposeSettings.enforce_eager),
            VllmComposeSettings.enforce_eager,
        ),
        enable_auto_tool_choice=_bool(
            pick("vllm_enable_auto_tool_choice", VllmComposeSettings.enable_auto_tool_choice),
            VllmComposeSettings.enable_auto_tool_choice,
        ),
        tool_call_parser=str(
            pick("vllm_tool_call_parser", VllmComposeSettings.tool_call_parser)
        ).strip(),
        reasoning_parser=str(
            pick("vllm_reasoning_parser", VllmComposeSettings.reasoning_parser)
        ).strip(),
        dtype=str(pick("vllm_dtype", VllmComposeSettings.dtype)).strip(),
        tokenizer=str(pick("vllm_tokenizer", VllmComposeSettings.tokenizer)).strip(),
        tokenizer_mode=str(
            pick("vllm_tokenizer_mode", VllmComposeSettings.tokenizer_mode)
        ).strip(),
        revision=str(pick("vllm_revision", VllmComposeSettings.revision)).strip(),
        load_format=str(pick("vllm_load_format", VllmComposeSettings.load_format)).strip(),
        quantization=str(pick("vllm_quantization", VllmComposeSettings.quantization)).strip(),
        download_dir=str(pick("vllm_download_dir", VllmComposeSettings.download_dir)).strip(),
        max_num_batched_tokens=str(
            pick("vllm_max_num_batched_tokens", VllmComposeSettings.max_num_batched_tokens)
        ).strip(),
        enable_chunked_prefill=_optional_bool(
            pick("vllm_enable_chunked_prefill", VllmComposeSettings.enable_chunked_prefill)
        ),
        enable_prefix_caching=_optional_bool(
            pick("vllm_enable_prefix_caching", VllmComposeSettings.enable_prefix_caching)
        ),
        kv_cache_dtype=str(
            pick("vllm_kv_cache_dtype", VllmComposeSettings.kv_cache_dtype)
        ).strip(),
        cpu_offload_gb=_nonnegative_float(
            pick("vllm_cpu_offload_gb", VllmComposeSettings.cpu_offload_gb),
            VllmComposeSettings.cpu_offload_gb,
        ),
        swap_space=_nonnegative_float(
            pick("vllm_swap_space", VllmComposeSettings.swap_space),
            VllmComposeSettings.swap_space,
        ),
        tensor_parallel_size=_positive_int(
            pick("vllm_tensor_parallel_size", VllmComposeSettings.tensor_parallel_size),
            VllmComposeSettings.tensor_parallel_size,
        ),
        pipeline_parallel_size=_positive_int(
            pick("vllm_pipeline_parallel_size", VllmComposeSettings.pipeline_parallel_size),
            VllmComposeSettings.pipeline_parallel_size,
        ),
        uvicorn_log_level=str(
            pick("vllm_uvicorn_log_level", VllmComposeSettings.uvicorn_log_level)
        ).strip(),
        disable_log_stats=_bool(
            pick("vllm_disable_log_stats", VllmComposeSettings.disable_log_stats),
            VllmComposeSettings.disable_log_stats,
        ),
        extra_args_json=str(
            pick("vllm_extra_args_json", VllmComposeSettings.extra_args_json)
        ).strip()
        or VllmComposeSettings.extra_args_json,
        http_proxy=str(pick("network_http_proxy", VllmComposeSettings.http_proxy)).strip(),
        https_proxy=str(pick("network_https_proxy", VllmComposeSettings.https_proxy)).strip(),
        no_proxy=str(pick("network_no_proxy", VllmComposeSettings.no_proxy)).strip(),
        ca_bundle=str(pick("network_ca_bundle", VllmComposeSettings.ca_bundle)).strip(),
        hf_endpoint=str(pick("huggingface_endpoint", VllmComposeSettings.hf_endpoint)).strip(),
        proxy_port=_positive_int(
            pick("omlx_proxy_port", VllmComposeSettings.proxy_port),
            VllmComposeSettings.proxy_port,
        ),
        proxy_api_key=str(pick("omlx_proxy_api_key", "")).strip(),
        backend_api_key=str(pick("omlx_backend_api_key", "")).strip(),
        context_scaling=_bool(
            pick("context_scaling_enabled", VllmComposeSettings.context_scaling),
            VllmComposeSettings.context_scaling,
        ),
        target_context_size=_positive_int(
            pick("target_context_size", VllmComposeSettings.target_context_size),
            VllmComposeSettings.target_context_size,
        ),
        sse_keepalive_mode=str(
            pick("omlx_sse_keepalive_mode", VllmComposeSettings.sse_keepalive_mode)
        ).strip()
        or VllmComposeSettings.sse_keepalive_mode,
        sampling_max_tokens=_positive_int(
            pick("sampling_max_tokens", VllmComposeSettings.sampling_max_tokens),
            VllmComposeSettings.sampling_max_tokens,
        ),
        sampling_temperature=_float(
            pick("sampling_temperature", VllmComposeSettings.sampling_temperature),
            VllmComposeSettings.sampling_temperature,
        ),
        sampling_top_p=_float(
            pick("sampling_top_p", VllmComposeSettings.sampling_top_p),
            VllmComposeSettings.sampling_top_p,
        ),
        sampling_top_k=_int(
            pick("sampling_top_k", VllmComposeSettings.sampling_top_k),
            VllmComposeSettings.sampling_top_k,
        ),
        sampling_repetition_penalty=_float(
            pick(
                "sampling_repetition_penalty",
                VllmComposeSettings.sampling_repetition_penalty,
            ),
            VllmComposeSettings.sampling_repetition_penalty,
        ),
    )


def vllm_environment(settings: VllmComposeSettings) -> dict[str, str]:
    return {
        "VLLM_IMAGE": settings.image,
        "VLLM_MODEL": settings.model,
        "VLLM_SERVED_MODEL_NAME": settings.served_model_name,
        "VLLM_MAX_MODEL_LEN": str(settings.max_model_len),
        "VLLM_GPU_MEMORY_UTILIZATION": str(settings.gpu_memory_utilization),
        "VLLM_MAX_NUM_SEQS": str(settings.max_num_seqs),
        "VLLM_PORT": str(settings.port),
        "VLLM_HF_HOME": settings.hf_home,
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
        "VLLM_HTTP_PROXY": settings.http_proxy,
        "VLLM_HTTPS_PROXY": settings.https_proxy,
        "VLLM_NO_PROXY": settings.no_proxy,
        "VLLM_CA_BUNDLE": settings.ca_bundle,
        "VLLM_HF_ENDPOINT": settings.hf_endpoint,
        "OMLX_PROXY_PORT": str(settings.proxy_port),
        "OMLX_PROXY_API_KEY": settings.proxy_api_key,
        "OMLX_BACKEND_API_KEY": settings.backend_api_key,
        "OMLX_CONTEXT_SCALING": _bool_str(settings.context_scaling),
        "OMLX_TARGET_CONTEXT_SIZE": str(settings.target_context_size),
        "OMLX_SSE_KEEPALIVE_MODE": settings.sse_keepalive_mode,
        "OMLX_SAMPLING_MAX_TOKENS": str(settings.sampling_max_tokens),
        "OMLX_SAMPLING_TEMPERATURE": str(settings.sampling_temperature),
        "OMLX_SAMPLING_TOP_P": str(settings.sampling_top_p),
        "OMLX_SAMPLING_TOP_K": str(settings.sampling_top_k),
        "OMLX_SAMPLING_REPETITION_PENALTY": str(settings.sampling_repetition_penalty),
    }


def default_vllm_environment(*, expand_hf_home: bool = False) -> dict[str, str]:
    defaults = VllmComposeSettings()
    settings = defaults
    if expand_hf_home:
        settings = VllmComposeSettings(hf_home=_host_path(defaults.hf_home))
    return vllm_environment(settings)


def vllm_settings_from_env(values: Mapping[str, str]) -> VllmComposeSettings:
    defaults = VllmComposeSettings()
    return VllmComposeSettings(
        image=values.get("VLLM_IMAGE", defaults.image),
        model=values.get("VLLM_MODEL", defaults.model),
        served_model_name=values.get("VLLM_SERVED_MODEL_NAME", defaults.served_model_name),
        max_model_len=_int_value(values.get("VLLM_MAX_MODEL_LEN"), defaults.max_model_len),
        gpu_memory_utilization=_float_value(
            values.get("VLLM_GPU_MEMORY_UTILIZATION"),
            defaults.gpu_memory_utilization,
        ),
        max_num_seqs=_int_value(values.get("VLLM_MAX_NUM_SEQS"), defaults.max_num_seqs),
        port=_int_value(values.get("VLLM_PORT"), defaults.port),
        hf_home=values.get("VLLM_HF_HOME", defaults.hf_home),
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
        http_proxy=values.get("VLLM_HTTP_PROXY", defaults.http_proxy),
        https_proxy=values.get("VLLM_HTTPS_PROXY", defaults.https_proxy),
        no_proxy=values.get("VLLM_NO_PROXY", defaults.no_proxy),
        ca_bundle=values.get("VLLM_CA_BUNDLE", defaults.ca_bundle),
        hf_endpoint=values.get("VLLM_HF_ENDPOINT", defaults.hf_endpoint),
        proxy_port=_int_value(values.get("OMLX_PROXY_PORT"), defaults.proxy_port),
        proxy_api_key=values.get("OMLX_PROXY_API_KEY", ""),
        backend_api_key=values.get("OMLX_BACKEND_API_KEY", ""),
        context_scaling=_bool_value(
            values.get("OMLX_CONTEXT_SCALING"),
            defaults.context_scaling,
        ),
        target_context_size=_int_value(
            values.get("OMLX_TARGET_CONTEXT_SIZE"),
            defaults.target_context_size,
        ),
        sse_keepalive_mode=values.get(
            "OMLX_SSE_KEEPALIVE_MODE",
            defaults.sse_keepalive_mode,
        ),
        sampling_max_tokens=_int_value(
            values.get("OMLX_SAMPLING_MAX_TOKENS"),
            defaults.sampling_max_tokens,
        ),
        sampling_temperature=_float_value(
            values.get("OMLX_SAMPLING_TEMPERATURE"),
            defaults.sampling_temperature,
        ),
        sampling_top_p=_float_value(
            values.get("OMLX_SAMPLING_TOP_P"),
            defaults.sampling_top_p,
        ),
        sampling_top_k=_int_or_zero_value(
            values.get("OMLX_SAMPLING_TOP_K"),
            defaults.sampling_top_k,
        ),
        sampling_repetition_penalty=_float_value(
            values.get("OMLX_SAMPLING_REPETITION_PENALTY"),
            defaults.sampling_repetition_penalty,
        ),
    )


def load_vllm_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
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


def write_vllm_env_file(path: str | os.PathLike[str], values: Mapping[str, str]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_vllm_env_file(values, include_header=True), encoding="utf-8")
    return output


def render_vllm_env_file(
    values: Mapping[str, str],
    *,
    include_header: bool = False,
) -> str:
    lines = []
    if include_header:
        lines.append(
            "# Generated by omni/admin. Edit with `omni serve` flags or the admin UI."
        )
    for key in VLLM_ENV_KEYS:
        value = str(values.get(key, ""))
        if "\n" in value or "\r" in value:
            raise ValueError(f"Environment value for {key} cannot contain newlines")
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def vllm_env_from_compose(path: str | os.PathLike[str]) -> dict[str, str]:
    compose_path = Path(path)
    if not compose_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in compose_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        for key in VLLM_ENV_KEYS:
            prefix = f"{key}:"
            if not stripped.startswith(prefix):
                continue
            raw_value = stripped[len(prefix) :].strip()
            parsed = _compose_default_value(key, raw_value)
            if parsed is not None:
                values[key] = parsed
    return values


def known_vllm_env(values: Mapping[str, str]) -> dict[str, str]:
    return {key: str(values[key]) for key in VLLM_ENV_KEYS if key in values}


def render_vllm_compose(
    settings: VllmComposeSettings,
    *,
    project_context: str = "..",
    compose_output_dir: str = "../docker",
) -> str:
    env = {
        "VLLM_IMAGE": settings.image,
        "VLLM_MODEL": settings.model,
        "VLLM_SERVED_MODEL_NAME": settings.served_model_name,
        "VLLM_MAX_MODEL_LEN": str(settings.max_model_len),
        "VLLM_GPU_MEMORY_UTILIZATION": str(settings.gpu_memory_utilization),
        "VLLM_MAX_NUM_SEQS": str(settings.max_num_seqs),
        "VLLM_PORT": str(settings.port),
        "VLLM_HF_HOME": settings.hf_home,
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
        "VLLM_HTTP_PROXY": settings.http_proxy,
        "VLLM_HTTPS_PROXY": settings.https_proxy,
        "VLLM_NO_PROXY": settings.no_proxy,
        "VLLM_CA_BUNDLE": settings.ca_bundle,
        "VLLM_HF_ENDPOINT": settings.hf_endpoint,
        "OMLX_PROXY_PORT": str(settings.proxy_port),
        "OMLX_PROXY_API_KEY": settings.proxy_api_key,
        "OMLX_BACKEND_API_KEY": settings.backend_api_key,
        "OMLX_CONTEXT_SCALING": _bool_str(settings.context_scaling),
        "OMLX_TARGET_CONTEXT_SIZE": str(settings.target_context_size),
        "OMLX_SSE_KEEPALIVE_MODE": settings.sse_keepalive_mode,
        "OMLX_SAMPLING_MAX_TOKENS": str(settings.sampling_max_tokens),
        "OMLX_SAMPLING_TEMPERATURE": str(settings.sampling_temperature),
        "OMLX_SAMPLING_TOP_P": str(settings.sampling_top_p),
        "OMLX_SAMPLING_TOP_K": str(settings.sampling_top_k),
        "OMLX_SAMPLING_REPETITION_PENALTY": str(settings.sampling_repetition_penalty),
    }
    env_lines = "\n".join(
        f"      {key}: {_yaml_quote(_compose_default_expr(key, value))}"
        for key, value in env.items()
    )
    return f'''# Generated by oMNI proxy admin. Edit docker-compose.vllm.template.yml or
# admin proxy settings, then regenerate instead of hand-editing this file.
services:
  omlx-proxy:
    build:
      context: {_yaml_quote(project_context)}
      dockerfile: docker/Dockerfile.proxy
    ports:
      - "{_compose_default_expr('OMLX_PROXY_PORT', str(settings.proxy_port))}:8080"
    environment:
      OMLX_BACKEND_URL: "http://vllm:8000/v1"
      OMLX_BACKEND_API_KEY: {_yaml_quote(_compose_default_expr('OMLX_BACKEND_API_KEY', settings.backend_api_key))}
      OMLX_PROXY_API_KEY: {_yaml_quote(_compose_default_expr('OMLX_PROXY_API_KEY', settings.proxy_api_key))}
      OMLX_PROXY_HOST: "0.0.0.0"
      OMLX_PROXY_PORT: "8080"
      OMLX_CONTEXT_SCALING: {_yaml_quote(_compose_default_expr('OMLX_CONTEXT_SCALING', _bool_str(settings.context_scaling)))}
      OMLX_TARGET_CONTEXT_SIZE: {_yaml_quote(_compose_default_expr('OMLX_TARGET_CONTEXT_SIZE', str(settings.target_context_size)))}
      OMLX_ACTUAL_CONTEXT_SIZE: {_yaml_quote(_compose_default_expr('VLLM_MAX_MODEL_LEN', str(settings.max_model_len)))}
      OMLX_SSE_KEEPALIVE_MODE: {_yaml_quote(_compose_default_expr('OMLX_SSE_KEEPALIVE_MODE', settings.sse_keepalive_mode))}
      OMLX_SAMPLING_MAX_TOKENS: {_yaml_quote(_compose_default_expr('OMLX_SAMPLING_MAX_TOKENS', str(settings.sampling_max_tokens)))}
      OMLX_SAMPLING_TEMPERATURE: {_yaml_quote(_compose_default_expr('OMLX_SAMPLING_TEMPERATURE', str(settings.sampling_temperature)))}
      OMLX_SAMPLING_TOP_P: {_yaml_quote(_compose_default_expr('OMLX_SAMPLING_TOP_P', str(settings.sampling_top_p)))}
      OMLX_SAMPLING_TOP_K: {_yaml_quote(_compose_default_expr('OMLX_SAMPLING_TOP_K', str(settings.sampling_top_k)))}
      OMLX_SAMPLING_REPETITION_PENALTY: {_yaml_quote(_compose_default_expr('OMLX_SAMPLING_REPETITION_PENALTY', str(settings.sampling_repetition_penalty)))}
      OMLX_PROXY_STATE_PATH: "/data/proxy-state.json"
      OMLX_VLLM_COMPOSE_OUTPUT_PATH: "/compose-output/docker-compose.vllm.yml"
      OMLX_VLLM_ENV_OUTPUT_PATH: "/compose-output/docker-compose.vllm.env"
      HTTP_PROXY: {_yaml_quote(_compose_default_expr('VLLM_HTTP_PROXY', settings.http_proxy))}
      HTTPS_PROXY: {_yaml_quote(_compose_default_expr('VLLM_HTTPS_PROXY', settings.https_proxy))}
      NO_PROXY: {_yaml_quote(_compose_default_expr('VLLM_NO_PROXY', settings.no_proxy))}
      REQUESTS_CA_BUNDLE: {_yaml_quote(_compose_default_expr('VLLM_CA_BUNDLE', settings.ca_bundle))}
      SSL_CERT_FILE: {_yaml_quote(_compose_default_expr('VLLM_CA_BUNDLE', settings.ca_bundle))}
    volumes:
      - proxy-state:/data
      - {_yaml_quote(f'{compose_output_dir}:/compose-output')}
    depends_on:
      - vllm

  vllm:
    image: {_yaml_quote(_compose_default_expr('VLLM_IMAGE', settings.image))}
    ports:
      - "{_compose_default_expr('VLLM_PORT', str(settings.port))}:8000"
    ipc: host
    gpus: all
    environment:
{env_lines}
      HF_TOKEN: "${{HF_TOKEN:-}}"
      HUGGING_FACE_HUB_TOKEN: "${{HUGGING_FACE_HUB_TOKEN:-${{HF_TOKEN:-}}}}"
      HF_HOME: "/root/.cache/huggingface"
    volumes:
      - "{_compose_default_expr('VLLM_HF_HOME', settings.hf_home)}:/root/.cache/huggingface"
    entrypoint: ["/bin/sh", "-lc"]
    command:
      - |
        set -eu
        if [ -n "$${{VLLM_HF_ENDPOINT:-}}" ]; then
          export HF_ENDPOINT="$${{VLLM_HF_ENDPOINT}}"
        else
          unset HF_ENDPOINT
        fi
        if [ -n "$${{VLLM_HTTP_PROXY:-}}" ]; then
          export HTTP_PROXY="$${{VLLM_HTTP_PROXY}}"
        else
          unset HTTP_PROXY
        fi
        if [ -n "$${{VLLM_HTTPS_PROXY:-}}" ]; then
          export HTTPS_PROXY="$${{VLLM_HTTPS_PROXY}}"
        else
          unset HTTPS_PROXY
        fi
        if [ -n "$${{VLLM_NO_PROXY:-}}" ]; then
          export NO_PROXY="$${{VLLM_NO_PROXY}}"
        else
          unset NO_PROXY
        fi
        if [ -n "$${{VLLM_CA_BUNDLE:-}}" ]; then
          export REQUESTS_CA_BUNDLE="$${{VLLM_CA_BUNDLE}}"
          export SSL_CERT_FILE="$${{VLLM_CA_BUNDLE}}"
        else
          unset REQUESTS_CA_BUNDLE
          unset SSL_CERT_FILE
        fi
        set -- "$${{VLLM_MODEL:?Set VLLM_MODEL to a Hugging Face model id or local container path}}"
        set -- "$${{@}}" --host 0.0.0.0
        set -- "$${{@}}" --port 8000
        set -- "$${{@}}" --served-model-name "$${{VLLM_SERVED_MODEL_NAME:-$${{VLLM_MODEL}}}}"
        set -- "$${{@}}" --max-model-len "$${{VLLM_MAX_MODEL_LEN:-8192}}"
        set -- "$${{@}}" --gpu-memory-utilization "$${{VLLM_GPU_MEMORY_UTILIZATION:-0.80}}"
        set -- "$${{@}}" --max-num-seqs "$${{VLLM_MAX_NUM_SEQS:-4}}"
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
        unset VLLM_IMAGE VLLM_MODEL VLLM_SERVED_MODEL_NAME VLLM_MAX_MODEL_LEN VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_NUM_SEQS VLLM_PORT VLLM_HF_HOME VLLM_GENERATION_CONFIG VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS VLLM_TRUST_REMOTE_CODE VLLM_ENFORCE_EAGER VLLM_ENABLE_AUTO_TOOL_CHOICE VLLM_TOOL_CALL_PARSER VLLM_REASONING_PARSER VLLM_DTYPE VLLM_TOKENIZER VLLM_TOKENIZER_MODE VLLM_REVISION VLLM_LOAD_FORMAT VLLM_QUANTIZATION VLLM_DOWNLOAD_DIR VLLM_MAX_NUM_BATCHED_TOKENS VLLM_ENABLE_CHUNKED_PREFILL VLLM_ENABLE_PREFIX_CACHING VLLM_KV_CACHE_DTYPE VLLM_CPU_OFFLOAD_GB VLLM_SWAP_SPACE VLLM_TENSOR_PARALLEL_SIZE VLLM_PIPELINE_PARALLEL_SIZE VLLM_UVICORN_LOG_LEVEL VLLM_DISABLE_LOG_STATS VLLM_EXTRA_ARGS_JSON VLLM_HTTP_PROXY VLLM_HTTPS_PROXY VLLM_NO_PROXY VLLM_CA_BUNDLE VLLM_HF_ENDPOINT VLLM_BUILD_COMMIT VLLM_BUILD_PIPELINE VLLM_BUILD_URL VLLM_IMAGE_TAG
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
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    output_dir = Path(path).parent.resolve()
    project_context = _relative_path(root.resolve(), output_dir)
    compose_output_dir = _relative_path((root / "docker").resolve(), output_dir)
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
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'
