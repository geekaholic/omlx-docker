# SPDX-License-Identifier: Apache-2.0
'''Render Docker Compose configuration for the vLLM proxy sidecar.'''

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    proxy_port: int = 8080
    proxy_api_key: str = ""
    backend_api_key: str = ""
    context_scaling: bool = False
    target_context_size: int = 200000
    sse_keepalive_mode: str = "ping"


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
    )


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
        "OMLX_PROXY_PORT": str(settings.proxy_port),
        "OMLX_PROXY_API_KEY": settings.proxy_api_key,
        "OMLX_BACKEND_API_KEY": settings.backend_api_key,
        "OMLX_CONTEXT_SCALING": _bool_str(settings.context_scaling),
        "OMLX_TARGET_CONTEXT_SIZE": str(settings.target_context_size),
        "OMLX_SSE_KEEPALIVE_MODE": settings.sse_keepalive_mode,
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
      OMLX_PROXY_STATE_PATH: "/data/proxy-state.json"
      OMLX_VLLM_COMPOSE_OUTPUT_PATH: "/compose-output/docker-compose.vllm.yml"
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
