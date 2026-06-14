# SPDX-License-Identifier: Apache-2.0
"""Linux/proxy-first oMNI command line interface."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from ._version import __version__
from .proxy import (  # noqa: F401  (registers backend specs)
    llamacpp_compose,
    vllm_compose,
)
from .proxy.config import BACKEND_URL_DEFAULTS
from .proxy.memory_fit import (
    auto_utilization,
    estimate_resident_bytes,
    evaluate_fit,
    format_gib,
    guard_disabled,
    host_reserve_bytes,
    kv_bytes_per_token,
    resolve_local_model_path,
)
from .proxy.metrics import host_memory_info
from .proxy.sidecar_compose import (
    backend_spec,
    derive_served_name,
    env_from_compose,
    known_env,
    load_env_file,
    render_env_file,
    write_env_file,
)
from .proxy.vllm_compose import (
    VllmComposeSettings,
    vllm_settings_from_env,
)
from .proxy.vllm_compose import (
    default_vllm_environment as _shared_default_vllm_environment,
)
from .proxy.vllm_compose import (
    write_vllm_compose_for_path as write_vllm_compose_for_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = REPO_ROOT / "docker"
DEFAULT_PROXY_COMPOSE = DOCKER_DIR / "docker-compose.proxy.yml"
DEFAULT_VLLM_COMPOSE = DOCKER_DIR / "docker-compose.vllm.yml"
DEFAULT_VLLM_ENV_FILE = DOCKER_DIR / "docker-compose.vllm.env"
DEFAULT_LLAMACPP_COMPOSE = DOCKER_DIR / "docker-compose.llamacpp.yml"
DEFAULT_LLAMACPP_ENV_FILE = DOCKER_DIR / "docker-compose.llamacpp.env"
DEFAULT_PROXY_ENV_FILE = DOCKER_DIR / "docker-compose.proxy.env"
DEFAULT_SERVE_STATE_FILE = DOCKER_DIR / "omni-serve.json"

SERVE_STATE_VERSION = 1
MANAGED_BACKENDS = {"vllm", "llamacpp"}
URL_BACKENDS = {"openai", "llamacpp"}
OPENAI_DEFAULT_BACKEND_URL = BACKEND_URL_DEFAULTS["openai-compatible"]
BACKENDS = MANAGED_BACKENDS | URL_BACKENDS
# Backwards-compatible alias; llamacpp belongs to both groups (managed sidecar
# without --backend-url, plain proxy with it).
PROXY_BACKENDS = URL_BACKENDS
SERVE_MODES = {"managed", "proxy"}
MANAGED_SERVICE_NAMES = ("vllm", "llamacpp")
PROXY_ENV_KEYS = (
    "OMLX_BACKEND_URL",
    "OMLX_BACKEND_API_KEY",
    "OMLX_PROXY_API_KEY",
    "OMLX_PROXY_PORT",
    "OMLX_CONTEXT_SCALING",
    "OMLX_TARGET_CONTEXT_SIZE",
    "OMLX_ACTUAL_CONTEXT_SIZE",
    "OMLX_SSE_KEEPALIVE_MODE",
)
PORTABLE_ARG_ATTRS = (
    "model",
    "served_model_name",
    "port",
    "context_length",
    "max_parallel",
    "hf_home",
    "hf_endpoint",
    "model_dir",
)
VLLM_SPECIFIC_ARG_ATTRS = (
    "vllm_image",
    "gpu_memory_utilization",
    "generation_config",
    "default_chat_template_kwargs",
    "trust_remote_code",
    "enforce_eager",
    "enable_auto_tool_choice",
    "tool_call_parser",
    "reasoning_parser",
    "dtype",
    "tokenizer",
    "tokenizer_mode",
    "revision",
    "load_format",
    "quantization",
    "download_dir",
    "max_num_batched_tokens",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "kv_cache_dtype",
    "cpu_offload_gb",
    "swap_space",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "uvicorn_log_level",
    "disable_log_stats",
    "extra_args_json",
)
LLAMACPP_SPECIFIC_ARG_ATTRS = (
    "llamacpp_image",
    "n_gpu_layers",
    "flash_attn",
    "cache_type_k",
    "cache_type_v",
    "threads",
    "batch_size",
    "ubatch_size",
    "jinja",
    "reasoning_format",
    "llamacpp_cache_dir",
    "llamacpp_model_dir",
    "llamacpp_extra_args",
)


TOP_LEVEL_HELP = """
Command quick reference:
  omni serve [options]
  omni launch <tool> [--model MODEL] [--port PORT] [--host HOST] [--api-key KEY]
  omni status [--compose-file PATH]
  omni logs [--target proxy|backend|both] [-f|--follow] [--compose-file PATH]
  omni restart [--target proxy|backend|both] [--compose-file PATH]
  omni stop --target proxy|backend|both [--compose-file PATH]

Backends:
  vllm       Generate docker/docker-compose.vllm.yml and launch vLLM + proxy.
  llamacpp   Generate docker/docker-compose.llamacpp.yml and launch llama.cpp + proxy.
             With --backend-url, proxy to an external llama.cpp server instead.
  openai     Launch the proxy against any OpenAI-compatible endpoint. Defaults to
             Ollama at http://host.docker.internal:11434/v1; use --backend-url to
             point elsewhere.

Common serve options:
  --backend {vllm,llamacpp,openai}
  --backend-url URL
  --backend-api-key KEY
  --api-key KEY
  --proxy-port PORT
  --compose-file PATH
  --foreground | --detach
  --no-build
  --generate-only
  --dry-run

Portable backend options (vLLM and llama.cpp):
  --model MODEL
  --served-model-name NAME
  --port PORT
  --context-length TOKENS
  --max-parallel COUNT
  --hf-home PATH

vLLM serve options:
  --vllm-image IMAGE
  --gpu-memory-utilization FRACTION
  --generation-config {vllm,auto}
  --default-chat-template-kwargs JSON
  --trust-remote-code | --no-trust-remote-code
  --enforce-eager
  --enable-auto-tool-choice
  --tool-call-parser NAME
  --reasoning-parser NAME
  --dtype {auto,bfloat16,float,float16,float32,half}
  --tokenizer TOKENIZER
  --tokenizer-mode MODE
  --revision REVISION
  --load-format FORMAT
  --quantization QUANTIZATION
  --download-dir PATH
  --max-num-batched-tokens TOKENS
  --enable-chunked-prefill | --no-enable-chunked-prefill
  --enable-prefix-caching | --no-enable-prefix-caching
  --kv-cache-dtype DTYPE
  --cpu-offload-gb GB
  --swap-space GB
  --tensor-parallel-size COUNT
  --pipeline-parallel-size COUNT
  --uvicorn-log-level LEVEL
  --disable-log-stats
  --extra-args-json JSON_ARRAY

llama.cpp serve options:
  --llamacpp-image IMAGE
  --n-gpu-layers COUNT
  --flash-attn {on,off,auto}
  --cache-type-k TYPE
  --cache-type-v TYPE
  --threads COUNT
  --batch-size SIZE
  --ubatch-size SIZE
  --jinja | --no-jinja
  --reasoning-format FORMAT
  --llamacpp-cache-dir PATH
  --llamacpp-model-dir PATH
  --llamacpp-extra-args "ARGS"

Proxy/network options:
  --http-proxy URL
  --https-proxy URL
  --no-proxy HOSTS
  --ca-bundle PATH
  --hf-endpoint URL

Proxy behavior options:
  --context-scaling
  --target-context-size TOKENS
  --sse-keepalive-mode {ping,comment,off}

Examples:
  omni serve
  omni serve --backend vllm --model Qwen/Qwen3-1.7B --served-model-name qwen3
  omni serve --backend llamacpp --model ggml-org/Qwen3-1.7B-GGUF:Q8_0
  omni serve --backend llamacpp --model /models/qwen3.gguf --context-length 16384
  omni serve --backend openai
  omni serve --backend openai --backend-url https://api.example.com/v1
  omni serve --backend llamacpp --backend-url http://host.docker.internal:8000/v1
  omni launch list
  omni launch claude
  omni launch codex --model qwen --port 8080
  omni launch claude --opus-model qwen-big --sonnet-model qwen --haiku-model qwen-fast
  omni status
  omni logs --target backend
  omni logs -f
  omni restart --target backend
  omni stop --target both

Run `omni serve --help`, `omni launch --help`, `omni status --help`,
`omni logs --help`, `omni restart --help`, or `omni stop --help` for
argparse's detailed option descriptions.
""".strip()

SERVE_DESCRIPTION = """
Bootstrap oMNI with Docker Compose. vLLM and llama.cpp are generated as local
sidecar compose files; openai (any OpenAI-compatible endpoint, Ollama by
default) and llamacpp with --backend-url use the proxy compose against an
external OpenAI-compatible backend. A first
`omni serve` uses the proxy compose; later runs reuse the last saved serve
backend unless flags choose another one.
""".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omni",
        description="omni: Docker-first oMNI launcher for proxy backends",
        epilog=TOP_LEVEL_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print the oMNI version and exit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser(
        "serve",
        help="Generate compose config and launch an oMNI backend",
        description=SERVE_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    serve.add_argument(
        "--backend",
        choices=["vllm", "llamacpp", "openai"],
        default=None,
        help=(
            "Backend to launch or proxy to (default: last used, then the "
            "openai proxy against Ollama)"
        ),
    )
    serve.add_argument(
        "--backend-url",
        default=None,
        help="OpenAI-compatible backend URL including /v1 for proxy backends",
    )
    serve.add_argument("--backend-api-key", default=None, help="Backend API key")
    serve.add_argument(
        "--api-key", default=None, help="API key required by the oMNI proxy"
    )
    serve.add_argument("--proxy-port", type=int, default=None, help="Host proxy port")
    serve.add_argument(
        "--compose-file",
        default=None,
        help="Compose file path. Defaults to the generated sidecar compose or proxy compose.",
    )
    serve.add_argument(
        "--foreground",
        action="store_true",
        help="Run docker compose in the foreground instead of detached mode",
    )
    serve.add_argument(
        "--detach",
        dest="foreground",
        action="store_false",
        help="Run docker compose detached (default)",
    )
    serve.set_defaults(foreground=False)
    serve.add_argument(
        "--no-build", action="store_true", help="Do not pass --build to compose up"
    )
    serve.add_argument(
        "--generate-only",
        action="store_true",
        help="Write generated files and print the docker compose command without launching",
    )
    serve.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated content and command without writing or launching",
    )
    serve.add_argument(
        "--force-memory",
        action="store_true",
        help=(
            "Launch even when the pre-flight memory check predicts the model "
            "will not fit in unified memory (can hard-lock the host)"
        ),
    )
    offline_group = serve.add_mutually_exclusive_group()
    offline_group.add_argument(
        "--offline",
        dest="hf_offline",
        action="store_true",
        default=None,
        help=(
            "Run the backend with HF_HUB_OFFLINE so startup never touches the "
            "network (default: auto-on when the model is already cached)"
        ),
    )
    offline_group.add_argument(
        "--online",
        dest="hf_offline",
        action="store_false",
        help="Force the backend to allow Hugging Face network access at startup",
    )

    serve.add_argument(
        "--model",
        default=None,
        help="Model id or container-local path (llama.cpp: owner/repo[:quant] or .gguf path)",
    )
    serve.add_argument(
        "--served-model-name", default=None, help="API-visible model name"
    )
    serve.add_argument("--port", type=int, default=None, help="Host backend port")
    serve.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Model context length (vLLM --max-model-len, llama.cpp --ctx-size)",
    )
    serve.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Max concurrent sequences (vLLM --max-num-seqs, llama.cpp --parallel)",
    )
    serve.add_argument(
        "--hf-home", default=None, help="Host Hugging Face cache directory"
    )
    serve.add_argument(
        "--scan-models",
        action="store_true",
        help=(
            "Let the admin UI scan the host model caches (read-only) and "
            "list downloaded models with a one-click sidecar switch"
        ),
    )
    serve.add_argument(
        "--model-dir",
        default=None,
        help=(
            "Host directory to scan for local models with --scan-models "
            "(default: the Hugging Face cache)"
        ),
    )

    serve.add_argument("--vllm-image", default=None, help="vLLM container image")
    serve.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM GPU memory utilization",
    )
    serve.add_argument(
        "--generation-config",
        choices=["vllm", "auto"],
        default=None,
        help="vLLM generation config source",
    )
    serve.add_argument(
        "--default-chat-template-kwargs",
        default=None,
        help="JSON passed to vLLM --default-chat-template-kwargs",
    )
    serve.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        default=None,
        help="Pass --trust-remote-code to vLLM",
    )
    serve.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Do not pass --trust-remote-code to vLLM",
    )
    serve.add_argument(
        "--enforce-eager",
        action="store_true",
        default=None,
        help="Pass --enforce-eager",
    )
    serve.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        default=None,
        help="Enable vLLM auto tool choice flags",
    )
    serve.add_argument("--tool-call-parser", default=None, help="vLLM tool call parser")
    serve.add_argument("--reasoning-parser", default=None, help="vLLM reasoning parser")
    serve.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float", "float16", "float32", "half"],
        default=None,
        help="vLLM model dtype",
    )
    serve.add_argument("--tokenizer", default=None, help="vLLM tokenizer id or path")
    serve.add_argument("--tokenizer-mode", default=None, help="vLLM tokenizer mode")
    serve.add_argument("--revision", default=None, help="Hugging Face model revision")
    serve.add_argument("--load-format", default=None, help="vLLM load format")
    serve.add_argument("--quantization", default=None, help="vLLM quantization mode")
    serve.add_argument("--download-dir", default=None, help="vLLM download directory")
    serve.add_argument(
        "--max-num-batched-tokens",
        default=None,
        help="vLLM max tokens processed per scheduler iteration",
    )
    serve.add_argument(
        "--enable-chunked-prefill",
        dest="enable_chunked_prefill",
        action="store_true",
        default=None,
        help="Pass --enable-chunked-prefill to vLLM",
    )
    serve.add_argument(
        "--no-enable-chunked-prefill",
        dest="enable_chunked_prefill",
        action="store_false",
        help="Pass --no-enable-chunked-prefill to vLLM",
    )
    serve.add_argument(
        "--enable-prefix-caching",
        dest="enable_prefix_caching",
        action="store_true",
        default=None,
        help="Pass --enable-prefix-caching to vLLM",
    )
    serve.add_argument(
        "--no-enable-prefix-caching",
        dest="enable_prefix_caching",
        action="store_false",
        help="Pass --no-enable-prefix-caching to vLLM",
    )
    serve.add_argument("--kv-cache-dtype", default=None, help="vLLM KV cache dtype")
    serve.add_argument(
        "--cpu-offload-gb", type=float, default=None, help="CPU offload GB per GPU"
    )
    serve.add_argument(
        "--swap-space", type=float, default=None, help="CPU swap space GB per GPU"
    )
    serve.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="vLLM tensor parallel size",
    )
    serve.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=None,
        help="vLLM pipeline parallel size",
    )
    serve.add_argument(
        "--uvicorn-log-level", default=None, help="vLLM API server log level"
    )
    serve.add_argument(
        "--disable-log-stats",
        action="store_true",
        default=None,
        help="Pass --disable-log-stats to vLLM",
    )
    serve.add_argument(
        "--extra-args-json",
        default=None,
        help='Raw vLLM args as a JSON array, appended last, e.g. ["--foo","bar"]',
    )

    serve.add_argument(
        "--llamacpp-image", default=None, help="llama.cpp server container image"
    )
    serve.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help="llama.cpp layers to offload to the GPU (999 = all)",
    )
    serve.add_argument(
        "--flash-attn",
        choices=["on", "off", "auto"],
        default=None,
        help="llama.cpp flash attention mode",
    )
    serve.add_argument(
        "--cache-type-k", default=None, help="llama.cpp KV cache K type (e.g. q8_0)"
    )
    serve.add_argument(
        "--cache-type-v", default=None, help="llama.cpp KV cache V type (e.g. q8_0)"
    )
    serve.add_argument(
        "--threads", type=int, default=None, help="llama.cpp CPU threads"
    )
    serve.add_argument(
        "--batch-size", type=int, default=None, help="llama.cpp logical batch size"
    )
    serve.add_argument(
        "--ubatch-size", type=int, default=None, help="llama.cpp physical batch size"
    )
    serve.add_argument(
        "--jinja",
        dest="jinja",
        action="store_true",
        default=None,
        help="Enable llama.cpp jinja chat templates and tool calling (default)",
    )
    serve.add_argument(
        "--no-jinja",
        dest="jinja",
        action="store_false",
        help="Disable llama.cpp jinja chat templates",
    )
    serve.add_argument(
        "--reasoning-format", default=None, help="llama.cpp reasoning format"
    )
    serve.add_argument(
        "--llamacpp-cache-dir",
        default=None,
        help="Host llama.cpp download cache directory (-hf downloads persist here)",
    )
    serve.add_argument(
        "--llamacpp-model-dir",
        default=None,
        help="Host directory mounted read-only at /models for local .gguf files",
    )
    serve.add_argument(
        "--llamacpp-extra-args",
        default=None,
        help="Raw llama-server args appended last (whitespace separated)",
    )

    serve.add_argument(
        "--http-proxy", default=None, help="HTTP proxy for proxy and backend containers"
    )
    serve.add_argument(
        "--https-proxy",
        default=None,
        help="HTTPS proxy for proxy and backend containers",
    )
    serve.add_argument(
        "--no-proxy", default=None, help="Comma-separated hosts that bypass proxy"
    )
    serve.add_argument(
        "--ca-bundle", default=None, help="CA bundle path for TLS verification"
    )
    serve.add_argument(
        "--hf-endpoint", default=None, help="Hugging Face endpoint for model downloads"
    )
    serve.add_argument(
        "--context-scaling",
        action="store_true",
        help="Enable Anthropic token usage scaling in the proxy",
    )
    serve.add_argument(
        "--target-context-size",
        type=int,
        default=None,
        help="Target Anthropic context size for usage scaling",
    )
    serve.add_argument(
        "--sse-keepalive-mode",
        choices=["ping", "comment", "off"],
        default=None,
        help="Proxy SSE keepalive mode",
    )

    launch = subparsers.add_parser(
        "launch",
        help="Launch an external coding agent pointed at the oMNI proxy",
        description=(
            "Connect an external coding agent (claude, codex, copilot, etc.) to "
            "the running oMNI proxy. Pass 'list' as the tool name to see all "
            "available integrations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    launch.add_argument(
        "tool",
        help="Tool to launch: claude, codex, opencode, openclaw, copilot, hermes, pi, or 'list'",
    )
    launch.add_argument(
        "--model", default=None, help="Model id to use (skips interactive selection)"
    )
    launch.add_argument(
        "--host",
        default=None,
        help="oMNI proxy host (default: localhost)",
    )
    launch.add_argument(
        "--port",
        type=int,
        default=None,
        help="oMNI proxy port (default: OMLX_PROXY_PORT env or 8080)",
    )
    launch.add_argument(
        "--api-key",
        default=None,
        help="oMNI proxy API key (default: OMLX_PROXY_API_KEY or OMLX_API_KEY env)",
    )
    launch.add_argument(
        "--opus-model", default=None, help="Claude Code Opus-tier model override"
    )
    launch.add_argument(
        "--sonnet-model", default=None, help="Claude Code Sonnet-tier model override"
    )
    launch.add_argument(
        "--haiku-model", default=None, help="Claude Code Haiku-tier model override"
    )

    status = subparsers.add_parser(
        "status",
        help="Show Docker Compose container status for the oMNI stack",
        description=(
            "Show Docker Compose container status. Defaults to the generated "
            "sidecar compose file when present, otherwise the proxy compose file."
        ),
    )
    status.add_argument(
        "--compose-file",
        default=None,
        help="Compose file path. Defaults to the generated sidecar compose or proxy compose.",
    )

    logs = subparsers.add_parser(
        "logs",
        help="Show Docker Compose logs for the oMNI stack",
        description=(
            "Show Docker Compose logs for the selected stack. Use --target to "
            "filter to the proxy or managed backend service."
        ),
    )
    logs.add_argument(
        "--target",
        choices=["proxy", "backend", "both"],
        default="both",
        help="Service group to show logs for (default: both)",
    )
    logs.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Follow log output",
    )
    logs.add_argument(
        "--compose-file",
        default=None,
        help="Compose file path. Defaults to the generated sidecar compose or proxy compose.",
    )

    restart = subparsers.add_parser(
        "restart",
        help="Restart the proxy, backend, or both services",
        description=(
            "Restart services in the selected Docker Compose stack. External "
            "OpenAI-compatible and external llama.cpp backends are not "
            "managed by this command."
        ),
    )
    restart.add_argument(
        "--target",
        choices=["proxy", "backend", "both"],
        default="both",
        help="Service group to restart (default: both)",
    )
    restart.add_argument(
        "--compose-file",
        default=None,
        help="Compose file path. Defaults to the generated sidecar compose or proxy compose.",
    )

    stop = subparsers.add_parser(
        "stop",
        help="Stop the proxy, backend, or both services",
        description=(
            "Stop services in the selected Docker Compose stack. External "
            "OpenAI-compatible and external llama.cpp backends are not "
            "managed by this command."
        ),
    )
    stop.add_argument(
        "--target",
        choices=["proxy", "backend", "both"],
        required=True,
        help="Service group to stop: proxy, managed backend, or both",
    )
    stop.add_argument(
        "--compose-file",
        default=None,
        help="Compose file path. Defaults to the generated sidecar compose or proxy compose.",
    )
    return parser


def default_compose_file(backend: str, mode: str = "managed") -> Path:
    if mode == "managed" and backend in MANAGED_BACKENDS:
        if backend == "vllm":
            return DEFAULT_VLLM_COMPOSE
        return DEFAULT_LLAMACPP_COMPOSE
    return DEFAULT_PROXY_COMPOSE


def _path_from_state(value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser()


def load_serve_state(path: Path | None = None) -> dict[str, str]:
    state_path = path or DEFAULT_SERVE_STATE_FILE
    if not state_path.exists():
        return {}
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    backend = raw.get("backend")
    if backend == "ollama":
        # ollama was folded into the openai backend; it keeps the Ollama
        # endpoint as its default URL.
        backend = "openai"
    if backend not in BACKENDS:
        return {}
    state: dict[str, str] = {"backend": str(backend)}
    mode = raw.get("mode")
    if mode not in SERVE_MODES:
        # State files written before llama.cpp became a managed sidecar lack a
        # mode; only vLLM was managed back then.
        mode = "managed" if backend == "vllm" else "proxy"
    state["mode"] = str(mode)
    compose_file = raw.get("compose_file")
    if isinstance(compose_file, str) and compose_file.strip():
        state["compose_file"] = compose_file
    return state


def save_serve_state(
    *,
    backend: str,
    compose_file: Path,
    mode: str | None = None,
    path: Path | None = None,
) -> Path:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    if mode is None:
        mode = "managed" if backend in MANAGED_BACKENDS else "proxy"
    if mode not in SERVE_MODES:
        raise ValueError(f"Unknown serve mode: {mode}")
    state_path = path or DEFAULT_SERVE_STATE_FILE
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SERVE_STATE_VERSION,
        "backend": backend,
        "mode": mode,
        "compose_file": str(compose_file),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    state_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return state_path


def compose_command(compose_file: Path, *, foreground: bool, build: bool) -> list[str]:
    command = ["docker", "compose", "-f", str(compose_file), "up"]
    if not foreground:
        command.append("-d")
    if build:
        command.append("--build")
    return command


def compose_env_command(
    compose_file: Path,
    env_file: Path,
    *,
    foreground: bool,
    build: bool,
) -> list[str]:
    command = [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "-f",
        str(compose_file),
        "up",
    ]
    if not foreground:
        command.append("-d")
    if build:
        command.append("--build")
    return command


def sidecar_env_file_for_compose(backend: str, compose_file: Path) -> Path:
    if backend == "vllm" and compose_file.resolve() == DEFAULT_VLLM_COMPOSE.resolve():
        return DEFAULT_VLLM_ENV_FILE
    if (
        backend == "llamacpp"
        and compose_file.resolve() == DEFAULT_LLAMACPP_COMPOSE.resolve()
    ):
        return DEFAULT_LLAMACPP_ENV_FILE
    return compose_file.with_suffix(".env")


def proxy_env_file_for_compose(compose_file: Path) -> Path:
    if compose_file.resolve() == DEFAULT_PROXY_COMPOSE.resolve():
        return DEFAULT_PROXY_ENV_FILE
    return compose_file.with_suffix(".env")


def default_vllm_environment() -> dict[str, str]:
    return _shared_default_vllm_environment(expand_hf_home=True)


def merged_sidecar_environment(
    backend: str,
    args: argparse.Namespace,
    *,
    env_file: Path | None = None,
    compose_file: Path | None = None,
) -> dict[str, str]:
    spec = backend_spec(backend)
    values = spec.default_environment(expand_hf_home=True)
    if env_file is not None:
        values.update(known_env(load_env_file(env_file), spec.env_keys))
    if env_file is not None and not env_file.exists() and compose_file is not None:
        values.update(
            known_env(env_from_compose(compose_file, spec.env_keys), spec.env_keys)
        )
    values.update(sidecar_cli_environment(backend, args))
    return values


def merged_vllm_environment(
    args: argparse.Namespace,
    *,
    env_file: Path | None = None,
    compose_file: Path | None = None,
) -> dict[str, str]:
    return merged_sidecar_environment(
        "vllm",
        args,
        env_file=env_file,
        compose_file=compose_file,
    )


def portable_cli_environment(args: argparse.Namespace) -> dict[str, str]:
    values: dict[str, str] = {}
    mappings = {
        "model": "OMNI_MODEL",
        "served_model_name": "OMNI_SERVED_MODEL_NAME",
        "context_length": "OMNI_CONTEXT_LENGTH",
        "max_parallel": "OMNI_MAX_PARALLEL",
        "port": "OMNI_BACKEND_PORT",
        "hf_endpoint": "OMNI_HF_ENDPOINT",
        "http_proxy": "OMNI_HTTP_PROXY",
        "https_proxy": "OMNI_HTTPS_PROXY",
        "no_proxy": "OMNI_NO_PROXY",
        "ca_bundle": "OMNI_CA_BUNDLE",
        "proxy_port": "OMLX_PROXY_PORT",
        "api_key": "OMLX_PROXY_API_KEY",
        "backend_api_key": "OMLX_BACKEND_API_KEY",
        "target_context_size": "OMLX_TARGET_CONTEXT_SIZE",
        "sse_keepalive_mode": "OMLX_SSE_KEEPALIVE_MODE",
    }
    for attr, key in mappings.items():
        value = getattr(args, attr, None)
        if value is not None:
            values[key] = str(value)
    # Changing --model without --served-model-name would otherwise keep the
    # previous session's served name, mislabeling the new model. Track it.
    if (
        getattr(args, "model", None) is not None
        and getattr(args, "served_model_name", None) is None
    ):
        values["OMNI_SERVED_MODEL_NAME"] = derive_served_name(args.model)
    if getattr(args, "hf_home", None) is not None:
        values["OMNI_HF_HOME"] = _host_path(args.hf_home)
    if getattr(args, "context_scaling", False):
        values["OMLX_CONTEXT_SCALING"] = "true"
    if getattr(args, "scan_models", False):
        values["OMLX_MODEL_SCAN"] = "true"
    if getattr(args, "model_dir", None) is not None:
        values["OMLX_MODEL_SCAN_HOST_DIR"] = _host_path(args.model_dir)
    return values


def vllm_cli_environment(args: argparse.Namespace) -> dict[str, str]:
    values = portable_cli_environment(args)
    mappings = {
        "vllm_image": "VLLM_IMAGE",
        "gpu_memory_utilization": "VLLM_GPU_MEMORY_UTILIZATION",
        "generation_config": "VLLM_GENERATION_CONFIG",
        "default_chat_template_kwargs": "VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS",
        "tool_call_parser": "VLLM_TOOL_CALL_PARSER",
        "reasoning_parser": "VLLM_REASONING_PARSER",
        "dtype": "VLLM_DTYPE",
        "tokenizer": "VLLM_TOKENIZER",
        "tokenizer_mode": "VLLM_TOKENIZER_MODE",
        "revision": "VLLM_REVISION",
        "load_format": "VLLM_LOAD_FORMAT",
        "quantization": "VLLM_QUANTIZATION",
        "download_dir": "VLLM_DOWNLOAD_DIR",
        "max_num_batched_tokens": "VLLM_MAX_NUM_BATCHED_TOKENS",
        "kv_cache_dtype": "VLLM_KV_CACHE_DTYPE",
        "cpu_offload_gb": "VLLM_CPU_OFFLOAD_GB",
        "swap_space": "VLLM_SWAP_SPACE",
        "tensor_parallel_size": "VLLM_TENSOR_PARALLEL_SIZE",
        "pipeline_parallel_size": "VLLM_PIPELINE_PARALLEL_SIZE",
        "uvicorn_log_level": "VLLM_UVICORN_LOG_LEVEL",
        "extra_args_json": "VLLM_EXTRA_ARGS_JSON",
    }
    for attr, key in mappings.items():
        value = getattr(args, attr, None)
        if value is not None:
            values[key] = str(value)
    if args.trust_remote_code is not None:
        values["VLLM_TRUST_REMOTE_CODE"] = _bool_str(args.trust_remote_code)
    if args.enforce_eager is not None:
        values["VLLM_ENFORCE_EAGER"] = _bool_str(args.enforce_eager)
    if args.enable_auto_tool_choice is not None:
        values["VLLM_ENABLE_AUTO_TOOL_CHOICE"] = _bool_str(args.enable_auto_tool_choice)
    if args.enable_chunked_prefill is not None:
        values["VLLM_ENABLE_CHUNKED_PREFILL"] = _bool_str(args.enable_chunked_prefill)
    if args.enable_prefix_caching is not None:
        values["VLLM_ENABLE_PREFIX_CACHING"] = _bool_str(args.enable_prefix_caching)
    if args.disable_log_stats is not None:
        values["VLLM_DISABLE_LOG_STATS"] = _bool_str(args.disable_log_stats)
    return values


def llamacpp_cli_environment(args: argparse.Namespace) -> dict[str, str]:
    values = portable_cli_environment(args)
    mappings = {
        "llamacpp_image": "LLAMACPP_IMAGE",
        "n_gpu_layers": "LLAMACPP_N_GPU_LAYERS",
        "flash_attn": "LLAMACPP_FLASH_ATTN",
        "cache_type_k": "LLAMACPP_CACHE_TYPE_K",
        "cache_type_v": "LLAMACPP_CACHE_TYPE_V",
        "threads": "LLAMACPP_THREADS",
        "batch_size": "LLAMACPP_BATCH_SIZE",
        "ubatch_size": "LLAMACPP_UBATCH_SIZE",
        "reasoning_format": "LLAMACPP_REASONING_FORMAT",
        "llamacpp_extra_args": "LLAMACPP_EXTRA_ARGS",
    }
    for attr, key in mappings.items():
        value = getattr(args, attr, None)
        if value is not None:
            values[key] = str(value)
    if args.jinja is not None:
        values["LLAMACPP_JINJA"] = _bool_str(args.jinja)
    if args.llamacpp_cache_dir is not None:
        values["LLAMACPP_CACHE_DIR"] = _host_path(args.llamacpp_cache_dir)
    if args.llamacpp_model_dir is not None:
        values["LLAMACPP_MODEL_DIR"] = _host_path(args.llamacpp_model_dir)
    return values


def sidecar_cli_environment(backend: str, args: argparse.Namespace) -> dict[str, str]:
    if backend == "llamacpp":
        return llamacpp_cli_environment(args)
    return vllm_cli_environment(args)


def _has_args(args: argparse.Namespace, attrs: Sequence[str]) -> bool:
    return any(getattr(args, attr, None) is not None for attr in attrs)


def has_portable_args(args: argparse.Namespace) -> bool:
    return _has_args(args, PORTABLE_ARG_ATTRS)


def has_vllm_specific_args(args: argparse.Namespace) -> bool:
    return _has_args(args, VLLM_SPECIFIC_ARG_ATTRS)


def has_llamacpp_specific_args(args: argparse.Namespace) -> bool:
    return _has_args(args, LLAMACPP_SPECIFIC_ARG_ATTRS)


def resolve_serve_backend(
    args: argparse.Namespace,
    state: Mapping[str, str] | None = None,
) -> str:
    if args.backend:
        return args.backend
    if has_llamacpp_specific_args(args):
        return "llamacpp"
    if has_vllm_specific_args(args):
        return "vllm"
    if args.backend_url:
        state_backend = (state or {}).get("backend")
        if state_backend in URL_BACKENDS:
            return str(state_backend)
        return "openai"
    state_backend = (state or {}).get("backend")
    if has_portable_args(args):
        if state_backend in MANAGED_BACKENDS:
            return str(state_backend)
        return "vllm"
    if state_backend in BACKENDS:
        return str(state_backend)
    return "openai"


def resolve_serve_mode(
    args: argparse.Namespace,
    backend: str,
    state: Mapping[str, str] | None = None,
) -> str:
    if backend not in MANAGED_BACKENDS:
        return "proxy"
    if backend == "vllm":
        return "managed"
    if args.backend_url:
        return "proxy"
    # Backend resolved from saved state with no explicit selection: honor the
    # saved mode so a proxy-to-external llama.cpp stays a proxy.
    if (
        args.backend is None
        and not has_llamacpp_specific_args(args)
        and not has_portable_args(args)
        and (state or {}).get("backend") == backend
    ):
        return str((state or {}).get("mode") or "managed")
    return "managed"


def compose_file_for_serve_backend(
    args: argparse.Namespace,
    backend: str,
    state: Mapping[str, str] | None = None,
    mode: str = "managed",
) -> Path:
    if args.compose_file:
        return Path(args.compose_file).expanduser()
    if (state or {}).get("backend") == backend and (state or {}).get(
        "mode", "managed"
    ) == mode:
        state_compose = _path_from_state((state or {}).get("compose_file"))
        if state_compose is not None:
            return state_compose
    return default_compose_file(backend, mode)


def default_proxy_environment(backend: str) -> dict[str, str]:
    return {
        "OMLX_BACKEND_URL": (OPENAI_DEFAULT_BACKEND_URL if backend == "openai" else ""),
        "OMLX_BACKEND_API_KEY": "",
        "OMLX_PROXY_API_KEY": "",
        "OMLX_PROXY_PORT": str(VllmComposeSettings.proxy_port),
        "OMLX_CONTEXT_SCALING": "false",
        "OMLX_TARGET_CONTEXT_SIZE": str(VllmComposeSettings.target_context_size),
        "OMLX_ACTUAL_CONTEXT_SIZE": "32768",
        "OMLX_SSE_KEEPALIVE_MODE": VllmComposeSettings.sse_keepalive_mode,
    }


def proxy_cli_environment(args: argparse.Namespace) -> dict[str, str]:
    values: dict[str, str] = {}
    mappings = {
        "backend_url": "OMLX_BACKEND_URL",
        "backend_api_key": "OMLX_BACKEND_API_KEY",
        "api_key": "OMLX_PROXY_API_KEY",
        "proxy_port": "OMLX_PROXY_PORT",
        "target_context_size": "OMLX_TARGET_CONTEXT_SIZE",
        "sse_keepalive_mode": "OMLX_SSE_KEEPALIVE_MODE",
    }
    for attr, key in mappings.items():
        value = getattr(args, attr, None)
        if value is not None:
            values[key] = str(value)
    if args.context_scaling:
        values["OMLX_CONTEXT_SCALING"] = "true"
    return values


def render_generic_env_file(
    values: Mapping[str, str],
    keys: Sequence[str],
    *,
    include_header: bool = False,
) -> str:
    lines: list[str] = []
    if include_header:
        lines.append("# Generated by omni. Edit with `omni serve` flags.")
    for key in keys:
        value = str(values.get(key, ""))
        if "\n" in value or "\r" in value:
            raise ValueError(f"Environment value for {key} cannot contain newlines")
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def write_generic_env_file(
    path: Path,
    values: Mapping[str, str],
    keys: Sequence[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_generic_env_file(values, keys, include_header=True),
        encoding="utf-8",
    )
    return path


def default_control_compose_file() -> Path:
    state = load_serve_state()
    state_compose = _path_from_state(state.get("compose_file"))
    if state_compose is not None and state_compose.exists():
        return state_compose
    if DEFAULT_VLLM_COMPOSE.exists():
        return DEFAULT_VLLM_COMPOSE
    if DEFAULT_LLAMACPP_COMPOSE.exists():
        return DEFAULT_LLAMACPP_COMPOSE
    return DEFAULT_PROXY_COMPOSE


def control_compose_file(args: argparse.Namespace) -> Path:
    compose_file = (
        Path(args.compose_file) if args.compose_file else default_control_compose_file()
    )
    if not compose_file.exists():
        raise SystemExit(f"Compose file not found: {compose_file}")
    return compose_file


def compose_services(compose_file: Path) -> list[str]:
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "config", "--services"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def services_for_target(target: str, services: Sequence[str]) -> list[str]:
    available = set(services)
    if target == "both":
        return []
    if target == "proxy":
        if "omlx-proxy" not in available:
            raise SystemExit("Proxy service not found in selected compose file")
        return ["omlx-proxy"]
    if target == "backend":
        for name in MANAGED_SERVICE_NAMES:
            if name in available:
                return [name]
        raise SystemExit(
            "Backend service is external or not managed by this compose stack"
        )
    raise SystemExit(f"Unknown service target: {target}")


def services_for_compose_target(target: str, compose_file: Path) -> list[str]:
    if target == "both":
        return []
    return services_for_target(target, compose_services(compose_file))


def restart_services_for_target(target: str, services: Sequence[str]) -> list[str]:
    return services_for_target(target, services)


def proxy_backend_url(backend: str, values: Mapping[str, str]) -> str:
    backend_url = values.get("OMLX_BACKEND_URL", "").strip()
    if backend == "openai":
        return backend_url or OPENAI_DEFAULT_BACKEND_URL
    if backend_url:
        return backend_url
    raise SystemExit(f"--backend-url is required for --backend {backend}")


def proxy_environment(
    args: argparse.Namespace,
    *,
    backend: str | None = None,
    existing_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    effective_backend = backend or args.backend or "openai"
    values = default_proxy_environment(effective_backend)
    if existing_env is not None:
        values.update(
            {
                key: str(existing_env[key])
                for key in PROXY_ENV_KEYS
                if key in existing_env
            }
        )
    values.update(proxy_cli_environment(args))
    values["OMLX_BACKEND_URL"] = proxy_backend_url(effective_backend, values)
    return values


def vllm_settings_from_args(
    args: argparse.Namespace,
    existing_env: Mapping[str, str] | None = None,
) -> VllmComposeSettings:
    if existing_env is None:
        return vllm_settings_from_env(
            merged_vllm_environment(args, env_file=None, compose_file=None)
        )
    spec = backend_spec("vllm")
    return vllm_settings_from_env(
        {
            **default_vllm_environment(),
            **known_env(existing_env, spec.env_keys),
            **vllm_cli_environment(args),
        }
    )


def run_compose(
    command: Sequence[str],
    env_overrides: Mapping[str, str],
    *,
    dry_run: bool,
    generate_only: bool,
) -> int:
    printable = " ".join(command)
    if dry_run or generate_only:
        print(printable)
        return 0

    env = os.environ.copy()
    env.update(env_overrides)
    subprocess.run(command, check=True, env=env)
    return 0


def status_command(args: argparse.Namespace) -> int:
    compose_file = control_compose_file(args)
    subprocess.run(["docker", "compose", "-f", str(compose_file), "ps"], check=True)
    return 0


def logs_command(args: argparse.Namespace) -> int:
    compose_file = control_compose_file(args)
    services = services_for_compose_target(args.target, compose_file)
    command = ["docker", "compose", "-f", str(compose_file), "logs"]
    if args.follow:
        command.append("-f")
    command.extend(services)
    subprocess.run(command, check=True)
    return 0


def restart_command(args: argparse.Namespace) -> int:
    compose_file = control_compose_file(args)
    services = services_for_compose_target(args.target, compose_file)
    command = ["docker", "compose", "-f", str(compose_file), "restart", *services]
    subprocess.run(command, check=True)
    return 0


def stop_command(args: argparse.Namespace) -> int:
    compose_file = control_compose_file(args)
    services = services_for_compose_target(args.target, compose_file)
    command = ["docker", "compose", "-f", str(compose_file), "stop", *services]
    subprocess.run(command, check=True)
    return 0


def launch_command(args: argparse.Namespace) -> int:
    from .integrations import IntegrationContext, get_integration, list_integrations

    tool_name = args.tool

    if tool_name == "list":
        print("Available integrations:")
        for integ in list_integrations():
            installed = "installed" if integ.is_installed() else "not installed"
            print(f"  {integ.name:12s} {integ.display_name} ({installed})")
        return 0

    integration = get_integration(tool_name)
    if integration is None:
        print(f"Unknown integration: {tool_name}")
        print("Available: " + ", ".join(i.name for i in list_integrations()))
        sys.exit(1)

    host = args.host or "localhost"
    port = args.port or int(os.environ.get("OMLX_PROXY_PORT", "8080"))
    api_key = (
        args.api_key
        or os.environ.get("OMLX_PROXY_API_KEY")
        or os.environ.get("OMLX_API_KEY")
        or ""
    )

    base_url = f"http://{host}:{port}"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Verify the proxy is reachable.
    try:
        req = urllib.request.Request(f"{base_url}/health", headers=headers)
        with urllib.request.urlopen(req, timeout=5):
            pass
    except (urllib.error.URLError, OSError):
        print(f"oMNI proxy is not reachable at {base_url}")
        print("Start the proxy first: omni serve")
        sys.exit(1)

    # Determine model.
    opus_model = args.opus_model or None
    sonnet_model = args.sonnet_model or None
    haiku_model = args.haiku_model or None
    claude_has_tier_models = tool_name == "claude" and any(
        (opus_model, sonnet_model, haiku_model)
    )

    model = args.model or ""
    if not model and claude_has_tier_models:
        model = sonnet_model or opus_model or haiku_model or ""

    if not model:
        try:
            req = urllib.request.Request(f"{base_url}/v1/models", headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            models = [
                m["id"]
                for m in data.get("data", [])
                if m.get("model_type") in ("llm", "vlm", None)
            ]
        except (urllib.error.URLError, OSError, KeyError, ValueError):
            models = []

        if not models:
            print("No models available. Load a model in the backend first.")
            sys.exit(1)

        if len(models) == 1:
            model = models[0]
            print(f"Using model: {model}")
        else:
            model = integration.select_model(
                [{"id": m} for m in models], integration.display_name
            )

    if not integration.is_installed():
        print(f"{integration.display_name} is not installed.")
        print(f"Install: {integration.install_hint}")
        sys.exit(1)

    ctx = IntegrationContext(
        host=host,
        port=port,
        api_key=api_key,
        model=model,
        opus_model=opus_model if tool_name == "claude" else None,
        sonnet_model=sonnet_model if tool_name == "claude" else None,
        haiku_model=haiku_model if tool_name == "claude" else None,
    )
    print(f"Launching {integration.display_name} with model {model}...")
    integration.launch(ctx)
    return 0


def _positive_int_or_none(value: str | None) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _vllm_memory_preflight(
    args: argparse.Namespace, merged_env: dict[str, str]
) -> None:
    """Set a unified-memory-aware util default and refuse oversized launches.

    Mutates ``merged_env``'s ``VLLM_GPU_MEMORY_UTILIZATION`` when the user did
    not set it explicitly, then runs the fit check. On a hard block it prints
    the numbers and raises ``SystemExit`` — unless overridden, or running under
    ``--dry-run``/``--generate-only`` (which only print the assessment).
    """
    host_total = int(host_memory_info().get("total_bytes") or 0)
    reserve = host_reserve_bytes()

    scan_dirs = [
        merged_env.get("OMNI_HF_HOME", ""),
        merged_env.get("OMLX_MODEL_SCAN_HOST_DIR", ""),
    ]
    model_path = resolve_local_model_path(merged_env.get("OMNI_MODEL", ""), scan_dirs)
    weights = estimate_resident_bytes(model_path) if model_path is not None else None
    kv_per_token = kv_bytes_per_token(model_path) if model_path is not None else None
    context = _positive_int_or_none(merged_env.get("OMNI_CONTEXT_LENGTH"))
    parallel = _positive_int_or_none(merged_env.get("OMNI_MAX_PARALLEL"))

    explicit_util = getattr(args, "gpu_memory_utilization", None) is not None
    if not explicit_util and host_total > 0:
        safe = auto_utilization(
            total_bytes=host_total,
            reserve_bytes=reserve,
            weights_bytes=weights,
            kv_per_token=kv_per_token,
            context_tokens=context,
            parallel=parallel,
        )
        merged_env["VLLM_GPU_MEMORY_UTILIZATION"] = f"{safe:.2f}"
        if kv_per_token and context and parallel:
            print(
                f"Using gpu-memory-utilization {safe:.2f}: sized KV cache for "
                f"{parallel} x {context} tokens on {format_gib(host_total)} "
                f"unified memory. Override with --gpu-memory-utilization."
            )
        else:
            print(
                f"Using gpu-memory-utilization {safe:.2f} for unified memory "
                f"({format_gib(host_total)} total, {format_gib(reserve)} host "
                f"reserve). Override with --gpu-memory-utilization."
            )

    try:
        util = float(merged_env.get("VLLM_GPU_MEMORY_UTILIZATION", "0.80"))
    except ValueError:
        util = 0.80

    result = evaluate_fit(
        total_bytes=host_total,
        util=util,
        reserve_bytes=reserve,
        weights_bytes=weights,
        kv_bytes_per_token=kv_per_token,
        context_tokens=context,
        parallel=parallel,
    )
    print(f"Memory check [{result.level}]: {result.reason}")

    if not result.blocked:
        return

    overridden = getattr(args, "force_memory", False) or guard_disabled()
    advisory = args.dry_run or args.generate_only
    if overridden or advisory:
        if not advisory:
            print("Overriding the memory guard (--force-memory) — this is unsafe.")
        return

    raise SystemExit(
        "Refusing to launch: the model is predicted to exhaust unified memory "
        "and hard-lock the host. Lower --context-length, pick a smaller model, "
        f"set --gpu-memory-utilization at most {result.recommended_util:.2f}, "
        "or pass --force-memory to override."
    )


def _apply_hf_offline(args: argparse.Namespace, merged_env: dict[str, str]) -> None:
    """Set OMNI_HF_OFFLINE so a cached model starts without HF network access.

    --offline/--online force the value; otherwise it auto-enables when the
    model resolves to a locally-cached path, so a broken DNS sandbox or an
    offline box can't crash startup for an already-downloaded model.
    """
    choice = getattr(args, "hf_offline", None)
    if choice is None:
        scan_dirs = [
            merged_env.get("OMNI_HF_HOME", ""),
            merged_env.get("OMLX_MODEL_SCAN_HOST_DIR", ""),
        ]
        cached = (
            resolve_local_model_path(merged_env.get("OMNI_MODEL", ""), scan_dirs)
            is not None
        )
        if cached:
            print(
                "Model is cached locally; starting the backend offline "
                "(HF_HUB_OFFLINE=true). Pass --online to allow network access."
            )
        choice = cached
    merged_env["OMNI_HF_OFFLINE"] = "true" if choice else "false"


def serve_command(args: argparse.Namespace) -> int:
    state = load_serve_state()
    backend = resolve_serve_backend(args, state)
    mode = resolve_serve_mode(args, backend, state)
    compose_file = compose_file_for_serve_backend(args, backend, state, mode)

    if mode == "managed":
        spec = backend_spec(backend)
        env_file = sidecar_env_file_for_compose(backend, compose_file)
        merged_env = merged_sidecar_environment(
            backend,
            args,
            env_file=env_file,
            compose_file=compose_file,
        )
        if backend == "vllm":
            _vllm_memory_preflight(args, merged_env)
        _apply_hf_offline(args, merged_env)
        settings = spec.settings_from_env(merged_env)
        compose_content = spec.render_compose_for_path(compose_file, settings)
        command = compose_env_command(
            compose_file,
            env_file,
            foreground=args.foreground,
            build=not args.no_build,
        )
        if args.dry_run:
            print(compose_content)
            print(f"# Env file: {env_file}")
            print(render_env_file(merged_env, spec.env_keys), end="")
        else:
            spec.write_compose_for_path(compose_file, settings)
            write_env_file(env_file, merged_env, spec.env_keys)
            save_serve_state(backend=backend, compose_file=compose_file, mode=mode)
            print(f"Wrote {compose_file}")
            print(f"Wrote {env_file}")
        return run_compose(
            command,
            {},
            dry_run=args.dry_run,
            generate_only=args.generate_only,
        )

    if not compose_file.exists():
        raise SystemExit(f"Compose file not found: {compose_file}")
    env_file = proxy_env_file_for_compose(compose_file)
    existing_env = load_env_file(env_file)
    env_overrides = proxy_environment(
        args,
        backend=backend,
        existing_env=existing_env,
    )
    command = compose_env_command(
        compose_file,
        env_file,
        foreground=args.foreground,
        build=not args.no_build,
    )
    if args.dry_run or args.generate_only:
        for key in sorted(env_overrides):
            print(f"{key}={env_overrides[key]}")
    if not args.dry_run:
        write_generic_env_file(env_file, env_overrides, PROXY_ENV_KEYS)
        save_serve_state(backend=backend, compose_file=compose_file, mode=mode)
        print(f"Wrote {env_file}")
    return run_compose(
        command,
        {},
        dry_run=args.dry_run,
        generate_only=args.generate_only,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return serve_command(args)
    if args.command == "launch":
        return launch_command(args)
    if args.command == "status":
        return status_command(args)
    if args.command == "logs":
        return logs_command(args)
    if args.command == "restart":
        return restart_command(args)
    if args.command == "stop":
        return stop_command(args)
    parser.print_help()
    return 1


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _host_path(value: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(value)))


if __name__ == "__main__":
    sys.exit(main())
