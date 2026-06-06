# SPDX-License-Identifier: Apache-2.0
"""Linux/proxy-first oMNI command line interface."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from ._version import __version__
from .proxy.vllm_compose import (
    VllmComposeSettings,
    default_vllm_environment as _shared_default_vllm_environment,
    known_vllm_env,
    load_vllm_env_file,
    render_vllm_compose_for_path,
    render_vllm_env_file,
    vllm_env_from_compose,
    vllm_environment,
    vllm_settings_from_env,
    write_vllm_compose_for_path,
    write_vllm_env_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = REPO_ROOT / "docker"
DEFAULT_PROXY_COMPOSE = DOCKER_DIR / "docker-compose.proxy.yml"
DEFAULT_VLLM_COMPOSE = DOCKER_DIR / "docker-compose.vllm.yml"
DEFAULT_VLLM_ENV_FILE = DOCKER_DIR / "docker-compose.vllm.env"


TOP_LEVEL_HELP = """
Command quick reference:
  omni serve [options]
  omni status [--compose-file PATH]
  omni logs [--target proxy|backend|both] [-f|--follow] [--compose-file PATH]
  omni restart [--target proxy|backend|both] [--compose-file PATH]
  omni stop --target proxy|backend|both [--compose-file PATH]

Backends:
  vllm       Generate docker/docker-compose.vllm.yml and launch vLLM + proxy.
  ollama     Launch the proxy against Ollama at http://host.docker.internal:11434/v1 by default.
  openai     Launch the proxy against an OpenAI-compatible HTTPS endpoint. Requires --backend-url.
  llamacpp   Launch the proxy against a llama.cpp OpenAI-compatible server. Requires --backend-url.

Common serve options:
  --backend {vllm,openai,ollama,llamacpp}
  --backend-url URL
  --backend-api-key KEY
  --api-key KEY
  --proxy-port PORT
  --compose-file PATH
  --foreground | --detach
  --no-build
  --generate-only
  --dry-run

vLLM serve options:
  --model MODEL
  --served-model-name NAME
  --vllm-image IMAGE
  --port PORT
  --max-model-len TOKENS
  --gpu-memory-utilization FRACTION
  --max-num-seqs COUNT
  --hf-home PATH
  --generation-config {vllm,auto}
  --default-chat-template-kwargs JSON
  --trust-remote-code | --no-trust-remote-code
  --enforce-eager
  --enable-auto-tool-choice
  --tool-call-parser NAME
  --reasoning-parser NAME

Proxy behavior options:
  --context-scaling
  --target-context-size TOKENS
  --sse-keepalive-mode {ping,comment,off}

Examples:
  omni serve --backend vllm --model Qwen/Qwen3-1.7B --served-model-name qwen3
  omni serve --backend ollama
  omni serve --backend llamacpp --backend-url http://host.docker.internal:8000/v1
  omni status
  omni logs --target backend
  omni logs -f
  omni restart --target backend
  omni stop --target both

Run `omni serve --help`, `omni status --help`, `omni logs --help`,
`omni restart --help`, or `omni stop --help` for argparse's detailed
option descriptions.
""".strip()

SERVE_DESCRIPTION = """
Bootstrap oMNI with Docker Compose. vLLM is generated as a local sidecar
compose file; OpenAI, Ollama, and llama.cpp use the proxy compose against an
external OpenAI-compatible backend.
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
        choices=["vllm", "openai", "ollama", "llamacpp"],
        default="vllm",
        help="Backend to launch or proxy to (default: vllm)",
    )
    serve.add_argument(
        "--backend-url",
        default=None,
        help="OpenAI-compatible backend URL including /v1 for proxy backends",
    )
    serve.add_argument("--backend-api-key", default=None, help="Backend API key")
    serve.add_argument("--api-key", default=None, help="API key required by the oMNI proxy")
    serve.add_argument("--proxy-port", type=int, default=None, help="Host proxy port")
    serve.add_argument(
        "--compose-file",
        default=None,
        help="Compose file path. Defaults to generated vLLM compose or proxy compose.",
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
    serve.add_argument("--no-build", action="store_true", help="Do not pass --build to compose up")
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

    serve.add_argument("--model", default=None, help="vLLM model id or container-local path")
    serve.add_argument("--served-model-name", default=None, help="API-visible model name")
    serve.add_argument("--vllm-image", default=None, help="vLLM container image")
    serve.add_argument("--port", type=int, default=None, help="Host vLLM port")
    serve.add_argument("--max-model-len", type=int, default=None, help="vLLM max model length")
    serve.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM GPU memory utilization",
    )
    serve.add_argument("--max-num-seqs", type=int, default=None, help="vLLM max sequences")
    serve.add_argument("--hf-home", default=None, help="Host Hugging Face cache directory")
    serve.add_argument(
        "--generation-config",
        choices=["vllm", "auto"],
        default=None,
        help="vLLM generation config source",
    )
    serve.add_argument(
        "--default-chat-template-kwargs",
        default=None,
        help='JSON passed to vLLM --default-chat-template-kwargs',
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
    serve.add_argument("--enforce-eager", action="store_true", default=None, help="Pass --enforce-eager")
    serve.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        default=None,
        help="Enable vLLM auto tool choice flags",
    )
    serve.add_argument("--tool-call-parser", default=None, help="vLLM tool call parser")
    serve.add_argument("--reasoning-parser", default=None, help="vLLM reasoning parser")
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

    status = subparsers.add_parser(
        "status",
        help="Show Docker Compose container status for the oMNI stack",
        description=(
            "Show Docker Compose container status. Defaults to the generated "
            "vLLM compose file when present, otherwise the proxy compose file."
        ),
    )
    status.add_argument(
        "--compose-file",
        default=None,
        help="Compose file path. Defaults to generated vLLM compose or proxy compose.",
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
        help="Compose file path. Defaults to generated vLLM compose or proxy compose.",
    )

    restart = subparsers.add_parser(
        "restart",
        help="Restart the proxy, backend, or both services",
        description=(
            "Restart services in the selected Docker Compose stack. External "
            "OpenAI, Ollama, and llama.cpp backends are not managed by this command."
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
        help="Compose file path. Defaults to generated vLLM compose or proxy compose.",
    )

    stop = subparsers.add_parser(
        "stop",
        help="Stop the proxy, backend, or both services",
        description=(
            "Stop services in the selected Docker Compose stack. External "
            "OpenAI, Ollama, and llama.cpp backends are not managed by this command."
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
        help="Compose file path. Defaults to generated vLLM compose or proxy compose.",
    )
    return parser


def default_compose_file(backend: str) -> Path:
    if backend == "vllm":
        return DEFAULT_VLLM_COMPOSE
    return DEFAULT_PROXY_COMPOSE


def compose_command(compose_file: Path, *, foreground: bool, build: bool) -> list[str]:
    command = ["docker", "compose", "-f", str(compose_file), "up"]
    if not foreground:
        command.append("-d")
    if build:
        command.append("--build")
    return command


def vllm_env_file_for_compose(compose_file: Path) -> Path:
    if compose_file.resolve() == DEFAULT_VLLM_COMPOSE.resolve():
        return DEFAULT_VLLM_ENV_FILE
    return compose_file.with_suffix(".env")


def vllm_compose_command(
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



load_env_file = load_vllm_env_file
write_env_file = write_vllm_env_file
render_env_file = render_vllm_env_file


def default_vllm_environment() -> dict[str, str]:
    return _shared_default_vllm_environment(expand_hf_home=True)


def merged_vllm_environment(
    args: argparse.Namespace,
    *,
    env_file: Path | None = None,
    compose_file: Path | None = None,
) -> dict[str, str]:
    values = default_vllm_environment()
    if env_file is not None:
        values.update(known_vllm_env(load_env_file(env_file)))
    if env_file is not None and not env_file.exists() and compose_file is not None:
        values.update(known_vllm_env(vllm_env_from_compose(compose_file)))
    values.update(vllm_cli_environment(args))
    return values


def vllm_cli_environment(args: argparse.Namespace) -> dict[str, str]:
    values: dict[str, str] = {}
    mappings = {
        "vllm_image": "VLLM_IMAGE",
        "model": "VLLM_MODEL",
        "served_model_name": "VLLM_SERVED_MODEL_NAME",
        "max_model_len": "VLLM_MAX_MODEL_LEN",
        "gpu_memory_utilization": "VLLM_GPU_MEMORY_UTILIZATION",
        "max_num_seqs": "VLLM_MAX_NUM_SEQS",
        "port": "VLLM_PORT",
        "generation_config": "VLLM_GENERATION_CONFIG",
        "default_chat_template_kwargs": "VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS",
        "tool_call_parser": "VLLM_TOOL_CALL_PARSER",
        "reasoning_parser": "VLLM_REASONING_PARSER",
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
    if args.hf_home is not None:
        values["VLLM_HF_HOME"] = _host_path(args.hf_home)
    if args.trust_remote_code is not None:
        values["VLLM_TRUST_REMOTE_CODE"] = _bool_str(args.trust_remote_code)
    if args.enforce_eager is not None:
        values["VLLM_ENFORCE_EAGER"] = _bool_str(args.enforce_eager)
    if args.enable_auto_tool_choice is not None:
        values["VLLM_ENABLE_AUTO_TOOL_CHOICE"] = _bool_str(args.enable_auto_tool_choice)
    if args.context_scaling:
        values["OMLX_CONTEXT_SCALING"] = "true"
    return values


def default_control_compose_file() -> Path:
    if DEFAULT_VLLM_COMPOSE.exists():
        return DEFAULT_VLLM_COMPOSE
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
        if "vllm" not in available:
            raise SystemExit(
                "Backend service is external or not managed by this compose stack"
            )
        return ["vllm"]
    raise SystemExit(f"Unknown service target: {target}")


def services_for_compose_target(target: str, compose_file: Path) -> list[str]:
    if target == "both":
        return []
    return services_for_target(target, compose_services(compose_file))


def restart_services_for_target(target: str, services: Sequence[str]) -> list[str]:
    return services_for_target(target, services)


def proxy_backend_url(args: argparse.Namespace) -> str:
    if args.backend == "ollama":
        return args.backend_url or "http://host.docker.internal:11434/v1"
    if args.backend_url:
        return args.backend_url
    raise SystemExit(f"--backend-url is required for --backend {args.backend}")


def proxy_environment(args: argparse.Namespace) -> dict[str, str]:
    return {
        "OMLX_BACKEND_URL": proxy_backend_url(args),
        "OMLX_BACKEND_API_KEY": args.backend_api_key or "",
        "OMLX_PROXY_API_KEY": args.api_key or "",
        "OMLX_PROXY_PORT": str(args.proxy_port or VllmComposeSettings.proxy_port),
        "OMLX_CONTEXT_SCALING": _bool_str(args.context_scaling),
        "OMLX_TARGET_CONTEXT_SIZE": str(
            args.target_context_size or VllmComposeSettings.target_context_size
        ),
        "OMLX_ACTUAL_CONTEXT_SIZE": "32768",
        "OMLX_SSE_KEEPALIVE_MODE": (
            args.sse_keepalive_mode or VllmComposeSettings.sse_keepalive_mode
        ),
    }


def vllm_settings_from_args(
    args: argparse.Namespace,
    existing_env: Mapping[str, str] | None = None,
) -> VllmComposeSettings:
    return vllm_settings_from_env(
        merged_vllm_environment(args, env_file=None, compose_file=None)
        if existing_env is None
        else {**default_vllm_environment(), **known_vllm_env(existing_env), **vllm_cli_environment(args)}
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


def serve_command(args: argparse.Namespace) -> int:
    compose_file = Path(args.compose_file) if args.compose_file else default_compose_file(args.backend)
    command = compose_command(compose_file, foreground=args.foreground, build=not args.no_build)

    if args.backend == "vllm":
        env_file = vllm_env_file_for_compose(compose_file)
        merged_env = merged_vllm_environment(
            args,
            env_file=env_file,
            compose_file=compose_file,
        )
        settings = vllm_settings_from_env(merged_env)
        compose_content = render_vllm_compose_for_path(compose_file, settings)
        command = vllm_compose_command(
            compose_file,
            env_file,
            foreground=args.foreground,
            build=not args.no_build,
        )
        if args.dry_run:
            print(compose_content)
            print(f"# Env file: {env_file}")
            print(render_env_file(merged_env), end="")
        else:
            write_vllm_compose_for_path(compose_file, settings)
            write_env_file(env_file, merged_env)
            print(f"Wrote {compose_file}")
            print(f"Wrote {env_file}")
        return run_compose(
            command,
            {},
            dry_run=args.dry_run,
            generate_only=args.generate_only,
        )

    env_overrides = proxy_environment(args)
    if not compose_file.exists():
        raise SystemExit(f"Compose file not found: {compose_file}")
    if args.dry_run or args.generate_only:
        for key in sorted(env_overrides):
            print(f"{key}={env_overrides[key]}")
    return run_compose(
        command,
        env_overrides,
        dry_run=args.dry_run,
        generate_only=args.generate_only,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return serve_command(args)
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
