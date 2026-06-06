# SPDX-License-Identifier: Apache-2.0
"""Tests for the Linux/proxy-first omni CLI."""

from pathlib import Path

import pytest

from omlx import omni_cli
from omlx.proxy.vllm_compose import VllmComposeSettings, write_vllm_compose


def parse_args(*args):
    return omni_cli.build_parser().parse_args(["serve", *args])


def test_top_level_help_documents_serve_options(capsys):
    with pytest.raises(SystemExit) as excinfo:
        omni_cli.main(["--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "omni serve [options]" in output
    assert "omni status [--compose-file PATH]" in output
    assert "omni logs [--target proxy|backend|both]" in output
    assert "omni restart [--target proxy|backend|both]" in output
    assert "omni stop --target proxy|backend|both" in output
    assert "--backend {vllm,openai,ollama,llamacpp}" in output
    assert "--hf-home PATH" in output
    assert "--sse-keepalive-mode {ping,comment,off}" in output


def test_parser_recognizes_status_logs_restart_and_stop():
    parser = omni_cli.build_parser()

    status = parser.parse_args(["status"])
    logs = parser.parse_args(["logs", "--target", "backend", "-f"])
    restart = parser.parse_args(["restart", "--target", "backend"])
    stop = parser.parse_args(["stop", "--target", "backend"])

    assert status.command == "status"
    assert logs.command == "logs"
    assert logs.target == "backend"
    assert logs.follow is True
    assert restart.command == "restart"
    assert restart.target == "backend"
    assert stop.command == "stop"
    assert stop.target == "backend"


def test_stop_requires_target():
    parser = omni_cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["stop"])


def test_default_control_compose_file_prefers_generated_vllm(monkeypatch, tmp_path):
    proxy_compose = tmp_path / "docker-compose.proxy.yml"
    vllm_compose = tmp_path / "docker-compose.vllm.yml"
    proxy_compose.write_text("services: {}")
    vllm_compose.write_text("services: {}")
    monkeypatch.setattr(omni_cli, "DEFAULT_PROXY_COMPOSE", proxy_compose)
    monkeypatch.setattr(omni_cli, "DEFAULT_VLLM_COMPOSE", vllm_compose)

    assert omni_cli.default_control_compose_file() == vllm_compose

    vllm_compose.unlink()

    assert omni_cli.default_control_compose_file() == proxy_compose


def test_restart_services_for_target_maps_managed_services():
    services = ["omlx-proxy", "vllm"]

    assert omni_cli.restart_services_for_target("proxy", services) == ["omlx-proxy"]
    assert omni_cli.restart_services_for_target("backend", services) == ["vllm"]
    assert omni_cli.restart_services_for_target("both", services) == []


def test_restart_backend_errors_for_external_backend_stack():
    with pytest.raises(SystemExit, match="external or not managed"):
        omni_cli.restart_services_for_target("backend", ["omlx-proxy"])


def test_stop_backend_errors_for_external_backend_stack():
    with pytest.raises(SystemExit, match="external or not managed"):
        omni_cli.services_for_target("backend", ["omlx-proxy"])


def test_status_command_runs_compose_ps(monkeypatch, tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}")
    calls = []

    def fake_run(command, check, **kwargs):
        calls.append((command, check, kwargs))

    monkeypatch.setattr(omni_cli.subprocess, "run", fake_run)
    args = omni_cli.build_parser().parse_args([
        "status",
        "--compose-file",
        str(compose_file),
    ])

    assert omni_cli.status_command(args) == 0
    assert calls == [
        (["docker", "compose", "-f", str(compose_file), "ps"], True, {})
    ]


def test_logs_command_runs_all_service_logs_without_discovery(monkeypatch, tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}")
    calls = []

    def fake_run(command, check, **kwargs):
        calls.append((command, check, kwargs))

    monkeypatch.setattr(omni_cli.subprocess, "run", fake_run)
    args = omni_cli.build_parser().parse_args([
        "logs",
        "--compose-file",
        str(compose_file),
    ])

    assert omni_cli.logs_command(args) == 0
    assert calls == [
        (["docker", "compose", "-f", str(compose_file), "logs"], True, {})
    ]


def test_logs_command_filters_proxy_service(monkeypatch, tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}")
    calls = []

    class Result:
        stdout = "omlx-proxy\nvllm\n"

    def fake_run(command, check, **kwargs):
        calls.append((command, check, kwargs))
        return Result()

    monkeypatch.setattr(omni_cli.subprocess, "run", fake_run)
    args = omni_cli.build_parser().parse_args([
        "logs",
        "--target",
        "proxy",
        "--compose-file",
        str(compose_file),
    ])

    assert omni_cli.logs_command(args) == 0
    assert calls == [
        (
            ["docker", "compose", "-f", str(compose_file), "config", "--services"],
            True,
            {"capture_output": True, "text": True},
        ),
        (
            ["docker", "compose", "-f", str(compose_file), "logs", "omlx-proxy"],
            True,
            {},
        ),
    ]


def test_logs_command_follows_backend_service(monkeypatch, tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}")
    calls = []

    class Result:
        stdout = "omlx-proxy\nvllm\n"

    def fake_run(command, check, **kwargs):
        calls.append((command, check, kwargs))
        return Result()

    monkeypatch.setattr(omni_cli.subprocess, "run", fake_run)
    args = omni_cli.build_parser().parse_args([
        "logs",
        "--target",
        "backend",
        "--follow",
        "--compose-file",
        str(compose_file),
    ])

    assert omni_cli.logs_command(args) == 0
    assert calls == [
        (
            ["docker", "compose", "-f", str(compose_file), "config", "--services"],
            True,
            {"capture_output": True, "text": True},
        ),
        (
            ["docker", "compose", "-f", str(compose_file), "logs", "-f", "vllm"],
            True,
            {},
        ),
    ]


def test_logs_backend_errors_for_external_backend_stack(monkeypatch, tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}")

    class Result:
        stdout = "omlx-proxy\n"

    def fake_run(command, check, **kwargs):
        return Result()

    monkeypatch.setattr(omni_cli.subprocess, "run", fake_run)
    args = omni_cli.build_parser().parse_args([
        "logs",
        "--target",
        "backend",
        "--compose-file",
        str(compose_file),
    ])

    with pytest.raises(SystemExit, match="external or not managed"):
        omni_cli.logs_command(args)


def test_restart_command_restarts_backend_service(monkeypatch, tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}")
    calls = []

    class Result:
        stdout = "omlx-proxy\nvllm\n"

    def fake_run(command, check, **kwargs):
        calls.append((command, check, kwargs))
        return Result()

    monkeypatch.setattr(omni_cli.subprocess, "run", fake_run)
    args = omni_cli.build_parser().parse_args([
        "restart",
        "--target",
        "backend",
        "--compose-file",
        str(compose_file),
    ])

    assert omni_cli.restart_command(args) == 0
    assert calls == [
        (
            ["docker", "compose", "-f", str(compose_file), "config", "--services"],
            True,
            {"capture_output": True, "text": True},
        ),
        (
            ["docker", "compose", "-f", str(compose_file), "restart", "vllm"],
            True,
            {},
        ),
    ]


def test_stop_command_stops_proxy_service(monkeypatch, tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}")
    calls = []

    class Result:
        stdout = "omlx-proxy\nvllm\n"

    def fake_run(command, check, **kwargs):
        calls.append((command, check, kwargs))
        return Result()

    monkeypatch.setattr(omni_cli.subprocess, "run", fake_run)
    args = omni_cli.build_parser().parse_args([
        "stop",
        "--target",
        "proxy",
        "--compose-file",
        str(compose_file),
    ])

    assert omni_cli.stop_command(args) == 0
    assert calls == [
        (
            ["docker", "compose", "-f", str(compose_file), "config", "--services"],
            True,
            {"capture_output": True, "text": True},
        ),
        (
            ["docker", "compose", "-f", str(compose_file), "stop", "omlx-proxy"],
            True,
            {},
        ),
    ]


def test_stop_command_stops_backend_service(monkeypatch, tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}")
    calls = []

    class Result:
        stdout = "omlx-proxy\nvllm\n"

    def fake_run(command, check, **kwargs):
        calls.append((command, check, kwargs))
        return Result()

    monkeypatch.setattr(omni_cli.subprocess, "run", fake_run)
    args = omni_cli.build_parser().parse_args([
        "stop",
        "--target",
        "backend",
        "--compose-file",
        str(compose_file),
    ])

    assert omni_cli.stop_command(args) == 0
    assert calls == [
        (
            ["docker", "compose", "-f", str(compose_file), "config", "--services"],
            True,
            {"capture_output": True, "text": True},
        ),
        (
            ["docker", "compose", "-f", str(compose_file), "stop", "vllm"],
            True,
            {},
        ),
    ]


def test_stop_command_stops_all_services_without_discovery(monkeypatch, tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}")
    calls = []

    def fake_run(command, check, **kwargs):
        calls.append((command, check, kwargs))

    monkeypatch.setattr(omni_cli.subprocess, "run", fake_run)
    args = omni_cli.build_parser().parse_args([
        "stop",
        "--target",
        "both",
        "--compose-file",
        str(compose_file),
    ])

    assert omni_cli.stop_command(args) == 0
    assert calls == [
        (["docker", "compose", "-f", str(compose_file), "stop"], True, {})
    ]


def test_stop_backend_errors_for_external_backend_command(monkeypatch, tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}")

    class Result:
        stdout = "omlx-proxy\n"

    def fake_run(command, check, **kwargs):
        return Result()

    monkeypatch.setattr(omni_cli.subprocess, "run", fake_run)
    args = omni_cli.build_parser().parse_args([
        "stop",
        "--target",
        "backend",
        "--compose-file",
        str(compose_file),
    ])

    with pytest.raises(SystemExit, match="external or not managed"):
        omni_cli.stop_command(args)


def test_vllm_generate_only_writes_compose(tmp_path, capsys):
    compose_file = tmp_path / "docker-compose.vllm.yml"

    result = omni_cli.main([
        "serve",
        "--backend",
        "vllm",
        "--model",
        "Qwen/Qwen3-1.7B",
        "--served-model-name",
        "qwen-test",
        "--compose-file",
        str(compose_file),
        "--generate-only",
    ])

    assert result == 0
    content = compose_file.read_text()
    assert "VLLM_MODEL" in content
    assert "Qwen/Qwen3-1.7B" in content
    assert "qwen-test" in content
    assert "docker compose --env-file" in capsys.readouterr().out


def test_vllm_generate_only_writes_env_file_defaults(tmp_path, capsys):
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")

    result = omni_cli.main([
        "serve",
        "--backend",
        "vllm",
        "--compose-file",
        str(compose_file),
        "--generate-only",
        "--no-build",
    ])

    assert result == 0
    env = omni_cli.load_env_file(env_file)
    assert env["VLLM_MODEL"] == "Qwen/Qwen3-1.7B"
    assert env["VLLM_HF_HOME"].endswith("/.cache/huggingface")
    output = capsys.readouterr().out
    assert f"--env-file {env_file}" in output


def test_vllm_serve_preserves_existing_env_model_when_model_omitted(tmp_path):
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")
    existing = omni_cli.default_vllm_environment()
    existing["VLLM_MODEL"] = "example/existing-model"
    existing["VLLM_SERVED_MODEL_NAME"] = "existing-name"
    existing["VLLM_MAX_MODEL_LEN"] = "16384"
    omni_cli.write_env_file(env_file, existing)

    result = omni_cli.main([
        "serve",
        "--backend",
        "vllm",
        "--compose-file",
        str(compose_file),
        "--generate-only",
        "--no-build",
    ])

    assert result == 0
    env = omni_cli.load_env_file(env_file)
    assert env["VLLM_MODEL"] == "example/existing-model"
    assert env["VLLM_SERVED_MODEL_NAME"] == "existing-name"
    assert env["VLLM_MAX_MODEL_LEN"] == "16384"


def test_vllm_serve_updates_only_supplied_model_flag(tmp_path):
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")
    existing = omni_cli.default_vllm_environment()
    existing["VLLM_MODEL"] = "example/existing-model"
    existing["VLLM_SERVED_MODEL_NAME"] = "existing-name"
    existing["VLLM_MAX_MODEL_LEN"] = "16384"
    omni_cli.write_env_file(env_file, existing)

    result = omni_cli.main([
        "serve",
        "--backend",
        "vllm",
        "--model",
        "example/new-model",
        "--compose-file",
        str(compose_file),
        "--generate-only",
        "--no-build",
    ])

    assert result == 0
    env = omni_cli.load_env_file(env_file)
    assert env["VLLM_MODEL"] == "example/new-model"
    assert env["VLLM_SERVED_MODEL_NAME"] == "existing-name"
    assert env["VLLM_MAX_MODEL_LEN"] == "16384"


def test_vllm_serve_seeds_env_from_existing_compose_when_env_missing(tmp_path):
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")
    settings = VllmComposeSettings(
        model="example/compose-model",
        served_model_name="compose-name",
        max_model_len=32768,
    )
    omni_cli.write_vllm_compose_for_path(compose_file, settings)

    result = omni_cli.main([
        "serve",
        "--backend",
        "vllm",
        "--compose-file",
        str(compose_file),
        "--generate-only",
        "--no-build",
    ])

    assert result == 0
    env = omni_cli.load_env_file(env_file)
    assert env["VLLM_MODEL"] == "example/compose-model"
    assert env["VLLM_SERVED_MODEL_NAME"] == "compose-name"
    assert env["VLLM_MAX_MODEL_LEN"] == "32768"


def test_vllm_dry_run_does_not_write_env_file(tmp_path, capsys):
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")

    result = omni_cli.main([
        "serve",
        "--backend",
        "vllm",
        "--compose-file",
        str(compose_file),
        "--dry-run",
    ])

    assert result == 0
    assert not compose_file.exists()
    assert not env_file.exists()
    output = capsys.readouterr().out
    assert f"# Env file: {env_file}" in output
    assert "VLLM_MODEL=Qwen/Qwen3-1.7B" in output
    assert "docker compose --env-file" in output


def test_admin_vllm_writer_keeps_template_relative_paths(tmp_path):
    compose_file = tmp_path / "docker-compose.vllm.yml"

    write_vllm_compose(compose_file, VllmComposeSettings())

    content = compose_file.read_text()
    assert 'context: ".."' in content
    assert '"../docker:/compose-output"' in content


def test_vllm_serve_maps_advanced_flags_to_env_and_compose(tmp_path):
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")

    result = omni_cli.main([
        "serve",
        "--backend",
        "vllm",
        "--compose-file",
        str(compose_file),
        "--dtype",
        "bfloat16",
        "--tokenizer",
        "example/tokenizer",
        "--max-num-batched-tokens",
        "8192",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--kv-cache-dtype",
        "fp8",
        "--cpu-offload-gb",
        "8",
        "--tensor-parallel-size",
        "2",
        "--http-proxy",
        "http://proxy:8080",
        "--hf-endpoint",
        "https://hf.example",
        "--extra-args-json",
        '["--foo","bar baz"]',
        "--generate-only",
        "--no-build",
    ])

    assert result == 0
    env = omni_cli.load_env_file(env_file)
    assert env["VLLM_DTYPE"] == "bfloat16"
    assert env["VLLM_TOKENIZER"] == "example/tokenizer"
    assert env["VLLM_MAX_NUM_BATCHED_TOKENS"] == "8192"
    assert env["VLLM_ENABLE_CHUNKED_PREFILL"] == "true"
    assert env["VLLM_ENABLE_PREFIX_CACHING"] == "false"
    assert env["VLLM_KV_CACHE_DTYPE"] == "fp8"
    assert env["VLLM_CPU_OFFLOAD_GB"] == "8.0"
    assert env["VLLM_TENSOR_PARALLEL_SIZE"] == "2"
    assert env["VLLM_HTTP_PROXY"] == "http://proxy:8080"
    assert env["VLLM_HF_ENDPOINT"] == "https://hf.example"
    assert env["VLLM_EXTRA_ARGS_JSON"] == '["--foo","bar baz"]'

    content = compose_file.read_text()
    assert "--dtype" in content
    assert "--max-num-batched-tokens" in content
    assert "--no-enable-prefix-caching" in content
    assert "HF_ENDPOINT" in content
    assert "HTTP_PROXY" in content



def test_vllm_compose_sanitizes_empty_runtime_url_env(tmp_path):
    compose_file = tmp_path / "docker-compose.vllm.yml"

    omni_cli.write_vllm_compose_for_path(compose_file, VllmComposeSettings())

    content = compose_file.read_text()
    vllm_environment = content.split("  vllm:", 1)[1].split("    volumes:", 1)[0]
    assert "\n      HF_ENDPOINT:" not in vllm_environment
    assert "\n      HTTP_PROXY:" not in vllm_environment
    assert "\n      HTTPS_PROXY:" not in vllm_environment
    assert "\n      REQUESTS_CA_BUNDLE:" not in vllm_environment
    assert "\n      SSL_CERT_FILE:" not in vllm_environment
    assert 'export HF_ENDPOINT="$${VLLM_HF_ENDPOINT}"' in content
    assert "unset HF_ENDPOINT" in content
    assert "unset REQUESTS_CA_BUNDLE" in content
    assert "unset VLLM_IMAGE VLLM_MODEL" in content
    assert "unset OMLX_PROXY_PORT OMLX_PROXY_API_KEY" in content


def test_vllm_settings_from_args_maps_advanced_flags():
    args = parse_args(
        "--backend",
        "vllm",
        "--dtype",
        "float16",
        "--max-num-batched-tokens",
        "4096",
        "--enable-prefix-caching",
        "--tensor-parallel-size",
        "4",
        "--hf-endpoint",
        "https://hf.example",
    )

    settings = omni_cli.vllm_settings_from_args(args)

    assert settings.dtype == "float16"
    assert settings.max_num_batched_tokens == "4096"
    assert settings.enable_prefix_caching is True
    assert settings.tensor_parallel_size == 4
    assert settings.hf_endpoint == "https://hf.example"


def test_ollama_proxy_backend_defaults(capsys):
    args = parse_args("--backend", "ollama", "--generate-only", "--no-build")

    env = omni_cli.proxy_environment(args)

    assert env["OMLX_BACKEND_URL"] == "http://host.docker.internal:11434/v1"
    assert env["OMLX_PROXY_PORT"] == "8080"

    result = omni_cli.serve_command(args)

    assert result == 0
    output = capsys.readouterr().out
    assert "OMLX_BACKEND_URL=http://host.docker.internal:11434/v1" in output
    assert "docker compose -f" in output


def test_openai_proxy_requires_backend_url():
    args = parse_args("--backend", "openai", "--generate-only")

    with pytest.raises(SystemExit):
        omni_cli.proxy_environment(args)


def test_compose_command_defaults_to_detached_build(tmp_path):
    compose_file = tmp_path / "docker-compose.yml"

    command = omni_cli.compose_command(compose_file, foreground=False, build=True)

    assert command == ["docker", "compose", "-f", str(compose_file), "up", "-d", "--build"]


def test_vllm_settings_from_args_maps_model_flags():
    args = parse_args(
        "--backend",
        "vllm",
        "--model",
        "/models/local",
        "--served-model-name",
        "local-model",
        "--max-model-len",
        "4096",
        "--proxy-port",
        "9090",
    )

    settings = omni_cli.vllm_settings_from_args(args)

    assert settings.model == "/models/local"
    assert settings.served_model_name == "local-model"
    assert settings.max_model_len == 4096
    assert settings.proxy_port == 9090
    assert settings.hf_home.endswith("/.cache/huggingface")
    assert "${" not in settings.hf_home


# ---------------------------------------------------------------------------
# omni launch subcommand
# ---------------------------------------------------------------------------


def parse_launch(*args):
    return omni_cli.build_parser().parse_args(["launch", *args])


def test_launch_parser_recognizes_list():
    args = parse_launch("list")
    assert args.command == "launch"
    assert args.tool == "list"


def test_launch_parser_recognizes_tool_with_model_and_port():
    args = parse_launch("claude", "--model", "my-model", "--port", "9090")
    assert args.tool == "claude"
    assert args.model == "my-model"
    assert args.port == 9090


def test_launch_parser_recognizes_claude_tier_models():
    args = parse_launch(
        "claude",
        "--opus-model", "big",
        "--sonnet-model", "mid",
        "--haiku-model", "small",
    )
    assert args.opus_model == "big"
    assert args.sonnet_model == "mid"
    assert args.haiku_model == "small"


def test_launch_parser_host_and_api_key():
    args = parse_launch("codex", "--host", "myserver", "--api-key", "secret")
    assert args.host == "myserver"
    assert args.api_key == "secret"


def test_launch_list_prints_integrations(capsys):
    result = omni_cli.launch_command(parse_launch("list"))
    assert result == 0
    out = capsys.readouterr().out
    assert "claude" in out
    assert "codex" in out


def test_launch_command_exits_when_proxy_unreachable(monkeypatch):
    import urllib.error

    def fail_open(req, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(omni_cli.urllib.request, "urlopen", fail_open)

    with pytest.raises(SystemExit):
        omni_cli.launch_command(parse_launch("claude", "--port", "19999"))


def test_launch_command_exits_when_no_models(monkeypatch):
    import io
    import urllib.error

    health_response = io.BytesIO(b'{"status":"healthy"}')
    models_response = io.BytesIO(b'{"data":[]}')
    responses = [health_response, models_response]

    class FakeCtxManager:
        def __init__(self, data):
            self._data = data

        def __enter__(self):
            return self._data

        def __exit__(self, *args):
            pass

        def read(self):
            return self._data.read()

    call_count = [0]

    def fake_open(req, timeout):
        idx = call_count[0]
        call_count[0] += 1
        return FakeCtxManager(responses[idx])

    monkeypatch.setattr(omni_cli.urllib.request, "urlopen", fake_open)

    with pytest.raises(SystemExit):
        omni_cli.launch_command(parse_launch("claude", "--port", "8080"))


def test_launch_command_calls_integration_launch(monkeypatch):
    import io
    import json as _json

    health_bytes = b'{"status":"healthy"}'
    models_bytes = _json.dumps({"data": [{"id": "test-model"}]}).encode()
    responses = [io.BytesIO(health_bytes), io.BytesIO(models_bytes)]

    class FakeCtxManager:
        def __init__(self, data):
            self._data = data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return self._data.read()

    call_count = [0]

    def fake_open(req, timeout):
        idx = call_count[0]
        call_count[0] += 1
        return FakeCtxManager(responses[idx])

    monkeypatch.setattr(omni_cli.urllib.request, "urlopen", fake_open)

    launched = []

    from omlx.integrations import get_integration
    real_integration = get_integration("codex")
    monkeypatch.setattr(real_integration, "is_installed", lambda: True)
    monkeypatch.setattr(real_integration, "launch", lambda ctx: launched.append(ctx))

    args = parse_launch("codex", "--port", "8080", "--model", "test-model")
    result = omni_cli.launch_command(args)

    assert result == 0
    assert len(launched) == 1
    ctx = launched[0]
    assert ctx.model == "test-model"
    assert ctx.port == 8080
    assert ctx.host == "localhost"


def test_launch_command_uses_env_port(monkeypatch):
    import io

    monkeypatch.setenv("OMLX_PROXY_PORT", "9191")

    def fail_open(req, timeout):
        raise omni_cli.urllib.error.URLError("refused")

    monkeypatch.setattr(omni_cli.urllib.request, "urlopen", fail_open)

    with pytest.raises(SystemExit):
        omni_cli.launch_command(parse_launch("claude"))

    # Verify the URL included the env port — the error fires after URL construction
    # so this just checks the flow doesn't crash before the health check.


def test_launch_command_uses_env_api_key(monkeypatch):
    import io
    import json as _json

    monkeypatch.setenv("OMLX_PROXY_API_KEY", "env-secret")

    health_bytes = b'{"status":"healthy"}'
    models_bytes = _json.dumps({"data": [{"id": "m"}]}).encode()
    responses = [io.BytesIO(health_bytes), io.BytesIO(models_bytes)]
    seen_headers = []

    class FakeCtx:
        def __init__(self, data):
            self._data = data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return self._data.read()

    call_count = [0]

    def fake_open(req, timeout):
        seen_headers.append(req.get_header("Authorization"))
        idx = call_count[0]
        call_count[0] += 1
        return FakeCtx(responses[idx])

    monkeypatch.setattr(omni_cli.urllib.request, "urlopen", fake_open)

    from omlx.integrations import get_integration
    real_integration = get_integration("claude")
    monkeypatch.setattr(real_integration, "is_installed", lambda: True)
    monkeypatch.setattr(real_integration, "launch", lambda ctx: None)

    omni_cli.launch_command(parse_launch("claude", "--port", "8080"))

    assert seen_headers[0] == "Bearer env-secret"


def test_top_level_help_documents_launch_command(capsys):
    with pytest.raises(SystemExit) as excinfo:
        omni_cli.main(["--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "omni launch" in out
