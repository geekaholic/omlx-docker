# SPDX-License-Identifier: Apache-2.0
"""Tests for the Linux/proxy-first omni CLI."""

import json

import pytest

from omlx import omni_cli
from omlx.proxy.vllm_compose import (
    VllmComposeSettings,
    write_vllm_compose,
    write_vllm_env_file,
)


def parse_args(*args):
    return omni_cli.build_parser().parse_args(["serve", *args])


@pytest.fixture(autouse=True)
def isolate_omni_local_state(monkeypatch, tmp_path):
    monkeypatch.setattr(
        omni_cli,
        "DEFAULT_SERVE_STATE_FILE",
        tmp_path / "omni-serve.json",
    )
    monkeypatch.setattr(
        omni_cli,
        "DEFAULT_PROXY_ENV_FILE",
        tmp_path / "docker-compose.proxy.env",
    )


def _final_serve_env(model):
    """Reproduce the managed-serve env pipeline: baseline -> merge -> resolve."""
    args = parse_args("--backend", "vllm", "--model", model)
    spec = omni_cli.backend_spec("vllm")
    merged = omni_cli.merged_sidecar_environment(
        "vllm", args, env_file=None, compose_file=None
    )
    settings = spec.settings_from_env(merged)
    return spec.environment(settings)


def test_fresh_serve_auto_detects_gemma4_tool_parser():
    # The reported bug: a fresh `omni serve --model <gemma-4>` must wire up the
    # gemma4 tool parser, not inherit the default model's hermes parser, and
    # must never write the literal "auto" sentinel into the env file.
    env = _final_serve_env("google/gemma-4-26B-A4B-it")
    assert env["VLLM_TOOL_CALL_PARSER"] == "gemma4"
    assert env["VLLM_REASONING_PARSER"] == "gemma4"
    assert env["VLLM_CHAT_TEMPLATE"].endswith("tool_chat_template_gemma4.jinja")


def test_fresh_serve_unknown_model_leaves_parser_empty():
    # An unrecognized family must not pass a bogus parser; the hardened launch
    # guard then skips --enable-auto-tool-choice so vLLM still starts.
    env = _final_serve_env("microsoft/phi-4")
    assert env["VLLM_TOOL_CALL_PARSER"] == ""


def test_max_output_tokens_flag_maps_to_env():
    args = parse_args(
        "--backend", "vllm", "--model", "m", "--max-output-tokens", "8192"
    )
    env = omni_cli.portable_cli_environment(args)
    assert env["OMLX_SAMPLING_MAX_TOKENS"] == "8192"


def test_max_output_tokens_omitted_is_auto():
    args = parse_args("--backend", "vllm", "--model", "m")
    env = omni_cli.portable_cli_environment(args)
    assert "OMLX_SAMPLING_MAX_TOKENS" not in env


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
    assert "--backend {vllm,llamacpp,openai}" in output
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


def test_optimize_flag_selects_vllm_backend():
    args = parse_args("--optimize")
    assert args.optimize is True
    assert omni_cli.resolve_serve_backend(args, {}) == "vllm"


def test_default_control_compose_file_prefers_generated_vllm(monkeypatch, tmp_path):
    proxy_compose = tmp_path / "docker-compose.proxy.yml"
    vllm_compose = tmp_path / "docker-compose.vllm.yml"
    llamacpp_compose = tmp_path / "docker-compose.llamacpp.yml"
    proxy_compose.write_text("services: {}")
    vllm_compose.write_text("services: {}")
    llamacpp_compose.write_text("services: {}")
    monkeypatch.setattr(omni_cli, "DEFAULT_PROXY_COMPOSE", proxy_compose)
    monkeypatch.setattr(omni_cli, "DEFAULT_VLLM_COMPOSE", vllm_compose)
    monkeypatch.setattr(omni_cli, "DEFAULT_LLAMACPP_COMPOSE", llamacpp_compose)

    assert omni_cli.default_control_compose_file() == vllm_compose

    vllm_compose.unlink()

    assert omni_cli.default_control_compose_file() == llamacpp_compose

    llamacpp_compose.unlink()

    assert omni_cli.default_control_compose_file() == proxy_compose


def test_default_control_compose_file_prefers_saved_state(monkeypatch, tmp_path):
    proxy_compose = tmp_path / "docker-compose.proxy.yml"
    vllm_compose = tmp_path / "docker-compose.vllm.yml"
    proxy_compose.write_text("services: {}")
    vllm_compose.write_text("services: {}")
    monkeypatch.setattr(omni_cli, "DEFAULT_PROXY_COMPOSE", proxy_compose)
    monkeypatch.setattr(omni_cli, "DEFAULT_VLLM_COMPOSE", vllm_compose)

    omni_cli.save_serve_state(backend="openai", compose_file=proxy_compose)

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
    args = omni_cli.build_parser().parse_args(
        [
            "status",
            "--compose-file",
            str(compose_file),
        ]
    )

    assert omni_cli.status_command(args) == 0
    assert calls == [(["docker", "compose", "-f", str(compose_file), "ps"], True, {})]


def test_logs_command_runs_all_service_logs_without_discovery(monkeypatch, tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}")
    calls = []

    def fake_run(command, check, **kwargs):
        calls.append((command, check, kwargs))

    monkeypatch.setattr(omni_cli.subprocess, "run", fake_run)
    args = omni_cli.build_parser().parse_args(
        [
            "logs",
            "--compose-file",
            str(compose_file),
        ]
    )

    assert omni_cli.logs_command(args) == 0
    assert calls == [(["docker", "compose", "-f", str(compose_file), "logs"], True, {})]


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
    args = omni_cli.build_parser().parse_args(
        [
            "logs",
            "--target",
            "proxy",
            "--compose-file",
            str(compose_file),
        ]
    )

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
    args = omni_cli.build_parser().parse_args(
        [
            "logs",
            "--target",
            "backend",
            "--follow",
            "--compose-file",
            str(compose_file),
        ]
    )

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
    args = omni_cli.build_parser().parse_args(
        [
            "logs",
            "--target",
            "backend",
            "--compose-file",
            str(compose_file),
        ]
    )

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
    args = omni_cli.build_parser().parse_args(
        [
            "restart",
            "--target",
            "backend",
            "--compose-file",
            str(compose_file),
        ]
    )

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
    args = omni_cli.build_parser().parse_args(
        [
            "stop",
            "--target",
            "proxy",
            "--compose-file",
            str(compose_file),
        ]
    )

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
    args = omni_cli.build_parser().parse_args(
        [
            "stop",
            "--target",
            "backend",
            "--compose-file",
            str(compose_file),
        ]
    )

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
    args = omni_cli.build_parser().parse_args(
        [
            "stop",
            "--target",
            "both",
            "--compose-file",
            str(compose_file),
        ]
    )

    assert omni_cli.stop_command(args) == 0
    assert calls == [(["docker", "compose", "-f", str(compose_file), "stop"], True, {})]


def test_stop_backend_errors_for_external_backend_command(monkeypatch, tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}")

    class Result:
        stdout = "omlx-proxy\n"

    def fake_run(command, check, **kwargs):
        return Result()

    monkeypatch.setattr(omni_cli.subprocess, "run", fake_run)
    args = omni_cli.build_parser().parse_args(
        [
            "stop",
            "--target",
            "backend",
            "--compose-file",
            str(compose_file),
        ]
    )

    with pytest.raises(SystemExit, match="external or not managed"):
        omni_cli.stop_command(args)


def test_vllm_generate_only_writes_compose(tmp_path, capsys):
    compose_file = tmp_path / "docker-compose.vllm.yml"

    result = omni_cli.main(
        [
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
        ]
    )

    assert result == 0
    content = compose_file.read_text()
    assert "OMNI_MODEL" in content
    assert "Qwen/Qwen3-1.7B" in content
    assert "qwen-test" in content
    assert "docker compose --env-file" in capsys.readouterr().out


def test_serve_without_backend_defaults_to_proxy_stack(capsys):
    result = omni_cli.main(["serve", "--generate-only", "--no-build"])

    assert result == 0
    output = capsys.readouterr().out
    assert "OMLX_BACKEND_URL=http://host.docker.internal:11434/v1" in output
    assert f"--env-file {omni_cli.DEFAULT_PROXY_ENV_FILE}" in output
    assert f"-f {omni_cli.DEFAULT_PROXY_COMPOSE}" in output

    env = omni_cli.load_env_file(omni_cli.DEFAULT_PROXY_ENV_FILE)
    assert env["OMLX_BACKEND_URL"] == "http://host.docker.internal:11434/v1"
    assert env["OMLX_PROXY_PORT"] == "8080"

    state = omni_cli.load_serve_state()
    assert state["backend"] == "openai"
    assert state["compose_file"] == str(omni_cli.DEFAULT_PROXY_COMPOSE)


def test_vllm_serve_persists_last_backend(tmp_path):
    compose_file = tmp_path / "docker-compose.vllm.yml"

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "vllm",
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )

    assert result == 0
    state = omni_cli.load_serve_state()
    assert state["backend"] == "vllm"
    assert state["compose_file"] == str(compose_file)


def test_plain_serve_reuses_saved_vllm_backend(tmp_path, capsys):
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")
    omni_cli.save_serve_state(backend="vllm", compose_file=compose_file)

    result = omni_cli.main(["serve", "--generate-only", "--no-build"])

    assert result == 0
    assert compose_file.exists()
    assert env_file.exists()
    output = capsys.readouterr().out
    assert f"--env-file {env_file}" in output
    assert f"-f {compose_file}" in output


def test_vllm_generate_only_writes_env_file_defaults(tmp_path, capsys):
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "vllm",
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )

    assert result == 0
    env = omni_cli.load_env_file(env_file)
    assert env["OMNI_MODEL"] == "Qwen/Qwen3-1.7B"
    assert env["OMNI_HF_HOME"].endswith("/.cache/huggingface")
    output = capsys.readouterr().out
    assert f"--env-file {env_file}" in output


def test_vllm_serve_preserves_existing_env_model_when_model_omitted(tmp_path):
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")
    existing = omni_cli.default_vllm_environment()
    existing["OMNI_MODEL"] = "example/existing-model"
    existing["OMNI_SERVED_MODEL_NAME"] = "existing-name"
    existing["OMNI_CONTEXT_LENGTH"] = "16384"
    write_vllm_env_file(env_file, existing)

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "vllm",
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )

    assert result == 0
    env = omni_cli.load_env_file(env_file)
    assert env["OMNI_MODEL"] == "example/existing-model"
    assert env["OMNI_SERVED_MODEL_NAME"] == "existing-name"
    assert env["OMNI_CONTEXT_LENGTH"] == "16384"


def test_vllm_serve_syncs_served_name_with_new_model(tmp_path):
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")
    existing = omni_cli.default_vllm_environment()
    existing["OMNI_MODEL"] = "example/existing-model"
    existing["OMNI_SERVED_MODEL_NAME"] = "existing-name"
    existing["OMNI_CONTEXT_LENGTH"] = "16384"
    write_vllm_env_file(env_file, existing)

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "vllm",
            "--model",
            "example/new-model",
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )

    assert result == 0
    env = omni_cli.load_env_file(env_file)
    assert env["OMNI_MODEL"] == "example/new-model"
    # Changing --model without --served-model-name re-derives the served name
    # so the new model isn't mislabeled with the previous session's name.
    assert env["OMNI_SERVED_MODEL_NAME"] == "new-model"
    # Other unsupplied fields are still preserved.
    assert env["OMNI_CONTEXT_LENGTH"] == "16384"


def test_vllm_serve_explicit_served_name_wins_over_derived(tmp_path):
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "vllm",
            "--model",
            "example/new-model",
            "--served-model-name",
            "custom-name",
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )

    assert result == 0
    env = omni_cli.load_env_file(env_file)
    assert env["OMNI_SERVED_MODEL_NAME"] == "custom-name"


def test_vllm_serve_seeds_env_from_existing_compose_when_env_missing(tmp_path):
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")
    settings = VllmComposeSettings(
        model="example/compose-model",
        served_model_name="compose-name",
        context_length=32768,
    )
    omni_cli.write_vllm_compose_for_path(compose_file, settings)

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "vllm",
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )

    assert result == 0
    env = omni_cli.load_env_file(env_file)
    assert env["OMNI_MODEL"] == "example/compose-model"
    assert env["OMNI_SERVED_MODEL_NAME"] == "compose-name"
    assert env["OMNI_CONTEXT_LENGTH"] == "32768"


def test_vllm_dry_run_does_not_write_env_file(tmp_path, capsys):
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "vllm",
            "--compose-file",
            str(compose_file),
            "--dry-run",
        ]
    )

    assert result == 0
    assert not compose_file.exists()
    assert not env_file.exists()
    output = capsys.readouterr().out
    assert f"# Env file: {env_file}" in output
    assert "OMNI_MODEL=Qwen/Qwen3-1.7B" in output
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

    result = omni_cli.main(
        [
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
        ]
    )

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
    assert env["OMNI_HTTP_PROXY"] == "http://proxy:8080"
    assert env["OMNI_HF_ENDPOINT"] == "https://hf.example"
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
    assert 'export HF_ENDPOINT="$${OMNI_HF_ENDPOINT}"' in content
    assert "unset HF_ENDPOINT" in content
    assert "unset REQUESTS_CA_BUNDLE" in content
    assert "unset OMNI_MODEL OMNI_SERVED_MODEL_NAME" in content
    assert "unset VLLM_IMAGE VLLM_GPU_MEMORY_UTILIZATION" in content
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


def test_openai_proxy_backend_defaults_to_ollama_url(capsys):
    args = parse_args("--backend", "openai", "--generate-only", "--no-build")

    env = omni_cli.proxy_environment(args)

    assert env["OMLX_BACKEND_URL"] == "http://host.docker.internal:11434/v1"
    assert env["OMLX_PROXY_PORT"] == "8080"

    result = omni_cli.serve_command(args)

    assert result == 0
    output = capsys.readouterr().out
    assert "OMLX_BACKEND_URL=http://host.docker.internal:11434/v1" in output
    assert "docker compose --env-file" in output


def test_proxy_serve_preserves_existing_env_when_flags_omitted(tmp_path):
    compose_file = tmp_path / "docker-compose.proxy.yml"
    env_file = compose_file.with_suffix(".env")
    compose_file.write_text("services: {}")
    omni_cli.write_generic_env_file(
        env_file,
        {
            "OMLX_BACKEND_URL": "https://api.example.test/v1",
            "OMLX_BACKEND_API_KEY": "backend-secret",
            "OMLX_PROXY_API_KEY": "proxy-secret",
            "OMLX_PROXY_PORT": "8080",
            "OMLX_CONTEXT_SCALING": "true",
            "OMLX_TARGET_CONTEXT_SIZE": "123456",
            "OMLX_ACTUAL_CONTEXT_SIZE": "32768",
            "OMLX_SSE_KEEPALIVE_MODE": "comment",
        },
        omni_cli.PROXY_ENV_KEYS,
    )

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "openai",
            "--compose-file",
            str(compose_file),
            "--proxy-port",
            "9090",
            "--generate-only",
            "--no-build",
        ]
    )

    assert result == 0
    env = omni_cli.load_env_file(env_file)
    assert env["OMLX_BACKEND_URL"] == "https://api.example.test/v1"
    assert env["OMLX_BACKEND_API_KEY"] == "backend-secret"
    assert env["OMLX_PROXY_API_KEY"] == "proxy-secret"
    assert env["OMLX_PROXY_PORT"] == "9090"
    assert env["OMLX_CONTEXT_SCALING"] == "true"


def test_proxy_serve_writes_backend_url_and_recalls_it(tmp_path):
    compose_file = tmp_path / "docker-compose.proxy.yml"
    env_file = compose_file.with_suffix(".env")
    compose_file.write_text("services: {}")

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "openai",
            "--backend-url",
            "https://api.example.test/v1",
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )

    assert result == 0
    env = omni_cli.load_env_file(env_file)
    assert env["OMLX_BACKEND_URL"] == "https://api.example.test/v1"

    result = omni_cli.main(
        ["serve", "--proxy-port", "9191", "--generate-only", "--no-build"]
    )

    assert result == 0
    env = omni_cli.load_env_file(env_file)
    assert env["OMLX_BACKEND_URL"] == "https://api.example.test/v1"
    assert env["OMLX_PROXY_PORT"] == "9191"


def test_proxy_dry_run_does_not_write_state_or_env(tmp_path):
    compose_file = tmp_path / "docker-compose.proxy.yml"
    env_file = compose_file.with_suffix(".env")
    compose_file.write_text("services: {}")

    result = omni_cli.main(
        [
            "serve",
            "--compose-file",
            str(compose_file),
            "--dry-run",
        ]
    )

    assert result == 0
    assert not env_file.exists()
    assert not omni_cli.DEFAULT_SERVE_STATE_FILE.exists()


def test_llamacpp_proxy_requires_backend_url():
    with pytest.raises(SystemExit):
        omni_cli.proxy_backend_url("llamacpp", {})


def test_legacy_ollama_serve_state_resolves_to_openai(tmp_path):
    state_path = tmp_path / "omni-serve.json"
    state_path.write_text(
        json.dumps(
            {
                "version": omni_cli.SERVE_STATE_VERSION,
                "backend": "ollama",
                "mode": "proxy",
                "compose_file": str(tmp_path / "docker-compose.proxy.yml"),
            }
        )
    )

    state = omni_cli.load_serve_state(state_path)

    assert state["backend"] == "openai"
    assert state["mode"] == "proxy"


def test_compose_command_defaults_to_detached_build(tmp_path):
    compose_file = tmp_path / "docker-compose.yml"

    command = omni_cli.compose_command(compose_file, foreground=False, build=True)

    assert command == [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "up",
        "-d",
        "--build",
    ]


def test_vllm_settings_from_args_maps_model_flags():
    args = parse_args(
        "--backend",
        "vllm",
        "--model",
        "/models/local",
        "--served-model-name",
        "local-model",
        "--context-length",
        "4096",
        "--proxy-port",
        "9090",
    )

    settings = omni_cli.vllm_settings_from_args(args)

    assert settings.model == "/models/local"
    assert settings.served_model_name == "local-model"
    assert settings.context_length == 4096
    assert settings.proxy_port == 9090
    assert settings.hf_home.endswith("/.cache/huggingface")
    assert "${" not in settings.hf_home


# ---------------------------------------------------------------------------
# llama.cpp managed sidecar
# ---------------------------------------------------------------------------


def test_llamacpp_generate_only_writes_compose_and_env(tmp_path, capsys):
    compose_file = tmp_path / "docker-compose.llamacpp.yml"
    env_file = compose_file.with_suffix(".env")

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "llamacpp",
            "--model",
            "ggml-org/Qwen3-1.7B-GGUF:Q8_0",
            "--served-model-name",
            "qwen-gguf",
            "--context-length",
            "16384",
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )

    assert result == 0
    content = compose_file.read_text()
    assert "  llamacpp:" in content
    assert 'OMLX_BACKEND_URL: "http://llamacpp:8000/v1"' in content
    assert 'OMLX_SIDECAR_BACKEND: "llamacpp"' in content
    assert "-hf" in content
    env = omni_cli.load_env_file(env_file)
    assert env["OMNI_MODEL"] == "ggml-org/Qwen3-1.7B-GGUF:Q8_0"
    assert env["OMNI_SERVED_MODEL_NAME"] == "qwen-gguf"
    assert env["OMNI_CONTEXT_LENGTH"] == "16384"
    assert env["LLAMACPP_IMAGE"] == "ghcr.io/ggml-org/llama.cpp:server-cuda"
    state = omni_cli.load_serve_state()
    assert state["backend"] == "llamacpp"
    assert state["mode"] == "managed"
    assert "docker compose --env-file" in capsys.readouterr().out


def test_llamacpp_backend_url_stays_proxy_only(tmp_path):
    compose_file = tmp_path / "docker-compose.proxy.yml"
    compose_file.write_text("services: {}")

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "llamacpp",
            "--backend-url",
            "http://host.docker.internal:9000/v1",
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )

    assert result == 0
    env = omni_cli.load_env_file(compose_file.with_suffix(".env"))
    assert env["OMLX_BACKEND_URL"] == "http://host.docker.internal:9000/v1"
    assert "OMNI_MODEL" not in env
    state = omni_cli.load_serve_state()
    assert state["backend"] == "llamacpp"
    assert state["mode"] == "proxy"


def test_plain_serve_reuses_saved_external_llamacpp_proxy(tmp_path):
    compose_file = tmp_path / "docker-compose.proxy.yml"
    env_file = compose_file.with_suffix(".env")
    compose_file.write_text("services: {}")
    omni_cli.save_serve_state(
        backend="llamacpp",
        compose_file=compose_file,
        mode="proxy",
    )
    omni_cli.write_generic_env_file(
        env_file,
        {"OMLX_BACKEND_URL": "http://host.docker.internal:9000/v1"},
        omni_cli.PROXY_ENV_KEYS,
    )

    result = omni_cli.main(["serve", "--generate-only", "--no-build"])

    assert result == 0
    env = omni_cli.load_env_file(env_file)
    assert env["OMLX_BACKEND_URL"] == "http://host.docker.internal:9000/v1"
    state = omni_cli.load_serve_state()
    assert state["backend"] == "llamacpp"
    assert state["mode"] == "proxy"


def test_llamacpp_specific_flag_resolves_backend(tmp_path):
    compose_file = tmp_path / "docker-compose.llamacpp.yml"

    result = omni_cli.main(
        [
            "serve",
            "--n-gpu-layers",
            "80",
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )

    assert result == 0
    env = omni_cli.load_env_file(compose_file.with_suffix(".env"))
    assert env["LLAMACPP_N_GPU_LAYERS"] == "80"
    state = omni_cli.load_serve_state()
    assert state["backend"] == "llamacpp"
    assert state["mode"] == "managed"


def test_portable_flags_reuse_saved_managed_backend(tmp_path):
    compose_file = tmp_path / "docker-compose.llamacpp.yml"
    omni_cli.save_serve_state(
        backend="llamacpp",
        compose_file=compose_file,
        mode="managed",
    )

    result = omni_cli.main(
        [
            "serve",
            "--model",
            "ggml-org/example-GGUF:Q4_K_M",
            "--generate-only",
            "--no-build",
        ]
    )

    assert result == 0
    env = omni_cli.load_env_file(compose_file.with_suffix(".env"))
    assert env["OMNI_MODEL"] == "ggml-org/example-GGUF:Q4_K_M"
    state = omni_cli.load_serve_state()
    assert state["backend"] == "llamacpp"


def test_llamacpp_local_gguf_model_uses_dash_m(tmp_path):
    compose_file = tmp_path / "docker-compose.llamacpp.yml"

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "llamacpp",
            "--model",
            "qwen3.gguf",
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )

    assert result == 0
    content = compose_file.read_text()
    assert '-m "/models/$$model"' in content
    env = omni_cli.load_env_file(compose_file.with_suffix(".env"))
    assert env["OMNI_MODEL"] == "qwen3.gguf"


def test_services_for_target_finds_llamacpp_backend():
    services = ["omlx-proxy", "llamacpp"]

    assert omni_cli.services_for_target("backend", services) == ["llamacpp"]
    assert omni_cli.services_for_target("proxy", services) == ["omlx-proxy"]


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
        "--opus-model",
        "big",
        "--sonnet-model",
        "mid",
        "--haiku-model",
        "small",
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


def test_launch_command_passes_served_window(monkeypatch):
    """The served context window from /v1/models reaches the IntegrationContext.

    Without this, Codex has no metadata for the custom provider and falls back to
    a ~256K window, overflowing vLLM's smaller served window mid-session.
    """
    import io
    import json as _json

    health_bytes = b'{"status":"healthy"}'
    models_bytes = _json.dumps(
        {
            "data": [
                {
                    "id": "test-model",
                    "model_type": "llm",
                    "max_context_window": 65536,
                    "max_tokens": 4096,
                }
            ]
        }
    ).encode()
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

    result = omni_cli.launch_command(
        parse_launch("codex", "--port", "8080", "--model", "test-model")
    )

    assert result == 0
    ctx = launched[0]
    assert ctx.context_window == 65536
    assert ctx.max_tokens == 4096
    assert ctx.model_type == "llm"


def test_launch_command_uses_env_port(monkeypatch):
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


def test_scan_models_flags_map_to_env():
    from omlx.omni_cli import build_parser, portable_cli_environment

    args = build_parser().parse_args(
        ["serve", "--backend", "vllm", "--scan-models", "--model-dir", "~/models"]
    )
    env = portable_cli_environment(args)
    assert env["OMLX_MODEL_SCAN"] == "true"
    assert env["OMLX_MODEL_SCAN_HOST_DIR"].endswith("/models")
    assert not env["OMLX_MODEL_SCAN_HOST_DIR"].startswith("~")


def test_scan_models_absent_by_default():
    from omlx.omni_cli import build_parser, portable_cli_environment

    args = build_parser().parse_args(["serve", "--backend", "vllm"])
    env = portable_cli_environment(args)
    assert "OMLX_MODEL_SCAN" not in env
    assert "OMLX_MODEL_SCAN_HOST_DIR" not in env


_GIB = 1024**3


def _make_cached_model(root, repo_id, weight_bytes):
    """Minimal HF-cache entry with a sparse safetensors shard under root/hub."""
    encoded = "models--" + repo_id.replace("/", "--")
    commit = "deadbeef"
    snapshot = root / "hub" / encoded / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"max_position_embeddings": 4096}))
    with open(snapshot / "model.safetensors", "wb") as handle:
        handle.truncate(weight_bytes)
    refs = root / "hub" / encoded / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(commit)


def test_force_memory_flag_parses():
    args = parse_args("--backend", "vllm", "--force-memory")
    assert args.force_memory is True
    assert parse_args("--backend", "vllm").force_memory is False


def test_serve_blocks_oversized_model(tmp_path, monkeypatch, capsys):
    _make_cached_model(tmp_path, "org/huge", 80 * _GIB)
    monkeypatch.setattr(
        omni_cli, "host_memory_info", lambda *a, **k: {"total_bytes": 122 * _GIB}
    )
    args = omni_cli.build_parser().parse_args(
        [
            "serve",
            "--backend",
            "vllm",
            "--model",
            "org/huge",
            "--hf-home",
            str(tmp_path),
            "--compose-file",
            str(tmp_path / "docker-compose.vllm.yml"),
            "--no-build",
        ]
    )
    with pytest.raises(SystemExit):
        omni_cli.serve_command(args)
    assert "Memory check [block]" in capsys.readouterr().out
    # The block fires before any compose file is written.
    assert not (tmp_path / "docker-compose.vllm.yml").exists()


def test_force_memory_overrides_block(tmp_path, monkeypatch):
    _make_cached_model(tmp_path, "org/huge", 80 * _GIB)
    monkeypatch.setattr(
        omni_cli, "host_memory_info", lambda *a, **k: {"total_bytes": 122 * _GIB}
    )
    launched = {}

    def _fake_run_compose(*a, **k):
        launched["ran"] = True
        return 0

    monkeypatch.setattr(omni_cli, "run_compose", _fake_run_compose)
    args = omni_cli.build_parser().parse_args(
        [
            "serve",
            "--backend",
            "vllm",
            "--model",
            "org/huge",
            "--hf-home",
            str(tmp_path),
            "--compose-file",
            str(tmp_path / "docker-compose.vllm.yml"),
            "--no-build",
            "--force-memory",
        ]
    )
    assert omni_cli.serve_command(args) == 0
    assert launched.get("ran") is True


def test_serve_generate_only_does_not_block(tmp_path, monkeypatch, capsys):
    _make_cached_model(tmp_path, "org/huge", 80 * _GIB)
    monkeypatch.setattr(
        omni_cli, "host_memory_info", lambda *a, **k: {"total_bytes": 122 * _GIB}
    )
    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "vllm",
            "--model",
            "org/huge",
            "--hf-home",
            str(tmp_path),
            "--compose-file",
            str(tmp_path / "docker-compose.vllm.yml"),
            "--generate-only",
            "--no-build",
        ]
    )
    assert result == 0
    assert "Memory check [block]" in capsys.readouterr().out


def test_serve_auto_offline_when_model_cached(tmp_path):
    _make_cached_model(tmp_path, "org/cached", 4 * _GIB)
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "vllm",
            "--model",
            "org/cached",
            "--hf-home",
            str(tmp_path),
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )
    assert result == 0
    assert omni_cli.load_env_file(env_file)["OMNI_HF_OFFLINE"] == "true"


def test_serve_stays_online_when_model_not_cached(tmp_path):
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "vllm",
            "--model",
            "org/not-cached",
            "--hf-home",
            str(tmp_path),
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )
    assert result == 0
    assert omni_cli.load_env_file(env_file)["OMNI_HF_OFFLINE"] == "false"


def test_serve_online_flag_overrides_cached_autodetect(tmp_path):
    _make_cached_model(tmp_path, "org/cached", 4 * _GIB)
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "vllm",
            "--model",
            "org/cached",
            "--online",
            "--hf-home",
            str(tmp_path),
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )
    assert result == 0
    assert omni_cli.load_env_file(env_file)["OMNI_HF_OFFLINE"] == "false"


def test_serve_offline_flag_forces_offline_when_uncached(tmp_path):
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")

    result = omni_cli.main(
        [
            "serve",
            "--backend",
            "vllm",
            "--model",
            "org/not-cached",
            "--offline",
            "--hf-home",
            str(tmp_path),
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )
    assert result == 0
    assert omni_cli.load_env_file(env_file)["OMNI_HF_OFFLINE"] == "true"


def _qwen_config(snapshot):
    import json as _json

    (snapshot / "config.json").write_text(
        _json.dumps(
            {
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 40960,
            }
        )
    )


def test_serve_auto_util_is_demand_sized_for_small_model(tmp_path, monkeypatch):
    _make_cached_model(tmp_path, "org/small", 4 * _GIB)
    snap = next((tmp_path / "hub").glob("models--org--small/snapshots/*"))
    _qwen_config(snap)
    monkeypatch.delenv("OMLX_HOST_MEMORY_RESERVE_GB", raising=False)
    monkeypatch.setattr(
        omni_cli, "host_memory_info", lambda *a, **k: {"total_bytes": 122 * _GIB}
    )
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")

    def util_for(parallel):
        omni_cli.main(
            [
                "serve",
                "--backend",
                "vllm",
                "--model",
                "org/small",
                "--context-length",
                "40960",
                "--max-parallel",
                str(parallel),
                "--hf-home",
                str(tmp_path),
                "--compose-file",
                str(compose_file),
                "--generate-only",
                "--no-build",
            ]
        )
        return float(omni_cli.load_env_file(env_file)["VLLM_GPU_MEMORY_UTILIZATION"])

    u2 = util_for(2)
    u4 = util_for(4)
    # Demand-sized: far below the safety ceiling (~0.83) and scales with parallel.
    assert u2 < 0.40
    assert u2 < u4


def test_serve_explicit_util_overrides_demand_sizing(tmp_path):
    _make_cached_model(tmp_path, "org/small", 4 * _GIB)
    snap = next((tmp_path / "hub").glob("models--org--small/snapshots/*"))
    _qwen_config(snap)
    compose_file = tmp_path / "docker-compose.vllm.yml"
    env_file = compose_file.with_suffix(".env")

    omni_cli.main(
        [
            "serve",
            "--backend",
            "vllm",
            "--model",
            "org/small",
            "--gpu-memory-utilization",
            "0.5",
            "--hf-home",
            str(tmp_path),
            "--compose-file",
            str(compose_file),
            "--generate-only",
            "--no-build",
        ]
    )
    assert omni_cli.load_env_file(env_file)["VLLM_GPU_MEMORY_UTILIZATION"] == "0.5"


def _make_fake_cached_model(
    cache_root, repo_id, *, native, layers=4, kv_heads=2, head_dim=64, weight_bytes=1024
):
    import json as _json

    encoded = "models--" + repo_id.replace("/", "--")
    commit = "deadbeef"
    snap = cache_root / "hub" / encoded / "snapshots" / commit
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(
        _json.dumps(
            {
                "num_hidden_layers": layers,
                "num_attention_heads": kv_heads,
                "num_key_value_heads": kv_heads,
                "head_dim": head_dim,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": native,
            }
        ),
        encoding="utf-8",
    )
    with open(snap / "model.safetensors", "wb") as fh:
        fh.truncate(weight_bytes)
    refs = cache_root / "hub" / encoded / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(commit, encoding="utf-8")


def test_serve_preflight_auto_sizes_context(tmp_path, monkeypatch):
    import argparse

    _make_fake_cached_model(tmp_path, "org/m", native=16384)
    # Big host so the native window (not memory) bounds the result -> deterministic.
    monkeypatch.setattr(
        omni_cli,
        "host_memory_info",
        lambda: {"total_bytes": 200 * 1024**3, "available_bytes": 200 * 1024**3},
    )
    args = argparse.Namespace(
        context_length=None,
        gpu_memory_utilization=None,
        dry_run=False,
        generate_only=True,
        force_memory=False,
    )
    merged = {
        "OMNI_MODEL": "org/m",
        "OMNI_HF_HOME": str(tmp_path),
        "OMNI_MAX_PARALLEL": "2",
        "OMNI_CONTEXT_LENGTH": "8192",
    }
    omni_cli._vllm_memory_preflight(args, merged)
    assert merged["OMNI_CONTEXT_LENGTH"] == "16384"  # auto, native-bounded


def test_serve_preflight_context_is_resource_based(tmp_path, monkeypatch):
    import argparse

    _make_fake_cached_model(
        tmp_path,
        "org/m",
        native=262144,
        layers=48,
        kv_heads=16,
        head_dim=256,
        weight_bytes=8 * 1024**3,
    )
    monkeypatch.setenv("OMLX_HOST_MEMORY_RESERVE_GB", "8")
    monkeypatch.setattr(
        omni_cli,
        "host_memory_info",
        lambda: {"total_bytes": 48 * 1024**3, "available_bytes": 48 * 1024**3},
    )
    args = argparse.Namespace(
        context_length=None,
        gpu_memory_utilization=None,
        dry_run=False,
        generate_only=True,
        force_memory=False,
        enable_chunked_prefill=None,
    )
    merged = {
        "OMNI_MODEL": "org/m",
        "OMNI_HF_HOME": str(tmp_path),
        "OMNI_MAX_PARALLEL": "2",
        "OMNI_CONTEXT_LENGTH": "65536",
    }
    omni_cli._vllm_memory_preflight(args, merged)
    assert int(merged["OMNI_CONTEXT_LENGTH"]) < 65536
    assert int(merged["OMNI_CONTEXT_LENGTH"]) % 4096 == 0


def test_serve_preflight_enables_chunked_prefill_for_long_auto_context(
    tmp_path, monkeypatch
):
    import argparse

    _make_fake_cached_model(tmp_path, "org/m", native=262144)
    monkeypatch.setattr(
        omni_cli,
        "host_memory_info",
        lambda: {"total_bytes": 200 * 1024**3, "available_bytes": 200 * 1024**3},
    )
    args = argparse.Namespace(
        context_length=None,
        gpu_memory_utilization=None,
        dry_run=False,
        generate_only=True,
        force_memory=False,
        enable_chunked_prefill=None,
    )
    merged = {
        "OMNI_MODEL": "org/m",
        "OMNI_HF_HOME": str(tmp_path),
        "OMNI_MAX_PARALLEL": "2",
        "OMNI_CONTEXT_LENGTH": "8192",
        "VLLM_ENABLE_CHUNKED_PREFILL": "false",
    }
    omni_cli._vllm_memory_preflight(args, merged)
    assert merged["OMNI_CONTEXT_LENGTH"] == "262144"
    assert merged["VLLM_ENABLE_CHUNKED_PREFILL"] == "true"


def test_serve_preflight_respects_explicit_chunked_prefill(tmp_path, monkeypatch):
    import argparse

    _make_fake_cached_model(tmp_path, "org/m", native=262144)
    monkeypatch.setattr(
        omni_cli,
        "host_memory_info",
        lambda: {"total_bytes": 200 * 1024**3, "available_bytes": 200 * 1024**3},
    )
    args = argparse.Namespace(
        context_length=None,
        gpu_memory_utilization=None,
        dry_run=False,
        generate_only=True,
        force_memory=False,
        enable_chunked_prefill=False,
    )
    merged = {
        "OMNI_MODEL": "org/m",
        "OMNI_HF_HOME": str(tmp_path),
        "OMNI_MAX_PARALLEL": "2",
        "OMNI_CONTEXT_LENGTH": "8192",
        "VLLM_ENABLE_CHUNKED_PREFILL": "false",
    }
    omni_cli._vllm_memory_preflight(args, merged)
    assert merged["OMNI_CONTEXT_LENGTH"] == "262144"
    assert merged["VLLM_ENABLE_CHUNKED_PREFILL"] == "false"


def test_vllm_optimize_resets_stale_model_specific_env():
    import argparse

    args = argparse.Namespace(
        optimize=True,
        model=None,
        served_model_name=None,
        context_length=None,
        max_output_tokens=None,
        max_parallel=None,
        port=None,
        hf_endpoint=None,
        http_proxy=None,
        https_proxy=None,
        no_proxy=None,
        ca_bundle=None,
        proxy_port=None,
        api_key=None,
        backend_api_key=None,
        target_context_size=None,
        sse_keepalive_mode=None,
        hf_home=None,
        context_scaling=False,
        scan_models=False,
        model_dir=None,
        vllm_image=None,
        gpu_memory_utilization=None,
        generation_config=None,
        default_chat_template_kwargs=None,
        tool_call_parser=None,
        reasoning_parser=None,
        chat_template=None,
        dtype=None,
        tokenizer=None,
        tokenizer_mode=None,
        revision=None,
        load_format=None,
        quantization=None,
        download_dir=None,
        max_num_batched_tokens=None,
        kv_cache_dtype=None,
        cpu_offload_gb=None,
        swap_space=None,
        tensor_parallel_size=None,
        pipeline_parallel_size=None,
        uvicorn_log_level=None,
        extra_args_json=None,
        trust_remote_code=None,
        enforce_eager=None,
        enable_auto_tool_choice=None,
        enable_chunked_prefill=None,
        enable_prefix_caching=None,
        disable_log_stats=None,
    )
    merged = {
        "VLLM_DTYPE": "float16",
        "VLLM_TOOL_CALL_PARSER": "hermes",
        "VLLM_ENABLE_CHUNKED_PREFILL": "false",
    }
    omni_cli._apply_vllm_optimize(args, merged)
    assert merged["VLLM_DTYPE"] == ""
    assert merged["VLLM_TOOL_CALL_PARSER"] == "auto"
    assert merged["VLLM_ENABLE_CHUNKED_PREFILL"] == ""


def test_serve_preflight_respects_explicit_context(tmp_path, monkeypatch):
    import argparse

    _make_fake_cached_model(tmp_path, "org/m", native=16384)
    monkeypatch.setattr(
        omni_cli,
        "host_memory_info",
        lambda: {"total_bytes": 200 * 1024**3, "available_bytes": 200 * 1024**3},
    )
    args = argparse.Namespace(
        context_length=4096,
        gpu_memory_utilization=None,
        dry_run=False,
        generate_only=True,
        force_memory=False,
    )
    merged = {
        "OMNI_MODEL": "org/m",
        "OMNI_HF_HOME": str(tmp_path),
        "OMNI_MAX_PARALLEL": "2",
        "OMNI_CONTEXT_LENGTH": "4096",
    }
    omni_cli._vllm_memory_preflight(args, merged)
    assert merged["OMNI_CONTEXT_LENGTH"] == "4096"  # explicit wins
