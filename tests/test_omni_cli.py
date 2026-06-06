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
    assert "--backend {vllm,openai,ollama,llamacpp}" in output
    assert "--hf-home PATH" in output
    assert "--sse-keepalive-mode {ping,comment,off}" in output


def test_parser_recognizes_status_logs_and_restart():
    parser = omni_cli.build_parser()

    status = parser.parse_args(["status"])
    logs = parser.parse_args(["logs", "--target", "backend", "-f"])
    restart = parser.parse_args(["restart", "--target", "backend"])

    assert status.command == "status"
    assert logs.command == "logs"
    assert logs.target == "backend"
    assert logs.follow is True
    assert restart.command == "restart"
    assert restart.target == "backend"


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
