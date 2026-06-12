# SPDX-License-Identifier: Apache-2.0
"""Tests for the llama.cpp sidecar compose renderer."""

import subprocess

import pytest

from omlx.proxy.llamacpp_compose import (
    LLAMACPP_ENV_KEYS,
    LlamacppComposeSettings,
    llamacpp_env_from_compose,
    llamacpp_environment,
    llamacpp_settings_from_env,
    llamacpp_settings_from_overrides,
    render_llamacpp_compose,
    render_llamacpp_env_file,
    write_llamacpp_compose_for_path,
    write_llamacpp_env_file,
)
from omlx.proxy.sidecar_compose import load_env_file


def test_render_compose_contains_model_dispatch_and_portable_args():
    content = render_llamacpp_compose(LlamacppComposeSettings())

    # -hf vs -m dispatch on the OMNI_MODEL value
    assert '-hf "$$model"' in content
    assert '-m "$$model"' in content
    assert '-m "/models/$$model"' in content
    assert "*.gguf)" in content
    # llama-server listens on container port 8000 to mirror the vLLM stack
    assert "--port 8000" in content
    assert 'OMLX_BACKEND_URL: "http://llamacpp:8000/v1"' in content
    # portable OMNI_* settings drive the launch flags
    assert '--alias "$${OMNI_SERVED_MODEL_NAME:-$$model}"' in content
    assert '--ctx-size "$${OMNI_CONTEXT_LENGTH:-8192}"' in content
    assert '--parallel "$${OMNI_MAX_PARALLEL:-4}"' in content
    assert '--n-gpu-layers "$${LLAMACPP_N_GPU_LAYERS:-999}"' in content
    assert "--jinja" in content
    # Prometheus endpoint for the dashboard's backend cache observability
    assert "--metrics" in content


def test_render_compose_mounts_hf_cache_and_llamacpp_cache():
    content = render_llamacpp_compose(LlamacppComposeSettings())

    assert ":/root/.cache/huggingface" in content
    assert ":/root/.cache/llama.cpp" in content
    assert ":/models:ro" in content
    assert 'LLAMA_CACHE: "/root/.cache/llama.cpp"' in content
    # model dir falls back to the llama.cpp cache dir when unset
    assert "${LLAMACPP_MODEL_DIR:-${LLAMACPP_CACHE_DIR:-" in content


def test_render_compose_extra_args_is_word_split_not_json():
    content = render_llamacpp_compose(LlamacppComposeSettings())

    # No python3 in the llama.cpp image: extra args are word-split, not JSON
    assert "$$LLAMACPP_EXTRA_ARGS" in content
    assert "python3" not in content


def test_render_compose_unsets_env_before_exec():
    content = render_llamacpp_compose(LlamacppComposeSettings())

    assert "unset OMNI_MODEL OMNI_SERVED_MODEL_NAME" in content
    assert "unset LLAMACPP_IMAGE LLAMACPP_N_GPU_LAYERS" in content
    assert "unset OMLX_PROXY_PORT OMLX_PROXY_API_KEY" in content
    assert 'exec /app/llama-server "$${@}"' in content


def test_entrypoint_passes_sh_syntax_check():
    content = render_llamacpp_compose(LlamacppComposeSettings())
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(content.replace("$$", "$"))
    command = data["services"]["llamacpp"]["command"][0]
    subprocess.run(["sh", "-n", "-c", command], check=True)


def test_environment_round_trips_through_settings():
    settings = LlamacppComposeSettings(
        model="ggml-org/Qwen3-1.7B-GGUF:Q8_0",
        served_model_name="qwen-gguf",
        context_length=16384,
        max_parallel=8,
        n_gpu_layers=80,
        flash_attn="on",
        cache_type_k="q8_0",
        jinja=False,
        extra_args="--mlock",
    )

    restored = llamacpp_settings_from_env(llamacpp_environment(settings))

    assert restored == settings


def test_settings_from_overrides_reads_omni_and_llamacpp_keys():
    settings = llamacpp_settings_from_overrides(
        {
            "omni_model": "ggml-org/example-GGUF:Q4_K_M",
            "omni_context_length": 32768,
            "llamacpp_n_gpu_layers": 40,
            "llamacpp_jinja": "false",
            "llamacpp_cache_dir": "/srv/llama-cache",
            "sampling_top_k": 7,
        }
    )

    assert settings.model == "ggml-org/example-GGUF:Q4_K_M"
    assert settings.context_length == 32768
    assert settings.n_gpu_layers == 40
    assert settings.jinja is False
    assert settings.cache_dir == "/srv/llama-cache"
    assert settings.sampling_top_k == 7


def test_env_file_round_trip(tmp_path):
    env_path = tmp_path / "docker-compose.llamacpp.env"
    values = llamacpp_environment(LlamacppComposeSettings(model="a/b:Q8_0"))

    write_llamacpp_env_file(env_path, values)
    loaded = load_env_file(env_path)

    assert loaded == values
    rendered = render_llamacpp_env_file(values)
    for key in LLAMACPP_ENV_KEYS:
        assert f"{key}=" in rendered


def test_env_from_compose_reseeds_settings(tmp_path):
    compose_path = tmp_path / "docker-compose.llamacpp.yml"
    settings = LlamacppComposeSettings(
        model="ggml-org/example-GGUF:Q4_K_M",
        context_length=65536,
        n_gpu_layers=33,
    )
    write_llamacpp_compose_for_path(compose_path, settings)

    values = llamacpp_env_from_compose(compose_path)

    assert values["OMNI_MODEL"] == "ggml-org/example-GGUF:Q4_K_M"
    assert values["OMNI_CONTEXT_LENGTH"] == "65536"
    assert values["LLAMACPP_N_GPU_LAYERS"] == "33"
    restored = llamacpp_settings_from_env(values)
    assert restored.model == settings.model
    assert restored.context_length == settings.context_length


def test_model_dir_override_replaces_cache_fallback():
    content = render_llamacpp_compose(
        LlamacppComposeSettings(model_dir="/srv/gguf-models")
    )

    assert "${LLAMACPP_MODEL_DIR:-/srv/gguf-models}" in content
