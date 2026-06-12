# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared sidecar compose layer."""

from omlx.proxy.llamacpp_compose import LlamacppComposeSettings, render_llamacpp_compose
from omlx.proxy.sidecar_compose import (
    BACKEND_SPECS,
    OMLX_PROXY_SIDECAR_KEYS,
    OMNI_ENV_KEYS,
    CommonSidecarSettings,
    backend_spec,
    common_settings_kwargs_from_env,
    common_settings_kwargs_from_overrides,
    env_from_compose,
    known_env,
    load_env_file,
    write_env_file,
)
from omlx.proxy.vllm_compose import VllmComposeSettings, render_vllm_compose


def test_backend_specs_register_both_backends():
    assert backend_spec("vllm").service_name == "vllm"
    assert backend_spec("llamacpp").service_name == "llamacpp"
    assert set(BACKEND_SPECS) == {"vllm", "llamacpp"}


def test_env_keys_share_portable_and_proxy_tuples():
    for name in ("vllm", "llamacpp"):
        keys = backend_spec(name).env_keys
        assert keys[: len(OMNI_ENV_KEYS)] == OMNI_ENV_KEYS
        assert keys[-len(OMLX_PROXY_SIDECAR_KEYS) :] == OMLX_PROXY_SIDECAR_KEYS


def test_common_settings_env_coercion():
    defaults = CommonSidecarSettings()
    kwargs = common_settings_kwargs_from_env(
        {
            "OMNI_MODEL": "a/b",
            "OMNI_CONTEXT_LENGTH": "16384",
            "OMNI_MAX_PARALLEL": "not-a-number",
            "OMNI_BACKEND_PORT": "-1",
            "OMLX_CONTEXT_SCALING": "true",
            "OMLX_SAMPLING_TOP_K": "0",
        },
        defaults,
    )

    assert kwargs["model"] == "a/b"
    assert kwargs["context_length"] == 16384
    assert kwargs["max_parallel"] == defaults.max_parallel
    assert kwargs["backend_port"] == defaults.backend_port
    assert kwargs["context_scaling"] is True
    assert kwargs["sampling_top_k"] == 0


def test_common_settings_override_keys():
    defaults = CommonSidecarSettings()
    kwargs = common_settings_kwargs_from_overrides(
        {
            "omni_model": "x/y",
            "omni_served_model_name": "alias",
            "network_http_proxy": "http://proxy:3128",
            "huggingface_endpoint": "https://hf.example",
            "target_context_size": 100000,
        },
        defaults,
    )

    assert kwargs["model"] == "x/y"
    assert kwargs["served_model_name"] == "alias"
    assert kwargs["http_proxy"] == "http://proxy:3128"
    assert kwargs["hf_endpoint"] == "https://hf.example"
    assert kwargs["target_context_size"] == 100000


def test_known_env_filters_to_keys():
    values = {"OMNI_MODEL": "a/b", "UNRELATED": "x", "OMLX_PROXY_PORT": "9090"}

    filtered = known_env(values, OMNI_ENV_KEYS + OMLX_PROXY_SIDECAR_KEYS)

    assert filtered == {"OMNI_MODEL": "a/b", "OMLX_PROXY_PORT": "9090"}


def test_env_file_round_trip(tmp_path):
    path = tmp_path / "test.env"
    values = {"OMNI_MODEL": "a/b", "OMNI_CONTEXT_LENGTH": "8192"}

    write_env_file(path, values, ("OMNI_MODEL", "OMNI_CONTEXT_LENGTH"))

    assert load_env_file(path) == values


def test_env_from_compose_parses_default_expressions(tmp_path):
    compose = tmp_path / "compose.yml"
    compose.write_text(
        "services:\n"
        "  x:\n"
        "    environment:\n"
        '      OMNI_MODEL: "${OMNI_MODEL:-a/b}"\n'
        '      OMNI_CONTEXT_LENGTH: "${OMNI_CONTEXT_LENGTH:-16384}"\n'
        '      OTHER: "literal"\n'
    )

    values = env_from_compose(compose, OMNI_ENV_KEYS)

    assert values == {"OMNI_MODEL": "a/b", "OMNI_CONTEXT_LENGTH": "16384"}


def _proxy_block(content: str, backend_service: str) -> str:
    return content.split("  omlx-proxy:", 1)[1].split(f"  {backend_service}:", 1)[0]


def test_proxy_service_parity_between_backends():
    vllm_proxy = _proxy_block(render_vllm_compose(VllmComposeSettings()), "vllm")
    lcpp_proxy = _proxy_block(
        render_llamacpp_compose(LlamacppComposeSettings()), "llamacpp"
    )

    # The proxy blocks differ only in backend URL, sidecar identity,
    # generated-file names, and depends_on.
    normalized = (
        lcpp_proxy.replace("http://llamacpp:8000/v1", "http://vllm:8000/v1")
        .replace('OMLX_SIDECAR_BACKEND: "llamacpp"', 'OMLX_SIDECAR_BACKEND: "vllm"')
        .replace("docker-compose.llamacpp.yml", "docker-compose.vllm.yml")
        .replace("docker-compose.llamacpp.env", "docker-compose.vllm.env")
        .replace("- llamacpp", "- vllm")
    )
    assert normalized == vllm_proxy
    assert (
        'OMLX_COMPOSE_OUTPUT_PATH: "/compose-output/docker-compose.vllm.yml"'
        in vllm_proxy
    )
    assert "OMLX_ACTUAL_CONTEXT_SIZE" in vllm_proxy
    assert "${OMNI_CONTEXT_LENGTH:-8192}" in vllm_proxy


def test_proxy_service_mounts_docker_socket_for_sidecar_restart():
    for content in (
        render_vllm_compose(VllmComposeSettings()),
        render_llamacpp_compose(LlamacppComposeSettings()),
    ):
        proxy_block = content.split("  omlx-proxy:", 1)[1]
        assert "- /var/run/docker.sock:/var/run/docker.sock" in proxy_block


def test_sidecar_reloads_env_file_on_restart():
    cases = (
        (render_vllm_compose(VllmComposeSettings()), "docker-compose.vllm.env"),
        (
            render_llamacpp_compose(LlamacppComposeSettings()),
            "docker-compose.llamacpp.env",
        ),
    )
    for content, env_name in cases:
        # The sidecar mounts the generated-files dir read-only and re-exports
        # the env file at start so a container restart applies saved settings.
        assert ":/compose-output:ro" in content
        assert f'if [ -f "/compose-output/{env_name}" ]' in content
        assert f'done < "/compose-output/{env_name}"' in content
        assert 'export "$$omni_env_line"' in content


def test_vllm_compose_enables_prompt_tokens_details():
    # Proxy serving stats need usage.prompt_tokens_details.cached_tokens,
    # which vLLM only reports with this flag.
    content = render_vllm_compose(VllmComposeSettings())
    assert "--enable-prompt-tokens-details" in content
