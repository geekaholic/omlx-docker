# SPDX-License-Identifier: Apache-2.0

import importlib
import json

import httpx
import pytest

from omlx.api.anthropic_models import AnthropicMessage, AnthropicTool, MessagesRequest
from omlx.proxy.app import anthropic_to_openai_chat_body, create_app
from omlx.proxy.backend import OpenAIBackend
from omlx.proxy.config import ProxyConfig
from omlx.proxy.metrics import parse_prometheus_text, select_prometheus_metrics
from omlx.proxy.scaling import scale_token_count
from omlx.proxy.llamacpp_compose import load_llamacpp_env_file
from omlx.proxy.vllm_compose import (
    VllmComposeSettings,
    load_vllm_env_file,
    vllm_environment,
    write_vllm_compose,
    write_vllm_env_file,
)


def test_proxy_app_import_does_not_require_env(monkeypatch):
    monkeypatch.delenv("OMLX_BACKEND_URL", raising=False)
    module = importlib.import_module("omlx.proxy.app")
    importlib.reload(module)


def test_anthropic_request_maps_to_openai_chat_body():
    request = MessagesRequest(
        model="claude-compatible",
        max_tokens=128,
        system="You are concise.",
        messages=[
            AnthropicMessage(role="user", content="hello"),
        ],
        tools=[
            AnthropicTool(
                name="lookup",
                description="Look up a value",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ],
        temperature=0.2,
        stop_sequences=["STOP"],
    )

    body = anthropic_to_openai_chat_body(request)

    assert body["model"] == "claude-compatible"
    assert body["max_tokens"] == 128
    assert body["temperature"] == 0.2
    assert body["stop"] == ["STOP"]
    assert body["messages"][0] == {"role": "system", "content": "You are concise."}
    assert body["messages"][1] == {"role": "user", "content": "hello"}
    assert body["tools"][0]["function"]["name"] == "lookup"


def test_scale_token_count_uses_target_context_ratio():
    config = ProxyConfig(
        backend_url="http://backend/v1",
        context_scaling_enabled=True,
        target_context_size=200000,
        actual_context_size=50000,
    )

    assert scale_token_count(25, config) == 100


@pytest.mark.asyncio
async def test_proxy_root_redirects_to_admin_dashboard():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            f"root redirect should not reach backend: {request.url.path}"
        )

    app = _app_with_mock_backend(handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == "/admin/dashboard"


@pytest.mark.asyncio
async def test_anthropic_endpoint_probes_do_not_return_405():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"probe should not reach backend: {request.url.path}")

    app = _app_with_mock_backend(handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        for path in ("/v1/messages", "/v1/messages/count_tokens"):
            get_response = await client.get(path)
            assert get_response.status_code == 200
            assert get_response.json()["endpoint"] == path

            head_response = await client.head(path)
            assert head_response.status_code == 204

            options_response = await client.options(path)
            assert options_response.status_code == 204


@pytest.mark.asyncio
async def test_proxy_auth_accepts_x_api_key_for_claude_code():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"object": "list", "data": []})

    config = ProxyConfig(backend_url="http://backend/v1", proxy_api_key="secret")
    backend = OpenAIBackend(
        config=config,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    app = create_app(config=config, backend=backend)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        missing = await client.get("/v1/models")
        assert missing.status_code == 401

        response = await client.get("/v1/models", headers={"x-api-key": "secret"})
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_anthropic_messages_non_stream_translates_response():
    seen_request = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": []})
        assert request.url.path == "/v1/chat/completions"
        seen_request.update(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi there"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
        )

    app = _app_with_mock_backend(handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "qwen",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert seen_request["messages"] == [{"role": "user", "content": "hello"}]
    data = response.json()
    assert data["type"] == "message"
    assert data["content"][0]["text"] == "hi there"
    assert data["usage"]["input_tokens"] == 10
    assert data["usage"]["output_tokens"] == 2


@pytest.mark.asyncio
async def test_anthropic_messages_stream_translates_openai_sse():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    app = _app_with_mock_backend(handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "qwen",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    text = response.text
    assert "message_start" in text
    assert "text_delta" in text
    assert "hello" in text
    assert "message_stop" in text


@pytest.mark.asyncio
async def test_openai_chat_completions_passthrough():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"model": body["model"], "choices": [], "usage": {}},
        )

    app = _app_with_mock_backend(handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [], "stream": False},
        )

    assert response.status_code == 200
    assert response.json()["model"] == "qwen"


@pytest.mark.asyncio
async def test_openai_chat_completions_stream_passthrough():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content.decode())
        assert body["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    app = _app_with_mock_backend(handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "hi" in response.text
    assert "[DONE]" in response.text


@pytest.mark.asyncio
async def test_proxy_serves_admin_chat_page():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": []})

    app = _app_with_mock_backend(handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/chat")

    assert response.status_code == 200
    assert "oMLX" in response.text
    assert "/v1/chat/completions" in response.text


@pytest.mark.asyncio
async def test_proxy_server_info_prefers_localhost_aliases():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": []})

    app = _app_with_mock_backend(handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/api/server-info")

    assert response.status_code == 200
    assert response.json()["aliases"][:3] == ["::1", "localhost", "127.0.0.1"]


@pytest.mark.asyncio
async def test_proxy_admin_models_reflect_backend_models():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "qwen",
                        "object": "model",
                        "owned_by": "vllm",
                        "max_model_len": 32768,
                    }
                ],
            },
        )

    app = _app_with_mock_backend(handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/api/models")

    assert response.status_code == 200
    models = response.json()["models"]
    assert models[0]["id"] == "qwen"
    assert models[0]["model_type"] == "llm"
    assert models[0]["loaded"] is True


def _app_with_mock_backend(handler):
    config = ProxyConfig(backend_url="http://backend/v1")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OpenAIBackend(config=config, client=client)
    return create_app(config=config, backend=backend)


@pytest.mark.asyncio
async def test_proxy_applies_saved_sampling_defaults_to_openai_passthrough(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(state_path))
    seen_request = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": "qwen"}]}
            )
        assert request.url.path == "/v1/chat/completions"
        seen_request.update(json.loads(request.content.decode()))
        return httpx.Response(200, json={"model": "qwen", "choices": [], "usage": {}})

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/admin/api/global-settings",
            json={
                "proxy_backend_url": "http://backend/v1",
                "proxy_backend_type": "vllm",
                "sampling_temperature": 0.25,
                "sampling_top_p": 0.8,
                "sampling_max_tokens": 77,
            },
        )
        assert response.status_code == 200

        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [], "stream": False},
        )

    assert response.status_code == 200
    assert seen_request["temperature"] == 0.25
    assert seen_request["top_p"] == 0.8
    assert seen_request["max_tokens"] == 77


@pytest.mark.asyncio
async def test_proxy_model_force_sampling_overrides_openai_request(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(state_path))
    seen_request = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": "qwen"}]}
            )
        seen_request.update(json.loads(request.content.decode()))
        return httpx.Response(200, json={"model": "qwen", "choices": [], "usage": {}})

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.put(
            "/admin/api/models/qwen/settings",
            json={
                "temperature": 0.1,
                "top_k": 20,
                "force_sampling": True,
                "enable_thinking": False,
                "thinking_budget_enabled": True,
                "thinking_budget_tokens": 512,
            },
        )
        assert response.status_code == 200

        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [],
                "stream": False,
                "temperature": 0.9,
                "chat_template_kwargs": {"foo": "bar"},
            },
        )

    assert response.status_code == 200
    assert seen_request["temperature"] == 0.1
    assert seen_request["top_k"] == 20
    assert seen_request["chat_template_kwargs"] == {
        "foo": "bar",
        "enable_thinking": False,
    }
    assert seen_request["thinking_budget"] == 512


@pytest.mark.asyncio
async def test_proxy_chat_page_is_unlocked_without_proxy_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": []})

    app = _app_with_mock_backend(handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/chat")

    assert response.status_code == 200
    assert "apiKeyRequired: false" in response.text
    assert "apiKeySet: true" in response.text


@pytest.mark.asyncio
async def test_proxy_status_endpoint_reports_backend_health(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "qwen"}]},
        )

    app = _app_with_mock_backend(handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/api/proxy/status")

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "proxy"
    assert data["backend_reachable"] is True
    assert data["model_count"] == 1
    assert data["capabilities"]["proxy_mode"] is True


def test_parse_prometheus_metrics_selects_vllm_counters():
    text = """
# HELP vllm:num_requests_running Running requests.
vllm:num_requests_running{model_name="qwen"} 2
vllm:num_requests_waiting{model_name="qwen"} 1
vllm:prompt_tokens_total{model_name="qwen"} 100
vllm:generation_tokens_total{model_name="qwen"} 25
vllm:gpu_cache_usage_perc{model_name="qwen"} 0.42
"""
    samples = parse_prometheus_text(text)
    selected = select_prometheus_metrics(samples)

    assert selected["running_requests"] == 2
    assert selected["waiting_requests"] == 1
    assert selected["prompt_tokens_total"] == 100
    assert selected["generation_tokens_total"] == 25
    assert selected["gpu_cache_usage_perc"] == 0.42


@pytest.mark.asyncio
async def test_proxy_metrics_endpoint_reports_prometheus(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                text=(
                    "vllm:num_requests_running 1\n"
                    "vllm:prompt_tokens_total 7\n"
                    "vllm:generation_tokens_total 3\n"
                ),
            )
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/api/proxy/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["prometheus"]["available"] is True
    assert data["summary"]["running_requests"] == 1
    assert data["summary"]["prompt_tokens_total"] == 7
    assert data["summary"]["generation_tokens_total"] == 3


@pytest.mark.asyncio
async def test_proxy_metrics_endpoint_reports_ollama(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(404)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "qwen3:4b", "size": 123}]},
            )
        if request.url.path == "/api/ps":
            return httpx.Response(
                200,
                json={"models": [{"name": "qwen3:4b", "size_vram": 456}]},
            )
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/api/proxy/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["backend_kind"] == "ollama"
    assert data["ollama"]["available"] is True
    assert data["ollama"]["models_count"] == 1
    assert data["ollama"]["loaded_count"] == 1


@pytest.mark.asyncio
async def test_proxy_backend_config_updates_runtime_and_persists(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(state_path))
    seen_requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": f"{request.url.host}-model",
                            "object": "model",
                        }
                    ],
                },
            )
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/admin/api/global-settings",
            json={
                "proxy_backend_url": "http://new-backend/v1/",
                "proxy_backend_type": "openai-compatible",
                "proxy_backend_api_key": "backend-secret",
            },
        )
        assert response.status_code == 200
        assert response.json()["requires_restart"] is False

        response = await client.get("/admin/api/proxy/config")
        assert response.status_code == 200
        config = response.json()
        assert config["backend_url"] == "http://new-backend/v1"
        assert config["backend_type"] == "openai-compatible"
        assert config["backend_api_key_set"] is True

        response = await client.get("/v1/models")
        assert response.status_code == 200
        assert response.json()["data"][0]["id"] == "new-backend-model"

    model_requests = [
        request for request in seen_requests if request.url.path == "/v1/models"
    ]
    assert model_requests[-1].url.host == "new-backend"
    assert model_requests[-1].headers["authorization"] == "Bearer backend-secret"

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/api/proxy/config")

    assert response.status_code == 200
    assert response.json()["backend_url"] == "http://new-backend/v1"


@pytest.mark.asyncio
async def test_proxy_backend_config_rejects_invalid_url(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": []})

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/admin/api/global-settings",
            json={"proxy_backend_url": "new-backend/v1"},
        )
        assert response.status_code == 422

        response = await client.get("/admin/api/proxy/config")
        assert response.status_code == 200
        assert response.json()["backend_url"] == "http://backend/v1"


@pytest.mark.asyncio
async def test_admin_vllm_settings_reflect_generated_env_file(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    compose_path = tmp_path / "docker-compose.vllm.yml"
    env_path = tmp_path / "docker-compose.vllm.env"
    monkeypatch.setenv("OMLX_COMPOSE_OUTPUT_PATH", str(compose_path))
    monkeypatch.setenv("OMLX_ENV_OUTPUT_PATH", str(env_path))

    write_vllm_compose(
        compose_path,
        VllmComposeSettings(
            model="example/compose-model",
            served_model_name="compose-name",
            context_length=16384,
        ),
    )
    write_vllm_env_file(
        env_path,
        vllm_environment(
            VllmComposeSettings(
                model="example/env-model",
                served_model_name="env-name",
                context_length=32768,
                hf_home="/cache/from-env",
            )
        ),
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": []})

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/api/global-settings")
        assert response.status_code == 200
        payload = response.json()["proxy"]
        assert payload["sidecar_backend"] == "vllm"
        sidecar = payload["sidecar"]
        assert sidecar["model"] == "example/env-model"
        assert sidecar["served_model_name"] == "env-name"
        assert sidecar["context_length"] == 32768
        assert sidecar["hf_home"] == "/cache/from-env"
        assert sidecar["compose_output_path"] == str(compose_path)
        assert sidecar["env_output_path"] == str(env_path)

        response = await client.get("/admin/api/proxy/sidecar-compose")
        assert response.status_code == 200
        data = response.json()
        assert data["backend"] == "vllm"
        assert data["settings"]["model"] == "example/env-model"
        assert data["env_output_path"] == str(env_path)
        assert "OMNI_MODEL=example/env-model" in data["env_content"]


@pytest.mark.asyncio
async def test_admin_vllm_settings_save_writes_env_and_compose(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    compose_path = tmp_path / "docker-compose.vllm.yml"
    env_path = tmp_path / "docker-compose.vllm.env"
    monkeypatch.setenv("OMLX_COMPOSE_OUTPUT_PATH", str(compose_path))
    monkeypatch.setenv("OMLX_ENV_OUTPUT_PATH", str(env_path))

    write_vllm_env_file(
        env_path,
        vllm_environment(
            VllmComposeSettings(
                model="example/previous-model",
                served_model_name="previous-name",
                tool_call_parser="existing-parser",
                reasoning_parser="existing-reasoner",
            )
        ),
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": []})

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/admin/api/global-settings",
            json={
                "proxy_backend_url": "http://backend/v1",
                "proxy_backend_type": "vllm",
                "omni_model": "example/admin-model",
                "omni_served_model_name": "admin-name",
                "omni_context_length": 24576,
                "vllm_gpu_memory_utilization": 0.72,
                "omni_max_parallel": 8,
                "omni_hf_home": "/cache/admin",
                "sampling_top_p": 0.77,
                "sampling_top_k": 42,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["requires_restart"] is True
    assert "sidecar_env_file" in data["runtime_applied"]
    assert "sidecar_compose_file" in data["runtime_applied"]
    assert data["compose"]["env_written"] is True
    assert data["compose"]["compose_written"] is True

    env = load_vllm_env_file(env_path)
    assert env["OMNI_MODEL"] == "example/admin-model"
    assert env["OMNI_SERVED_MODEL_NAME"] == "admin-name"
    assert env["OMNI_CONTEXT_LENGTH"] == "24576"
    assert env["OMNI_HF_HOME"] == "/cache/admin"
    assert env["OMLX_SAMPLING_TOP_P"] == "0.77"
    assert env["OMLX_SAMPLING_TOP_K"] == "42"
    assert env["VLLM_TOOL_CALL_PARSER"] == "existing-parser"
    assert env["VLLM_REASONING_PARSER"] == "existing-reasoner"

    content = compose_path.read_text(encoding="utf-8")
    assert 'OMLX_ENV_OUTPUT_PATH: "/compose-output/docker-compose.vllm.env"' in content
    assert "example/admin-model" in content
    assert "OMLX_SAMPLING_TOP_P" in content
    assert "0.77" in content


@pytest.mark.asyncio
async def test_admin_llamacpp_settings_save_writes_env_and_compose(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    compose_path = tmp_path / "docker-compose.llamacpp.yml"
    env_path = tmp_path / "docker-compose.llamacpp.env"
    monkeypatch.setenv("OMLX_SIDECAR_BACKEND", "llamacpp")
    monkeypatch.setenv("OMLX_COMPOSE_OUTPUT_PATH", str(compose_path))
    monkeypatch.setenv("OMLX_ENV_OUTPUT_PATH", str(env_path))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": []})

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/api/proxy/sidecar-compose")
        assert response.status_code == 200
        assert response.json()["backend"] == "llamacpp"

        response = await client.post(
            "/admin/api/global-settings",
            json={
                "proxy_backend_url": "http://backend/v1",
                "proxy_backend_type": "llama.cpp",
                "omni_model": "ggml-org/Qwen3-1.7B-GGUF:Q8_0",
                "omni_context_length": 16384,
                "llamacpp_n_gpu_layers": 80,
                "llamacpp_flash_attn": "on",
                "llamacpp_jinja": False,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["requires_restart"] is True
    assert data["compose"]["backend"] == "llamacpp"
    assert data["compose"]["env_written"] is True
    assert data["compose"]["compose_written"] is True

    env = load_llamacpp_env_file(env_path)
    assert env["OMNI_MODEL"] == "ggml-org/Qwen3-1.7B-GGUF:Q8_0"
    assert env["OMNI_CONTEXT_LENGTH"] == "16384"
    assert env["LLAMACPP_N_GPU_LAYERS"] == "80"
    assert env["LLAMACPP_FLASH_ATTN"] == "on"
    assert env["LLAMACPP_JINJA"] == "false"

    content = compose_path.read_text(encoding="utf-8")
    assert "  llamacpp:" in content
    assert 'OMLX_BACKEND_URL: "http://llamacpp:8000/v1"' in content
    assert "ggml-org/Qwen3-1.7B-GGUF:Q8_0" in content


@pytest.mark.asyncio
async def test_proxy_sampling_defaults_seed_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("OMLX_SAMPLING_TOP_P", "0.66")
    monkeypatch.setenv("OMLX_SAMPLING_TOP_K", "13")
    seen_request = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": "qwen"}]}
            )
        seen_request.update(json.loads(request.content.decode()))
        return httpx.Response(200, json={"model": "qwen", "choices": [], "usage": {}})

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [], "stream": False},
        )

    assert response.status_code == 200
    assert seen_request["top_p"] == 0.66
    assert seen_request["top_k"] == 13


@pytest.mark.asyncio
async def test_proxy_admin_model_settings_persist(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(state_path))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "qwen"}]},
        )

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.put(
            "/admin/api/models/qwen/settings",
            json={"model_alias": "local-qwen", "is_default": True},
        )
        assert response.status_code == 200

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/api/models")

    models = response.json()["models"]
    assert models[0]["settings"]["model_alias"] == "local-qwen"
    assert models[0]["model_alias"] == "local-qwen"
    assert models[0]["is_default"] is True


@pytest.mark.asyncio
async def test_proxy_state_migrates_ollama_backend_type(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "model_settings": {},
                "global_overrides": {
                    "proxy_backend_type": "ollama",
                    "proxy_backend_url": "http://my-ollama:11434/v1",
                },
            }
        )
    )
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(state_path))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/api/proxy/config")

    assert response.status_code == 200
    config = response.json()
    assert config["backend_type"] == "openai-compatible"
    assert config["backend_url"] == "http://my-ollama:11434/v1"


@pytest.mark.asyncio
async def test_posting_ollama_backend_type_normalizes(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/admin/api/global-settings",
            json={"proxy_backend_type": "ollama"},
        )
        assert response.status_code == 200

        response = await client.get("/admin/api/proxy/config")

    assert response.json()["backend_type"] == "openai-compatible"


@pytest.mark.asyncio
async def test_sidecar_backend_url_is_enforced(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/admin/api/global-settings",
            json={
                "proxy_backend_type": "vllm",
                "proxy_backend_url": "http://bogus:9999/v1",
            },
        )
        assert response.status_code == 200
        assert response.json()["requires_restart"] is True

        response = await client.get("/admin/api/proxy/config")

    config = response.json()
    assert config["backend_type"] == "vllm"
    assert config["backend_url"] == "http://vllm:8000/v1"


@pytest.mark.asyncio
async def test_backend_profiles_round_trip(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(state_path))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/admin/api/global-settings",
            json={
                "proxy_backend_type": "openai-compatible",
                "proxy_backend_url": "http://my-ollama:11434/v1",
                "proxy_backend_api_key": "secret",
            },
        )
        assert response.status_code == 200

        response = await client.post(
            "/admin/api/global-settings",
            json={"proxy_backend_type": "llama.cpp"},
        )
        assert response.status_code == 200
        response = await client.get("/admin/api/proxy/config")
        assert response.json()["backend_url"] == "http://llamacpp:8000/v1"

        response = await client.post(
            "/admin/api/global-settings",
            json={"proxy_backend_type": "openai-compatible"},
        )
        assert response.status_code == 200
        response = await client.get("/admin/api/proxy/config")

    config = response.json()
    assert config["backend_url"] == "http://my-ollama:11434/v1"
    assert config["backend_api_key_set"] is True

    saved = json.loads(state_path.read_text())
    profiles = saved["backend_profiles"]
    assert (
        profiles["openai-compatible"]["proxy_backend_url"]
        == "http://my-ollama:11434/v1"
    )
    assert profiles["llama.cpp"]["proxy_backend_url"] == "http://llamacpp:8000/v1"


@pytest.mark.asyncio
async def test_global_settings_payload_exposes_backend_url_defaults(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/api/global-settings")

    proxy = response.json()["proxy"]
    assert proxy["backend_url_defaults"] == {
        "vllm": "http://vllm:8000/v1",
        "llama.cpp": "http://llamacpp:8000/v1",
        "openai-compatible": "http://host.docker.internal:11434/v1",
    }
    assert proxy["backend_url_locked"] == ["vllm", "llama.cpp"]
    profiles = proxy["backend_profiles"]
    assert (
        profiles["openai-compatible"]["backend_url"]
        == "http://host.docker.internal:11434/v1"
    )
    assert profiles["vllm"]["backend_url"] == "http://vllm:8000/v1"


@pytest.mark.asyncio
async def test_sidecar_restart_conflicts_for_openai_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/admin/api/sidecar/restart")

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_sidecar_restart_unavailable_without_docker_socket(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("OMLX_DOCKER_SOCK", str(tmp_path / "missing.sock"))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/admin/api/global-settings",
            json={"proxy_backend_type": "vllm"},
        )
        assert response.status_code == 200

        response = await client.post("/admin/api/sidecar/restart")

    assert response.status_code == 501
    assert "docker.sock" in response.json()["detail"]


@pytest.mark.asyncio
async def test_sidecar_restart_happy_path(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    restarted = []

    async def fake_restart(service: str) -> str:
        restarted.append(service)
        return "abc123def456"

    monkeypatch.setattr("omlx.proxy.admin.restart_compose_service", fake_restart)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/admin/api/global-settings",
            json={"proxy_backend_type": "llama.cpp"},
        )
        assert response.status_code == 200

        response = await client.post("/admin/api/sidecar/restart")

    assert response.status_code == 202
    body = response.json()
    assert body["service"] == "llamacpp"
    assert body["container_id"] == "abc123def456"
    assert restarted == ["llamacpp"]


@pytest.mark.asyncio
async def test_global_settings_payload_reports_docker_socket(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async def fetch_payload() -> dict:
        app = _app_with_mock_backend(handler)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/admin/api/global-settings")
        return response.json()["proxy"]

    monkeypatch.setenv("OMLX_DOCKER_SOCK", str(tmp_path / "missing.sock"))
    assert (await fetch_payload())["docker_socket_available"] is False

    socket_path = tmp_path / "docker.sock"
    socket_path.touch()
    monkeypatch.setenv("OMLX_DOCKER_SOCK", str(socket_path))
    assert (await fetch_payload())["docker_socket_available"] is True


@pytest.mark.asyncio
async def test_sidecar_launch_overrides_stale_backend_state(monkeypatch, tmp_path):
    """OMLX_SIDECAR_BACKEND wins over a state file from another stack."""
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "model_settings": {},
                "global_overrides": {
                    "proxy_backend_type": "llama.cpp",
                    "proxy_backend_url": "http://llamacpp:8000/v1",
                },
            }
        )
    )
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(state_path))
    monkeypatch.setenv("OMLX_SIDECAR_BACKEND", "vllm")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/api/proxy/config")

    assert response.status_code == 200
    config = response.json()
    assert config["backend_type"] == "vllm"
    assert config["backend_url"] == "http://vllm:8000/v1"
    # The state file was rewritten so the next boot agrees.
    saved = json.loads(state_path.read_text())
    assert saved["global_overrides"]["proxy_backend_type"] == "vllm"
    # The llama.cpp profile was archived for switching back.
    assert (
        saved["backend_profiles"]["llama.cpp"]["proxy_backend_url"]
        == "http://llamacpp:8000/v1"
    )


@pytest.mark.asyncio
async def test_sidecar_env_matching_state_leaves_overrides_alone(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "model_settings": {},
                "global_overrides": {
                    "proxy_backend_type": "vllm",
                    "proxy_backend_api_key": "sk-keep",
                },
            }
        )
    )
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(state_path))
    monkeypatch.setenv("OMLX_SIDECAR_BACKEND", "vllm")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/api/proxy/config")

    assert response.status_code == 200
    config = response.json()
    assert config["backend_type"] == "vllm"
    assert config["backend_api_key_set"] is True


@pytest.mark.asyncio
async def test_standalone_launch_clears_stale_sidecar_state(monkeypatch, tmp_path):
    """Without OMLX_SIDECAR_BACKEND, a persisted sidecar type is stale."""
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "model_settings": {},
                "global_overrides": {
                    "proxy_backend_type": "vllm",
                    "proxy_backend_url": "http://vllm:8000/v1",
                },
            }
        )
    )
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(state_path))
    monkeypatch.delenv("OMLX_SIDECAR_BACKEND", raising=False)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/api/proxy/config")

    assert response.status_code == 200
    config = response.json()
    assert config["backend_type"] == "openai-compatible"
    # Falls back to the env/config-provided URL, not the dead vllm host.
    assert config["backend_url"] != "http://vllm:8000/v1"


@pytest.mark.asyncio
async def test_proxy_config_reports_backend_context_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": "gemma", "max_model_len": 65000}]}
            )
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        config = (await client.get("/admin/api/proxy/config")).json()
        settings = (await client.get("/admin/api/global-settings")).json()

    assert config["backend_context_limit"] == 65000
    assert settings["proxy"]["backend_context_limit"] == 65000


@pytest.mark.asyncio
async def test_global_settings_never_prefill_max_tokens(monkeypatch, tmp_path):
    """The Max Tokens field must not default to the context size."""
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        settings = (await client.get("/admin/api/global-settings")).json()

    assert settings["sampling"]["max_tokens"] is None
