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
