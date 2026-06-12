# SPDX-License-Identifier: Apache-2.0
"""Tests for proxy-side serving stats accounting."""

import json

import httpx
import pytest

from omlx.proxy.app import create_app
from omlx.proxy.backend import OpenAIBackend
from omlx.proxy.config import ProxyConfig
from omlx.proxy.stats import track_usage_stream
from omlx.server_metrics import ServerMetrics


def _sse(*events: str) -> bytes:
    return "".join(f"data: {e}\n\n" for e in events).encode()


def _chat_chunk(model="qwen", content=None, finish=None, usage=None):
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [],
    }
    if content is not None or finish is not None:
        chunk["choices"] = [
            {
                "index": 0,
                "delta": {"content": content} if content is not None else {},
                "finish_reason": finish,
            }
        ]
    if usage is not None:
        chunk["usage"] = usage
    return json.dumps(chunk)


_USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 40,
    "prompt_tokens_details": {"cached_tokens": 25},
}


def _make_app(handler, monkeypatch, tmp_path, config=None):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("OMLX_PROXY_STATS_PATH", str(tmp_path / "stats.json"))
    config = config or ProxyConfig(backend_url="http://backend/v1")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OpenAIBackend(config=config, client=client)
    return create_app(config=config, backend=backend)


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def _get_stats(client, **params):
    response = await client.get("/admin/api/stats", params=params)
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
async def test_streaming_chat_passthrough_records_usage(monkeypatch, tmp_path):
    seen_body = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        seen_body.update(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            content=_sse(
                _chat_chunk(content="hel"),
                _chat_chunk(content="lo", finish="stop"),
                _chat_chunk(usage=_USAGE),
                "[DONE]",
            ),
            headers={"content-type": "text/event-stream"},
        )

    app = _make_app(handler, monkeypatch, tmp_path)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [], "stream": True},
        )
        assert response.status_code == 200
        body = response.text

        stats = await _get_stats(client)

    # Proxy injects include_usage so the backend reports usage.
    assert seen_body["stream_options"] == {"include_usage": True}
    # The client did not ask for usage, so the usage-only chunk is stripped.
    assert '"prompt_tokens": 100' not in body
    assert "hel" in body and "lo" in body

    assert stats["total_prompt_tokens"] == 100
    assert stats["total_completion_tokens"] == 40
    assert stats["total_cached_tokens"] == 25
    assert stats["total_requests"] == 1
    assert stats["cache_efficiency"] == 25.0


@pytest.mark.asyncio
async def test_streaming_passthrough_keeps_usage_chunk_when_client_asks(
    monkeypatch, tmp_path
):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                _chat_chunk(content="hi", finish="stop"),
                _chat_chunk(usage=_USAGE),
                "[DONE]",
            ),
            headers={"content-type": "text/event-stream"},
        )

    app = _make_app(handler, monkeypatch, tmp_path)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert response.status_code == 200
        assert '"prompt_tokens": 100' in response.text

        stats = await _get_stats(client)
    assert stats["total_prompt_tokens"] == 100
    assert stats["total_requests"] == 1


@pytest.mark.asyncio
async def test_nonstreaming_chat_passthrough_records_usage(monkeypatch, tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _USAGE,
            },
        )

    app = _make_app(handler, monkeypatch, tmp_path)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": []},
        )
        assert response.status_code == 200

        stats = await _get_stats(client)
    assert stats["total_prompt_tokens"] == 100
    assert stats["total_completion_tokens"] == 40
    assert stats["total_cached_tokens"] == 25
    assert stats["total_requests"] == 1
    # Non-streaming requests record zero durations and must not skew speeds.
    assert stats["avg_prefill_tps"] == 0.0
    assert stats["avg_generation_tps"] == 0.0


@pytest.mark.asyncio
async def test_stats_per_model_filter(monkeypatch, tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "x"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _USAGE,
            },
        )

    app = _make_app(handler, monkeypatch, tmp_path)
    async with _client(app) as client:
        await client.post(
            "/v1/chat/completions", json={"model": "qwen", "messages": []}
        )

        matching = await _get_stats(client, model="qwen")
        other = await _get_stats(client, model="other-model")

    assert matching["total_prompt_tokens"] == 100
    assert other["total_prompt_tokens"] == 0
    assert other["total_requests"] == 0


@pytest.mark.asyncio
async def test_anthropic_streaming_records_usage_and_durations(monkeypatch, tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                _chat_chunk(content="hi there"),
                _chat_chunk(finish="stop", usage=_USAGE),
                "[DONE]",
            ),
            headers={"content-type": "text/event-stream"},
        )

    app = _make_app(handler, monkeypatch, tmp_path)
    async with _client(app) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-compatible",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200
        assert "message_stop" in response.text

        stats = await _get_stats(client)
    assert stats["total_prompt_tokens"] == 100
    assert stats["total_completion_tokens"] == 40
    assert stats["total_cached_tokens"] == 25
    assert stats["total_requests"] == 1
    # Streaming requests carry real wall-clock durations.
    assert stats["avg_generation_tps"] > 0.0


@pytest.mark.asyncio
async def test_anthropic_nonstreaming_records_unscaled_usage(monkeypatch, tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _USAGE,
            },
        )

    config = ProxyConfig(
        backend_url="http://backend/v1",
        context_scaling_enabled=True,
        target_context_size=200000,
        actual_context_size=50000,
    )
    app = _make_app(handler, monkeypatch, tmp_path, config=config)
    async with _client(app) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-compatible",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200
        # The client-visible usage is scaled 4x...
        assert response.json()["usage"]["input_tokens"] == 400

        stats = await _get_stats(client)
    # ...but stats record the real backend token counts.
    assert stats["total_prompt_tokens"] == 100
    assert stats["total_completion_tokens"] == 40


@pytest.mark.asyncio
async def test_responses_endpoint_records_usage(monkeypatch, tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _USAGE,
            },
        )

    app = _make_app(handler, monkeypatch, tmp_path)
    async with _client(app) as client:
        response = await client.post(
            "/v1/responses",
            json={"model": "qwen", "input": "hello"},
        )
        assert response.status_code == 200

        stats = await _get_stats(client)
    assert stats["total_prompt_tokens"] == 100
    assert stats["total_requests"] == 1


@pytest.mark.asyncio
async def test_embeddings_passthrough_not_counted(monkeypatch, tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [],
                "usage": {"prompt_tokens": 7, "total_tokens": 7},
            },
        )

    app = _make_app(handler, monkeypatch, tmp_path)
    async with _client(app) as client:
        response = await client.post(
            "/v1/embeddings", json={"model": "embed", "input": "hello"}
        )
        assert response.status_code == 200

        stats = await _get_stats(client)
    assert stats["total_requests"] == 0
    assert stats["total_prompt_tokens"] == 0


@pytest.mark.asyncio
async def test_clear_endpoints_split_session_and_alltime(monkeypatch, tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "x"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _USAGE,
            },
        )

    app = _make_app(handler, monkeypatch, tmp_path)
    async with _client(app) as client:
        await client.post(
            "/v1/chat/completions", json={"model": "qwen", "messages": []}
        )

        # Clearing the session keeps all-time totals.
        await client.post("/admin/api/stats/clear")
        session = await _get_stats(client)
        alltime = await _get_stats(client, scope="alltime")
        assert session["total_requests"] == 0
        assert alltime["total_requests"] == 1

        # Clearing all-time wipes the persisted totals too.
        await client.post("/admin/api/stats/clear-alltime")
        alltime = await _get_stats(client, scope="alltime")
        assert alltime["total_requests"] == 0


@pytest.mark.asyncio
async def test_alltime_stats_persist_across_app_instances(monkeypatch, tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "x"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _USAGE,
            },
        )

    app = _make_app(handler, monkeypatch, tmp_path)
    async with _client(app) as client:
        await client.post(
            "/v1/chat/completions", json={"model": "qwen", "messages": []}
        )
    app.state.server_metrics.save_alltime()

    second = _make_app(handler, monkeypatch, tmp_path)
    async with _client(second) as client:
        alltime = await _get_stats(client, scope="alltime")
        session = await _get_stats(client)
    assert alltime["total_requests"] == 1
    assert alltime["total_prompt_tokens"] == 100
    assert session["total_requests"] == 0


@pytest.mark.asyncio
async def test_track_usage_stream_handles_split_sse_byte_boundaries():
    payload = _sse(
        _chat_chunk(content="hello", finish="stop"),
        _chat_chunk(usage=_USAGE),
        "[DONE]",
    )

    async def chunked():
        # Feed the stream in 7-byte slices so lines split mid-JSON.
        for i in range(0, len(payload), 7):
            yield payload[i : i + 7]

    metrics = ServerMetrics()
    out = b""
    async for chunk in track_usage_stream(
        chunked(), metrics=metrics, model_id="qwen", strip_usage_chunk=True
    ):
        out += chunk

    assert b"hello" in out
    assert b'"prompt_tokens": 100' not in out
    assert b"[DONE]" in out
    snapshot = metrics.get_snapshot()
    assert snapshot["total_prompt_tokens"] == 100
    assert snapshot["total_completion_tokens"] == 40
    assert snapshot["total_cached_tokens"] == 25
    assert snapshot["total_requests"] == 1


@pytest.mark.asyncio
async def test_streaming_injection_retries_without_stream_options(
    monkeypatch, tmp_path
):
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        calls.append(body)
        if "stream_options" in body:
            return httpx.Response(400, json={"error": "unknown field"})
        return httpx.Response(
            200,
            content=_sse(_chat_chunk(content="ok", finish="stop"), "[DONE]"),
            headers={"content-type": "text/event-stream"},
        )

    app = _make_app(handler, monkeypatch, tmp_path)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [], "stream": True},
        )
        assert response.status_code == 200
        assert "ok" in response.text

    assert len(calls) == 2
    assert "stream_options" not in calls[1]
