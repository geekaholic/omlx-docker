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
async def test_anthropic_streaming_injects_include_usage_for_stats(
    monkeypatch, tmp_path
):
    seen_body = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_body.update(json.loads(request.content.decode()))
        events = [
            _chat_chunk(content="hi there"),
            _chat_chunk(finish="stop"),
        ]
        if seen_body.get("stream_options", {}).get("include_usage") is True:
            events.append(_chat_chunk(usage=_USAGE))
        events.append("[DONE]")
        return httpx.Response(
            200,
            content=_sse(*events),
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

        stats = await _get_stats(client)

    assert seen_body["stream_options"] == {"include_usage": True}
    assert stats["total_prompt_tokens"] == 100
    assert stats["total_completion_tokens"] == 40
    assert stats["total_cached_tokens"] == 25
    assert stats["cache_efficiency"] == 25.0
    assert stats["avg_prefill_tps"] > 0.0
    assert stats["avg_generation_tps"] > 0.0


@pytest.mark.asyncio
async def test_anthropic_streaming_retries_when_include_usage_is_rejected(
    monkeypatch, tmp_path
):
    seen_bodies = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen_bodies.append(body)
        if body.get("stream_options", {}).get("include_usage") is True:
            return httpx.Response(
                400,
                json={"error": {"message": "stream_options unsupported"}},
            )
        return httpx.Response(
            200,
            content=_sse(
                _chat_chunk(content="hi"),
                _chat_chunk(finish="stop"),
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

    assert len(seen_bodies) == 2
    assert seen_bodies[0]["stream_options"] == {"include_usage": True}
    assert "stream_options" not in seen_bodies[1]
    assert stats["total_requests"] == 1


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


def _ollama_handler_factory(expires_in_seconds=240):
    import datetime

    expires = (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(seconds=expires_in_seconds)
    ).isoformat()

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": "llama3:8b"}]}
            )
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})
        if path == "/api/ps":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "llama3:8b",
                            "model": "llama3:8b",
                            "size": 8 * 1024**3,
                            "expires_at": expires,
                        }
                    ]
                },
            )
        return httpx.Response(404, json={"error": "not found"})

    return handler


@pytest.mark.asyncio
async def test_stats_active_models_ollama_shows_size_and_ttl(monkeypatch, tmp_path):
    app = _make_app(_ollama_handler_factory(), monkeypatch, tmp_path)
    async with _client(app) as client:
        stats = await _get_stats(client)

    active = stats["active_models"]
    rows = active["models"]
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "llama3:8b"
    assert row["estimated_size"] == 8 * 1024**3
    assert "GB" in row["estimated_size_formatted"]
    assert row["ttl_remaining_seconds"] is not None
    assert 0 < row["ttl_remaining_seconds"] <= 300
    assert active["model_memory_used"] == 8 * 1024**3


@pytest.mark.asyncio
async def test_stats_memory_pressure_populated_from_ollama(monkeypatch, tmp_path):
    import omlx.proxy.admin as proxy_admin

    monkeypatch.setattr(
        proxy_admin,
        "host_memory_info",
        lambda: {"total_bytes": 32 * 1024**3, "available_bytes": 16 * 1024**3},
    )
    app = _make_app(_ollama_handler_factory(), monkeypatch, tmp_path)
    async with _client(app) as client:
        stats = await _get_stats(client)

    active = stats["active_models"]
    mp = active["memory_pressure"]
    assert mp["enabled"] is True
    assert mp["current_bytes"] == 8 * 1024**3
    assert mp["soft_bytes"] == 0
    assert mp["hard_bytes"] == 32 * 1024**3
    assert mp["pressure_level"] == "ok"
    assert active["model_memory_max"] == 32 * 1024**3


def test_active_models_memory_pressure_sidecar_branch():
    from omlx.proxy.admin import _active_models_payload

    metrics = {
        "summary": {"running_requests": 1, "waiting_requests": 0},
        "ollama": {},
    }
    payload = _active_models_payload(
        [{"id": "qwen"}],
        metrics,
        memory={
            "host_total_bytes": 32 * 1024**3,
            "sidecar_bytes": 8 * 1024**3,
            "soft_fraction": 0.8,
        },
    )

    mp = payload["memory_pressure"]
    assert mp["enabled"] is True
    assert mp["current_bytes"] == 8 * 1024**3
    assert mp["soft_bytes"] == int(32 * 1024**3 * 0.8)
    assert mp["hard_bytes"] == 32 * 1024**3
    assert mp["pressure_level"] == "ok"
    row = payload["models"][0]
    assert row["estimated_size"] == 8 * 1024**3
    assert "GB" in row["estimated_size_formatted"]
    assert payload["model_memory_used"] == 8 * 1024**3


def test_active_models_memory_pressure_disabled_without_signals():
    from omlx.proxy.admin import _active_models_payload

    payload = _active_models_payload([{"id": "qwen"}], {"summary": {}}, memory=None)

    mp = payload["memory_pressure"]
    assert mp["enabled"] is False
    assert mp["hard_bytes"] == 0
    assert payload["models"][0]["estimated_size_formatted"] == "remote"


def test_memory_pressure_levels():
    from omlx.proxy.admin import _memory_pressure_payload

    gib = 1024**3
    ok = _memory_pressure_payload(10 * gib, 24 * gib, 32 * gib)
    assert ok["pressure_level"] == "ok"
    soft = _memory_pressure_payload(28 * gib, 24 * gib, 32 * gib)
    assert soft["pressure_level"] == "soft"
    hard = _memory_pressure_payload(33 * gib, 24 * gib, 32 * gib)
    assert hard["pressure_level"] == "hard"


_VLLM_METRICS_TEXT = """\
# HELP vllm:num_requests_running Number of requests currently running.
vllm:num_requests_running{model_name="qwen"} 2.0
vllm:num_requests_waiting{model_name="qwen"} 3.0
vllm:prompt_tokens_total{model_name="qwen"} 1000.0
vllm:generation_tokens_total{model_name="qwen"} 500.0
"""


@pytest.mark.asyncio
async def test_stats_active_models_vllm_attaches_request_counts(monkeypatch, tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": "qwen"}]}
            )
        if path == "/metrics":
            return httpx.Response(
                200, text=_VLLM_METRICS_TEXT, headers={"content-type": "text/plain"}
            )
        return httpx.Response(404, json={"error": "not found"})

    app = _make_app(handler, monkeypatch, tmp_path)
    async with _client(app) as client:
        stats = await _get_stats(client)

    active = stats["active_models"]
    assert active["total_active_requests"] == 2
    assert active["total_waiting_requests"] == 3
    row = active["models"][0]
    assert row["id"] == "qwen"
    assert row["active_requests"] == 2
    assert row["waiting_requests"] == 3


@pytest.mark.asyncio
async def test_collect_backend_metrics_cached_respects_ttl(monkeypatch, tmp_path):
    from omlx.proxy.metrics import collect_backend_metrics_cached

    hits = {"metrics": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            hits["metrics"] += 1
            return httpx.Response(
                200, text=_VLLM_METRICS_TEXT, headers={"content-type": "text/plain"}
            )
        return httpx.Response(404, json={"error": "not found"})

    config = ProxyConfig(backend_url="http://backend/v1")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OpenAIBackend(config=config, client=client)

    first = await collect_backend_metrics_cached(backend, ttl=60.0)
    second = await collect_backend_metrics_cached(backend, ttl=60.0)
    assert hits["metrics"] == 1
    assert first["summary"]["running_requests"] == 2.0
    assert second is first

    third = await collect_backend_metrics_cached(backend, ttl=0.0)
    assert hits["metrics"] == 2
    assert third["summary"]["running_requests"] == 2.0
    await client.aclose()


def _state_with_max_tokens(value):
    from omlx.proxy.admin import ProxyAdminState

    state = ProxyAdminState()
    state.global_overrides["sampling_max_tokens"] = value
    return state


def test_max_tokens_injection_skipped_at_context_limit():
    from omlx.proxy.app import apply_proxy_request_defaults

    body = {"model": "m", "messages": []}
    injected = apply_proxy_request_defaults(
        body, _state_with_max_tokens(65000), context_limit=65000
    )
    assert injected is False
    assert "max_tokens" not in body


def test_max_tokens_injected_below_context_limit():
    from omlx.proxy.app import apply_proxy_request_defaults

    body = {"model": "m", "messages": []}
    injected = apply_proxy_request_defaults(
        body, _state_with_max_tokens(16384), context_limit=65000
    )
    assert injected is True
    assert body["max_tokens"] == 16384


def test_max_tokens_zero_or_unset_never_injected():
    from omlx.proxy.app import apply_proxy_request_defaults

    for value in (0, "0", None, ""):
        body = {"model": "m", "messages": []}
        injected = apply_proxy_request_defaults(
            body, _state_with_max_tokens(value), context_limit=None
        )
        assert injected is False, value
        assert "max_tokens" not in body, value


def test_max_tokens_client_value_never_overridden():
    from omlx.proxy.app import apply_proxy_request_defaults

    body = {"model": "m", "messages": [], "max_tokens": 50}
    injected = apply_proxy_request_defaults(
        body, _state_with_max_tokens(16384), context_limit=65000
    )
    assert injected is False
    assert body["max_tokens"] == 50


@pytest.mark.asyncio
async def test_passthrough_retries_without_injected_max_tokens(monkeypatch, tmp_path):
    """A 400 caused by the injected cap retries once without it."""
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            # context-limit probe: no max_model_len reported
            return httpx.Response(200, json={"data": [{"id": "qwen"}]})
        body = json.loads(request.content.decode())
        calls.append(body)
        if body.get("max_tokens"):
            return httpx.Response(
                400,
                json={
                    "error": {"message": "This model's maximum context length is..."}
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _USAGE,
            },
        )

    app = _make_app(handler, monkeypatch, tmp_path)
    app.state.proxy_admin_state.global_overrides["sampling_max_tokens"] = 4096
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": []},
        )

    assert response.status_code == 200
    assert "ok" in response.text
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 4096
    assert "max_tokens" not in calls[1]


@pytest.mark.asyncio
async def test_streaming_passthrough_retries_without_injected_max_tokens(
    monkeypatch, tmp_path
):
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen"}]})
        body = json.loads(request.content.decode())
        calls.append(body)
        if body.get("max_tokens"):
            return httpx.Response(400, json={"error": "too long"})
        return httpx.Response(
            200,
            content=_sse(_chat_chunk(content="ok", finish="stop"), "[DONE]"),
            headers={"content-type": "text/event-stream"},
        )

    app = _make_app(handler, monkeypatch, tmp_path)
    app.state.proxy_admin_state.global_overrides["sampling_max_tokens"] = 4096
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [], "stream": True},
        )

    assert response.status_code == 200
    assert "ok" in response.text
    assert any("max_tokens" not in c for c in calls)


@pytest.mark.asyncio
async def test_context_limit_probe_vllm_max_model_len(monkeypatch, tmp_path):
    from omlx.proxy.metrics import backend_context_limit

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": "gemma", "max_model_len": 65000}]}
            )
        return httpx.Response(404)

    config = ProxyConfig(backend_url="http://backend/v1")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OpenAIBackend(config=config, client=client)
    assert await backend_context_limit(backend) == 65000
    await client.aclose()


@pytest.mark.asyncio
async def test_context_limit_probe_llamacpp_props(monkeypatch, tmp_path):
    from omlx.proxy.metrics import backend_context_limit

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen"}]})
        if request.url.path == "/props":
            return httpx.Response(
                200, json={"default_generation_settings": {"n_ctx": 16384}}
            )
        return httpx.Response(404)

    config = ProxyConfig(backend_url="http://backend/v1")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OpenAIBackend(config=config, client=client)
    assert await backend_context_limit(backend) == 16384
    await client.aclose()


@pytest.mark.asyncio
async def test_context_limit_probe_none_when_unreported(monkeypatch, tmp_path):
    from omlx.proxy.metrics import backend_context_limit

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        return httpx.Response(404)

    config = ProxyConfig(backend_url="http://backend/v1")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = OpenAIBackend(config=config, client=client)
    assert await backend_context_limit(backend) is None
    await client.aclose()
