# SPDX-License-Identifier: Apache-2.0

import importlib
import json

import httpx
import pytest

from omlx.api.anthropic_models import AnthropicMessage, AnthropicTool, MessagesRequest
from omlx.proxy.app import anthropic_to_openai_chat_body, create_app
from omlx.proxy.backend import OpenAIBackend
from omlx.proxy.config import ProxyConfig
from omlx.proxy.llamacpp_compose import load_llamacpp_env_file
from omlx.proxy.metrics import parse_prometheus_text, select_prometheus_metrics
from omlx.proxy.scaling import scale_token_count
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


def test_anthropic_request_preserves_tool_history_for_proxy():
    request = MessagesRequest(
        model="claude-compatible",
        max_tokens=128,
        messages=[
            AnthropicMessage(
                role="assistant",
                content=[
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"file_path": "/tmp/a.css"},
                    }
                ],
            ),
            AnthropicMessage(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "body { color: red; }",
                    }
                ],
            ),
            AnthropicMessage(role="user", content="continue"),
        ],
        tools=[AnthropicTool(**_read_tool_schema())],
    )

    body = anthropic_to_openai_chat_body(request)

    serialized_messages = json.dumps(body["messages"])
    assert "[Calling tool:" not in serialized_messages
    assert body["messages"][0] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "toolu_1",
                "type": "function",
                "function": {
                    "name": "Read",
                    "arguments": '{"file_path": "/tmp/a.css"}',
                },
            }
        ],
    }
    assert body["messages"][1] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "body { color: red; }",
    }
    assert body["messages"][2] == {"role": "user", "content": "continue"}


def test_anthropic_request_serializes_empty_tool_args_for_vllm():
    request = MessagesRequest(
        model="claude-compatible",
        max_tokens=128,
        messages=[
            AnthropicMessage(
                role="assistant",
                content=[
                    {
                        "type": "tool_use",
                        "id": "chatcmpl-tool-92bc0253e16bb5aa",
                        "name": "EnterPlanMode",
                        "input": {},
                    }
                ],
            ),
        ],
        tools=[
            AnthropicTool(
                name="EnterPlanMode",
                input_schema={"type": "object", "properties": {}},
            )
        ],
    )

    body = anthropic_to_openai_chat_body(request)

    assert body["messages"][0]["tool_calls"] == [
        {
            "id": "chatcmpl-tool-92bc0253e16bb5aa",
            "type": "function",
            "function": {
                "name": "EnterPlanMode",
                "arguments": "{}",
            },
        }
    ]


def test_scale_token_count_uses_target_context_ratio():
    config = ProxyConfig(
        backend_url="http://backend/v1",
        context_scaling_enabled=True,
        target_context_size=200000,
        actual_context_size=50000,
    )

    assert scale_token_count(25, config) == 100


def _read_tool_schema():
    return {
        "name": "Read",
        "description": "Read a file",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    }


def _write_tool_schema():
    return {
        "name": "Write",
        "description": "Write a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    }


def _edit_tool_schema():
    return {
        "name": "Edit",
        "description": "Edit a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    }


def _anthropic_sse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ")
        events.append(json.loads(payload))
    return events


def _sse_payload(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


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
async def test_anthropic_messages_non_stream_accepts_vllm_reasoning_field():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gemma",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "answer",
                            "reasoning": "plan",
                        },
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
                "model": "gemma",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    blocks = response.json()["content"]
    assert blocks[0]["type"] == "thinking"
    assert blocks[0]["thinking"] == "plan"
    assert blocks[1] == {"type": "text", "text": "answer"}


@pytest.mark.asyncio
async def test_anthropic_messages_non_stream_recovers_bracket_tool_call():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content.decode())
        assert body["tools"][0]["function"]["name"] == "Read"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "[Calling tool: Read({"
                                '"file_path":"/home/bud/Work/test-omni/css/style.css"'
                                "})]"
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
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
                "messages": [{"role": "user", "content": "read the css"}],
                "tools": [_read_tool_schema()],
            },
        )

    assert response.status_code == 200
    data = response.json()
    text_blocks = [block for block in data["content"] if block["type"] == "text"]
    tool_use_blocks = [
        block for block in data["content"] if block["type"] == "tool_use"
    ]
    assert all("[Calling tool:" not in block["text"] for block in text_blocks)
    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0]["name"] == "Read"
    assert tool_use_blocks[0]["input"] == {
        "file_path": "/home/bud/Work/test-omni/css/style.css"
    }
    assert data["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_anthropic_messages_non_stream_does_not_promote_unknown_tool():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "[Calling tool: Write({"
                                '"file_path":"/tmp/a.css","content":"x"'
                                "})]"
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
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
                "messages": [{"role": "user", "content": "read the css"}],
                "tools": [_read_tool_schema()],
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert not [block for block in data["content"] if block["type"] == "tool_use"]
    assert data["stop_reason"] == "end_turn"


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
async def test_anthropic_messages_stream_accepts_vllm_reasoning_delta():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {"reasoning": "plan"},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {"content": "answer"},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": 4,
                            "completion_tokens": 2,
                            "total_tokens": 6,
                        },
                    }
                )
                + "data: [DONE]\n\n"
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
                "model": "gemma",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    events = _anthropic_sse_events(response.text)
    thinking = "".join(
        event["delta"]["thinking"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "thinking_delta"
    )
    text = "".join(
        event["delta"]["text"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "text_delta"
    )

    assert thinking == "plan"
    assert text == "answer"


@pytest.mark.asyncio
async def test_anthropic_messages_stream_strips_markers_without_tools():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {"content": "Visible <|channel>thou"},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {"content": "ght\nsecret<channel|> text"},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                )
                + "data: [DONE]\n\n"
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
    events = _anthropic_sse_events(response.text)
    text = "".join(
        event["delta"]["text"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "text_delta"
    )

    assert text == "Visible  text"
    assert "<|channel>" not in text
    assert "<channel|>" not in text
    assert "secret" not in text


@pytest.mark.asyncio
async def test_anthropic_messages_stream_recovers_bracket_tool_call():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content.decode())
        assert body["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                _sse_payload(
                    {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}
                )
                + _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {"content": "[Calling tool:"},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": (
                                        ' Read({"file_path":"/tmp/a.css"})]'
                                    )
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": 4,
                            "completion_tokens": 2,
                            "total_tokens": 6,
                        },
                    }
                )
                + "data: [DONE]\n\n"
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
                "messages": [{"role": "user", "content": "read the css"}],
                "tools": [_read_tool_schema()],
            },
        )

    assert response.status_code == 200
    events = _anthropic_sse_events(response.text)
    text = "".join(
        event["delta"]["text"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "text_delta"
    )
    tool_use_blocks = [
        event["content_block"]
        for event in events
        if event.get("type") == "content_block_start"
        and event.get("content_block", {}).get("type") == "tool_use"
    ]
    tool_json = "".join(
        event["delta"]["partial_json"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "input_json_delta"
    )
    stop_reasons = [
        event["delta"]["stop_reason"]
        for event in events
        if event.get("type") == "message_delta"
    ]

    assert "[Calling tool:" not in text
    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0]["name"] == "Read"
    assert json.loads(tool_json) == {"file_path": "/tmp/a.css"}
    assert stop_reasons == ["tool_use"]


@pytest.mark.asyncio
async def test_anthropic_messages_stream_recovers_split_reasoning_content_tool_call():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "reasoning_content": (
                                        '[Calling tool: Write({"content":"body"'
                                    )
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": ',"file_path":"/tmp/a.css"})]'
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": 4,
                            "completion_tokens": 2,
                            "total_tokens": 6,
                        },
                    }
                )
                + "data: [DONE]\n\n"
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
                "messages": [{"role": "user", "content": "write the css"}],
                "tools": [_write_tool_schema()],
            },
        )

    assert response.status_code == 200
    events = _anthropic_sse_events(response.text)
    text = "".join(
        event["delta"]["text"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "text_delta"
    )
    tool_use_blocks = [
        event["content_block"]
        for event in events
        if event.get("type") == "content_block_start"
        and event.get("content_block", {}).get("type") == "tool_use"
    ]
    tool_json = "".join(
        event["delta"]["partial_json"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "input_json_delta"
    )
    stop_reasons = [
        event["delta"]["stop_reason"]
        for event in events
        if event.get("type") == "message_delta"
    ]

    assert text == ""
    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0]["name"] == "Write"
    assert json.loads(tool_json) == {
        "content": "body",
        "file_path": "/tmp/a.css",
    }
    assert stop_reasons == ["tool_use"]


@pytest.mark.asyncio
async def test_anthropic_messages_stream_preserves_split_reasoning_edit_string():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "reasoning_content": (
                                        '[Calling tool: Edit({"file_path":'
                                        '"js/window-manager.js","old_string":"'
                                        "const val"
                                    )
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "reasoning_content": (
                                        'ue = 1;","new_string":"const value = 2;"})]'
                                    )
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": 4,
                            "completion_tokens": 2,
                            "total_tokens": 6,
                        },
                    }
                )
                + "data: [DONE]\n\n"
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
                "messages": [{"role": "user", "content": "edit the js"}],
                "tools": [_edit_tool_schema()],
            },
        )

    assert response.status_code == 200
    events = _anthropic_sse_events(response.text)
    tool_json = "".join(
        event["delta"]["partial_json"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "input_json_delta"
    )

    assert json.loads(tool_json) == {
        "file_path": "js/window-manager.js",
        "old_string": "const value = 1;",
        "new_string": "const value = 2;",
    }


@pytest.mark.asyncio
async def test_anthropic_messages_stream_suppresses_malformed_split_tool_tail():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "reasoning_content": (
                                        '[Calling tool: Write({"content":"body"'
                                    )
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": '`,file_path: "/tmp/a.css"})]'
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": 4,
                            "completion_tokens": 2,
                            "total_tokens": 6,
                        },
                    }
                )
                + "data: [DONE]\n\n"
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
                "messages": [{"role": "user", "content": "write the css"}],
                "tools": [_write_tool_schema()],
            },
        )

    assert response.status_code == 200
    events = _anthropic_sse_events(response.text)
    text = "".join(
        event["delta"]["text"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "text_delta"
    )
    tool_use_blocks = [
        event
        for event in events
        if event.get("type") == "content_block_start"
        and event.get("content_block", {}).get("type") == "tool_use"
    ]
    stop_reasons = [
        event["delta"]["stop_reason"]
        for event in events
        if event.get("type") == "message_delta"
    ]

    assert "file_path" not in text
    assert "Calling tool" not in text
    assert tool_use_blocks == []
    assert stop_reasons == ["end_turn"]


@pytest.mark.asyncio
async def test_anthropic_messages_stream_strips_split_gemma_channel_markers():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                _sse_payload(
                    {
                        "model": "gemma-4-26B-A4B-it",
                        "choices": [
                            {
                                "delta": {"content": "Plan ready.\n\n<|channel>thou"},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                + _sse_payload(
                    {
                        "model": "gemma-4-26B-A4B-it",
                        "choices": [
                            {
                                "delta": {
                                    "content": "ght\ninternal notes<channel|> done"
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                + _sse_payload(
                    {
                        "model": "gemma-4-26B-A4B-it",
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                    }
                )
                + "data: [DONE]\n\n"
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
                "messages": [{"role": "user", "content": "continue"}],
                "tools": [_read_tool_schema()],
            },
        )

    assert response.status_code == 200
    events = _anthropic_sse_events(response.text)
    text = "".join(
        event["delta"]["text"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "text_delta"
    )

    assert text == "Plan ready.\n\n done"
    assert "<|channel>" not in text
    assert "<channel|>" not in text
    assert "internal notes" not in text


@pytest.mark.asyncio
async def test_anthropic_messages_stream_recovers_gemma_edit_tool_call():
    old_string = """#desktop {
    width: 100%;
    height: calc(100% - var(--taskbar-height));
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    background-size: cover;
    position: relative;
    overflow: hidden;
}
"""
    new_string = """#desktop {
    width: 100%;
    height: calc(100% - var(--taskbar-height));
    background: #121212;
    background-size: cover;
    position: relative;
    overflow: hidden;
}
"""
    leaked_call = (
        '<|tool_call>call:Edit{file_path:<|"|>css/style.css<|"|>,'
        f'new_string:<|"|>{new_string}<|"|>,'
        f'old_string:<|"|>{old_string}<|"|>,'
        "replace_all:false}<tool_call|>"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                _sse_payload(
                    {
                        "model": "gemma-4-26B-A4B-it",
                        "choices": [
                            {
                                "delta": {"content": leaked_call[:45]},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                + _sse_payload(
                    {
                        "model": "gemma-4-26B-A4B-it",
                        "choices": [
                            {
                                "delta": {"content": leaked_call[45:]},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                + _sse_payload(
                    {
                        "model": "gemma-4-26B-A4B-it",
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                    }
                )
                + "data: [DONE]\n\n"
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
                "messages": [{"role": "user", "content": "edit the css"}],
                "tools": [_edit_tool_schema()],
            },
        )

    assert response.status_code == 200
    events = _anthropic_sse_events(response.text)
    text = "".join(
        event["delta"]["text"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "text_delta"
    )
    tool_use_blocks = [
        event["content_block"]
        for event in events
        if event.get("type") == "content_block_start"
        and event.get("content_block", {}).get("type") == "tool_use"
    ]
    tool_json = "".join(
        event["delta"]["partial_json"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "input_json_delta"
    )
    stop_reasons = [
        event["delta"]["stop_reason"]
        for event in events
        if event.get("type") == "message_delta"
    ]

    assert text == ""
    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0]["name"] == "Edit"
    assert json.loads(tool_json) == {
        "file_path": "css/style.css",
        "new_string": new_string,
        "old_string": old_string,
        "replace_all": False,
    }
    assert stop_reasons == ["tool_use"]


@pytest.mark.asyncio
async def test_anthropic_messages_stream_drops_unknown_gemma_tool_call():
    leaked_call = '<|tool_call>call:Unknown{path:<|"|>x<|"|>}<tool_call|>'

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                _sse_payload(
                    {
                        "model": "gemma-4-26B-A4B-it",
                        "choices": [
                            {
                                "delta": {"content": leaked_call},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                + _sse_payload(
                    {
                        "model": "gemma-4-26B-A4B-it",
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                    }
                )
                + "data: [DONE]\n\n"
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
                "messages": [{"role": "user", "content": "edit the css"}],
                "tools": [_edit_tool_schema()],
            },
        )

    assert response.status_code == 200
    events = _anthropic_sse_events(response.text)
    text = "".join(
        event["delta"]["text"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "text_delta"
    )
    tool_use_blocks = [
        event
        for event in events
        if event.get("type") == "content_block_start"
        and event.get("content_block", {}).get("type") == "tool_use"
    ]

    assert text == ""
    assert "<|tool_call>" not in response.text
    assert "<tool_call|>" not in response.text
    assert tool_use_blocks == []


@pytest.mark.asyncio
async def test_anthropic_messages_stream_translates_backend_tool_deltas():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "Read",
                                                "arguments": '{"file',
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "arguments": '_path":"/tmp/a.css"}'
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {
                        "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                        "usage": {
                            "prompt_tokens": 4,
                            "completion_tokens": 2,
                            "total_tokens": 6,
                        },
                    }
                )
                + "data: [DONE]\n\n"
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
                "messages": [{"role": "user", "content": "read the css"}],
                "tools": [_read_tool_schema()],
            },
        )

    assert response.status_code == 200
    events = _anthropic_sse_events(response.text)
    tool_use_blocks = [
        event["content_block"]
        for event in events
        if event.get("type") == "content_block_start"
        and event.get("content_block", {}).get("type") == "tool_use"
    ]
    tool_json = "".join(
        event["delta"]["partial_json"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "input_json_delta"
    )

    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0]["id"] == "call_1"
    assert tool_use_blocks[0]["name"] == "Read"
    assert json.loads(tool_json) == {"file_path": "/tmp/a.css"}


@pytest.mark.asyncio
async def test_anthropic_messages_stream_preserves_literal_bracket_text():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {"content": "Heads up: [Calling tool:"},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {"content": " maybe later]"},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + _sse_payload(
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": 4,
                            "completion_tokens": 2,
                            "total_tokens": 6,
                        },
                    }
                )
                + "data: [DONE]\n\n"
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
                "messages": [{"role": "user", "content": "heads up"}],
                "tools": [_read_tool_schema()],
            },
        )

    assert response.status_code == 200
    events = _anthropic_sse_events(response.text)
    text = "".join(
        event["delta"]["text"]
        for event in events
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "text_delta"
    )
    tool_use_blocks = [
        event
        for event in events
        if event.get("type") == "content_block_start"
        and event.get("content_block", {}).get("type") == "tool_use"
    ]

    assert text == "Heads up: [Calling tool: maybe later]"
    assert tool_use_blocks == []


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
async def test_openai_chat_completions_passthrough_aliases_vllm_reasoning():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "done",
                            "reasoning": "plan",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            },
        )

    app = _app_with_mock_backend(handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "gemma", "messages": [], "stream": False},
        )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["reasoning"] == "plan"
    assert message["reasoning_content"] == "plan"


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
async def test_openai_chat_completions_stream_aliases_vllm_reasoning():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content.decode())
        assert body["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                _sse_payload(
                    {
                        "choices": [
                            {
                                "delta": {"reasoning": "plan"},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + "data: [DONE]\n\n"
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
                "model": "gemma",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert '"reasoning":"plan"' in response.text
    assert '"reasoning_content":"plan"' in response.text
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


async def _ok_models_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"object": "list", "data": []})


@pytest.mark.asyncio
async def test_proxy_logs_default_returns_in_memory_buffer(monkeypatch, tmp_path):
    # No Docker socket: only the in-memory proxy buffer is advertised.
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("OMLX_DOCKER_SOCK", str(tmp_path / "missing.sock"))
    app = _app_with_mock_backend(_ok_models_handler)
    app.state.proxy_admin_state.log("hello world")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/admin/api/logs")

    assert response.status_code == 200
    data = response.json()
    assert "hello world" in data["logs"]
    assert data["available_files"] == ["server.log"]


@pytest.mark.asyncio
async def test_proxy_logs_advertises_and_serves_container_sources(
    monkeypatch, tmp_path
):
    from omlx.proxy import admin as proxy_admin

    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    sock = tmp_path / "docker.sock"
    sock.touch()  # make docker_socket_path() report present
    monkeypatch.setenv("OMLX_DOCKER_SOCK", str(sock))

    async def fake_container_logs(service, *, tail=200, timestamps=True, client=None):
        return f"=== {service} tail={tail} ===\nstarting up\n"

    monkeypatch.setattr(proxy_admin, "container_logs", fake_container_logs)

    app = _app_with_mock_backend(_ok_models_handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        available = (await client.get("/admin/api/logs")).json()["available_files"]
        response = await client.get(
            "/admin/api/logs",
            params={"file": "omlx-proxy (container)", "lines": 25},
        )

    assert available == [
        "server.log",
        "omlx-proxy (container)",
        "vllm (container)",
    ]
    data = response.json()
    assert data["file"] == "omlx-proxy (container)"
    assert "=== omlx-proxy tail=25 ===" in data["logs"]


@pytest.mark.asyncio
async def test_proxy_logs_container_source_without_socket_returns_501(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("OMLX_DOCKER_SOCK", str(tmp_path / "missing.sock"))
    app = _app_with_mock_backend(_ok_models_handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/admin/api/logs", params={"file": "vllm (container)"}
        )

    assert response.status_code == 501
    assert "docker.sock" in response.json()["detail"]


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
    # The tool-call / reasoning parser are per-model now: switching models resets
    # them to "auto" and re-detects from the new model's family. "admin-model" is
    # not a recognized family, so detection leaves tool calling off.
    assert env["VLLM_TOOL_CALL_PARSER"] == ""
    assert env["VLLM_REASONING_PARSER"] == ""

    content = compose_path.read_text(encoding="utf-8")
    assert 'OMLX_ENV_OUTPUT_PATH: "/compose-output/docker-compose.vllm.env"' in content
    assert "example/admin-model" in content
    assert "OMLX_SAMPLING_TOP_P" in content
    assert "0.77" in content


@pytest.mark.asyncio
async def test_admin_model_switch_resets_model_specific_vllm_overrides(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    compose_path = tmp_path / "docker-compose.vllm.yml"
    env_path = tmp_path / "docker-compose.vllm.env"
    monkeypatch.setenv("OMLX_COMPOSE_OUTPUT_PATH", str(compose_path))
    monkeypatch.setenv("OMLX_ENV_OUTPUT_PATH", str(env_path))

    # Previous model left model-specific knobs tuned for it: a quantization /
    # dtype / chunked-prefill setting that can break or mislabel the next model
    # (the gemma-4 VLM, e.g., does not support disabling chunked prefill).
    write_vllm_env_file(
        env_path,
        vllm_environment(
            VllmComposeSettings(
                model="example/previous-model",
                served_model_name="previous-name",
                dtype="float16",
                quantization="awq",
                kv_cache_dtype="fp8",
                enable_chunked_prefill=False,
                enforce_eager=True,  # GB10-required; must be preserved
                tool_call_parser="existing-parser",
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
                "omni_model": "example/next-model",
                # a knob the user explicitly sets in THIS save must win over the
                # reset-to-default.
                "vllm_kv_cache_dtype": "fp8_e5m2",
            },
        )

    assert response.status_code == 200
    env = load_vllm_env_file(env_path)
    assert env["OMNI_MODEL"] == "example/next-model"
    # Model-specific knobs are cleared back to vLLM's auto defaults.
    assert env["VLLM_DTYPE"] == ""
    assert env["VLLM_QUANTIZATION"] == ""
    assert env["VLLM_ENABLE_CHUNKED_PREFILL"] == ""
    # Cross-model / hardware prefs are preserved.
    assert env["VLLM_ENFORCE_EAGER"] == "true"
    # The tool-call parser is per-model: it resets to "auto" and re-detects from
    # the new model's family (unrecognized here, so tool calling stays off).
    assert env["VLLM_TOOL_CALL_PARSER"] == ""
    # An explicit override in the same save wins over the reset.
    assert env["VLLM_KV_CACHE_DTYPE"] == "fp8_e5m2"


@pytest.mark.asyncio
async def test_admin_model_switch_auto_detects_tool_parser(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    compose_path = tmp_path / "docker-compose.vllm.yml"
    env_path = tmp_path / "docker-compose.vllm.env"
    monkeypatch.setenv("OMLX_COMPOSE_OUTPUT_PATH", str(compose_path))
    monkeypatch.setenv("OMLX_ENV_OUTPUT_PATH", str(env_path))

    # Previous model auto-resolved to the Hermes parser.
    write_vllm_env_file(
        env_path,
        vllm_environment(VllmComposeSettings(model="Qwen/Qwen3-8B")),
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
                "omni_model": "google/gemma-4-26B-A4B-it",
            },
        )

    assert response.status_code == 200
    env = load_vllm_env_file(env_path)
    # Switching to Gemma 4 re-detects its native vLLM parser, reasoning parser,
    # and tool chat template — tool calling works out of the box.
    assert env["VLLM_ENABLE_AUTO_TOOL_CHOICE"] == "true"
    assert env["VLLM_TOOL_CALL_PARSER"] == "gemma4"
    assert env["VLLM_REASONING_PARSER"] == "gemma4"
    assert env["VLLM_CHAT_TEMPLATE"].endswith("tool_chat_template_gemma4.jinja")


@pytest.mark.asyncio
async def test_use_with_sidecar_reapplies_optimal_on_same_model(monkeypatch, tmp_path):
    # After a failed load the proxy already considers the model "current", so
    # re-clicking "Use with sidecar" sends the same model. The button asks for
    # optimal defaults (reset_optimal), so stale per-model knobs must still be
    # cleared even though the model id is unchanged.
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    compose_path = tmp_path / "docker-compose.vllm.yml"
    env_path = tmp_path / "docker-compose.vllm.env"
    monkeypatch.setenv("OMLX_COMPOSE_OUTPUT_PATH", str(compose_path))
    monkeypatch.setenv("OMLX_ENV_OUTPUT_PATH", str(env_path))

    write_vllm_env_file(
        env_path,
        vllm_environment(
            VllmComposeSettings(
                model="example/stuck-model",
                served_model_name="stuck",
                dtype="float16",
                enable_chunked_prefill=False,
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
        # Plain save of the same model leaves per-model knobs untouched.
        await client.post(
            "/admin/api/global-settings",
            json={
                "proxy_backend_url": "http://backend/v1",
                "proxy_backend_type": "vllm",
                "omni_model": "example/stuck-model",
            },
        )
        assert load_vllm_env_file(env_path)["VLLM_DTYPE"] == "float16"

        # The button re-applies optimal defaults even for the same model.
        response = await client.post(
            "/admin/api/global-settings",
            json={
                "proxy_backend_url": "http://backend/v1",
                "proxy_backend_type": "vllm",
                "omni_model": "example/stuck-model",
                "reset_optimal": True,
            },
        )

    assert response.status_code == 200
    env = load_vllm_env_file(env_path)
    assert env["VLLM_DTYPE"] == ""
    assert env["VLLM_ENABLE_CHUNKED_PREFILL"] == ""


@pytest.mark.asyncio
async def test_use_with_sidecar_enables_chunked_prefill_for_computed_long_context(
    monkeypatch, tmp_path
):
    import omlx.proxy.admin as admin

    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("OMLX_MODEL_SCAN_DIR", str(tmp_path))
    compose_path = tmp_path / "docker-compose.vllm.yml"
    env_path = tmp_path / "docker-compose.vllm.env"
    monkeypatch.setenv("OMLX_COMPOSE_OUTPUT_PATH", str(compose_path))
    monkeypatch.setenv("OMLX_ENV_OUTPUT_PATH", str(env_path))
    monkeypatch.setattr(
        admin,
        "host_memory_info",
        lambda *a, **k: {"total_bytes": 200 * 1024**3},
    )
    _make_cached_model_hub(
        tmp_path,
        "org/long",
        1 * 1024**3,
        native=262144,
        layers=4,
        kv_heads=2,
        head_dim=64,
    )

    write_vllm_env_file(
        env_path,
        vllm_environment(
            VllmComposeSettings(
                model="org/long",
                served_model_name="long",
                enable_chunked_prefill=False,
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
                "omni_model": "org/long",
                "reset_optimal": True,
            },
        )

    assert response.status_code == 200
    env = load_vllm_env_file(env_path)
    assert env["OMNI_CONTEXT_LENGTH"] == "262144"
    assert env["OMNI_MAX_PARALLEL"] == "2"
    assert env["VLLM_ENABLE_CHUNKED_PREFILL"] == "true"


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


@pytest.mark.asyncio
async def test_local_models_endpoint_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.delenv("OMLX_MODEL_SCAN", raising=False)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        data = (await client.get("/admin/api/proxy/local-models")).json()

    assert data["enabled"] is False
    assert data["models"] == []


@pytest.mark.asyncio
async def test_local_models_endpoint_scans_when_enabled(monkeypatch, tmp_path):
    import json as jsonlib

    scan_root = tmp_path / "scan"
    model = scan_root / "tiny-model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        jsonlib.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    )
    (model / "model.safetensors").write_bytes(b"\0" * 128)

    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("OMLX_MODEL_SCAN", "true")
    monkeypatch.setenv("OMLX_MODEL_SCAN_DIR", str(scan_root))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        data = (await client.get("/admin/api/proxy/local-models")).json()
        # Cache is reused until refresh=1 rescans.
        (model / "config2.json").write_text("{}")
        second = scan_root / "second-model"
        second.mkdir()
        (second / "config.json").write_text(
            jsonlib.dumps(
                {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
            )
        )
        (second / "model.safetensors").write_bytes(b"\0" * 64)
        cached = (await client.get("/admin/api/proxy/local-models")).json()
        refreshed = (await client.get("/admin/api/proxy/local-models?refresh=1")).json()

    assert data["enabled"] is True
    assert [m["repo_id"] for m in data["models"]] == ["tiny-model"]
    assert data["models"][0]["backends"] == ["vllm"]
    assert len(cached["models"]) == 1
    assert len(refreshed["models"]) == 2


def _make_cached_model_hub(
    root,
    repo_id,
    weight_bytes,
    *,
    native=4096,
    layers=None,
    kv_heads=None,
    head_dim=None,
):
    encoded = "models--" + repo_id.replace("/", "--")
    commit = "feedface"
    snapshot = root / "hub" / encoded / "snapshots" / commit
    snapshot.mkdir(parents=True)
    config = {"max_position_embeddings": native}
    if layers and kv_heads and head_dim:
        config.update(
            {
                "num_hidden_layers": layers,
                "num_attention_heads": kv_heads,
                "num_key_value_heads": kv_heads,
                "head_dim": head_dim,
                "torch_dtype": "bfloat16",
            }
        )
    (snapshot / "config.json").write_text(json.dumps(config))
    with open(snapshot / "model.safetensors", "wb") as handle:
        handle.truncate(weight_bytes)
    refs = root / "hub" / encoded / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(commit)
    return snapshot


@pytest.mark.asyncio
async def test_sidecar_restart_blocks_oversized_model(monkeypatch, tmp_path):
    import omlx.proxy.admin as admin

    state_path = tmp_path / "state.json"
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(state_path))
    monkeypatch.setenv("OMLX_MODEL_SCAN_DIR", str(tmp_path))
    monkeypatch.setattr(
        admin, "host_memory_info", lambda *a, **k: {"total_bytes": 122 * 1024**3}
    )
    _make_cached_model_hub(tmp_path, "org/huge", 80 * 1024**3)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": [{"id": "huge"}]})

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        # Saving a vLLM sidecar pointed at the oversized model is refused.
        bad = {
            "proxy_backend_type": "vllm",
            "omni_model": "org/huge",
            "omni_served_model_name": "huge",
        }
        blocked = await client.post("/admin/api/global-settings", json=bad)
        assert blocked.status_code == 409
        body = blocked.json()
        assert body["blocked"] is True
        assert body["memory"]["weights_bytes"] > 0

        # The override persists it (so the dangerous config exists on disk).
        forced = await client.post(
            "/admin/api/global-settings", json={**bad, "force_memory": True}
        )
        assert forced.status_code == 200

        # Restarting that persisted config is refused again...
        restart_blocked = await client.post("/admin/api/sidecar/restart", json={})
        assert restart_blocked.status_code == 409
        assert restart_blocked.json()["blocked"] is True

        # ...and the override gets past the guard (then fails only on Docker).
        restart_forced = await client.post(
            "/admin/api/sidecar/restart", json={"force_memory": True}
        )
        assert restart_forced.status_code != 409


@pytest.mark.asyncio
async def test_admin_resyncs_served_name_on_model_change(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(state_path))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": []})

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        # Seed a vLLM sidecar with an auto-derived served name.
        seed = await client.post(
            "/admin/api/global-settings",
            json={
                "proxy_backend_type": "vllm",
                "omni_model": "org/old-model",
                "omni_served_model_name": "old-model",
            },
        )
        assert seed.status_code == 200

        # Changing the model while the form still carries the stale served
        # name re-derives it from the new model.
        changed = await client.post(
            "/admin/api/global-settings",
            json={
                "proxy_backend_type": "vllm",
                "omni_model": "org/new-model",
                "omni_served_model_name": "old-model",
            },
        )
        assert changed.status_code == 200
        settings = (await client.get("/admin/api/global-settings")).json()
        assert settings["proxy"]["sidecar"]["served_model_name"] == "new-model"

        # A custom served name is preserved across a model change.
        custom = await client.post(
            "/admin/api/global-settings",
            json={
                "proxy_backend_type": "vllm",
                "omni_model": "org/third-model",
                "omni_served_model_name": "my-custom",
            },
        )
        assert custom.status_code == 200
        settings = (await client.get("/admin/api/global-settings")).json()
        assert settings["proxy"]["sidecar"]["served_model_name"] == "my-custom"


@pytest.mark.asyncio
async def test_admin_sets_offline_for_cached_model_switch(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(state_path))
    monkeypatch.setenv("OMLX_MODEL_SCAN_DIR", str(tmp_path))
    _make_cached_model_hub(tmp_path, "org/cached", 4 * 1024**3)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": []})

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        # Switch to a cached model -> offline enabled.
        cached = await client.post(
            "/admin/api/global-settings",
            json={"proxy_backend_type": "vllm", "omni_model": "org/cached"},
        )
        assert cached.status_code == 200
        settings = (await client.get("/admin/api/global-settings")).json()
        assert settings["proxy"]["sidecar"]["hf_offline"] is True

        # Switch to an uncached model -> back online so it can download.
        uncached = await client.post(
            "/admin/api/global-settings",
            json={"proxy_backend_type": "vllm", "omni_model": "org/not-cached"},
        )
        assert uncached.status_code == 200
        settings = (await client.get("/admin/api/global-settings")).json()
        assert settings["proxy"]["sidecar"]["hf_offline"] is False


def _make_cached_vllm_model(root, repo_id, weight_bytes):
    """HF-cache entry under root/hub with a sparse shard + KV-geometry config."""
    encoded = "models--" + repo_id.replace("/", "--")
    commit = "cafef00d"
    snapshot = root / "hub" / encoded / "snapshots" / commit
    snapshot.mkdir(parents=True)
    snapshot.joinpath("config.json").write_text(
        json.dumps(
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
    with open(snapshot / "model.safetensors", "wb") as handle:
        handle.truncate(weight_bytes)
    refs = root / "hub" / encoded / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(commit)


@pytest.mark.asyncio
async def test_admin_resizes_util_on_model_switch(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("OMLX_PROXY_STATE_PATH", str(state_path))
    monkeypatch.setenv("OMLX_MODEL_SCAN_DIR", str(tmp_path))
    monkeypatch.setattr(
        "omlx.proxy.admin.host_memory_info",
        lambda *a, **k: {"total_bytes": 122 * 1024**3},
    )
    _make_cached_vllm_model(tmp_path, "org/small", 4 * 1024**3)
    _make_cached_vllm_model(tmp_path, "org/other", 4 * 1024**3)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": []})

    app = _app_with_mock_backend(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        # Switch to a small cached model without touching util -> demand-sized.
        r = await client.post(
            "/admin/api/global-settings",
            json={
                "proxy_backend_type": "vllm",
                "omni_model": "org/small",
                "omni_context_length": 40960,
                "omni_max_parallel": 2,
            },
        )
        assert r.status_code == 200
        settings = (await client.get("/admin/api/global-settings")).json()
        auto_util = settings["proxy"]["sidecar"]["gpu_memory_utilization"]
        assert auto_util < 0.40  # not the ~0.83 safety ceiling

        # Switch model while explicitly changing util -> the explicit value wins.
        r = await client.post(
            "/admin/api/global-settings",
            json={
                "proxy_backend_type": "vllm",
                "omni_model": "org/other",
                "omni_context_length": 40960,
                "omni_max_parallel": 2,
                "vllm_gpu_memory_utilization": 0.7,
            },
        )
        assert r.status_code == 200
        settings = (await client.get("/admin/api/global-settings")).json()
        assert settings["proxy"]["sidecar"]["gpu_memory_utilization"] == 0.7
