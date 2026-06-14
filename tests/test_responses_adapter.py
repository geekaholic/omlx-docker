# SPDX-License-Identifier: Apache-2.0
"""Tests for the oMNI proxy Responses API <-> Chat Completions adapter."""

import json

import pytest

from omlx.proxy.responses_adapter import (
    responses_to_chat_body,
    stream_responses_events,
)


class _FakeBackend:
    def __init__(self, chunks):
        self._chunks = chunks

    async def stream_chat_completion(self, cc_body, authorization):
        for chunk in self._chunks:
            yield chunk
        yield "[DONE]"


def _cc_chunk(content=None, finish=None):
    delta = {} if content is None else {"content": content}
    choice = {"delta": delta}
    if finish:
        choice["finish_reason"] = finish
    return {"model": "m", "choices": [choice]}


def _output_text(events):
    """Concatenate the deltas of all response.output_text.delta SSE events."""
    out = ""
    for ev in events:
        for line in ev.splitlines():
            if line.startswith("data:"):
                try:
                    data = json.loads(line[len("data:") :].strip())
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "response.output_text.delta":
                    out += data.get("delta", "")
    return out


@pytest.mark.asyncio
async def test_stream_strips_reasoning_markers_split_across_chunks():
    chunks = [
        _cc_chunk(content="Plan ready.\n\n"),
        _cc_chunk(content="<|channel>thou"),  # marker split across deltas
        _cc_chunk(content="ght\nreasoning<channel|>"),
        _cc_chunk(content=" done"),
        _cc_chunk(finish="stop"),
    ]
    events = []
    async for ev in stream_responses_events(
        _FakeBackend(chunks), {"stream": True}, None, "resp_1", "m", 0
    ):
        events.append(ev)
    visible = _output_text(events)
    assert "<|channel>" not in visible
    assert "<channel|>" not in visible
    assert "reasoning" not in visible  # withheld channel content not leaked
    assert "Plan ready." in visible and "done" in visible


def test_function_call_and_output_round_trip():
    # Codex re-sends the full conversation each turn, including the prior tool
    # call and its result as typed items. Both must reach the backend or the
    # model never sees the result and re-requests the same command in a loop.
    payload = {
        "model": "m",
        "instructions": "be helpful",
        "input": [
            {"type": "message", "role": "user", "content": "list files"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "shell",
                "arguments": '{"cmd":"ls -la"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "a.txt\nb.txt",
            },
        ],
    }
    msgs = responses_to_chat_body(payload)["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]

    assistant = msgs[2]
    tc = assistant["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "shell"
    # arguments stay a JSON *string* (OpenAI chat-completions wire format)
    assert tc["function"]["arguments"] == '{"cmd":"ls -la"}'

    tool = msgs[3]
    assert tool["tool_call_id"] == "call_1"
    assert tool["content"] == "a.txt\nb.txt"


def test_unknown_item_types_are_skipped():
    payload = {
        "input": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "x"}]},
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "item_reference", "id": "x"},
        ]
    }
    msgs = responses_to_chat_body(payload)["messages"]
    # reasoning + item_reference dropped, no stray empty user messages
    assert msgs == [{"role": "user", "content": "hi"}]


def test_assistant_text_then_tool_call_merge_into_one_turn():
    payload = {
        "input": [
            {"type": "message", "role": "assistant", "content": "I'll list files"},
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "shell",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "c1", "output": "done"},
        ]
    }
    msgs = responses_to_chat_body(payload)["messages"]
    assert [m["role"] for m in msgs] == ["assistant", "tool"]
    assert msgs[0]["content"] == "I'll list files"
    assert msgs[0]["tool_calls"][0]["id"] == "c1"


def test_tool_call_without_preceding_message_gets_its_own_assistant_turn():
    payload = {
        "input": [
            {"type": "message", "role": "user", "content": "go"},
            {
                "type": "function_call",
                "call_id": "c9",
                "name": "shell",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "c9", "output": "ok"},
        ]
    }
    msgs = responses_to_chat_body(payload)["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[1]["tool_calls"][0]["id"] == "c9"


def test_dict_arguments_are_serialized_to_string():
    payload = {
        "input": [
            {
                "type": "function_call",
                "call_id": "c",
                "name": "f",
                "arguments": {"a": 1},
            },
            {"type": "function_call_output", "call_id": "c", "output": "r"},
        ]
    }
    msgs = responses_to_chat_body(payload)["messages"]
    args = msgs[0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str) and args == '{"a": 1}'


def test_plain_text_conversation_unchanged():
    payload = {"input": [{"type": "message", "role": "user", "content": "hello"}]}
    assert responses_to_chat_body(payload)["messages"] == [
        {"role": "user", "content": "hello"}
    ]


def test_image_content_still_flattens():
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,x"},
                    },
                ],
            }
        ]
    }
    parts = responses_to_chat_body(payload)["messages"][0]["content"]
    assert isinstance(parts, list)
    assert {"type": "text", "text": "what is this"} in parts
    assert any(p.get("type") == "image_url" for p in parts)
