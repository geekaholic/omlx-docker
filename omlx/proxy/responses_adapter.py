# SPDX-License-Identifier: Apache-2.0
"""Translate OpenAI Responses API (/v1/responses) ↔ Chat Completions API.

Codex CLI uses the Responses API exclusively. This adapter lets the oMNI proxy
accept Responses API requests and forward them as Chat Completions to any
OpenAI-compatible backend (vLLM, Ollama, llama.cpp, etc.).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from omlx.server_metrics import ServerMetrics

from .backend import OpenAIBackend
from .protocol_markers import MarkerStripper, strip_markers
from .stats import record_request


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _flatten_content(content: Any) -> Any:
    """Flatten Responses content parts to the Chat Completions shape.

    Returns a plain string when the content is a single text part, otherwise a
    list of `{"type": "text"|"image_url", ...}` parts (so VLMs keep images).
    """
    if not isinstance(content, list):
        return content
    cc_parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")
        if ptype in ("text", "input_text", "output_text"):
            cc_parts.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "image_url":
            cc_parts.append(part)
        elif ptype == "refusal":
            cc_parts.append({"type": "text", "text": part.get("refusal", "")})
    if len(cc_parts) == 1 and cc_parts[0].get("type") == "text":
        return cc_parts[0]["text"]
    return cc_parts or ""


def _stringify_output(output: Any) -> str:
    """Coerce a function_call_output payload to a string tool-message content."""
    if isinstance(output, str):
        return output
    if output is None:
        return ""
    if isinstance(output, list):
        parts: list[str] = []
        for part in output:
            if isinstance(part, dict):
                parts.append(part.get("text") or part.get("output") or "")
            else:
                parts.append(str(part))
        return "".join(parts)
    if isinstance(output, dict):
        return output.get("text") or output.get("output") or json.dumps(output)
    return str(output)


def _input_to_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    instructions = payload.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    input_val = payload.get("input", [])
    if isinstance(input_val, str):
        messages.append({"role": "user", "content": input_val})
        return messages
    if not isinstance(input_val, list):
        return messages

    # Accumulate the model's tool calls so they can be grouped into one
    # assistant message (and merged onto a preceding assistant text turn),
    # mirroring omlx.api.responses_utils.convert_responses_input_to_messages.
    pending_tool_calls: list[dict[str, Any]] = []

    def flush_tool_calls() -> None:
        if not pending_tool_calls:
            return
        if (
            messages
            and messages[-1].get("role") == "assistant"
            and "tool_calls" not in messages[-1]
        ):
            messages[-1]["tool_calls"] = list(pending_tool_calls)
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": list(pending_tool_calls),
                }
            )
        pending_tool_calls.clear()

    for item in input_val:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type is None and item.get("role") is not None:
            item_type = "message"

        if item_type == "function_call":
            # Assistant tool call. Keep arguments a JSON *string* per the
            # OpenAI chat-completions wire format; reuse the Responses call_id
            # so the matching tool result correlates.
            arguments = item.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            messages_call_id = item.get("call_id") or item.get("id") or _new_id("call")
            pending_tool_calls.append(
                {
                    "id": messages_call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": arguments,
                    },
                }
            )
        elif item_type == "function_call_output":
            # The tool result. Flush the assistant tool-call turn first so the
            # tool message has a preceding tool_calls message to correlate to.
            flush_tool_calls()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": _stringify_output(item.get("output", "")),
                }
            )
        elif item_type == "message":
            flush_tool_calls()
            role = item.get("role", "user")
            messages.append(
                {"role": role, "content": _flatten_content(item.get("content", ""))}
            )
        # Unknown item types (reasoning, item_reference, ...) are skipped so they
        # don't become empty user messages that pollute the model's context.

    flush_tool_calls()
    return messages


def _tools_to_cc(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Responses API tool definitions to Chat Completions format.

    Only "function" type tools are translated; built-in Responses API tools
    (computer_use, code_interpreter, etc.) are executed client-side by Codex
    and are not forwarded to the backend.
    """
    cc_tools: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn: dict[str, Any] = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        }
        if "strict" in tool:
            fn["strict"] = tool["strict"]
        cc_tools.append({"type": "function", "function": fn})
    return cc_tools


def responses_to_chat_body(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate a Responses API request body to Chat Completions format."""
    body: dict[str, Any] = {
        "model": payload.get("model", ""),
        "messages": _input_to_messages(payload),
        "stream": bool(payload.get("stream", False)),
    }
    if payload.get("max_output_tokens") is not None:
        body["max_tokens"] = payload["max_output_tokens"]
    for field in (
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
    ):
        if payload.get(field) is not None:
            body[field] = payload[field]
    tools = payload.get("tools")
    if tools:
        cc_tools = _tools_to_cc(tools)
        if cc_tools:
            body["tools"] = cc_tools
            tool_choice = payload.get("tool_choice")
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
    if body["stream"]:
        body["stream_options"] = {"include_usage": True}
    return body


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


async def stream_responses_events(
    backend: OpenAIBackend,
    cc_body: dict[str, Any],
    authorization: str | None,
    resp_id: str,
    model: str,
    created_at: int,
    metrics: ServerMetrics | None = None,
) -> AsyncIterator[str]:
    """Translate a Chat Completions SSE stream to Responses API SSE events."""
    text_item_id = _new_id("item")
    # Sequential output index counter
    next_out_idx = 0
    text_out_idx: int | None = None
    content_index = 0
    accumulated_text = ""
    # tc_idx → {item_id, output_index, call_id, name, arguments}
    tc_state: dict[int, dict[str, Any]] = {}
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    text_item_opened = False
    content_part_opened = False
    request_start = time.monotonic()
    first_chunk_at: float | None = None
    seen_model = ""
    # Scrub stray Gemma-4 reasoning-channel markers that vLLM's parser leaves in
    # content (empty/mid-stream channels); buffers across deltas.
    stripper = MarkerStripper()

    def _base_resp(status: str, output: list[dict]) -> dict[str, Any]:
        return {
            "id": resp_id,
            "object": "response",
            "created_at": created_at,
            "status": status,
            "model": model,
            "output": output,
        }

    async def _emit_text(chunk: str) -> AsyncIterator[str]:
        nonlocal text_item_opened, content_part_opened, accumulated_text
        nonlocal text_out_idx, next_out_idx
        if not chunk:
            return
        if not text_item_opened:
            text_out_idx = next_out_idx
            next_out_idx += 1
            yield _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": text_out_idx,
                    "item": {
                        "id": text_item_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                },
            )
            text_item_opened = True
        if not content_part_opened:
            yield _sse(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "item_id": text_item_id,
                    "output_index": text_out_idx,
                    "content_index": content_index,
                    "part": {"type": "output_text", "text": ""},
                },
            )
            content_part_opened = True
        accumulated_text += chunk
        yield _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": text_item_id,
                "output_index": text_out_idx,
                "content_index": content_index,
                "delta": chunk,
            },
        )

    yield _sse(
        "response.created",
        {"type": "response.created", "response": _base_resp("in_progress", [])},
    )

    try:
        async for item in backend.stream_chat_completion(cc_body, authorization):
            if item == "[DONE]":
                break
            if not isinstance(item, dict):
                continue

            if first_chunk_at is None and item.get("choices"):
                first_chunk_at = time.monotonic()
            if item.get("model"):
                seen_model = str(item["model"])

            usage = item.get("usage")
            if usage:
                input_tokens = usage.get("prompt_tokens", input_tokens)
                output_tokens = usage.get("completion_tokens", output_tokens)
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = details.get("cached_tokens") or cached_tokens

            for choice in item.get("choices") or []:
                delta = choice.get("delta") or {}
                finish_reason = choice.get("finish_reason")

                # ── Text content ──────────────────────────────────────────────
                text = stripper.feed(delta.get("content") or "")
                if text:
                    async for ev in _emit_text(text):
                        yield ev

                # ── Tool calls ────────────────────────────────────────────────
                for tc_delta in delta.get("tool_calls") or []:
                    tc_idx = tc_delta.get("index", 0)
                    if tc_idx not in tc_state:
                        tc_item_id = _new_id("item")
                        tc_out = next_out_idx
                        next_out_idx += 1
                        tc_state[tc_idx] = {
                            "item_id": tc_item_id,
                            "output_index": tc_out,
                            "call_id": tc_delta.get("id", ""),
                            "name": (tc_delta.get("function") or {}).get("name", ""),
                            "arguments": "",
                        }
                        yield _sse(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": tc_out,
                                "item": {
                                    "id": tc_item_id,
                                    "type": "function_call",
                                    "status": "in_progress",
                                    "call_id": tc_delta.get("id", ""),
                                    "name": (tc_delta.get("function") or {}).get(
                                        "name", ""
                                    ),
                                    "arguments": "",
                                },
                            },
                        )
                    tc = tc_state[tc_idx]
                    fn = tc_delta.get("function") or {}
                    if fn.get("name"):
                        tc["name"] = fn["name"]
                    if tc_delta.get("id"):
                        tc["call_id"] = tc_delta["id"]
                    args_delta = fn.get("arguments", "")
                    if args_delta:
                        tc["arguments"] += args_delta
                        yield _sse(
                            "response.function_call_arguments.delta",
                            {
                                "type": "response.function_call_arguments.delta",
                                "item_id": tc["item_id"],
                                "output_index": tc["output_index"],
                                "delta": args_delta,
                            },
                        )

                # ── Finish ────────────────────────────────────────────────────
                if finish_reason:
                    tail = stripper.flush()
                    if tail:
                        async for ev in _emit_text(tail):
                            yield ev
                    if content_part_opened:
                        assert text_out_idx is not None
                        yield _sse(
                            "response.output_text.done",
                            {
                                "type": "response.output_text.done",
                                "item_id": text_item_id,
                                "output_index": text_out_idx,
                                "content_index": content_index,
                                "text": accumulated_text,
                            },
                        )
                        yield _sse(
                            "response.content_part.done",
                            {
                                "type": "response.content_part.done",
                                "item_id": text_item_id,
                                "output_index": text_out_idx,
                                "content_index": content_index,
                                "part": {
                                    "type": "output_text",
                                    "text": accumulated_text,
                                },
                            },
                        )
                        yield _sse(
                            "response.output_item.done",
                            {
                                "type": "response.output_item.done",
                                "output_index": text_out_idx,
                                "item": {
                                    "id": text_item_id,
                                    "type": "message",
                                    "status": "completed",
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": accumulated_text,
                                        }
                                    ],
                                },
                            },
                        )
                    for tc in tc_state.values():
                        yield _sse(
                            "response.function_call_arguments.done",
                            {
                                "type": "response.function_call_arguments.done",
                                "item_id": tc["item_id"],
                                "output_index": tc["output_index"],
                                "arguments": tc["arguments"],
                            },
                        )
                        yield _sse(
                            "response.output_item.done",
                            {
                                "type": "response.output_item.done",
                                "output_index": tc["output_index"],
                                "item": {
                                    "id": tc["item_id"],
                                    "type": "function_call",
                                    "status": "completed",
                                    "call_id": tc["call_id"],
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            },
                        )
    except Exception as exc:
        yield _sse(
            "error",
            {"type": "error", "error": {"message": str(exc), "code": "server_error"}},
        )
        return
    finally:
        end = time.monotonic()
        prefill_duration = 0.0
        generation_duration = 0.0
        if first_chunk_at is not None:
            prefill_duration = max(0.0, first_chunk_at - request_start)
            generation_duration = max(0.0, end - first_chunk_at)
        record_request(
            metrics,
            model_id=seen_model or model,
            prompt_tokens=int(input_tokens or 0),
            completion_tokens=int(output_tokens or 0),
            cached_tokens=int(cached_tokens or 0),
            prefill_duration=prefill_duration,
            generation_duration=generation_duration,
        )

    # ── response.completed ────────────────────────────────────────────────────
    output: list[dict[str, Any]] = []
    if content_part_opened:
        assert text_out_idx is not None
        output.append(
            {
                "id": text_item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": accumulated_text}],
            }
        )
    for tc in sorted(tc_state.values(), key=lambda t: t["output_index"]):
        output.append(
            {
                "id": tc["item_id"],
                "type": "function_call",
                "status": "completed",
                "call_id": tc["call_id"],
                "name": tc["name"],
                "arguments": tc["arguments"],
            }
        )

    completed = _base_resp("completed", output)
    completed["usage"] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    yield _sse(
        "response.completed", {"type": "response.completed", "response": completed}
    )


def non_streaming_responses_response(
    cc_data: dict[str, Any],
    resp_id: str,
    created_at: int,
) -> dict[str, Any]:
    """Translate a non-streaming Chat Completions response to Responses API format."""
    model = cc_data.get("model", "")
    usage = cc_data.get("usage") or {}
    output: list[dict[str, Any]] = []

    for choice in cc_data.get("choices") or []:
        message = choice.get("message") or {}
        role = message.get("role", "assistant")
        content_text = strip_markers(message.get("content") or "")
        if content_text:
            output.append(
                {
                    "id": _new_id("item"),
                    "type": "message",
                    "role": role,
                    "status": "completed",
                    "content": [{"type": "output_text", "text": content_text}],
                }
            )
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            output.append(
                {
                    "id": _new_id("item"),
                    "type": "function_call",
                    "status": "completed",
                    "call_id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", ""),
                }
            )

    return {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }
