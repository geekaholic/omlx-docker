# SPDX-License-Identifier: Apache-2.0
"""HTTP backend adapter for OpenAI-compatible inference servers."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from omlx.api.adapters.base import InternalResponse, StreamChunk
from omlx.api.openai_models import FunctionCall, ToolCall
from omlx.api.thinking import extract_thinking
from omlx.api.tool_calling import extract_tool_calls_with_thinking
from omlx.utils.tokenizer import is_gemma4_model

from .config import ProxyConfig
from .protocol_markers import recover_tool_calls, strip_markers


class BackendError(RuntimeError):
    """Raised when the remote backend returns an unusable response."""


@dataclass
class OpenAIBackend:
    config: ProxyConfig
    client: httpx.AsyncClient | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.request_timeout_seconds),
            )

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    def url(self, path: str) -> str:
        return f"{self.config.normalized_backend_url}/{path.lstrip('/')}"

    @property
    def root_base_url(self) -> str:
        url = self.config.normalized_backend_url
        if url.endswith("/v1"):
            return url[:-3]
        return url

    def headers(self, inbound_authorization: str | None = None) -> dict[str, str]:
        headers = {"accept": "application/json"}
        token = self.config.backend_api_key
        if token:
            headers["authorization"] = f"Bearer {token}"
        elif inbound_authorization:
            headers["authorization"] = inbound_authorization
        return headers

    async def get_models(
        self, inbound_authorization: str | None = None
    ) -> dict[str, Any]:
        assert self.client is not None
        response = await self.client.get(
            self.url("models"),
            headers=self.headers(inbound_authorization),
        )
        response.raise_for_status()
        return response.json()

    async def get_root_json(
        self,
        path: str,
        inbound_authorization: str | None = None,
    ) -> dict[str, Any]:
        assert self.client is not None
        response = await self.client.get(
            f"{self.root_base_url}/{path.lstrip('/')}",
            headers=self.headers(inbound_authorization),
        )
        response.raise_for_status()
        return response.json()

    async def get_root_text(
        self,
        path: str,
        inbound_authorization: str | None = None,
    ) -> str:
        assert self.client is not None
        headers = self.headers(inbound_authorization)
        headers["accept"] = "text/plain, */*"
        response = await self.client.get(
            f"{self.root_base_url}/{path.lstrip('/')}",
            headers=headers,
        )
        response.raise_for_status()
        return response.text

    async def chat_completion(
        self,
        body: dict[str, Any],
        inbound_authorization: str | None = None,
    ) -> dict[str, Any]:
        assert self.client is not None
        response = await self.client.post(
            self.url("chat/completions"),
            headers=self.headers(inbound_authorization),
            json=body,
        )
        response.raise_for_status()
        return response.json()

    async def stream_chat_completion(
        self,
        body: dict[str, Any],
        inbound_authorization: str | None = None,
    ) -> AsyncIterator[dict[str, Any] | str]:
        assert self.client is not None
        async with self.client.stream(
            "POST",
            self.url("chat/completions"),
            headers=self.headers(inbound_authorization),
            json=body,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                parsed = parse_sse_line(line)
                if parsed is not None:
                    yield parsed

    async def raw_request(
        self,
        method: str,
        path: str,
        body: bytes,
        headers: dict[str, str],
    ) -> httpx.Response:
        assert self.client is not None
        return await self.client.request(
            method,
            self.url(path),
            content=body,
            headers=headers,
        )

    async def raw_stream(
        self,
        method: str,
        path: str,
        body: bytes,
        headers: dict[str, str],
    ) -> httpx.Response:
        assert self.client is not None
        request = self.client.build_request(
            method,
            self.url(path),
            content=body,
            headers=headers,
        )
        return await self.client.send(request, stream=True)


def parse_sse_line(line: str) -> dict[str, Any] | str | None:
    """Parse a single OpenAI-style SSE line."""
    if not line.startswith("data:"):
        return None
    data = line.removeprefix("data:").strip()
    if not data:
        return None
    if data == "[DONE]":
        return "[DONE]"
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        raise BackendError(f"Malformed backend SSE data: {data[:120]!r}") from exc


def openai_response_to_internal(
    data: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
) -> InternalResponse:
    choices = data.get("choices") or []
    if not choices:
        raise BackendError("Backend response did not include choices")

    choice = choices[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}

    tool_calls = _coerce_tool_calls(message.get("tool_calls"))
    raw_content = message.get("content") or ""
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("reasoning")
    text = strip_markers(raw_content)
    # Recover a tool call vLLM left in content (Gemma-4, when it didn't already
    # produce a structured one); otherwise just strip protocol markers.
    if not tool_calls and tools:
        recovered_text, recovered_reasoning, recovered_calls = recover_text_tool_calls(
            raw_content,
            reasoning_content=reasoning,
            tools=tools,
        )
        if recovered_calls:
            text = recovered_text
            reasoning = recovered_reasoning
            tool_calls = recovered_calls
        elif recovered_reasoning:
            text = recovered_text
            reasoning = recovered_reasoning

    if not tool_calls and is_gemma4_model(str(data.get("model") or "")):
        text, recovered = recover_tool_calls(raw_content)
        if recovered:
            tool_calls = [
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    type="function",
                    function=FunctionCall(name=rc["name"], arguments=rc["arguments"]),
                )
                for rc in recovered
            ]
    if reasoning and not text:
        text = ""

    return InternalResponse(
        text=text,
        reasoning_content=reasoning,
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        cached_tokens=int(prompt_details.get("cached_tokens") or 0),
        tool_calls=tool_calls,
        request_id=data.get("id"),
        model=data.get("model"),
    )


def openai_chunk_to_stream_chunk(
    data: dict[str, Any],
    *,
    is_first: bool,
) -> StreamChunk:
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    delta = choice.get("delta") or {}
    usage = data.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}

    return StreamChunk(
        text=delta.get("content") or "",
        reasoning_content=delta.get("reasoning_content")
        if delta.get("reasoning_content") is not None
        else delta.get("reasoning"),
        tool_call_delta=delta.get("tool_calls"),
        finish_reason=choice.get("finish_reason"),
        is_first=is_first,
        is_last=bool(choice.get("finish_reason")),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        cached_tokens=int(prompt_details.get("cached_tokens") or 0),
    )


def recover_text_tool_calls(
    text: str,
    *,
    reasoning_content: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, str | None, list[ToolCall] | None]:
    """Recover structured tool calls from text-formatted model output.

    Some local backends emit the conversation-history bracket form, e.g.
    ``[Calling tool: Read({...})]``, as ordinary assistant text.  Claude Code
    needs Anthropic ``tool_use`` blocks instead, so the proxy promotes these
    envelopes only when they match a tool from the request.
    """
    thinking_content, regular_content = extract_thinking(text or "")
    thinking_parts = [part for part in (reasoning_content, thinking_content) if part]
    combined_thinking = "\n".join(thinking_parts)
    extraction = extract_tool_calls_with_thinking(
        combined_thinking,
        regular_content,
        tokenizer=None,
        tools=tools,
    )
    tool_calls = filter_tool_calls_by_tools(extraction.tool_calls, tools)
    if not tool_calls:
        split_calls = _recover_split_bracket_tool_calls(
            combined_thinking,
            regular_content,
            tools,
        )
        if split_calls:
            return "", None, split_calls
        if _has_unresolved_bracket_tool_prefix(
            combined_thinking
        ) and _looks_like_bracket_tool_tail(regular_content):
            return "", None, None
    return (
        extraction.cleaned_text,
        extraction.cleaned_thinking if extraction.cleaned_thinking else None,
        tool_calls,
    )


def _recover_split_bracket_tool_calls(
    thinking_content: str,
    regular_content: str,
    tools: list[dict[str, Any]] | None,
) -> list[ToolCall] | None:
    if not thinking_content or not regular_content:
        return None
    if not _has_unresolved_bracket_tool_prefix(thinking_content):
        return None

    extraction = extract_tool_calls_with_thinking(
        "",
        f"{thinking_content}{regular_content}",
        tokenizer=None,
        tools=tools,
    )
    return filter_tool_calls_by_tools(extraction.tool_calls, tools)


def _has_unresolved_bracket_tool_prefix(text: str) -> bool:
    for prefix in ("[Calling tool:", "[Tool call:"):
        idx = text.rfind(prefix)
        if idx >= 0 and "]" not in text[idx:]:
            return True
    return False


def _looks_like_bracket_tool_tail(text: str) -> bool:
    tail = (text or "").lstrip()
    return bool(tail and "]" in tail and (")]" in tail or "}]" in tail))


def coalesce_tool_call_deltas(deltas: list[Any]) -> list[ToolCall] | None:
    """Merge OpenAI streaming ``delta.tool_calls`` fragments."""
    if not deltas:
        return None

    merged: dict[int, dict[str, Any]] = {}
    fallback_index = 0
    for delta in deltas:
        if not isinstance(delta, dict):
            continue
        raw_index = delta.get("index")
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = fallback_index
            fallback_index += 1

        current = merged.setdefault(
            index,
            {"id": "", "type": "function", "name": "", "arguments": ""},
        )
        if delta.get("id"):
            current["id"] = delta["id"]
        if delta.get("type"):
            current["type"] = delta["type"]

        function = delta.get("function") or {}
        if isinstance(function, dict):
            if function.get("name"):
                current["name"] = function["name"]
            if function.get("arguments"):
                current["arguments"] += function["arguments"]

    result: list[ToolCall] = []
    for index in sorted(merged):
        item = merged[index]
        if not item["name"]:
            continue
        result.append(
            ToolCall(
                id=item["id"] or f"call_{uuid.uuid4().hex[:8]}",
                type=item["type"] or "function",
                function=FunctionCall(
                    name=item["name"],
                    arguments=item["arguments"] or "{}",
                ),
            )
        )
    return result or None


def _coerce_tool_calls(value: Any) -> list[ToolCall] | None:
    if not value:
        return None
    result: list[ToolCall] = []
    for item in value:
        if isinstance(item, ToolCall):
            result.append(item)
            continue
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        result.append(
            ToolCall(
                id=item.get("id") or "",
                type=item.get("type") or "function",
                function=FunctionCall(
                    name=function.get("name") or "",
                    arguments=function.get("arguments") or "{}",
                ),
            )
        )
    return result or None


def filter_tool_calls_by_tools(
    tool_calls: list[ToolCall] | None,
    tools: list[dict[str, Any]] | None,
) -> list[ToolCall] | None:
    if not tool_calls:
        return None
    if tools is None:
        return tool_calls

    tools_by_name = {
        function.get("name"): tool
        for tool in tools
        if isinstance(tool, dict)
        and isinstance(function := tool.get("function"), dict)
        and function.get("name")
    }
    filtered = [
        tc
        for tc in tool_calls
        if (tool := tools_by_name.get(tc.function.name))
        and _tool_call_matches_requested_tool(tc, tool)
    ]
    return filtered or None


def _tool_call_matches_requested_tool(
    tool_call: ToolCall,
    tool: dict[str, Any],
) -> bool:
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(arguments, dict):
        return False

    function = tool.get("function") if isinstance(tool, dict) else None
    parameters = function.get("parameters") if isinstance(function, dict) else None
    if not isinstance(parameters, dict):
        return set(arguments) != {"raw"}

    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        properties = {}

    if set(arguments) == {"raw"} and "raw" not in properties:
        return False

    required = parameters.get("required")
    if isinstance(required, list):
        missing = [
            name
            for name in required
            if isinstance(name, str) and name not in arguments
        ]
        if missing:
            return False

    return True
