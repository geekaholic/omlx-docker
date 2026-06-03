# SPDX-License-Identifier: Apache-2.0
"""HTTP backend adapter for OpenAI-compatible inference servers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from omlx.api.adapters.base import InternalResponse, StreamChunk
from omlx.api.openai_models import FunctionCall, ToolCall

from .config import ProxyConfig


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

    async def get_models(self, inbound_authorization: str | None = None) -> dict[str, Any]:
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


def openai_response_to_internal(data: dict[str, Any]) -> InternalResponse:
    choices = data.get("choices") or []
    if not choices:
        raise BackendError("Backend response did not include choices")

    choice = choices[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}

    tool_calls = _coerce_tool_calls(message.get("tool_calls"))
    text = message.get("content") or ""
    reasoning = message.get("reasoning_content")
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
        reasoning_content=delta.get("reasoning_content"),
        tool_call_delta=delta.get("tool_calls"),
        finish_reason=choice.get("finish_reason"),
        is_first=is_first,
        is_last=bool(choice.get("finish_reason")),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        cached_tokens=int(prompt_details.get("cached_tokens") or 0),
    )


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
