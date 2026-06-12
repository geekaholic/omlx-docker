# SPDX-License-Identifier: Apache-2.0
"""Proxy-side serving stats: per-request usage and timing capture.

The proxy cannot see engine internals, so serving stats are accumulated
from the ``usage`` payloads that OpenAI-compatible backends attach to
responses, plus wall-clock timing (time-to-first-token approximates the
prefill duration, first-token-to-close the generation duration).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from omlx.server_metrics import ServerMetrics

from .backend import parse_sse_line


def stats_path_from_env() -> Path:
    """Resolve where all-time proxy stats persist.

    ``OMLX_PROXY_STATS_PATH`` wins; otherwise the stats file lives next to
    the proxy admin state file.
    """
    explicit = os.getenv("OMLX_PROXY_STATS_PATH", "").strip()
    if explicit:
        return Path(explicit)
    state_path = Path(os.getenv("OMLX_PROXY_STATE_PATH", "/data/proxy-state.json"))
    return state_path.parent / "proxy-stats.json"


def usage_from_chat_data(data: dict[str, Any]) -> tuple[int, int, int]:
    """Extract (prompt, completion, cached) token counts from a response."""
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        int(details.get("cached_tokens") or 0),
    )


def timings_from_chat_data(data: dict[str, Any]) -> tuple[float, float]:
    """Extract (prefill, generation) durations from llama.cpp ``timings``."""
    timings = data.get("timings") or {}
    try:
        prompt_ms = float(timings.get("prompt_ms") or 0.0)
        predicted_ms = float(timings.get("predicted_ms") or 0.0)
    except (TypeError, ValueError):
        return 0.0, 0.0
    return prompt_ms / 1000.0, predicted_ms / 1000.0


def record_request(
    metrics: ServerMetrics | None,
    *,
    model_id: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    prefill_duration: float = 0.0,
    generation_duration: float = 0.0,
) -> None:
    """Record one completed proxied request. No-op without metrics."""
    if metrics is None:
        return
    metrics.record_request_complete(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        prefill_duration=prefill_duration,
        generation_duration=generation_duration,
        model_id=model_id,
    )


def record_chat_response(
    metrics: ServerMetrics | None,
    data: Any,
    fallback_model: str = "",
) -> None:
    """Record a non-streaming chat/completions response body."""
    if metrics is None or not isinstance(data, dict):
        return
    prompt, completion, cached = usage_from_chat_data(data)
    prefill_duration, generation_duration = timings_from_chat_data(data)
    record_request(
        metrics,
        model_id=str(data.get("model") or fallback_model or ""),
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
        prefill_duration=prefill_duration,
        generation_duration=generation_duration,
    )


def _is_usage_only_chunk(parsed: Any) -> bool:
    return (
        isinstance(parsed, dict)
        and not parsed.get("choices")
        and bool(parsed.get("usage"))
    )


async def track_usage_stream(
    stream: AsyncIterator[bytes],
    *,
    metrics: ServerMetrics | None,
    model_id: str = "",
    request_start: float | None = None,
    strip_usage_chunk: bool = False,
) -> AsyncIterator[bytes]:
    """Relay a backend SSE byte stream while extracting usage and timing.

    When the proxy injected ``stream_options.include_usage`` on the client's
    behalf (``strip_usage_chunk=True``), the trailing usage-only chunk is
    consumed for stats and not relayed downstream.
    """
    started = request_start if request_start is not None else time.monotonic()
    first_chunk_at: float | None = None
    prompt = completion = cached = 0
    seen_model = ""
    drop_next_blank = False
    buffer = b""

    try:
        async for raw in stream:
            buffer += raw
            out = b""
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                text = line.decode("utf-8", errors="replace")
                if drop_next_blank and not text.strip():
                    drop_next_blank = False
                    continue
                try:
                    parsed = parse_sse_line(text)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    if parsed.get("choices") and first_chunk_at is None:
                        first_chunk_at = time.monotonic()
                    if parsed.get("model"):
                        seen_model = str(parsed["model"])
                    if parsed.get("usage"):
                        prompt, completion, cached = usage_from_chat_data(parsed)
                    if strip_usage_chunk and _is_usage_only_chunk(parsed):
                        drop_next_blank = True
                        continue
                out += line + b"\n"
            if out:
                yield out
        if buffer:
            yield buffer
    finally:
        end = time.monotonic()
        prefill_duration = 0.0
        generation_duration = 0.0
        if first_chunk_at is not None:
            prefill_duration = max(0.0, first_chunk_at - started)
            generation_duration = max(0.0, end - first_chunk_at)
        record_request(
            metrics,
            model_id=seen_model or model_id,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached,
            prefill_duration=prefill_duration,
            generation_duration=generation_duration,
        )


def inject_include_usage(body: bytes) -> tuple[bytes, bool]:
    """Add ``stream_options.include_usage`` to a streaming request body.

    Returns the (possibly rewritten) body and whether injection happened.
    Bodies that already carry ``stream_options`` are left untouched.
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return body, False
    if not isinstance(payload, dict) or not payload.get("stream"):
        return body, False
    if "stream_options" in payload:
        return body, False
    payload["stream_options"] = {"include_usage": True}
    return json.dumps(payload, separators=(",", ":")).encode("utf-8"), True


def model_from_body(body: bytes) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return ""
    if isinstance(payload, dict):
        return str(payload.get("model") or "")
    return ""
