# SPDX-License-Identifier: Apache-2.0
"""Claude Code usage scaling and SSE keepalive helpers."""

from __future__ import annotations

from .config import ProxyConfig

KEEPALIVE_COMMENT = ": keep-alive\n\n"
KEEPALIVE_ANTHROPIC_PING = 'event: ping\ndata: {"type":"ping"}\n\n'


def scale_token_count(token_count: int, config: ProxyConfig) -> int:
    """Scale Anthropic usage counts to a configured target context window."""
    if not config.context_scaling_enabled:
        return token_count
    actual = config.actual_context_size
    target = config.target_context_size
    if actual <= 0 or target <= 0 or actual >= target:
        return token_count
    return int(token_count * target / actual)


def anthropic_keepalive_frame(config: ProxyConfig) -> str | None:
    mode = config.sse_keepalive_mode.strip().lower()
    if mode == "off":
        return None
    if mode == "comment":
        return KEEPALIVE_COMMENT
    return KEEPALIVE_ANTHROPIC_PING

