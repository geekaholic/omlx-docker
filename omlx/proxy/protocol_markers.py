# SPDX-License-Identifier: Apache-2.0
"""Strip stray Gemma-4 protocol markers that leak into visible content.

vLLM's gemma4 tool/reasoning parsers cover the well-formed cases, but empty or
mid-stream reasoning-channel scaffolding (``<|channel>thought\\n<channel|>``) and
stray delimiters still reach ``delta.content``. These tokens are never legitimate
model output, so we scrub them as a backend-agnostic safety net.

Mirrors the marker set in ``omlx/adapter/gemma4.py`` (which isn't importable from
the proxy venv — its package ``__init__`` pulls in ``openai_harmony``).
"""

from __future__ import annotations

import re

# Channel/tool-response spans (open ... close), possibly empty. Removed whole.
_SPAN_RE = re.compile(
    r"<\|channel>.*?<channel\|>|<\|tool_response>.*?<tool_response\|>",
    re.DOTALL,
)
# Any single protocol delimiter (stray opens/closes, turn, tool-call tokens).
_SINGLE_RE = re.compile(
    r"<\|channel>|<channel\|>|<turn\|>|<\|tool_response>|<tool_response\|>"
    r"|<\|tool_call>|<tool_call\|>"
)
# A span open that never closed, through end of text — drop it and its trailing
# (reasoning) content rather than leaking the content with only the open stripped.
_UNTERMINATED_RE = re.compile(r"<\|channel>.*\Z|<\|tool_response>.*\Z", re.DOTALL)

# Span markers, used by the streaming stripper to detect an unclosed open.
_SPANS = (("<|channel>", "<channel|>"), ("<|tool_response>", "<tool_response|>"))
_ALL_MARKERS = (
    "<|channel>",
    "<channel|>",
    "<turn|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|tool_call>",
    "<tool_call|>",
)
_MAX_MARKER_LEN = max(len(m) for m in _ALL_MARKERS)


def strip_markers(text: str) -> str:
    """Remove complete spans and every stray protocol delimiter from ``text``.

    Safe for any model: a no-op when none of the markers are present, and the
    tokens it removes never occur in legitimate output.
    """
    if not text or "<" not in text:
        return text
    return _SINGLE_RE.sub("", _SPAN_RE.sub("", text))


def _safe_cut(buf: str) -> int:
    """Index up to which ``buf`` is safe to emit during streaming.

    Everything before the returned index contains only *complete* markers/spans;
    the suffix is held because it may be an unclosed span open or a partial
    marker split across deltas.
    """
    cut = len(buf)
    # Earliest unclosed span open (its close may arrive in a later delta).
    for open_tok, close_tok in _SPANS:
        start = 0
        while True:
            pos = buf.find(open_tok, start)
            if pos == -1:
                break
            if buf.find(close_tok, pos + len(open_tok)) == -1:
                cut = min(cut, pos)
                break
            start = pos + len(open_tok)
    # Trailing run that is a proper prefix of some marker (a partial token).
    region = buf[:cut]
    for i in range(max(0, len(region) - _MAX_MARKER_LEN + 1), len(region)):
        suffix = region[i:]
        if any(m != suffix and m.startswith(suffix) for m in _ALL_MARKERS):
            cut = min(cut, i)
            break
    return cut


class MarkerStripper:
    """Incremental :func:`strip_markers` for a streamed content channel.

    ``feed`` returns the safe-to-emit prefix and buffers an ambiguous suffix;
    ``flush`` returns whatever remains once the stream ends.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> str:
        if not text:
            return ""
        self._buf += text
        cut = _safe_cut(self._buf)
        head, self._buf = self._buf[:cut], self._buf[cut:]
        return strip_markers(head)

    def flush(self) -> str:
        # An open span that never closed: drop it through end of text so its
        # reasoning content isn't leaked with only the open delimiter removed.
        buf = _UNTERMINATED_RE.sub("", self._buf)
        self._buf = ""
        return strip_markers(buf)
