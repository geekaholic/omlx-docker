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

import json
import re
from typing import Any

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


# A leaked Gemma-4 tool call (the `<|tool_call>` special token is usually gone
# from decoded content). Group 1 is `call:name{` so we can find the args brace.
_TOOLCALL_START_RE = re.compile(r"(?:<\|tool_call>\s*)?(\bcall:[\w.:-]+\s*\{)")
# A partial tool-call start at the buffer tail (no `{` yet) to hold across deltas.
_TOOLCALL_PARTIAL_RE = re.compile(r"(?:<\|tool_call>\s*)?\bcall:[\w.:-]*\Z")
_CALL_KEYWORD = "call:"


def _is_word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _partial_toolcall_cut(buf: str) -> int:
    """Index of a trailing run that could grow into a tool-call start.

    Holds both ``call:<partial-name>`` and the ``call:`` keyword itself forming
    char-by-char (``c`` / ``ca`` / ``cal`` / ``call``) at a word boundary — real
    token streaming delivers one or a few characters per delta, so the keyword
    must be allowed to accrete instead of leaking out a character at a time.
    """
    match = _TOOLCALL_PARTIAL_RE.search(buf)
    if match is not None:
        return match.start()
    for i in range(max(0, len(buf) - len(_CALL_KEYWORD) + 1), len(buf)):
        suffix = buf[i:]
        if _CALL_KEYWORD.startswith(suffix) and (
            i == 0 or not _is_word_char(buf[i - 1])
        ):
            return i
    return len(buf)


def _match_braces(s: str, open_idx: int) -> int | None:
    """Index just past the `}` balancing the `{` at ``open_idx``, or None if the
    buffer doesn't yet contain a balancing `}`. Counts every brace, matching
    omlx's balanced-brace parser — so braces inside the JS payload are handled."""
    depth = 0
    for i in range(open_idx, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def parse_leaked_tool_calls(call_text: str) -> list[dict[str, str]]:
    """Parse a leaked ``call:name{...}`` span into OpenAI tool-call dicts.

    Reuses omlx's balanced-brace fallback parser, which handles `}` inside the
    arguments and needs no trailing marker. ``arguments`` is returned as a JSON
    string (OpenAI wire format). Returns [] when nothing parses.
    """
    try:
        from omlx.api.tool_calling import _parse_gemma4_tool_call_fallback

        parsed = _parse_gemma4_tool_call_fallback(call_text)
    except Exception:
        return []
    items = parsed if isinstance(parsed, list) else [parsed]
    out: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        args = item.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        out.append({"name": str(item["name"]), "arguments": args})
    return out


class Gemma4StreamProcessor:
    """Streaming content processor for Gemma-4 output.

    Strips reasoning-channel markers AND reassembles tool calls that vLLM's
    parser leaked into the content channel (it needs a clean ``<tool_call|>``
    terminator and truncates brace-heavy arguments; this recovers them via
    omlx's balanced-brace parser).

    ``feed`` / ``flush`` return a list of events:
      * ``("text", str)`` — clean text to forward.
      * ``("tool_call", {"name": str, "arguments": str})`` — a recovered call.

    Set ``enable_tool_recovery=False`` (non-Gemma models) to do marker stripping
    only — then it behaves exactly like :class:`MarkerStripper`.
    """

    def __init__(self, *, enable_tool_recovery: bool = True) -> None:
        self._buf = ""
        self._recover = enable_tool_recovery

    def feed(self, text: str) -> list[tuple[str, Any]]:
        if text:
            self._buf += text
        return self._drain(final=False)

    def flush(self) -> list[tuple[str, Any]]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[tuple[str, Any]]:
        events: list[tuple[str, Any]] = []
        while True:
            match = _TOOLCALL_START_RE.search(self._buf) if self._recover else None
            if match is not None:
                end = _match_braces(self._buf, match.end(1) - 1)
                if end is not None:
                    pre = self._buf[: match.start()]
                    if pre:
                        events.append(("text", strip_markers(pre)))
                    calls = parse_leaked_tool_calls(self._buf[match.start(1) : end])
                    if calls:
                        events.extend(("tool_call", tc) for tc in calls)
                    else:
                        # Unparseable — surface the raw text rather than drop it.
                        events.append(
                            ("text", strip_markers(self._buf[match.start(1) : end]))
                        )
                    self._buf = self._buf[end:]
                    continue
                # Incomplete tool call: emit any text before it.
                pre = self._buf[: match.start()]
                if pre:
                    events.append(("text", strip_markers(pre)))
                if final:
                    self._buf = ""  # truncated call can't be executed — drop it
                else:
                    self._buf = self._buf[match.start() :]  # hold for more deltas
                break
            # No (complete) tool-call start in the buffer.
            if final:
                rest = _UNTERMINATED_RE.sub("", self._buf)
                self._buf = ""
                cleaned = strip_markers(rest)
                if cleaned:
                    events.append(("text", cleaned))
            else:
                cut = self._safe_text_cut(self._buf)
                head, self._buf = self._buf[:cut], self._buf[cut:]
                if head:
                    events.append(("text", strip_markers(head)))
            break
        return [e for e in events if not (e[0] == "text" and e[1] == "")]

    def _safe_text_cut(self, buf: str) -> int:
        cut = _safe_cut(buf)
        if self._recover:
            cut = min(cut, _partial_toolcall_cut(buf))
        return cut


def recover_tool_calls(text: str) -> tuple[str, list[dict[str, str]]]:
    """Non-streaming split of leaked gemma-4 tool calls out of ``text``.

    Returns ``(cleaned_text, [{"name", "arguments"}])``. Intended for Gemma-4
    output where vLLM left a `call:name{...}` in the content. A no-op (empty
    list) when no recoverable tool call is present.
    """
    proc = Gemma4StreamProcessor(enable_tool_recovery=True)
    events = proc.feed(text) + proc.flush()
    cleaned = "".join(p for k, p in events if k == "text")
    calls = [p for k, p in events if k == "tool_call"]
    return cleaned, calls
