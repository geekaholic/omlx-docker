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
# Name + opening brace, used to split a recovered span into name / args object.
_CALL_NAME_RE = re.compile(r"\bcall:([\w.:-]+)\s*\{")
# A partial tool-call start at the buffer tail (no `{` yet) to hold across deltas.
_TOOLCALL_PARTIAL_RE = re.compile(r"(?:<\|tool_call>\s*)?\bcall:[\w.:-]*\Z")
_CALL_KEYWORD = "call:"
# Gemma-4 string-value delimiter; its content is opaque (braces, colons, commas).
_STRING_DELIM = '<|"|>'
_STRING_DELIM_LEN = len(_STRING_DELIM)


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
    """Index just past the `}` closing the args object opened at ``open_idx``.

    ``<|"|>``-aware: Gemma-4 string values are wrapped in ``<|"|>…<|"|>`` and may
    contain *unbalanced* braces (a patch, a code snippet). We toggle an in-string
    flag on each ``<|"|>`` and count braces only outside strings, so the real
    args-closing `}` is found regardless of braces in the value. Returns None when
    the buffer doesn't yet contain that closing `}` (incomplete / still streaming).
    """
    depth = 0
    in_string = False
    i = open_idx
    n = len(s)
    while i < n:
        if s.startswith(_STRING_DELIM, i):
            in_string = not in_string
            i += _STRING_DELIM_LEN
            continue
        if not in_string:
            c = s[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return None


def _coerce_gemma4_bare_value(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return ""
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null":
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def _parse_gemma4_args_stdlib(args_str: str) -> dict[str, Any]:
    """Small stdlib parser for Gemma-4 ``{key:<|"|>value<|"|>}`` args.

    The full fallback in :mod:`omlx.api.tool_calling` uses the third-party
    ``regex`` package. The proxy environment may not have it, so this covers the
    streamed recovery cases here without introducing that dependency.
    """
    text = args_str.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError("Gemma-4 args must be wrapped in braces")
    out: dict[str, Any] = {}
    i = 1
    end = len(text) - 1
    while i < end:
        while i < end and (text[i].isspace() or text[i] == ","):
            i += 1
        if i >= end:
            break

        key_start = i
        while i < end and text[i] != ":":
            i += 1
        if i >= end:
            raise ValueError("Gemma-4 arg key missing ':'")
        key = text[key_start:i].strip()
        if not key:
            raise ValueError("Gemma-4 arg key is empty")
        i += 1

        while i < end and text[i].isspace():
            i += 1
        if text.startswith(_STRING_DELIM, i):
            i += _STRING_DELIM_LEN
            value_start = i
            value_end = text.find(_STRING_DELIM, i)
            if value_end == -1 or value_end > end:
                raise ValueError("Unterminated Gemma-4 string arg")
            out[key] = text[value_start:value_end]
            i = value_end + _STRING_DELIM_LEN
            continue

        value_start = i
        depth = 0
        in_json_string = False
        escape = False
        while i < end:
            c = text[i]
            if in_json_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_json_string = False
            elif c == '"':
                in_json_string = True
            elif c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
            elif c == "," and depth == 0:
                break
            i += 1
        out[key] = _coerce_gemma4_bare_value(text[value_start:i])
    return out


def _parse_gemma4_args(args_str: str) -> Any:
    """Parse a Gemma-4 ``{...}`` args object to a Python value, or None on failure.

    ``args_str`` already has the correct (``<|"|>``-aware) closing brace. omlx's
    ``_gemma4_args_to_json_robust`` extracts ``<|"|>…<|"|>`` string content into
    placeholders *before* touching braces, so it copes with patches/code whose
    values contain braces, colons and commas — the cases the brace-balanced outer
    regex of ``_parse_gemma4_tool_call_fallback`` chokes on.
    """
    try:
        return json.loads(args_str)
    except (ValueError, TypeError):
        pass
    try:
        from omlx.api.tool_calling import _gemma4_args_to_json_robust

        return _gemma4_args_to_json_robust(args_str)
    except Exception:
        pass
    try:
        return _parse_gemma4_args_stdlib(args_str)
    except Exception:
        return None



def parse_leaked_tool_calls(call_text: str) -> list[dict[str, str]]:
    """Parse leaked ``call:name{...}`` span(s) into OpenAI tool-call dicts.

    Locates each ``call:name{`` and finds the matching `}` with the
    ``<|"|>``-aware :func:`_match_braces`, so values containing braces (apply_patch
    diffs, code) don't break end-detection. ``arguments`` is returned as a JSON
    string (OpenAI wire format). Returns [] when nothing parses.
    """
    out: list[dict[str, str]] = []
    pos = 0
    n = len(call_text)
    while pos < n:
        m = _CALL_NAME_RE.search(call_text, pos)
        if m is None:
            break
        open_idx = call_text.index("{", m.end() - 1)
        end = _match_braces(call_text, open_idx)
        if end is None:
            break
        args = _parse_gemma4_args(call_text[open_idx:end])
        if args is not None:
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            out.append({"name": m.group(1), "arguments": args})
        pos = end
    return out


class Gemma4StreamProcessor:
    """Streaming content processor for Gemma-4 output.

    Strips reasoning-channel markers AND reassembles tool calls that vLLM's
    parser leaked into the content channel (it needs a clean ``<tool_call|>``
    terminator and truncates brace-heavy arguments; this recovers them via the
    ``<|"|>``-aware :func:`parse_leaked_tool_calls`, so apply_patch/code values
    containing braces survive).

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
