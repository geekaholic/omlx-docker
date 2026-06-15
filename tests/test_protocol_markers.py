# SPDX-License-Identifier: Apache-2.0
"""Tests for the proxy Gemma-4 protocol-marker sanitizer."""

import json

import pytest

from omlx.proxy.protocol_markers import (
    Gemma4StreamProcessor,
    MarkerStripper,
    strip_markers,
)


def _texts(events):
    return "".join(p for k, p in events if k == "text")


def _tool_calls(events):
    return [p for k, p in events if k == "tool_call"]


def test_strips_empty_thought_channel():
    assert strip_markers("Step 1: go\n\n<|channel>thought\n<channel|>\n\ndone") == (
        "Step 1: go\n\n\n\ndone"
    )


def test_strips_nonempty_thought_span():
    assert strip_markers("a<|channel>thought\nsecret reasoning<channel|>b") == "ab"


@pytest.mark.parametrize(
    "marker",
    ["<|channel>", "<channel|>", "<turn|>", "<|tool_call>", "<tool_call|>"],
)
def test_strips_each_stray_marker(marker):
    assert strip_markers(f"x{marker}y") == "xy"


def test_noop_on_clean_text():
    text = "def f(x):\n    return x < 3 and x > 1"
    assert strip_markers(text) == text


def test_noop_when_no_angle_bracket():
    assert strip_markers("just plain words") == "just plain words"


def test_streaming_marker_split_across_deltas():
    s = MarkerStripper()
    out = ""
    # "<|channel>thought\n<channel|>" arrives in awkward fragments.
    for chunk in [
        "before ",
        "<|cha",
        "nnel>thou",
        "ght\nmid",
        "<chan",
        "nel|>",
        " after",
    ]:
        out += s.feed(chunk)
    out += s.flush()
    assert out == "before  after"


def test_streaming_emits_clean_text_promptly():
    s = MarkerStripper()
    # A delta with no marker risk should pass straight through.
    assert s.feed("hello world") == "hello world"
    assert s.flush() == ""


def test_streaming_holds_unclosed_channel_until_close():
    s = MarkerStripper()
    # Open arrives; reasoning must be withheld (not leaked) until the close.
    first = s.feed("answer <|channel>thought\nthinking...")
    assert first == "answer "
    second = s.feed(" still thinking<channel|> visible")
    assert second == " visible"
    assert s.flush() == ""


def test_streaming_unclosed_open_at_end_is_dropped_on_flush():
    s = MarkerStripper()
    # Clean prefix emits immediately; the open channel is held back.
    assert s.feed("ok <|channel>thought\ndangling reasoning") == "ok "
    # No close ever arrives; the dangling channel content is scrubbed at flush
    # (not leaked with only the open delimiter removed).
    assert s.flush() == ""


def test_streaming_does_not_split_real_less_than():
    s = MarkerStripper()
    out = s.feed("if a < b and c > d: pass")
    out += s.flush()
    assert out == "if a < b and c > d: pass"


# ── Gemma4StreamProcessor: tool-call reassembly ──────────────────────────────


_HEREDOC_CALL = (
    'call:exec_command{cmd:<|"|>cat <<EOF > js/os.js\n'
    "class WebOS { constructor() { this.apps = {}; } init() { return {}; } }\n"
    "EOF\n"
    '<|"|>}'
)


def test_processor_recovers_brace_heavy_call_without_terminator():
    # The reported failure: a heredoc writing JS (with braces) and NO trailing
    # <tool_call|>. vLLM's non-greedy fallback truncates at the first '}'; the
    # balanced-brace recovery must reassemble the whole thing.
    p = Gemma4StreamProcessor()
    events = p.feed(_HEREDOC_CALL) + p.flush()
    calls = _tool_calls(events)
    assert len(calls) == 1
    assert calls[0]["name"] == "exec_command"
    cmd = json.loads(calls[0]["arguments"])["cmd"]
    assert "class WebOS {" in cmd and "this.apps = {}" in cmd and "EOF" in cmd
    assert "call:exec_command" not in _texts(events)


def test_processor_recovers_call_split_across_deltas():
    p = Gemma4StreamProcessor()
    events = []
    # Awkward fragmentation, including a split inside the braces.
    for chunk in ["call:exec_comm", 'and{cmd:<|"|>echo {a:1}', ' {b:2}<|"|>', "}"]:
        events += p.feed(chunk)
    events += p.flush()
    calls = _tool_calls(events)
    assert len(calls) == 1 and calls[0]["name"] == "exec_command"


def test_processor_keeps_text_around_a_call():
    p = Gemma4StreamProcessor()
    events = p.feed('before <|tool_call>call:ls{path:<|"|>.<|"|>}<tool_call|> after')
    events += p.flush()
    assert len(_tool_calls(events)) == 1
    text = _texts(events)
    assert "before" in text and "after" in text
    assert "<|tool_call>" not in text and "call:ls" not in text


def test_processor_holds_partial_call_start_until_brace():
    p = Gemma4StreamProcessor()
    # "call:foo" with no brace yet must be held, not emitted as text.
    first = p.feed("answer: call:foo")
    assert "call:foo" not in _texts(first)
    rest = p.feed('{x:<|"|>1<|"|>}')
    events = first + rest + p.flush()
    assert len(_tool_calls(events)) == 1


def test_processor_drops_truncated_unbalanced_call_on_flush():
    p = Gemma4StreamProcessor()
    events = p.feed('call:exec_command{cmd:<|"|>cat <<EOF\nclass X {')  # never closes
    events += p.flush()
    assert _tool_calls(events) == []
    assert "call:exec_command" not in _texts(events)


_APPLY_PATCH_CALL = (
    'call:apply_patch{command:<|"|>*** Begin Patch\n'
    "@@ .start-menu {\n"
    "+  transition: transform cubic-bezier(0.4, 0, 0.2, 1);\n"
    "+}\n"
    "+if (x) { y(); {\n"  # deliberately unbalanced braces inside the value
    '*** End Patch<|"|>}'
)


def test_processor_recovers_apply_patch_with_braces_in_value():
    # apply_patch values are diffs/code: they contain braces (balanced *and*
    # unbalanced), colons and commas. The <|"|>-aware recovery must take the value
    # verbatim and not let those characters break end-detection or parsing.
    p = Gemma4StreamProcessor()
    events = p.feed(_APPLY_PATCH_CALL) + p.flush()
    calls = _tool_calls(events)
    assert len(calls) == 1 and calls[0]["name"] == "apply_patch"
    command = json.loads(calls[0]["arguments"])["command"]
    assert "*** Begin Patch" in command and "*** End Patch" in command
    assert "if (x) { y(); {" in command  # unbalanced braces preserved verbatim
    assert "cubic-bezier(0.4, 0, 0.2, 1)" in command
    assert "call:apply_patch" not in _texts(events)


def test_processor_recovers_apply_patch_streamed_char_by_char():
    p = Gemma4StreamProcessor()
    events = []
    for ch in _APPLY_PATCH_CALL:
        events += p.feed(ch)
    events += p.flush()
    calls = _tool_calls(events)
    assert len(calls) == 1 and calls[0]["name"] == "apply_patch"
    assert "call:apply_patch" not in _texts(events)


def test_processor_recovery_disabled_passes_call_as_text():
    # Non-Gemma models: behave like marker stripping only.
    p = Gemma4StreamProcessor(enable_tool_recovery=False)
    events = p.feed("see call:foo{a:1} here") + p.flush()
    assert _tool_calls(events) == []
    assert "call:foo{a:1}" in _texts(events)


def test_processor_still_strips_reasoning_markers():
    p = Gemma4StreamProcessor()
    events = p.feed("hi <|channel>thought\nreasoning<channel|> there") + p.flush()
    assert _tool_calls(events) == []
    assert _texts(events) == "hi  there"


def test_processor_does_not_misfire_on_recall():
    p = Gemma4StreamProcessor()
    events = p.feed("I recall:foo was mentioned.") + p.flush()
    assert _tool_calls(events) == []
    assert "recall:foo was mentioned." in _texts(events)


def test_processor_recovers_call_streamed_one_char_at_a_time():
    # Real token streaming delivers ~1 char per delta; the call: keyword must
    # accrete instead of leaking out a character at a time.
    p = Gemma4StreamProcessor()
    leaked = (
        'call:exec_command{cmd:<|"|>sed -i '
        "'s/a(`X', () => b.init());/c(`X', () => d.init());/' js/os.js<|\"|>}"
    )
    events = []
    for ch in leaked:
        events += p.feed(ch)
    events += p.flush()
    calls = _tool_calls(events)
    assert len(calls) == 1 and calls[0]["name"] == "exec_command"
    assert "call:exec_command" not in _texts(events)


def test_processor_char_stream_keeps_surrounding_text():
    p = Gemma4StreamProcessor()
    stream = 'I recall the plan. call:ls{p:<|"|>.<|"|>} done'
    events = []
    for ch in stream:
        events += p.feed(ch)
    events += p.flush()
    assert len(_tool_calls(events)) == 1
    text = _texts(events)
    assert "I recall the plan." in text and "done" in text  # 'recall' not eaten
    assert "call:ls" not in text
