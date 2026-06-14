# SPDX-License-Identifier: Apache-2.0
"""Tests for the proxy Gemma-4 protocol-marker sanitizer."""

import pytest

from omlx.proxy.protocol_markers import MarkerStripper, strip_markers


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
