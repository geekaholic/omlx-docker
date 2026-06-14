# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM sidecar tool-calling auto-configuration."""

import pytest

from omlx.proxy.vllm_compose import (
    VllmComposeSettings,
    render_vllm_compose,
    resolve_tool_env,
    settings_from_overrides,
    vllm_environment,
    vllm_settings_from_env,
)


def _tool_env(model="Qwen/Qwen3-1.7B", **overrides):
    return resolve_tool_env(VllmComposeSettings(model=model, **overrides))


def test_gemma4_auto_selects_native_parser_reasoning_and_template():
    env = _tool_env("google/gemma-4-26B-A4B-it")
    assert env["VLLM_ENABLE_AUTO_TOOL_CHOICE"] == "true"
    assert env["VLLM_TOOL_CALL_PARSER"] == "gemma4"
    assert env["VLLM_REASONING_PARSER"] == "gemma4"
    assert env["VLLM_CHAT_TEMPLATE"].endswith("tool_chat_template_gemma4.jinja")


def test_qwen_auto_selects_hermes_without_reasoning_or_template():
    env = _tool_env("Qwen/Qwen3-8B")
    assert env["VLLM_ENABLE_AUTO_TOOL_CHOICE"] == "true"
    assert env["VLLM_TOOL_CALL_PARSER"] == "hermes"
    assert env["VLLM_REASONING_PARSER"] == ""
    assert env["VLLM_CHAT_TEMPLATE"] == ""


def test_unknown_family_leaves_parser_empty():
    # An unrecognized family resolves to an empty parser; the launch shell only
    # passes --enable-auto-tool-choice when a concrete parser is present, so
    # vLLM (which refuses the flag without a parser) still starts.
    env = _tool_env("microsoft/phi-4")
    assert env["VLLM_TOOL_CALL_PARSER"] == ""


def test_explicit_parser_overrides_detection():
    env = _tool_env("google/gemma-4-26B-A4B-it", tool_call_parser="hermes")
    assert env["VLLM_ENABLE_AUTO_TOOL_CHOICE"] == "true"
    assert env["VLLM_TOOL_CALL_PARSER"] == "hermes"


def test_explicit_gemma4_parser_pairs_reasoning_parser():
    # An explicit gemma4 tool parser must still get its matching reasoning parser
    # so the `<|channel>thought ... <channel|>` markers don't leak.
    env = _tool_env("google/gemma-4-26B-A4B-it", tool_call_parser="gemma4")
    assert env["VLLM_TOOL_CALL_PARSER"] == "gemma4"
    assert env["VLLM_REASONING_PARSER"] == "gemma4"


def test_explicit_user_reasoning_parser_wins_over_pairing():
    env = _tool_env(
        "google/gemma-4-26B-A4B-it",
        tool_call_parser="gemma4",
        reasoning_parser="custom",
    )
    assert env["VLLM_REASONING_PARSER"] == "custom"


def test_explicit_non_gemma_parser_has_no_reasoning_pairing():
    env = _tool_env("Qwen/Qwen3-8B", tool_call_parser="hermes")
    assert env["VLLM_REASONING_PARSER"] == ""


@pytest.mark.parametrize("off_value", ["none", "off", "disabled", "false"])
def test_explicit_off_disables_tools(off_value):
    env = _tool_env("google/gemma-4-26B-A4B-it", tool_call_parser=off_value)
    assert env["VLLM_TOOL_CALL_PARSER"] == ""


def test_master_switch_off_disables_even_with_known_family():
    env = _tool_env("google/gemma-4-26B-A4B-it", enable_auto_tool_choice=False)
    assert env["VLLM_ENABLE_AUTO_TOOL_CHOICE"] == "false"
    assert env["VLLM_TOOL_CALL_PARSER"] == ""


def test_user_reasoning_parser_is_not_clobbered_by_detection():
    env = _tool_env("google/gemma-4-26B-A4B-it", reasoning_parser="deepseek_r1")
    assert env["VLLM_REASONING_PARSER"] == "deepseek_r1"


def test_environment_round_trips_resolved_values():
    settings = VllmComposeSettings(model="google/gemma-4-26B-A4B-it")
    env = vllm_environment(settings)
    back = vllm_settings_from_env(env)
    assert back.tool_call_parser == "gemma4"
    assert back.reasoning_parser == "gemma4"
    assert back.chat_template.endswith("tool_chat_template_gemma4.jinja")


def test_overrides_resolve_default_model_parser():
    # settings_from_overrides resolves the parser for the model: the default
    # Qwen model maps to the Hermes parser, with tools on by default.
    settings = settings_from_overrides({})
    assert settings.enable_auto_tool_choice is True
    assert settings.tool_call_parser == "hermes"
    assert settings.chat_template == ""


def test_render_emits_hardened_tool_guard_and_chat_template_block():
    content = render_vllm_compose(
        VllmComposeSettings(model="google/gemma-4-26B-A4B-it")
    )
    # Hardened guard: tools only enabled when a concrete parser is present.
    assert (
        '[ "$${VLLM_ENABLE_AUTO_TOOL_CHOICE:-true}" = "true" ] '
        '&& [ -n "$${VLLM_TOOL_CALL_PARSER:-}" ]'
    ) in content
    assert "--tool-call-parser" in content
    assert "--chat-template" in content
    # The gemma4 template path is baked into the compose default.
    assert "tool_chat_template_gemma4.jinja" in content
