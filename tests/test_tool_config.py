# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-model vLLM tool-calling registry."""

import pytest

from omlx.proxy.tool_config import (
    GEMMA4_CHAT_TEMPLATE,
    VllmToolConfig,
    resolve_vllm_tool_config,
)


def test_gemma4_maps_to_native_parser_template_and_reasoning():
    cfg = resolve_vllm_tool_config("google/gemma-4-26B-A4B-it")
    assert cfg == VllmToolConfig(
        tool_call_parser="gemma4",
        reasoning_parser="gemma4",
        chat_template=GEMMA4_CHAT_TEMPLATE,
    )


@pytest.mark.parametrize(
    "model,parser",
    [
        ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "qwen3_xml"),
        ("Qwen/Qwen3-8B", "hermes"),
        ("Qwen/Qwen2.5-7B-Instruct", "hermes"),
        ("Qwen/QwQ-32B", "hermes"),
        ("meta-llama/Llama-4-Scout-17B-16E-Instruct", "llama4_pythonic"),
        ("meta-llama/Llama-3.1-8B-Instruct", "llama3_json"),
        ("meta-llama/Llama-3.3-70B-Instruct", "llama3_json"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", "mistral"),
        ("mistralai/Devstral-Small-2507", "mistral"),
        ("deepseek-ai/DeepSeek-V3.1", "deepseek_v31"),
        ("deepseek-ai/DeepSeek-V3", "deepseek_v3"),
        ("zai-org/GLM-4.5", "glm45"),
        ("zai-org/GLM-4.7", "glm47"),
        ("ibm-granite/granite-3.3-8b-instruct", "granite"),
        ("moonshotai/Kimi-K2-Instruct", "kimi_k2"),
        ("NousResearch/Hermes-3-Llama-3.1-8B", "hermes"),
    ],
)
def test_known_families_select_expected_parser(model, parser):
    cfg = resolve_vllm_tool_config(model)
    assert cfg is not None, f"expected a tool config for {model}"
    assert cfg.tool_call_parser == parser


def test_qwen3_coder_takes_precedence_over_generic_qwen():
    # The generic "qwen" matcher must not shadow the more specific coder match.
    cfg = resolve_vllm_tool_config("Qwen/Qwen3-Coder-480B-A35B-Instruct")
    assert cfg is not None
    assert cfg.tool_call_parser == "qwen3_xml"


@pytest.mark.parametrize(
    "model",
    [
        "",
        "google/gemma-3-27b-it",  # plain Gemma 3 has no native vLLM parser
        "some-internal/mystery-model-7b",
        "microsoft/phi-4",
    ],
)
def test_unknown_families_return_none(model):
    assert resolve_vllm_tool_config(model) is None


def test_non_gemma4_families_have_no_reasoning_or_chat_template():
    cfg = resolve_vllm_tool_config("Qwen/Qwen3-8B")
    assert cfg is not None
    assert cfg.reasoning_parser == ""
    assert cfg.chat_template == ""
