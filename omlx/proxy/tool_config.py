# SPDX-License-Identifier: Apache-2.0
"""Per-model tool-calling configuration for backend sidecars.

vLLM needs ``--enable-auto-tool-choice`` plus a model-appropriate
``--tool-call-parser`` (and sometimes ``--reasoning-parser`` / ``--chat-template``)
to turn a model's native tool-call syntax into structured ``tool_calls``. No single
parser works for every family, so this module maps a model id to the right vLLM
parser and turns tool calling on automatically. Unknown families return ``None``:
tool parsing is left off, because vLLM refuses to start when
``--enable-auto-tool-choice`` is passed without a valid parser.

llama.cpp needs none of this — its ``--jinja`` flag (on by default) drives tool
calling straight from the model's embedded chat template.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from omlx.utils.tokenizer import is_gemma4_model

# In-container path to vLLM's bundled Gemma-4 tool chat template. The
# ``vllm/vllm-openai`` image ships its examples under ``/vllm-workspace/examples``;
# the launch shell skips ``--chat-template`` when the file is absent so a missing
# file degrades to the model's default template instead of crashing startup.
GEMMA4_CHAT_TEMPLATE = "/vllm-workspace/examples/tool_chat_template_gemma4.jinja"

# Sentinel values for ``tool_call_parser`` meaning "detect from the model".
AUTO_VALUES = frozenset({"", "auto"})
# Values that explicitly disable tool-call parsing.
OFF_VALUES = frozenset({"none", "off", "disabled", "false", "no"})


@dataclass(frozen=True)
class VllmToolConfig:
    """vLLM tool-calling flags for a model family."""

    tool_call_parser: str
    reasoning_parser: str = ""
    chat_template: str = ""


# Tool-call parsers whose family also emits reasoning in a matching format, so
# the reasoning parser should be paired automatically — even when the user picks
# the tool parser explicitly. Without this, Gemma-4's `<|channel>thought ...
# <channel|>` reasoning markers leak into the visible output.
_REASONING_FOR_TOOL_PARSER = {"gemma4": "gemma4"}


def reasoning_parser_for(tool_call_parser: str) -> str:
    """Matched vLLM ``--reasoning-parser`` for a tool parser, or "" if none."""
    return _REASONING_FOR_TOOL_PARSER.get((tool_call_parser or "").strip(), "")


def _has(*needles: str) -> Callable[[str], bool]:
    return lambda name: any(needle in name for needle in needles)


# Ordered most-specific first; the first matching family wins. Names are matched
# against the lower-cased model id. Parser names must match those accepted by the
# pinned ``vllm/vllm-openai`` image's ``--tool-call-parser`` flag.
_REGISTRY: list[tuple[Callable[[str], bool], VllmToolConfig]] = [
    # Hermes/Nous fine-tunes carry their base model's name (often Llama/Qwen) but
    # emit the Hermes <tool_call> format, so match the "hermes" signal first.
    (_has("hermes"), VllmToolConfig("hermes")),
    # Qwen3-Coder emits XML tool calls (distinct from the Qwen instruct models).
    (_has("qwen3-coder", "qwen3coder"), VllmToolConfig("qwen3_xml")),
    # Llama 4 — pythonic tool calls.
    (_has("llama-4", "llama4"), VllmToolConfig("llama4_pythonic")),
    # Llama 3.1 / 3.2 / 3.3 — JSON tool calls.
    (
        _has(
            "llama-3.1",
            "llama-3.2",
            "llama-3.3",
            "llama3.1",
            "llama3.2",
            "llama3.3",
        ),
        VllmToolConfig("llama3_json"),
    ),
    # DeepSeek V3.1 then V3 (more specific first).
    (_has("deepseek-v3.1", "deepseek-v31"), VllmToolConfig("deepseek_v31")),
    (_has("deepseek-v3", "deepseek-v2.5"), VllmToolConfig("deepseek_v3")),
    # GLM 4.7 then 4.5/4.6.
    (_has("glm-4.7", "glm4.7"), VllmToolConfig("glm47")),
    (_has("glm-4.5", "glm-4.6", "glm4.5", "glm4.6"), VllmToolConfig("glm45")),
    # Mistral family (incl. Mixtral, Devstral, Ministral, Magistral).
    (
        _has("mistral", "mixtral", "devstral", "ministral", "magistral"),
        VllmToolConfig("mistral"),
    ),
    # IBM Granite.
    (_has("granite"), VllmToolConfig("granite")),
    # Moonshot Kimi K2.
    (_has("kimi-k2", "kimi_k2"), VllmToolConfig("kimi_k2")),
    # The Qwen instruct line (Qwen2.5 / QwQ / Qwen3) uses the Hermes-style parser.
    # Keep this last so the more specific Qwen3-Coder match above takes precedence.
    (
        _has("qwq", "qwen2.5", "qwen-2.5", "qwen3", "qwen"),
        VllmToolConfig("hermes"),
    ),
]


def resolve_vllm_tool_config(model: str) -> VllmToolConfig | None:
    """Pick vLLM tool-calling flags for ``model`` by family, or ``None``.

    Matching is name-based: at serve time the model may not be downloaded yet, so
    ``config.json`` is not reliably available. Returns ``None`` for families vLLM
    cannot parse natively, in which case the caller must leave tool parsing off.
    """
    name = (model or "").strip().lower()
    if not name:
        return None

    # Gemma 4 — vLLM ships a dedicated gemma4 tool + reasoning parser and a tool
    # chat template (per the official Gemma 4 vLLM recipe).
    if is_gemma4_model(model):
        return VllmToolConfig(
            tool_call_parser="gemma4",
            reasoning_parser="gemma4",
            chat_template=GEMMA4_CHAT_TEMPLATE,
        )

    for matcher, config in _REGISTRY:
        if matcher(name):
            return config
    return None
