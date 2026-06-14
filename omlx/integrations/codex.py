"""Codex (OpenAI Codex CLI) integration."""

from __future__ import annotations

import os
import re

from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_command_prefix


class CodexIntegration(Integration):
    """Codex integration that points Codex at oMLX for a single session.

    Rather than mutating the user's ``~/.codex/config.toml`` (which would
    persist after the session and break later upstream ``codex`` runs that
    expect ``OMLX_API_KEY``), this passes the whole oMLX provider via Codex's
    ``-c key=value`` inline overrides. Codex layers these on top of the
    existing config for that one run only, so the user's default provider,
    model, and per-project trust levels are read and honored, and the override
    evaporates when Codex exits.
    """

    def __init__(self):
        super().__init__(
            name="codex",
            display_name="Codex",
            type="env_var",
            install_check="codex",
            install_hint="npm install -g @openai/codex",
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        return (
            f"{get_cli_command_prefix()} "
            f"launch codex --model {ctx.model or 'select-a-model'}"
        )

    def _is_reasoning(self, ctx: IntegrationContext) -> bool:
        if ctx.reasoning is not None:
            return bool(ctx.reasoning)
        return bool(re.search(r"\b(thinking|o1|o3|r1)\b", ctx.model.lower()))

    def launch(self, ctx: IntegrationContext) -> None:
        env = self._scrubbed_env()
        env["OMLX_API_KEY"] = ctx.auth_token

        # Pass the oMLX provider as ephemeral `-c` overrides. String values are
        # wrapped in literal double quotes so Codex's TOML override parser
        # treats them as strings regardless of `:` / `/` content. No shell is
        # involved (os.execvpe takes an argv list), so the quotes pass verbatim.
        args = ["codex"]
        if ctx.model:
            args.extend(["-m", ctx.model])
        args.extend(
            [
                "-c",
                "model_provider=omlx",
                "-c",
                'model_providers.omlx.name="oMLX"',
                "-c",
                f'model_providers.omlx.base_url="{ctx.openai_base_url}"',
                "-c",
                'model_providers.omlx.env_key="OMLX_API_KEY"',
                "-c",
                'model_providers.omlx.wire_api="responses"',
            ]
        )
        if self._is_reasoning(ctx):
            args.extend(["-c", 'model_reasoning_effort="high"'])
        # Tell Codex the model's real limits. Without these, Codex has no
        # metadata for a custom-provider model ("Model metadata not found"),
        # so it skips auto-compaction and requests a fixed large output that,
        # with a growing prompt, overflows the context window — vLLM then 400s
        # and the response stream dies. Bare ints are parsed as TOML integers.
        if ctx.context_window:
            args.extend(["-c", f"model_context_window={int(ctx.context_window)}"])
        if ctx.max_tokens:
            args.extend(["-c", f"model_max_output_tokens={int(ctx.max_tokens)}"])
        args.extend(ctx.extra_args)

        os.execvpe("codex", args, env)
