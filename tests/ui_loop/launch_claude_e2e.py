# SPDX-License-Identifier: Apache-2.0
"""End-to-end probe: drive `claude -p` through the proxy like `omni launch claude`."""

from __future__ import annotations

import os
import subprocess


def claude_env(base_url: str, auth_token: str, model: str) -> dict[str, str]:
    """Mirror omlx/integrations/claude.py's env wiring for an automated probe."""
    env = dict(os.environ)
    env["ANTHROPIC_BASE_URL"] = base_url
    env["ANTHROPIC_AUTH_TOKEN"] = auth_token
    env["ANTHROPIC_API_KEY"] = auth_token
    env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    env["API_TIMEOUT_MS"] = "3000000"
    env["ANTHROPIC_MODEL"] = model
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    env["ANTHROPIC_SMALL_FAST_MODEL"] = model
    return env


def run_claude_ping(
    base_url: str, auth_token: str, model: str, timeout: int = 300
) -> tuple[bool, str]:
    """Run `claude -p 'reply with the single word: pong'`; return (ok, output)."""
    env = claude_env(base_url, auth_token, model)
    try:
        proc = subprocess.run(
            ["claude", "-p", "Reply with exactly the single word: pong"],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return False, "claude CLI not found on PATH"
    except subprocess.TimeoutExpired:
        return False, f"claude timed out after {timeout}s"
    out = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0 and "pong" in out.lower()
    return ok, out.strip()
