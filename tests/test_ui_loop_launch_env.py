# SPDX-License-Identifier: Apache-2.0
from tests.ui_loop.launch_claude_e2e import claude_env


def test_claude_env_sets_anthropic_base_and_token():
    env = claude_env(
        base_url="http://localhost:8080", auth_token="omlx", model="Qwen/Qwen3-1.7B"
    )
    assert env["ANTHROPIC_BASE_URL"] == "http://localhost:8080"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "omlx"
    assert env["ANTHROPIC_API_KEY"] == "omlx"
    assert env["ANTHROPIC_MODEL"] == "Qwen/Qwen3-1.7B"


def test_claude_env_disables_attribution_and_telemetry():
    env = claude_env(base_url="http://x", auth_token="t", model="m")
    assert env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
