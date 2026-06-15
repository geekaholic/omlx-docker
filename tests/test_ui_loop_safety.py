# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from tests.ui_loop.safety import (
    SettingsSnapshot,
    HeavyOpRefused,
    is_whitelisted_model,
    assert_heavy_op_allowed,
)

WHITELIST = ("Qwen/Qwen3-1.7B",)


def test_snapshot_restore_round_trips_file(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"a": 1}))
    snap = SettingsSnapshot([f])
    snap.capture()
    f.write_text(json.dumps({"a": 999}))
    snap.restore()
    assert json.loads(f.read_text()) == {"a": 1}


def test_snapshot_restore_deletes_file_created_during_test(tmp_path):
    f = tmp_path / "compose.env"  # absent at capture time
    snap = SettingsSnapshot([f])
    snap.capture()
    f.write_text("VLLM_MODEL=foo")
    snap.restore()
    assert not f.exists()


def test_is_whitelisted_model_matches_exact():
    assert is_whitelisted_model("Qwen/Qwen3-1.7B", WHITELIST) is True
    assert is_whitelisted_model("openai/gpt-oss-120b", WHITELIST) is False


def test_assert_heavy_op_allows_whitelisted():
    assert_heavy_op_allowed("Qwen/Qwen3-1.7B", WHITELIST)  # no raise


def test_assert_heavy_op_refuses_non_whitelisted():
    with pytest.raises(HeavyOpRefused):
        assert_heavy_op_allowed("openai/gpt-oss-120b", WHITELIST)
