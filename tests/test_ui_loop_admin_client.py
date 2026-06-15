# SPDX-License-Identifier: Apache-2.0
from tests.ui_loop.admin_client import diff_settings


def test_diff_settings_reports_changed_leaf():
    before = {"sampling": {"max_tokens": None, "temperature": 0.7}}
    after = {"sampling": {"max_tokens": 2048, "temperature": 0.7}}
    assert diff_settings(before, after) == {"sampling.max_tokens": (None, 2048)}


def test_diff_settings_empty_when_identical():
    d = {"backend": {"type": "vllm"}}
    assert diff_settings(d, dict(d)) == {}


def test_diff_settings_reports_added_and_removed_keys():
    before = {"a": 1}
    after = {"b": 2}
    assert diff_settings(before, after) == {"a": (1, None), "b": (None, 2)}
