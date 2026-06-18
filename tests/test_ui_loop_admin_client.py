# SPDX-License-Identifier: Apache-2.0
from tests.ui_loop.admin_client import diff_settings, flat_settings_key


def test_flat_settings_key_section_field():
    # The admin GET payload nests display values, but POST expects FLAT keys.
    assert flat_settings_key("sampling.temperature") == "sampling_temperature"
    assert flat_settings_key("network.http_proxy") == "network_http_proxy"
    assert flat_settings_key("huggingface.endpoint") == "huggingface_endpoint"


def test_flat_settings_key_vllm_display_paths():
    # vLLM advanced settings live under proxy.sidecar in GET but POST as vllm_*.
    assert flat_settings_key("vllm.dtype") == "vllm_dtype"
    assert flat_settings_key("vllm.gpu_memory_utilization") == "vllm_gpu_memory_utilization"


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
