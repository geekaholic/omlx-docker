# SPDX-License-Identifier: Apache-2.0
from tests.ui_loop.backend_assert import vllm_metric_families, served_model_listed

SAMPLE = """\
# HELP vllm:prefix_cache_hits_total ...
vllm:prefix_cache_hits_total 12.0
vllm:prefix_cache_queries_total 19.0
vllm:kv_cache_usage_perc 0.04
"""


def test_vllm_metric_families_extracts_names():
    fams = vllm_metric_families(SAMPLE)
    assert "vllm:prefix_cache_hits_total" in fams
    assert "vllm:kv_cache_usage_perc" in fams


def test_served_model_listed_true_when_present():
    models = {"data": [{"id": "Qwen/Qwen3-1.7B"}]}
    assert served_model_listed(models, "Qwen/Qwen3-1.7B") is True
    assert served_model_listed(models, "other") is False
