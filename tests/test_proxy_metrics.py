# SPDX-License-Identifier: Apache-2.0
"""Tests for backend Prometheus metric selection (vLLM, llama.cpp)."""

from omlx.proxy.metrics import (
    parse_prometheus_text,
    select_prometheus_metrics,
    summarize_selected_metrics,
)

_VLLM_V1_TEXT = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
vllm:num_requests_running{model_name="qwen"} 1.0
vllm:num_requests_waiting{model_name="qwen"} 2.0
vllm:prompt_tokens_total{model_name="qwen"} 4000.0
vllm:generation_tokens_total{model_name="qwen"} 900.0
vllm:gpu_cache_usage_perc{model_name="qwen"} 0.42
vllm:gpu_prefix_cache_queries_total{model_name="qwen"} 1000.0
vllm:gpu_prefix_cache_hits_total{model_name="qwen"} 250.0
"""

_VLLM_V0_TEXT = """\
vllm:num_requests_running{model_name="qwen"} 0.0
vllm:gpu_prefix_cache_hit_rate{model_name="qwen"} 0.4
"""

# Observed on vllm/vllm-openai:latest (June 2026, DGX Spark): the gpu_
# prefix is gone and external_* families must not be double-counted.
_VLLM_2026_TEXT = """\
vllm:num_requests_running{engine="0",model_name="gemma"} 1.0
vllm:kv_cache_usage_perc{engine="0",model_name="gemma"} 0.37
vllm:prefix_cache_queries_total{engine="0",model_name="gemma"} 800.0
vllm:prefix_cache_hits_total{engine="0",model_name="gemma"} 200.0
vllm:external_prefix_cache_queries_total{engine="0",model_name="gemma"} 999.0
vllm:external_prefix_cache_hits_total{engine="0",model_name="gemma"} 999.0
"""

_LLAMACPP_TEXT = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
llamacpp:prompt_tokens_total 1024.0
llamacpp:tokens_predicted_total 256.0
llamacpp:prompt_tokens_seconds 120.5
llamacpp:predicted_tokens_seconds 35.2
llamacpp:kv_cache_usage_ratio 0.25
llamacpp:kv_cache_tokens 2048.0
llamacpp:requests_processing 1.0
llamacpp:requests_deferred 3.0
"""


def _selected(text):
    return select_prometheus_metrics(parse_prometheus_text(text))


def test_vllm_v1_prefix_cache_counters_selected():
    selected = _selected(_VLLM_V1_TEXT)
    assert selected["prefix_cache_queries"] == 1000.0
    assert selected["prefix_cache_hits"] == 250.0


def test_vllm_v1_summary_computes_prefix_hit_rate():
    summary = summarize_selected_metrics(_selected(_VLLM_V1_TEXT))
    assert summary["prefix_cache_hit_rate"] == 25.0
    assert summary["gpu_cache_usage_perc"] == 0.42


def test_vllm_v0_hit_rate_gauge_fallback():
    summary = summarize_selected_metrics(_selected(_VLLM_V0_TEXT))
    assert summary["prefix_cache_hit_rate"] == 40.0


def test_vllm_2026_renamed_metrics_selected_without_external_families():
    summary = summarize_selected_metrics(_selected(_VLLM_2026_TEXT))
    assert summary["prefix_cache_queries"] == 800.0
    assert summary["prefix_cache_hits"] == 200.0
    assert summary["prefix_cache_hit_rate"] == 25.0
    assert summary["gpu_cache_usage_perc"] == 0.37


def test_llamacpp_metrics_selected():
    selected = _selected(_LLAMACPP_TEXT)
    assert selected["prompt_tokens_total"] == 1024.0
    assert selected["generation_tokens_total"] == 256.0
    assert selected["running_requests"] == 1.0
    assert selected["waiting_requests"] == 3.0
    assert selected["kv_cache_usage_ratio"] == 0.25
    assert selected["kv_cache_tokens"] == 2048.0
    assert selected["prompt_tokens_seconds"] == 120.5
    assert selected["predicted_tokens_seconds"] == 35.2


def test_llamacpp_summary_exposes_kv_cache():
    summary = summarize_selected_metrics(_selected(_LLAMACPP_TEXT))
    assert summary["kv_cache_usage_ratio"] == 0.25
    assert summary["kv_cache_tokens"] == 2048.0
    assert summary["prompt_tokens_seconds"] == 120.5
    assert summary["predicted_tokens_seconds"] == 35.2
    assert summary["prefix_cache_hit_rate"] is None


# Current llama.cpp builds (b9570+) dropped the kv_cache_* families; the
# context high-water mark is the remaining occupancy signal.
_LLAMACPP_MODERN_TEXT = """\
llamacpp:prompt_tokens_total 38.0
llamacpp:tokens_predicted_total 3336.0
llamacpp:n_tokens_max 3373.0
llamacpp:prompt_tokens_seconds 1.22
llamacpp:predicted_tokens_seconds 35.76
llamacpp:requests_processing 0.0
llamacpp:requests_deferred 0.0
"""


def test_llamacpp_modern_build_without_kv_cache_metrics():
    summary = summarize_selected_metrics(_selected(_LLAMACPP_MODERN_TEXT))
    assert summary["kv_cache_usage_ratio"] is None
    assert summary["kv_cache_tokens"] is None
    assert summary["context_tokens_peak"] == 3373.0
    assert summary["prompt_tokens_seconds"] == 1.22
    assert summary["predicted_tokens_seconds"] == 35.76


def test_host_memory_info_parses_meminfo(tmp_path):
    from omlx.proxy.metrics import host_memory_info

    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       127600524 kB\n"
        "MemFree:        99159460 kB\n"
        "MemAvailable:   111584904 kB\n"
    )
    info = host_memory_info(str(meminfo))
    assert info["total_bytes"] == 127600524 * 1024
    assert info["available_bytes"] == 111584904 * 1024


def test_host_memory_info_zero_when_unreadable(tmp_path):
    from omlx.proxy.metrics import host_memory_info

    info = host_memory_info(str(tmp_path / "missing"))
    assert info == {"total_bytes": 0, "available_bytes": 0}
