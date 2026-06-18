# SPDX-License-Identifier: Apache-2.0
"""Tests for the unified-memory pre-flight fit guard."""

from __future__ import annotations

import json
from pathlib import Path

from omlx.proxy.memory_fit import (
    DEFAULT_UNKNOWN_UTIL,
    auto_utilization,
    estimate_resident_bytes,
    evaluate_fit,
    recommended_utilization,
    resolve_local_model_path,
)

_GIB = 1024**3


def _make_hf_cache_model(root: Path, repo_id: str, weight_bytes: int) -> Path:
    """Create a minimal HF-cache entry with a sparse safetensors shard."""
    encoded = "models--" + repo_id.replace("/", "--")
    commit = "abc123"
    snapshot = root / "hub" / encoded / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"max_position_embeddings": 4096}), encoding="utf-8"
    )
    shard = snapshot / "model.safetensors"
    with open(shard, "wb") as handle:
        handle.truncate(weight_bytes)  # sparse: st_size == weight_bytes
    refs = root / "hub" / encoded / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(commit, encoding="utf-8")
    return snapshot


def test_recommended_utilization_is_host_level():
    total = 122 * _GIB
    reserve = 16 * _GIB
    # The ceiling only reserves the host headroom; it is independent of the
    # weight size. The load-time page-cache transient is reclaimable, so it is
    # NOT subtracted from vLLM's steady-state budget (oversize models are caught
    # by evaluate_fit, not by crushing the util). (122-16)/122 = 0.868 -> 0.86.
    util = recommended_utilization(total, reserve, 50 * _GIB)
    assert 0.85 <= util <= 0.87
    # Same ceiling no matter how large the weights are.
    assert recommended_utilization(total, reserve, 4 * _GIB) == util
    # Unknown footprint -> conservative fallback.
    assert recommended_utilization(total, reserve, None) == DEFAULT_UNKNOWN_UTIL
    # Floored to 2 decimals and clamped.
    assert util <= 0.92


def test_large_model_auto_util_leaves_runtime_and_kv_headroom():
    # Regression for gemma-4-26B on the DGX Spark: ~50 GiB weights on a ~121 GiB
    # unified pool. The old safety ceiling forced util ~0.45, whose budget
    # (~54 GiB) barely cleared the resident weights, so vLLM's ~8.6 GiB runtime
    # peak (activations + CUDA context + multimodal encoder profiling) pushed the
    # KV pool negative and the engine aborted with "No available memory for the
    # cache blocks". The auto util must leave room for weights + runtime overhead
    # + a usable KV pool.
    total, reserve, weights = 121 * _GIB, 16 * _GIB, 50 * _GIB
    vllm_runtime_peak = 8.6 * _GIB  # observed for this VLM at ctx 32768
    util = auto_utilization(
        total_bytes=total,
        reserve_bytes=reserve,
        weights_bytes=weights,
        kv_per_token=393216,
        context_tokens=32768,
        parallel=2,
    )
    budget = util * total
    assert budget - weights - vllm_runtime_peak >= 4 * _GIB


def test_auto_utilization_floor_covers_runtime_overhead():
    # A large model with a negligible workload (tiny context/parallel) has almost
    # no KV demand, so the demand path alone could pick a util whose budget can't
    # even hold the weights + vLLM's runtime overhead. A floor must guarantee a
    # minimum viable budget so the engine still starts with positive KV.
    total, reserve, weights = 121 * _GIB, 16 * _GIB, 50 * _GIB
    vllm_runtime_peak = 8.6 * _GIB
    util = auto_utilization(
        total_bytes=total,
        reserve_bytes=reserve,
        weights_bytes=weights,
        kv_per_token=1024,
        context_tokens=256,
        parallel=1,
    )
    budget = util * total
    assert budget - weights - vllm_runtime_peak > 0


def test_small_model_fits():
    result = evaluate_fit(
        total_bytes=122 * _GIB,
        util=0.83,
        reserve_bytes=16 * _GIB,
        weights_bytes=4 * _GIB,
    )
    assert result.level == "ok"
    assert not result.blocked


def test_oversized_model_blocks_intrinsically():
    # 80 GiB weights: 2*80 + 16 = 176 GiB > 122 GiB total -> unloadable at any util.
    result = evaluate_fit(
        total_bytes=122 * _GIB,
        util=0.20,
        reserve_bytes=16 * _GIB,
        weights_bytes=80 * _GIB,
    )
    assert result.blocked
    assert "too large" in result.reason.lower()


def test_high_util_blocks_on_reserve():
    # Mid model that could fit at a sane util, but 0.95 starves the OS.
    result = evaluate_fit(
        total_bytes=122 * _GIB,
        util=0.95,
        reserve_bytes=16 * _GIB,
        weights_bytes=10 * _GIB,
    )
    assert result.blocked
    assert result.recommended_util < 0.95


def test_large_model_loads_when_transient_reclaimable():
    # 40 GiB weights at 0.80 util: the page-cache "load transient" is reclaimable
    # (clean file pages the kernel drops under pressure), not pinned memory. As
    # long as the budget leaves the OS reserve (Rule A) and the load peak
    # (~2*weights) fits intrinsically, the launch must be allowed, not blocked on
    # a transient that never coexists with the full KV pool.
    result = evaluate_fit(
        total_bytes=122 * _GIB,
        util=0.80,
        reserve_bytes=16 * _GIB,
        weights_bytes=40 * _GIB,
    )
    assert result.level == "ok"
    assert not result.blocked


def test_unknown_model_warns():
    result = evaluate_fit(
        total_bytes=122 * _GIB,
        util=0.80,
        reserve_bytes=16 * _GIB,
        weights_bytes=None,
    )
    assert result.level == "warn"
    assert not result.blocked
    assert result.model_size_known is False


def test_zero_total_warns():
    result = evaluate_fit(
        total_bytes=0,
        util=0.80,
        reserve_bytes=16 * _GIB,
        weights_bytes=10 * _GIB,
    )
    assert result.level == "warn"


def test_resolve_and_estimate_hf_cache(tmp_path):
    snapshot = _make_hf_cache_model(tmp_path, "org/demo-model", 5 * _GIB)
    resolved = resolve_local_model_path("org/demo-model", [tmp_path / "hub"])
    assert resolved == snapshot
    size = estimate_resident_bytes(resolved)
    assert size is not None
    # estimate_model_size adds ~5% overhead.
    assert 5 * _GIB <= size <= 6 * _GIB


def test_resolve_missing_model_returns_none(tmp_path):
    assert resolve_local_model_path("org/not-here", [tmp_path]) is None
    assert resolve_local_model_path("", [tmp_path]) is None


def _write_config(snapshot, **fields):
    (snapshot / "config.json").write_text(json.dumps(fields), encoding="utf-8")


def test_kv_bytes_per_token_reads_config(tmp_path):
    from omlx.proxy.memory_fit import kv_bytes_per_token

    snap = _make_hf_cache_model(tmp_path, "org/m", 1 * _GIB)
    # Qwen3-1.7B geometry: 28 layers, 8 kv heads, head_dim 128, bf16.
    _write_config(
        snap,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        torch_dtype="bfloat16",
        max_position_embeddings=40960,
    )
    assert kv_bytes_per_token(snap) == 2 * 28 * 8 * 128 * 2  # 114688


def test_kv_bytes_per_token_missing_config_returns_none(tmp_path):
    from omlx.proxy.memory_fit import kv_bytes_per_token

    assert kv_bytes_per_token(tmp_path) is None


def test_kv_bytes_per_token_sliding_window_counts_only_full_layers(tmp_path):
    from omlx.proxy.memory_fit import kv_bytes_per_token

    snap = _make_hf_cache_model(tmp_path, "org/gemma4", 1 * _GIB)
    # Gemma-4 geometry: 30 layers, but only 5 full-attention layers grow with
    # context (the other 25 are window-capped sliding-attention). KV per token
    # must be sized off the 5 full layers, not all 30.
    _write_config(
        snap,
        num_hidden_layers=30,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        torch_dtype="bfloat16",
        sliding_window=1024,
        layer_types=["sliding_attention"] * 25 + ["full_attention"] * 5,
        max_position_embeddings=262144,
    )
    assert kv_bytes_per_token(snap) == 2 * 5 * 8 * 256 * 2  # 40960, not 245760


def test_kv_bytes_per_token_all_full_layers_counts_every_layer(tmp_path):
    from omlx.proxy.memory_fit import kv_bytes_per_token

    snap = _make_hf_cache_model(tmp_path, "org/dense", 1 * _GIB)
    # No sliding layers (or no sliding_window) -> every layer grows with context.
    _write_config(
        snap,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        torch_dtype="bfloat16",
        layer_types=["full_attention"] * 8,
        max_position_embeddings=8192,
    )
    assert kv_bytes_per_token(snap) == 2 * 8 * 8 * 256 * 2


def test_sliding_window_unlocks_native_context_for_gemma4(tmp_path):
    """End-to-end: sliding-aware KV lets the auto-context reach the native window.

    Counting all 30 layers (240KB/token) starves the context to ~64K; counting
    only the 5 full layers (40KB/token) fits the model's native 256K on a 121GB
    unified-memory host, which is what Codex should then be told.
    """
    from omlx.proxy.memory_fit import kv_bytes_per_token, recommended_context_length

    snap = _make_hf_cache_model(tmp_path, "org/gemma4", 1 * _GIB)
    _write_config(
        snap,
        num_hidden_layers=30,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        torch_dtype="bfloat16",
        sliding_window=1024,
        layer_types=["sliding_attention"] * 25 + ["full_attention"] * 5,
        max_position_embeddings=262144,
    )
    ctx = recommended_context_length(
        total_bytes=121 * _GIB,
        reserve_bytes=16 * _GIB,
        weights_bytes=52 * _GIB,
        kv_per_token=kv_bytes_per_token(snap),
        parallel=2,
        native_max=262144,
    )
    assert ctx == 262144


def test_demand_utilization_scales_with_workload():
    from omlx.proxy.memory_fit import demand_utilization

    kw = dict(
        total_bytes=122 * _GIB,
        reserve_bytes=16 * _GIB,
        weights_bytes=3 * _GIB,
        kv_per_token=114688,
    )
    small = demand_utilization(context_tokens=40960, parallel=2, **kw)
    big = demand_utilization(context_tokens=40960, parallel=4, **kw)
    assert small is not None and big is not None
    # More parallelism needs a bigger KV pool -> higher util, far below 0.83.
    assert small < big < 0.40
    assert demand_utilization(context_tokens=0, parallel=2, **kw) is None


def test_auto_utilization_is_min_of_safety_and_demand():
    from omlx.proxy.memory_fit import auto_utilization, recommended_utilization

    total, reserve = 122 * _GIB, 16 * _GIB
    # Small model: demand caps well below the safety ceiling.
    small = auto_utilization(
        total_bytes=total,
        reserve_bytes=reserve,
        weights_bytes=3 * _GIB,
        kv_per_token=114688,
        context_tokens=40960,
        parallel=2,
    )
    safety = recommended_utilization(total, reserve, 3 * _GIB)
    assert small < safety
    # Without KV geometry it falls back to the safety ceiling.
    assert (
        auto_utilization(
            total_bytes=total, reserve_bytes=reserve, weights_bytes=3 * _GIB
        )
        == safety
    )


def test_kv_bytes_per_token_reads_nested_text_config(tmp_path):
    # Multimodal models (Gemma-4 VLM) keep the LM geometry under text_config.
    from omlx.proxy.memory_fit import kv_bytes_per_token

    snap = _make_hf_cache_model(tmp_path, "org/vlm", 1 * _GIB)
    _write_config(
        snap,
        model_type="gemma4",
        text_config={
            "num_hidden_layers": 30,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "torch_dtype": "bfloat16",
        },
    )
    assert kv_bytes_per_token(snap) == 2 * 30 * 8 * 256 * 2  # 245760


def test_recommended_context_length_bounded_by_native():
    from omlx.proxy.memory_fit import recommended_context_length

    # Plenty of memory, small native window -> native bounds the result.
    ctx = recommended_context_length(
        total_bytes=200 * _GIB,
        reserve_bytes=16 * _GIB,
        weights_bytes=10 * _GIB,
        kv_per_token=100_000,
        parallel=1,
        native_max=8192,
        headroom=1.0,
    )
    assert ctx == 8192


def test_recommended_context_length_scales_inversely_with_parallel():
    from omlx.proxy.memory_fit import recommended_context_length

    kw = dict(
        total_bytes=120 * _GIB,
        reserve_bytes=16 * _GIB,
        weights_bytes=50 * _GIB,
        kv_per_token=240 * 1024,
        native_max=262144,
        headroom=1.5,
    )
    c1 = recommended_context_length(parallel=1, **kw)
    c2 = recommended_context_length(parallel=2, **kw)
    assert c1 and c2 and c1 > c2
    assert c1 % 4096 == 0 and c2 % 4096 == 0


def test_recommended_context_length_honors_cap():
    from omlx.proxy.memory_fit import MAX_AUTO_CONTEXT_CAP, recommended_context_length

    capped = recommended_context_length(
        total_bytes=10_000 * _GIB,
        reserve_bytes=16 * _GIB,
        weights_bytes=10 * _GIB,
        kv_per_token=1024,
        parallel=1,
        native_max=10**9,
        headroom=1.0,
    )
    assert capped == MAX_AUTO_CONTEXT_CAP


def test_recommended_context_length_none_when_unknown_or_too_big():
    from omlx.proxy.memory_fit import recommended_context_length

    base = dict(
        total_bytes=120 * _GIB,
        reserve_bytes=16 * _GIB,
        weights_bytes=50 * _GIB,
        kv_per_token=240 * 1024,
        parallel=2,
        native_max=262144,
    )
    assert recommended_context_length(**{**base, "kv_per_token": None}) is None
    assert recommended_context_length(**{**base, "native_max": None}) is None
    # Weights exceed the budget -> no KV room -> None (caller keeps fallback).
    assert recommended_context_length(**{**base, "weights_bytes": 200 * _GIB}) is None
