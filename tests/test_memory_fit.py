# SPDX-License-Identifier: Apache-2.0
"""Tests for the unified-memory pre-flight fit guard."""

from __future__ import annotations

import json
from pathlib import Path

from omlx.proxy.memory_fit import (
    DEFAULT_UNKNOWN_UTIL,
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


def test_recommended_utilization_is_model_aware():
    total = 122 * _GIB
    reserve = 16 * _GIB
    # 4 GiB model -> high util; 80 GiB model -> very low; unknown -> fallback.
    assert recommended_utilization(total, reserve, 4 * _GIB) > 0.80
    assert recommended_utilization(total, reserve, 80 * _GIB) < 0.30
    assert recommended_utilization(total, reserve, None) == DEFAULT_UNKNOWN_UTIL
    # Floored to 2 decimals and clamped.
    assert recommended_utilization(total, reserve, 4 * _GIB) <= 0.92


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


def test_util_plus_transient_blocks():
    # 40 GiB weights fit intrinsically (2*40+16=96<122) but a 0.80 budget
    # (97.6 GiB) + 40 GiB transient + 16 GiB reserve overruns 122 GiB.
    result = evaluate_fit(
        total_bytes=122 * _GIB,
        util=0.80,
        reserve_bytes=16 * _GIB,
        weights_bytes=40 * _GIB,
    )
    assert result.blocked
    # The same model at the recommended util is OK.
    ok = evaluate_fit(
        total_bytes=122 * _GIB,
        util=result.recommended_util,
        reserve_bytes=16 * _GIB,
        weights_bytes=40 * _GIB,
    )
    assert ok.level == "ok"


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
