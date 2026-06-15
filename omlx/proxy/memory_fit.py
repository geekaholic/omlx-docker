# SPDX-License-Identifier: Apache-2.0
"""Pre-flight memory-fit checks for vLLM sidecar launches on unified memory.

On a discrete GPU ``--gpu-memory-utilization`` is a fraction of separate VRAM,
and overshooting just fails a CUDA allocation cleanly. On unified-memory hosts
(DGX Spark GB10) the GPU and CPU share one pool, the driver reports the *whole*
pool as "GPU memory", and an overshoot eats the RAM the kernel needs — the OOM
killer cannot reclaim driver-pinned allocations, so the box hard-locks.

This module estimates whether a model can load within the configured budget and
still leave an absolute reserve for the OS, Docker, the proxy, and the
page-cache transient of streaming the weight files. It reuses the MLX-free
helpers in :mod:`omlx.model_discovery` and ``metrics.host_memory_info`` so it
works both on the host (``omni serve``) and inside the proxy container (admin).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from omlx.model_discovery import _resolve_hf_cache_entry, estimate_model_size

from .metrics import host_memory_info

# Absolute host headroom left outside vLLM's budget: OS, Docker, the proxy
# container, and steady-state slack.
DEFAULT_HOST_RESERVE_GB = 16.0

# Conservative utilization to suggest when the model footprint can't be
# measured (model not cached yet) — lower than vLLM's discrete-GPU 0.80.
DEFAULT_UNKNOWN_UTIL = 0.70

# vLLM fills its whole gpu-memory-utilization budget with KV cache regardless
# of model size. For small models we instead size the KV pool to what the
# workload can use (max_parallel x context) plus this multiplier of slack for
# prefix caching / bursts, then take the lower of that and the safety ceiling.
DEFAULT_KV_HEADROOM = 1.5

# Allowance for vLLM's non-weight runtime peak on top of weights + KV: the CUDA
# context, activations, and the profiling forward pass. This is much larger than
# plain activations for multimodal models, whose vision encoder is profiled with
# many image/video items (a Gemma-4 VLM at ctx 32768 peaked ~8.6 GiB above the
# resident weights). Under-sizing it lets the chosen util's budget come out below
# weights + overhead, leaving the KV pool negative.
ACTIVATION_HEADROOM_GB = 8.0

# Smallest KV pool worth launching with. The auto utilization is floored so its
# budget always clears weights + runtime overhead + this much KV — otherwise
# vLLM aborts with "No available memory for the cache blocks".
MIN_KV_GB = 1.0

# Upper bound on the auto-computed context window (env ``OMLX_MAX_AUTO_CONTEXT``).
# Memory usually binds well below this; the cap just avoids pathological prefill
# latency when a model declares a huge native window (e.g. 1M tokens).
MAX_AUTO_CONTEXT_CAP = 262144

# Auto context is floored to a multiple of this (clean vLLM block sizing).
_CONTEXT_GRANULARITY = 4096

_GIB = 1024**3


@dataclass(frozen=True)
class FitResult:
    """Outcome of a memory-fit evaluation.

    ``level`` is one of ``ok`` / ``warn`` / ``block``. ``warn`` is used when the
    footprint cannot be verified (model not cached locally yet) — the launch is
    allowed but flagged. ``block`` means the launch is predicted to exhaust
    memory and should be refused unless explicitly overridden.
    """

    level: str
    reason: str
    total_bytes: int
    budget_bytes: int
    weights_bytes: int
    reserve_bytes: int
    recommended_util: float
    model_size_known: bool

    @property
    def ok(self) -> bool:
        return self.level == "ok"

    @property
    def blocked(self) -> bool:
        return self.level == "block"

    def as_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "reason": self.reason,
            "total_bytes": self.total_bytes,
            "total_formatted": format_gib(self.total_bytes),
            "budget_bytes": self.budget_bytes,
            "budget_formatted": format_gib(self.budget_bytes),
            "weights_bytes": self.weights_bytes,
            "weights_formatted": (
                format_gib(self.weights_bytes) if self.model_size_known else "unknown"
            ),
            "reserve_bytes": self.reserve_bytes,
            "reserve_formatted": format_gib(self.reserve_bytes),
            "recommended_util": round(self.recommended_util, 3),
            "model_size_known": self.model_size_known,
        }


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / _GIB:.1f} GiB"


def host_reserve_bytes() -> int:
    """Absolute OS reserve in bytes (``OMLX_HOST_MEMORY_RESERVE_GB`` override)."""
    raw = os.getenv("OMLX_HOST_MEMORY_RESERVE_GB", "").strip()
    reserve_gb = DEFAULT_HOST_RESERVE_GB
    if raw:
        try:
            parsed = float(raw)
            if parsed >= 0:
                reserve_gb = parsed
        except ValueError:
            pass
    return int(reserve_gb * _GIB)


def guard_disabled() -> bool:
    return os.getenv("OMLX_SKIP_MEMORY_GUARD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def recommended_utilization(
    total_bytes: int, reserve_bytes: int, weights_bytes: int | None = None
) -> float:
    """Largest safe gpu-memory-utilization for this host.

    vLLM loads the weights *inside* its ``util*total`` budget (the pool also
    holds activations and the KV cache), so the only hard ceiling is that the
    pinned budget must leave the host reserve for the OS/Docker/proxy:

        budget (util*total) + reserve <= total   =>   util <= (total-reserve)/total

    Earlier this also subtracted the weights a second time to reserve room for
    the load-time page-cache copy of the weight files. That copy is *reclaimable*
    page cache, not pinned memory, and subtracting it double-counted the weights
    (which already live in the budget) — for a model whose weights are a large
    fraction of unified memory it dragged the ceiling *below* the budget vLLM
    needs for weights + runtime overhead + KV, so the engine aborted with
    "No available memory for the cache blocks". The genuine load-time peak
    (~2x weights, resident + page cache) is guarded separately by
    :func:`evaluate_fit` (the intrinsic ``2*weights + reserve <= total`` rule).

    ``weights_bytes`` is now only used to distinguish a measurable model from an
    uncached one: when the footprint is unknown we fall back to a fixed
    conservative value. Floored to 2 decimals so the emitted value never rounds
    back up past the safe limit.
    """
    if total_bytes <= 0:
        return DEFAULT_UNKNOWN_UTIL
    if weights_bytes is None or weights_bytes <= 0:
        return DEFAULT_UNKNOWN_UTIL
    util = (total_bytes - reserve_bytes) / total_bytes
    util = math.floor(util * 100) / 100
    return max(0.10, min(0.92, util))


def kv_headroom() -> float:
    """KV slack multiplier above the strict max (OMLX_KV_HEADROOM override)."""
    raw = os.getenv("OMLX_KV_HEADROOM", "").strip()
    if raw:
        try:
            parsed = float(raw)
            if parsed >= 1.0:
                return parsed
        except ValueError:
            pass
    return DEFAULT_KV_HEADROOM


def _dtype_bytes(torch_dtype: object) -> int:
    name = str(torch_dtype or "").lower()
    if "fp8" in name or "float8" in name or "int8" in name:
        return 1
    if "32" in name:  # float32 / bfloat32-ish
        return 4
    return 2  # bfloat16 / float16 / half (default)


def _nested_get(config: dict, key: str) -> object:
    """Read ``key`` from config.json, falling back to the multimodal text config.

    Multimodal models (e.g. Gemma-4 VLMs) keep the language-model geometry under
    ``text_config`` / ``language_config``, so a top-level-only read misses it and
    KV sizing silently degrades to the safety ceiling.
    """
    if key in config:
        return config[key]
    for nest in ("text_config", "language_config"):
        sub = config.get(nest)
        if isinstance(sub, dict) and key in sub:
            return sub[key]
    return None


def _growing_layer_count(config: dict, total_layers: int) -> int:
    """How many layers' KV actually grows with context length.

    Sliding-window models (e.g. Gemma-4: ``layer_types`` is 25 ``sliding_attention``
    + 5 ``full_attention``) only grow KV on their full-attention layers; the sliding
    layers are capped at ``sliding_window`` tokens, so their KV is ~constant and
    negligible for context sizing. Counting all 30 layers there overstates per-token
    KV ~6x and starves the auto-context. Returns the full-attention count when the
    config advertises a genuine sliding/full mix, else ``total_layers``.
    """
    layer_types = _nested_get(config, "layer_types")
    sliding_window = _nested_get(config, "sliding_window")
    if not isinstance(layer_types, (list, tuple)) or not sliding_window:
        return total_layers
    full = sum(1 for t in layer_types if isinstance(t, str) and "full" in t.lower())
    sliding = sum(1 for t in layer_types if isinstance(t, str) and "slid" in t.lower())
    # Trust the split only when it's a real mix (some sliding *and* some full).
    return full if full and sliding else total_layers


def kv_bytes_per_token(model_path: Path) -> int | None:
    """Bytes of KV cache one token occupies: 2 * layers * kv_heads * head_dim * dtype.

    Reads the model's ``config.json`` (incl. nested ``text_config``); returns None
    when it can't be parsed so callers fall back to the safety-ceiling utilization.
    For sliding-window models only the full-attention layers count (see
    :func:`_growing_layer_count`) — the sliding layers' KV doesn't grow with context.
    """
    try:
        import json

        config = json.loads(
            (Path(model_path) / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None

    layers = _nested_get(config, "num_hidden_layers") or _nested_get(config, "n_layers")
    heads = _nested_get(config, "num_attention_heads") or _nested_get(config, "n_heads")
    kv_heads = _nested_get(config, "num_key_value_heads") or heads
    head_dim = _nested_get(config, "head_dim")
    hidden = _nested_get(config, "hidden_size")
    if not head_dim and hidden and heads:
        head_dim = hidden // heads
    if not layers or not kv_heads or not head_dim:
        return None

    layers = _growing_layer_count(config, int(layers))
    dtype_bytes = _dtype_bytes(_nested_get(config, "torch_dtype"))
    return int(2 * layers * kv_heads * head_dim * dtype_bytes)


def demand_utilization(
    *,
    total_bytes: int,
    reserve_bytes: int,
    weights_bytes: int,
    kv_per_token: int,
    context_tokens: int,
    parallel: int,
    headroom: float | None = None,
) -> float | None:
    """Utilization that fits weights + the KV the workload can actually use.

    The most KV that can ever be resident is ``parallel * context`` tokens;
    we size for that times a headroom multiplier (prefix-cache / burst slack),
    plus weights and a fixed activation allowance. Rounded **up** to 2 decimals
    so vLLM never gets less than it needs. Returns None when inputs are missing.
    """
    if total_bytes <= 0 or kv_per_token <= 0 or context_tokens <= 0 or parallel <= 0:
        return None
    mult = kv_headroom() if headroom is None else headroom
    kv_pool = kv_per_token * context_tokens * parallel * mult
    activation = ACTIVATION_HEADROOM_GB * _GIB
    demand = (weights_bytes + kv_pool + activation) / total_bytes
    demand = math.ceil(demand * 100) / 100
    return max(0.05, min(0.92, demand))


def max_auto_context() -> int:
    """Cap for the auto-computed context (``OMLX_MAX_AUTO_CONTEXT`` override)."""
    raw = os.getenv("OMLX_MAX_AUTO_CONTEXT", "").strip()
    if raw:
        try:
            parsed = int(raw)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return MAX_AUTO_CONTEXT_CAP


def recommended_context_length(
    *,
    total_bytes: int,
    reserve_bytes: int,
    weights_bytes: int | None,
    kv_per_token: int | None,
    parallel: int | None,
    native_max: int | None,
    headroom: float | None = None,
) -> int | None:
    """Largest context that fits memory, bounded by the model's native window.

    The inverse of :func:`demand_utilization`: size the context so weights + the
    KV it can use (``context * parallel`` tokens, with the same headroom slack) +
    the activation allowance just fit the host budget, then clamp to the model's
    native max and the global cap. Floored to a clean block multiple. Returns
    None when the geometry/weights/native window are unknown — the caller then
    keeps its existing (conservative) context and recomputes once the model is
    cached.
    """
    if (
        not total_bytes
        or not weights_bytes
        or not kv_per_token
        or not parallel
        or not native_max
    ):
        return None
    if total_bytes <= 0 or kv_per_token <= 0 or parallel <= 0 or native_max <= 0:
        return None
    mult = kv_headroom() if headroom is None else headroom
    kv_budget = (
        (total_bytes - reserve_bytes)
        - weights_bytes
        - int(ACTIVATION_HEADROOM_GB * _GIB)
    )
    if kv_budget <= 0:
        return None
    max_fit = int(kv_budget / (kv_per_token * parallel * mult))
    context = min(int(native_max), max_fit, max_auto_context())
    floored = (context // _CONTEXT_GRANULARITY) * _CONTEXT_GRANULARITY
    return floored or context


def auto_utilization(
    *,
    total_bytes: int,
    reserve_bytes: int,
    weights_bytes: int | None,
    kv_per_token: int | None = None,
    context_tokens: int | None = None,
    parallel: int | None = None,
    headroom: float | None = None,
) -> float:
    """Safe, demand-aware utilization: ``min(safety ceiling, demand)``, floored.

    vLLM fills its whole budget with KV regardless of model size, so for small
    models we cap the budget to what the workload can use; large models hit the
    safety ceiling. Falls back to the safety ceiling when the KV geometry or the
    workload shape is unknown.

    A floor then guarantees the chosen budget can still hold the weights plus
    vLLM's runtime overhead plus a minimal KV pool (``MIN_KV_GB``): a large model
    with a tiny workload has almost no KV demand, so ``demand`` alone could pick a
    budget that can't even fit weights + overhead, and the engine would abort. The
    floor is capped by the safety ceiling — if even that can't satisfy it the
    model is genuinely too big and :func:`evaluate_fit` blocks it.
    """
    safety = recommended_utilization(total_bytes, reserve_bytes, weights_bytes)
    if (
        weights_bytes is None
        or kv_per_token is None
        or context_tokens is None
        or parallel is None
    ):
        return safety
    demand = demand_utilization(
        total_bytes=total_bytes,
        reserve_bytes=reserve_bytes,
        weights_bytes=weights_bytes,
        kv_per_token=kv_per_token,
        context_tokens=context_tokens,
        parallel=parallel,
        headroom=headroom,
    )
    util = safety if demand is None else min(safety, demand)
    return _apply_runtime_floor(util, total_bytes, weights_bytes, safety)


def _apply_runtime_floor(
    util: float, total_bytes: int, weights_bytes: int, safety: float
) -> float:
    """Raise ``util`` so its budget clears weights + runtime overhead + min KV.

    Capped by the safety ceiling so the floor can never push the budget past the
    host reserve. Returns ``util`` unchanged when the floor can't be computed.
    """
    if total_bytes <= 0 or weights_bytes <= 0:
        return util
    needed = weights_bytes + (ACTIVATION_HEADROOM_GB + MIN_KV_GB) * _GIB
    floor = math.ceil(needed / total_bytes * 100) / 100
    return max(util, min(safety, floor))


def resolve_local_model_path(model_id: str, scan_dirs: list[Path | str]) -> Path | None:
    """Resolve an HF repo id or local path to its on-disk snapshot directory.

    Returns None when the model is not present locally (e.g. it will be
    downloaded on first launch) — callers treat that as an unverifiable
    footprint rather than a hard failure.
    """
    if not model_id:
        return None

    candidate = Path(os.path.expanduser(model_id))
    if candidate.is_dir() and (candidate / "config.json").exists():
        return candidate

    encoded = "models--" + model_id.replace("/", "--")
    for raw in scan_dirs:
        if not raw:
            continue
        base = Path(os.path.expanduser(str(raw)))
        for cache_dir in (base / encoded, base / "hub" / encoded):
            if cache_dir.is_dir():
                entry = _resolve_hf_cache_entry(cache_dir)
                if entry is not None:
                    return entry.snapshot_path
    return None


def estimate_resident_bytes(model_path: Path) -> int | None:
    """Weight footprint ≈ on-disk size; None when no weights are found."""
    try:
        return estimate_model_size(model_path)
    except (ValueError, OSError):
        return None


def evaluate_fit(
    *,
    total_bytes: int,
    util: float,
    reserve_bytes: int,
    weights_bytes: int | None,
    kv_bytes_per_token: int | None = None,
    context_tokens: int | None = None,
    parallel: int | None = None,
) -> FitResult:
    """Decide whether a model fits under the configured vLLM budget.

    On unified memory vLLM pins ``util*total`` (weights + activations + KV) at
    steady state. Loading transiently holds the weight *files* in the page cache
    on top of the resident copy, but that page cache is reclaimable, so the only
    pinned constraint at steady state is that the budget leaves the host reserve.

    Block rules:
      A — ``util*total`` alone already leaves less than the reserve for the OS.
      B — the load-time peak (~2x weights: resident copy + page cache, before the
          KV pool is allocated) plus the reserve won't fit (``2*weights + reserve
          > total``), so the model can't even finish loading safely.
    A missing ``weights_bytes`` (uncached model) downgrades B to a warning.
    ``recommended_util`` is demand-aware when the KV geometry + workload shape
    are supplied (so a small model recommends a small util, not the ceiling).
    """
    rec_util = auto_utilization(
        total_bytes=total_bytes,
        reserve_bytes=reserve_bytes,
        weights_bytes=weights_bytes,
        kv_per_token=kv_bytes_per_token,
        context_tokens=context_tokens,
        parallel=parallel,
    )
    budget = int(total_bytes * util) if total_bytes > 0 else 0

    if total_bytes <= 0:
        return FitResult(
            level="warn",
            reason="Host memory total is unavailable; skipping the fit check.",
            total_bytes=total_bytes,
            budget_bytes=budget,
            weights_bytes=weights_bytes or 0,
            reserve_bytes=reserve_bytes,
            recommended_util=rec_util,
            model_size_known=weights_bytes is not None,
        )

    # Rule A: the utilization itself starves the OS, regardless of model.
    if budget > total_bytes - reserve_bytes:
        return FitResult(
            level="block",
            reason=(
                f"gpu-memory-utilization {util:.2f} targets "
                f"{format_gib(budget)} of {format_gib(total_bytes)} unified "
                f"memory, leaving less than the {format_gib(reserve_bytes)} "
                f"host reserve. Use {rec_util:.2f} or lower."
            ),
            total_bytes=total_bytes,
            budget_bytes=budget,
            weights_bytes=weights_bytes or 0,
            reserve_bytes=reserve_bytes,
            recommended_util=rec_util,
            model_size_known=weights_bytes is not None,
        )

    if weights_bytes is None:
        return FitResult(
            level="warn",
            reason=(
                "Model is not cached locally; its memory footprint cannot be "
                "verified before launch."
            ),
            total_bytes=total_bytes,
            budget_bytes=budget,
            weights_bytes=0,
            reserve_bytes=reserve_bytes,
            recommended_util=rec_util,
            model_size_known=False,
        )

    # Rule B (intrinsic): the load-time peak holds the weights twice (the
    # resident copy plus the page-cache copy of the files, before the KV pool is
    # allocated), so 2*weights + reserve must fit in total or the model can't even
    # finish loading without hard-locking the box.
    if 2 * weights_bytes + reserve_bytes > total_bytes:
        return FitResult(
            level="block",
            reason=(
                f"Model weights are ~{format_gib(weights_bytes)}; loading them "
                f"transiently needs about twice that on unified memory, which "
                f"with the {format_gib(reserve_bytes)} host reserve exceeds the "
                f"{format_gib(total_bytes)} total. This model will hard-lock "
                f"the machine — it is too large to serve here safely."
            ),
            total_bytes=total_bytes,
            budget_bytes=budget,
            weights_bytes=weights_bytes,
            reserve_bytes=reserve_bytes,
            recommended_util=rec_util,
            model_size_known=True,
        )

    return FitResult(
        level="ok",
        reason=(
            f"~{format_gib(weights_bytes)} weights + {format_gib(budget)} "
            f"budget fit within {format_gib(total_bytes)} with the "
            f"{format_gib(reserve_bytes)} reserve."
        ),
        total_bytes=total_bytes,
        budget_bytes=budget,
        weights_bytes=weights_bytes,
        reserve_bytes=reserve_bytes,
        recommended_util=rec_util,
        model_size_known=True,
    )


def assess_model_fit(
    *,
    model_id: str,
    util: float,
    scan_dirs: list[Path | str],
    total_bytes: int | None = None,
    reserve_bytes: int | None = None,
) -> FitResult:
    """High-level helper: resolve the model, read host memory, evaluate fit."""
    if total_bytes is None:
        total_bytes = int(host_memory_info().get("total_bytes") or 0)
    if reserve_bytes is None:
        reserve_bytes = host_reserve_bytes()

    model_path = resolve_local_model_path(model_id, scan_dirs)
    weights_bytes = (
        estimate_resident_bytes(model_path) if model_path is not None else None
    )
    return evaluate_fit(
        total_bytes=total_bytes,
        util=util,
        reserve_bytes=reserve_bytes,
        weights_bytes=weights_bytes,
    )
