# SPDX-License-Identifier: Apache-2.0
"""Backend observability helpers for proxy mode."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any

from .backend import OpenAIBackend

_PROM_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


async def collect_backend_metrics(backend: OpenAIBackend) -> dict[str, Any]:
    """Collect best-effort metrics from common OpenAI-compatible backends."""
    prometheus = await _collect_prometheus(backend)
    ollama = await _collect_ollama(backend)
    backend_kind = "unknown"
    if ollama["available"]:
        backend_kind = "ollama"
    elif prometheus["available"]:
        backend_kind = "prometheus"

    return {
        "backend_kind": backend_kind,
        "prometheus": prometheus,
        "ollama": ollama,
        "summary": _summary(prometheus, ollama),
    }


async def collect_backend_metrics_cached(
    backend: OpenAIBackend, ttl: float = 5.0
) -> dict[str, Any]:
    """Collect backend metrics with a short per-backend cache.

    The dashboard polls /admin/api/stats and /admin/api/proxy/metrics on
    short intervals; the cache keeps that from hammering the backend. The
    cache lives on the backend instance and invalidates when the backend
    URL changes (the admin UI can repoint it live).
    """
    url = backend.config.normalized_backend_url
    cached = getattr(backend, "_metrics_cache", None)
    now = time.monotonic()
    if cached is not None:
        cached_at, cached_url, result = cached
        if cached_url == url and now - cached_at < ttl:
            return result
    result = await collect_backend_metrics(backend)
    backend._metrics_cache = (now, url, result)
    return result


async def _collect_prometheus(backend: OpenAIBackend) -> dict[str, Any]:
    try:
        text = await backend.get_root_text("metrics")
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "sample_count": 0,
            "metric_count": 0,
            "selected": {},
        }

    samples = parse_prometheus_text(text)
    selected = select_prometheus_metrics(samples)
    return {
        "available": True,
        "error": None,
        "sample_count": len(samples),
        "metric_count": len({sample["name"] for sample in samples}),
        "selected": selected,
    }


async def _collect_ollama(backend: OpenAIBackend) -> dict[str, Any]:
    tags: dict[str, Any] | None = None
    ps: dict[str, Any] | None = None
    tag_error = None
    ps_error = None

    try:
        tags = await backend.get_root_json("api/tags")
    except Exception as exc:
        tag_error = str(exc)

    try:
        ps = await backend.get_root_json("api/ps")
    except Exception as exc:
        ps_error = str(exc)

    available = tags is not None or ps is not None
    models = tags.get("models", []) if isinstance(tags, dict) else []
    loaded = ps.get("models", []) if isinstance(ps, dict) else []
    return {
        "available": available,
        "error": None if available else (tag_error or ps_error),
        "models_count": len(models) if isinstance(models, list) else 0,
        "loaded_count": len(loaded) if isinstance(loaded, list) else 0,
        "models": _summarize_ollama_models(models),
        "loaded_models": _summarize_ollama_models(loaded),
    }


def parse_prometheus_text(text: str) -> list[dict[str, Any]]:
    samples = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_LINE.match(line)
        if match is None:
            continue
        labels = parse_prometheus_labels(match.group("labels") or "")
        samples.append(
            {
                "name": match.group("name"),
                "labels": labels,
                "value": float(match.group("value")),
            }
        )
    return samples


def parse_prometheus_labels(text: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not text:
        return labels
    for part in _split_label_parts(text):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1].replace(r"\"", '"').replace(r"\\", "\\")
        labels[key.strip()] = value
    return labels


def select_prometheus_metrics(samples: list[dict[str, Any]]) -> dict[str, float]:
    by_name: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        by_name[sample["name"]].append(float(sample["value"]))

    def pick(*candidates: str, aggregate: str = "sum") -> float | None:
        for candidate in candidates:
            if candidate in by_name:
                values = by_name[candidate]
                return max(values) if aggregate == "max" else sum(values)
        for candidate in candidates:
            suffix_matches = [
                value
                for name, values in by_name.items()
                if name.endswith(candidate)
                for value in values
            ]
            if suffix_matches:
                return (
                    max(suffix_matches) if aggregate == "max" else sum(suffix_matches)
                )
        return None

    selected: dict[str, float] = {}
    candidates = {
        "requests_total": (
            "vllm:num_requests_total",
            "vllm_requests_total",
            "requests_total",
        ),
        "prompt_tokens_total": (
            "vllm:prompt_tokens_total",
            "vllm_prompt_tokens_total",
            "llamacpp:prompt_tokens_total",
            "prompt_tokens_total",
        ),
        "generation_tokens_total": (
            "vllm:generation_tokens_total",
            "vllm_generation_tokens_total",
            "llamacpp:tokens_predicted_total",
            "generation_tokens_total",
        ),
        "running_requests": (
            "vllm:num_requests_running",
            "vllm_num_requests_running",
            "llamacpp:requests_processing",
            "num_requests_running",
        ),
        "waiting_requests": (
            "vllm:num_requests_waiting",
            "vllm_num_requests_waiting",
            "llamacpp:requests_deferred",
            "num_requests_waiting",
        ),
        "gpu_cache_usage_perc": (
            "vllm:gpu_cache_usage_perc",
            "vllm_gpu_cache_usage_perc",
            "gpu_cache_usage_perc",
        ),
        # vLLM prefix cache: v1 exposes hit/query counters (the client
        # appends _total); v0 exposed a 0..1 hit-rate gauge instead.
        "prefix_cache_queries": (
            "vllm:gpu_prefix_cache_queries_total",
            "vllm:gpu_prefix_cache_queries",
            "gpu_prefix_cache_queries_total",
            "gpu_prefix_cache_queries",
        ),
        "prefix_cache_hits": (
            "vllm:gpu_prefix_cache_hits_total",
            "vllm:gpu_prefix_cache_hits",
            "gpu_prefix_cache_hits_total",
            "gpu_prefix_cache_hits",
        ),
        "prefix_cache_hit_rate_gauge": (
            "vllm:gpu_prefix_cache_hit_rate",
            "gpu_prefix_cache_hit_rate",
        ),
        # llama.cpp server --metrics
        "kv_cache_usage_ratio": ("llamacpp:kv_cache_usage_ratio",),
        "kv_cache_tokens": ("llamacpp:kv_cache_tokens",),
        "prompt_tokens_seconds": ("llamacpp:prompt_tokens_seconds",),
        "predicted_tokens_seconds": ("llamacpp:predicted_tokens_seconds",),
    }
    _MAX_KEYS = {
        "gpu_cache_usage_perc",
        "prefix_cache_hit_rate_gauge",
        "kv_cache_usage_ratio",
        "prompt_tokens_seconds",
        "predicted_tokens_seconds",
    }
    for key, names in candidates.items():
        value = pick(*names, aggregate="max" if key in _MAX_KEYS else "sum")
        if value is not None:
            selected[key] = value
    return selected


def summarize_selected_metrics(selected: dict[str, float]) -> dict[str, Any]:
    """Flatten selected Prometheus metrics into the dashboard summary shape."""
    hits = selected.get("prefix_cache_hits")
    queries = selected.get("prefix_cache_queries")
    hit_rate: float | None = None
    if hits is not None and queries is not None and queries > 0:
        hit_rate = round(hits / queries * 100, 1)
    elif selected.get("prefix_cache_hit_rate_gauge") is not None:
        hit_rate = round(selected["prefix_cache_hit_rate_gauge"] * 100, 1)
    return {
        "requests_total": selected.get("requests_total"),
        "prompt_tokens_total": selected.get("prompt_tokens_total"),
        "generation_tokens_total": selected.get("generation_tokens_total"),
        "running_requests": selected.get("running_requests"),
        "waiting_requests": selected.get("waiting_requests"),
        "gpu_cache_usage_perc": selected.get("gpu_cache_usage_perc"),
        "prefix_cache_hits": hits,
        "prefix_cache_queries": queries,
        "prefix_cache_hit_rate": hit_rate,
        "kv_cache_usage_ratio": selected.get("kv_cache_usage_ratio"),
        "kv_cache_tokens": selected.get("kv_cache_tokens"),
        "prompt_tokens_seconds": selected.get("prompt_tokens_seconds"),
        "predicted_tokens_seconds": selected.get("predicted_tokens_seconds"),
    }


def _split_label_parts(text: str) -> list[str]:
    parts = []
    current = []
    in_quote = False
    escaped = False
    for char in text:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char == '"':
            in_quote = not in_quote
            current.append(char)
            continue
        if char == "," and not in_quote:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def _summarize_ollama_models(models: Any) -> list[dict[str, Any]]:
    if not isinstance(models, list):
        return []
    result = []
    for item in models:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "name": item.get("name") or item.get("model"),
                "model": item.get("model") or item.get("name"),
                "size": item.get("size") or item.get("size_vram") or 0,
                "expires_at": item.get("expires_at"),
                "details": item.get("details") or {},
            }
        )
    return result


def _summary(prometheus: dict[str, Any], ollama: dict[str, Any]) -> dict[str, Any]:
    selected = prometheus.get("selected") or {}
    summary = summarize_selected_metrics(selected)
    summary["ollama_models_count"] = (
        ollama.get("models_count") if ollama.get("available") else None
    )
    summary["ollama_loaded_count"] = (
        ollama.get("loaded_count") if ollama.get("available") else None
    )
    return summary
