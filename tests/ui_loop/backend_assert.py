# SPDX-License-Identifier: Apache-2.0
"""Assertions about backend state for the QA loop (pure where possible)."""

from __future__ import annotations

from omlx.proxy.metrics import parse_prometheus_text


def vllm_metric_families(metrics_text: str) -> set[str]:
    """Return the set of metric family names present in a /metrics dump."""
    samples = parse_prometheus_text(metrics_text)
    return {s["name"] for s in samples}


def served_model_listed(models_json: dict, served_name: str) -> bool:
    """True if served_name appears in an OpenAI /v1/models style payload."""
    return any(m.get("id") == served_name for m in models_json.get("data", []))
