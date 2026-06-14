# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/models context-window enrichment (single source of truth)."""

from types import SimpleNamespace

from omlx.proxy.app import _enrich_model_list


def test_advertises_backend_window():
    data = {"data": [{"id": "m", "max_model_len": 32768}]}
    item = _enrich_model_list(data, admin_state=None)["data"][0]
    assert item["context_window"] == 32768
    assert item["max_context_window"] == 32768


def test_admin_overrides_do_not_shrink_advertised_window():
    # A stale per-model / global cap must never make /v1/models advertise less
    # than what the backend actually enforces — that caused Codex to overflow.
    state = SimpleNamespace(
        model_settings={"m": {"max_context_window": 16384}},
        global_overrides={"sampling_max_context_window": 16384},
    )
    data = {"data": [{"id": "m", "max_model_len": 32768}]}
    item = _enrich_model_list(data, state)["data"][0]
    assert item["context_window"] == 32768


def test_no_max_model_len_leaves_item_untouched():
    data = {"data": [{"id": "m"}]}
    item = _enrich_model_list(data, admin_state=None)["data"][0]
    assert "context_window" not in item
