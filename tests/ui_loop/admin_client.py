# SPDX-License-Identifier: Apache-2.0
"""HTTP client for the live oMNI admin API + a pure settings-diff helper."""

from __future__ import annotations

import os
from typing import Any

import httpx

BASE_URL = os.environ.get("UI_LOOP_BASE_URL", "http://localhost:8080")
ADMIN = "/admin/api"


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def diff_settings(before: dict, after: dict) -> dict[str, tuple[Any, Any]]:
    """Return {dotted_key: (before, after)} for every leaf that changed."""
    fb, fa = _flatten(before), _flatten(after)
    changed: dict[str, tuple[Any, Any]] = {}
    for key in set(fb) | set(fa):
        b, a = fb.get(key), fa.get(key)
        if b != a:
            changed[key] = (b, a)
    return changed


class AdminClient:
    """Thin wrapper over the running proxy's admin API.

    Auth: if the admin password / api key is set, pass it via UI_LOOP_API_KEY;
    it is sent as a Bearer token (matches verify_proxy_key in proxy/app.py).
    """

    def __init__(self, base_url: str = BASE_URL, api_key: str | None = None):
        key = api_key or os.environ.get("UI_LOOP_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._c = httpx.Client(base_url=base_url, headers=headers, timeout=30.0)

    def close(self) -> None:
        self._c.close()

    def __enter__(self) -> "AdminClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- settings -------------------------------------------------------
    def get_global_settings(self) -> dict:
        r = self._c.get(f"{ADMIN}/global-settings")
        r.raise_for_status()
        return r.json()

    def post_global_settings(self, payload: dict) -> httpx.Response:
        return self._c.post(f"{ADMIN}/global-settings", json=payload)

    # --- proxy / backend ------------------------------------------------
    def get_proxy_config(self) -> dict:
        r = self._c.get(f"{ADMIN}/proxy/config")
        r.raise_for_status()
        return r.json()

    def get_proxy_status(self) -> dict:
        r = self._c.get(f"{ADMIN}/proxy/status")
        r.raise_for_status()
        return r.json()

    def get_proxy_metrics(self) -> dict:
        r = self._c.get(f"{ADMIN}/proxy/metrics")
        r.raise_for_status()
        return r.json()

    def get_sidecar_compose(self) -> dict:
        r = self._c.get(f"{ADMIN}/proxy/sidecar-compose")
        r.raise_for_status()
        return r.json()

    def regenerate_sidecar_compose(
        self, payload: dict | None = None
    ) -> httpx.Response:
        return self._c.post(
            f"{ADMIN}/proxy/sidecar-compose/regenerate", json=payload or {}
        )

    def get_local_models(self) -> dict:
        r = self._c.get(f"{ADMIN}/proxy/local-models")
        r.raise_for_status()
        return r.json()

    # --- stats ----------------------------------------------------------
    def get_stats(self, scope: str = "session") -> dict:
        r = self._c.get(f"{ADMIN}/stats", params={"scope": scope})
        r.raise_for_status()
        return r.json()

    def clear_stats(self, alltime: bool = False) -> httpx.Response:
        path = "stats/clear-alltime" if alltime else "stats/clear"
        return self._c.post(f"{ADMIN}/{path}")

    def get_logs(self, **params) -> httpx.Response:
        return self._c.get(f"{ADMIN}/logs", params=params)

    # --- lifecycle ------------------------------------------------------
    def restart_sidecar(self) -> httpx.Response:
        return self._c.post(f"{ADMIN}/sidecar/restart")
