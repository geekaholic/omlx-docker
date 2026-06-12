# SPDX-License-Identifier: Apache-2.0
"""Scan local directories for models the proxy sidecars can serve.

Opt-in via ``omni serve --scan-models``: the proxy container gets the host
HF cache (and llama.cpp cache) mounted read-only and lists what is already
downloaded — safetensors repos for vLLM, GGUF repos/files for llama.cpp —
so the admin UI can offer one-click sidecar model switching.

Reuses the MLX-free helpers from :mod:`omlx.model_discovery` (HF cache
layout resolution, model-type detection, size/context probes); the scan
loop itself is proxy-specific because the native ``discover_models`` is
MLX-filtered.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from omlx.model_discovery import (
    _is_model_dir,
    _read_model_context_length,
    _resolve_hf_cache_entry,
    detect_model_type,
    estimate_model_size,
)

logger = logging.getLogger(__name__)


def scan_local_models(scan_root: Path | str) -> list[dict[str, Any]]:
    """Scan ``scan_root`` (and its immediate subdirectories) for models.

    Handles three layouts at each location:
    - HF Hub caches (``models--Org--Repo`` entries, also ``<root>/hub``)
    - plain model directories (``config.json`` present)
    - loose ``*.gguf`` files

    Returns plain dict rows sorted by repo id; never raises on unreadable
    entries (skips and logs instead).
    """
    root = Path(scan_root)
    if not root.is_dir():
        return []

    rows: dict[str, dict[str, Any]] = {}
    for location in _scan_locations(root):
        for entry in _iter_entries(location):
            try:
                row = _classify_entry(entry)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("skipping %s: %s", entry, exc)
                continue
            if row is not None:
                # First sighting wins (e.g. the same repo reachable via
                # both the scan root and a hub/ subdir).
                rows.setdefault(row["repo_id"], row)
    return sorted(rows.values(), key=lambda r: r["repo_id"].lower())


def _scan_locations(root: Path) -> list[Path]:
    """Directories whose entries should be classified.

    Covers the mount layout ``/models-scan/<source>`` where each source is
    either an HF cache root (``.../hub/models--*``), a hub dir itself, or
    a plain folder of model dirs / GGUF files.
    """
    locations = [root]
    for child in _safe_iterdir(root):
        if child.is_dir() and not child.name.startswith("models--"):
            locations.append(child)
            hub = child / "hub"
            if hub.is_dir():
                locations.append(hub)
    return locations


def _iter_entries(location: Path) -> list[Path]:
    return [p for p in _safe_iterdir(location)]


def _safe_iterdir(path: Path) -> list[Path]:
    try:
        return sorted(path.iterdir())
    except OSError:
        return []


def _classify_entry(entry: Path) -> dict[str, Any] | None:
    if entry.is_file():
        if entry.suffix.lower() == ".gguf":
            return _gguf_file_row(entry)
        return None
    if not entry.is_dir():
        return None
    if entry.name.startswith("models--"):
        cache_entry = _resolve_hf_cache_entry(entry)
        if cache_entry is None:
            return None
        return _model_dir_row(
            cache_entry.snapshot_path,
            repo_id=cache_entry.source_repo_id,
        )
    if _is_model_dir(entry) or _contains_gguf(entry):
        return _model_dir_row(entry, repo_id=entry.name)
    return None


def _contains_gguf(path: Path) -> bool:
    return any(p.suffix.lower() == ".gguf" for p in _safe_iterdir(path) if p.is_file())


def _model_dir_row(path: Path, *, repo_id: str) -> dict[str, Any] | None:
    has_config = (path / "config.json").exists()
    safetensors = [p for p in _safe_iterdir(path) if p.suffix.lower() == ".safetensors"]
    ggufs = [p for p in path.glob("**/*.gguf") if p.is_file()]

    backends: list[str] = []
    if has_config and safetensors:
        backends.append("vllm")
    if ggufs:
        backends.append("llama.cpp")
    if not backends:
        return None

    model_format = "safetensors" if "vllm" in backends else "gguf"
    model_type = "llm"
    context_length = None
    if has_config:
        try:
            model_type = detect_model_type(path)
        except Exception:
            model_type = "llm"
        context_length = _read_model_context_length(path)

    size_bytes = _size_of(path, safetensors, ggufs)
    return {
        "model_id": repo_id.replace("/", "--"),
        "repo_id": repo_id,
        "path": str(path),
        "model_format": model_format,
        "model_type": model_type,
        "size_bytes": size_bytes,
        "size_formatted": _format_size(size_bytes),
        "context_length": context_length,
        "backends": backends,
    }


def _gguf_file_row(path: Path) -> dict[str, Any]:
    size = _stat_size(path)
    return {
        "model_id": path.stem,
        "repo_id": path.stem,
        "path": str(path),
        "model_format": "gguf",
        "model_type": "llm",
        "size_bytes": size,
        "size_formatted": _format_size(size),
        "context_length": None,
        "backends": ["llama.cpp"],
    }


def _size_of(path: Path, safetensors: list[Path], ggufs: list[Path]) -> int:
    if safetensors:
        try:
            # estimate_model_size adds runtime overhead and resolves the
            # HF blob symlinks via stat().
            return estimate_model_size(path)
        except (ValueError, OSError):
            pass
    total = sum(_stat_size(p) for p in safetensors) or sum(_stat_size(p) for p in ggufs)
    return total


def _stat_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _format_size(num: float) -> str:
    if num >= 1024**3:
        return f"{num / 1024**3:.2f} GB"
    if num >= 1024**2:
        return f"{num / 1024**2:.1f} MB"
    if num > 0:
        return f"{num / 1024:.0f} KB"
    return "0 B"


def scan_enabled() -> bool:
    return os.getenv("OMLX_MODEL_SCAN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def scan_dir() -> Path:
    return Path(os.getenv("OMLX_MODEL_SCAN_DIR", "/models-scan"))
