# SPDX-License-Identifier: Apache-2.0
"""Snapshot/restore and heavy-operation guardrails for the QA loop."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

# Models the loop is allowed to ACTUALLY launch/swap to. Everything else is
# dry-run only. Keep this tiny and small-memory (Spark power-cycle history).
DEFAULT_WHITELIST: tuple[str, ...] = ("Qwen/Qwen3-1.7B",)


class HeavyOpRefused(RuntimeError):
    """Raised when the loop tries to really launch a non-whitelisted model."""


class SettingsSnapshot:
    """Capture file contents (or absence) and restore them verbatim."""

    def __init__(self, paths: Iterable[Path]):
        self._paths = [Path(p) for p in paths]
        self._captured: dict[Path, bytes | None] = {}

    def capture(self) -> None:
        for p in self._paths:
            self._captured[p] = p.read_bytes() if p.exists() else None

    def restore(self) -> None:
        for p, data in self._captured.items():
            if data is None:
                if p.exists():
                    p.unlink()
            else:
                p.write_bytes(data)

    def __enter__(self) -> "SettingsSnapshot":
        self.capture()
        return self

    def __exit__(self, *exc) -> None:
        self.restore()


def is_whitelisted_model(
    model: str, whitelist: Sequence[str] = DEFAULT_WHITELIST
) -> bool:
    return model in whitelist


def assert_heavy_op_allowed(
    model: str, whitelist: Sequence[str] = DEFAULT_WHITELIST
) -> None:
    if not is_whitelisted_model(model, whitelist):
        raise HeavyOpRefused(
            f"Refusing real launch/swap to non-whitelisted model {model!r}; "
            f"use --dry-run / compose-gen only. Whitelist={tuple(whitelist)}"
        )
