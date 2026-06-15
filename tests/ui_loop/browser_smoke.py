# SPDX-License-Identifier: Apache-2.0
"""Headless-Chromium smoke pass: load a page, collect console errors."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("UI_LOOP_BASE_URL", "http://localhost:8080")
ARTIFACT_DIR = Path(os.environ.get("UI_LOOP_ARTIFACTS", "docs/qa/artifacts"))


@dataclass
class SmokeResult:
    path: str
    console_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    screenshot: str | None = None

    @property
    def ok(self) -> bool:
        return not self.console_errors and not self.page_errors


def smoke_page(path: str, base_url: str = BASE_URL) -> SmokeResult:
    """Load base_url+path headless; return console/page errors + a screenshot."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result = SmokeResult(path=path)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.on(
            "console",
            lambda msg: result.console_errors.append(msg.text)
            if msg.type == "error"
            else None,
        )
        page.on("pageerror", lambda exc: result.page_errors.append(str(exc)))
        # The admin dashboard streams logs / polls metrics, so the network never
        # goes idle. Wait for the DOM, then settle briefly to let JS run and any
        # console/page errors fire before we judge the page.
        page.goto(f"{base_url}{path}", wait_until="domcontentloaded", timeout=20000)
        page.wait_for_timeout(2500)
        shot = ARTIFACT_DIR / (path.strip("/").replace("/", "_") or "root")
        shot = shot.with_suffix(".png")
        page.screenshot(path=str(shot), full_page=True)
        result.screenshot = str(shot)
        browser.close()
    return result
