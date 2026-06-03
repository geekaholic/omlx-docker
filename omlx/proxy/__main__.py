# SPDX-License-Identifier: Apache-2.0
"""Run the oMLX proxy with `python -m omlx.proxy`."""

from __future__ import annotations

import uvicorn

from .config import ProxyConfig


def main() -> None:
    config = ProxyConfig.from_env()
    uvicorn.run(
        "omlx.proxy.app:create_app",
        factory=True,
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
