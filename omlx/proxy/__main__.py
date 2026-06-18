# SPDX-License-Identifier: Apache-2.0
"""Run the oMLX proxy with `python -m omlx.proxy`."""

from __future__ import annotations

import uvicorn

from .config import ProxyConfig


def main() -> None:
    config = ProxyConfig.from_env()
    if not config.proxy_api_key:
        print(
            "WARNING: OMLX_PROXY_API_KEY is not set; proxy and admin API "
            "requests will not require authentication. Keep the published "
            "port bound to localhost or set OMLX_PROXY_API_KEY.",
            flush=True,
        )
    uvicorn.run(
        "omlx.proxy.app:create_app",
        factory=True,
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
