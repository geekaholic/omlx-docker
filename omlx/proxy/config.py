# SPDX-License-Identifier: Apache-2.0
"""Configuration for the MLX-free oMLX proxy."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


@dataclass(frozen=True)
class ProxyConfig:
    """Runtime configuration for the proxy.

    `backend_url` should normally include the OpenAI-compatible `/v1` prefix,
    for example `http://vllm:8000/v1`.
    """

    backend_url: str
    backend_api_key: str | None = None
    proxy_api_key: str | None = None
    host: str = "0.0.0.0"
    port: int = 8080
    request_timeout_seconds: float = 600.0
    context_scaling_enabled: bool = False
    target_context_size: int = 200000
    actual_context_size: int = 32768
    sse_keepalive_mode: str = "ping"

    @classmethod
    def from_env(cls) -> "ProxyConfig":
        backend_url = os.getenv("OMLX_BACKEND_URL", "").strip()
        if not backend_url:
            raise RuntimeError(
                "OMLX_BACKEND_URL is required, e.g. http://vllm:8000/v1"
            )
        return cls(
            backend_url=backend_url,
            backend_api_key=os.getenv("OMLX_BACKEND_API_KEY") or None,
            proxy_api_key=(
                os.getenv("OMLX_PROXY_API_KEY")
                or os.getenv("OMLX_API_KEY")
                or None
            ),
            host=os.getenv("OMLX_PROXY_HOST", "0.0.0.0"),
            port=_env_int("OMLX_PROXY_PORT", 8080),
            request_timeout_seconds=float(os.getenv("OMLX_PROXY_TIMEOUT", "600")),
            context_scaling_enabled=_env_bool("OMLX_CONTEXT_SCALING", False),
            target_context_size=_env_int("OMLX_TARGET_CONTEXT_SIZE", 200000),
            actual_context_size=_env_int("OMLX_ACTUAL_CONTEXT_SIZE", 32768),
            sse_keepalive_mode=os.getenv("OMLX_SSE_KEEPALIVE_MODE", "ping"),
        )

    @property
    def normalized_backend_url(self) -> str:
        return self.backend_url.rstrip("/")

