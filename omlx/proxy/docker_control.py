# SPDX-License-Identifier: Apache-2.0
"""Minimal Docker Engine API client for managing sidecar containers.

Talks to the Docker daemon over its unix socket with httpx, so the proxy
image needs neither the docker CLI nor the docker SDK. Used by the admin UI
to restart the managed vLLM / llama.cpp sidecar container.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import httpx

DEFAULT_SOCKET_PATH = "/var/run/docker.sock"

_COMPOSE_SERVICE_LABEL = "com.docker.compose.service"
_COMPOSE_PROJECT_LABEL = "com.docker.compose.project"


class DockerControlError(RuntimeError):
    """Docker is reachable but the requested operation failed."""


class DockerUnavailableError(DockerControlError):
    """The Docker socket is not available to this process."""


def docker_socket_path() -> str:
    value = os.getenv("OMLX_DOCKER_SOCK", "").strip()
    return value or DEFAULT_SOCKET_PATH


async def restart_compose_service(
    service: str,
    *,
    timeout_seconds: int = 10,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Restart the container backing a Docker Compose service.

    Returns the restarted container id. Raises DockerUnavailableError when
    the Docker socket is not mounted and DockerControlError for any other
    failure (container missing, daemon error).
    """
    if client is not None:
        return await _restart_with_client(client, service, timeout_seconds)
    path = docker_socket_path()
    if not Path(path).exists():
        raise DockerUnavailableError(f"Docker socket {path} is not available.")
    transport = httpx.AsyncHTTPTransport(uds=path)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://docker",
        timeout=max(30.0, timeout_seconds + 20.0),
    ) as owned_client:
        try:
            return await _restart_with_client(owned_client, service, timeout_seconds)
        except httpx.HTTPError as exc:
            raise DockerUnavailableError(
                f"Docker socket {path} is not reachable: {exc}."
            ) from exc


async def _restart_with_client(
    client: httpx.AsyncClient,
    service: str,
    timeout_seconds: int,
) -> str:
    container = await _find_service_container(client, service)
    container_id = str(container.get("Id") or "")
    response = await client.post(
        f"/containers/{container_id}/restart",
        params={"t": timeout_seconds},
    )
    if response.status_code != 204:
        raise DockerControlError(
            f"Docker restart of {service} failed with status "
            f"{response.status_code}: {response.text}"
        )
    return container_id


async def _find_service_container(
    client: httpx.AsyncClient,
    service: str,
) -> dict[str, Any]:
    filters = json.dumps({"label": [f"{_COMPOSE_SERVICE_LABEL}={service}"]})
    response = await client.get(
        "/containers/json",
        params={"all": "1", "filters": filters},
    )
    if response.status_code != 200:
        raise DockerControlError(
            f"Docker container listing failed with status "
            f"{response.status_code}: {response.text}"
        )
    containers = [item for item in response.json() if isinstance(item, dict)]
    if not containers:
        raise DockerControlError(f"No container found for Compose service '{service}'.")
    if len(containers) == 1:
        return containers[0]
    project = await _own_compose_project(client)
    if project:
        for container in containers:
            labels = container.get("Labels") or {}
            if labels.get(_COMPOSE_PROJECT_LABEL) == project:
                return container
    return containers[0]


async def _own_compose_project(client: httpx.AsyncClient) -> str | None:
    """Compose project of the calling container, if it runs under Compose.

    Inside a container the hostname defaults to the container id, which the
    Engine API accepts as an identifier.
    """
    try:
        response = await client.get(f"/containers/{socket.gethostname()}/json")
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    config = response.json().get("Config") or {}
    labels = config.get("Labels") or {}
    return labels.get(_COMPOSE_PROJECT_LABEL)
