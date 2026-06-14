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


async def service_gpu_memory_bytes(
    service: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> int | None:
    """Total GPU memory (bytes) used by processes in a sidecar container.

    Runs ``nvidia-smi --query-compute-apps`` inside the container via the
    Docker exec API — the NVIDIA container runtime injects nvidia-smi into
    GPU containers, and on unified-memory machines (DGX Spark) this is the
    only per-process model-memory signal: CUDA allocations appear neither
    in the container cgroup nor in process RSS.

    Returns None when the query is unavailable (no Docker socket, no such
    container, no nvidia-smi, or no compute processes reported).
    """
    try:
        exit_code, output = await exec_in_service(
            service,
            [
                "nvidia-smi",
                "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
            ],
            client=client,
        )
    except DockerControlError:
        return None
    if exit_code != 0:
        return None
    total = 0
    seen = False
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            total += int(float(line)) * 1024 * 1024
        except ValueError:
            continue
        seen = True
    return total if seen else None


async def exec_in_service(
    service: str,
    cmd: list[str],
    *,
    client: httpx.AsyncClient | None = None,
) -> tuple[int, str]:
    """Run a command in a Compose service's container; (exit_code, stdout)."""
    if client is not None:
        return await _exec_with_client(client, service, cmd)
    path = docker_socket_path()
    if not Path(path).exists():
        raise DockerUnavailableError(f"Docker socket {path} is not available.")
    transport = httpx.AsyncHTTPTransport(uds=path)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://docker",
        timeout=30.0,
    ) as owned_client:
        try:
            return await _exec_with_client(owned_client, service, cmd)
        except httpx.HTTPError as exc:
            raise DockerUnavailableError(
                f"Docker socket {path} is not reachable: {exc}."
            ) from exc


async def container_logs(
    service: str,
    *,
    tail: int = 200,
    timestamps: bool = True,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Return the last ``tail`` lines of a Compose service's container logs.

    Mirrors ``docker compose logs <service>`` (stdout + stderr). Raises
    DockerUnavailableError when the Docker socket is not mounted and
    DockerControlError for any other failure (container missing, daemon error).
    """
    if client is not None:
        return await _logs_with_client(client, service, tail, timestamps)
    path = docker_socket_path()
    if not Path(path).exists():
        raise DockerUnavailableError(f"Docker socket {path} is not available.")
    transport = httpx.AsyncHTTPTransport(uds=path)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://docker",
        timeout=30.0,
    ) as owned_client:
        try:
            return await _logs_with_client(owned_client, service, tail, timestamps)
        except httpx.HTTPError as exc:
            raise DockerUnavailableError(
                f"Docker socket {path} is not reachable: {exc}."
            ) from exc


async def _logs_with_client(
    client: httpx.AsyncClient,
    service: str,
    tail: int,
    timestamps: bool,
) -> str:
    container = await _find_service_container(client, service)
    container_id = str(container.get("Id") or "")
    response = await client.get(
        f"/containers/{container_id}/logs",
        params={
            "stdout": "1",
            "stderr": "1",
            "tail": str(max(1, tail)),
            "timestamps": "1" if timestamps else "0",
        },
    )
    if response.status_code != 200:
        raise DockerControlError(
            f"Docker logs for {service} failed with status "
            f"{response.status_code}: {response.text}"
        )
    return _demux_docker_stream(response.content)


async def _exec_with_client(
    client: httpx.AsyncClient,
    service: str,
    cmd: list[str],
) -> tuple[int, str]:
    container = await _find_service_container(client, service)
    container_id = str(container.get("Id") or "")
    response = await client.post(
        f"/containers/{container_id}/exec",
        json={"AttachStdout": True, "AttachStderr": True, "Cmd": cmd},
    )
    if response.status_code != 201:
        raise DockerControlError(
            f"Docker exec create in {service} failed with status "
            f"{response.status_code}: {response.text}"
        )
    exec_id = str(response.json().get("Id") or "")
    response = await client.post(
        f"/exec/{exec_id}/start",
        json={"Detach": False, "Tty": False},
    )
    if response.status_code != 200:
        raise DockerControlError(
            f"Docker exec start in {service} failed with status "
            f"{response.status_code}: {response.text}"
        )
    output = _demux_docker_stream(response.content, streams=(1,))
    inspect = await client.get(f"/exec/{exec_id}/json")
    exit_code = 0
    if inspect.status_code == 200:
        exit_code = int(inspect.json().get("ExitCode") or 0)
    return exit_code, output


def _demux_docker_stream(raw: bytes, streams: tuple[int, ...] = (1, 2)) -> str:
    """Decode Docker's multiplexed attach/log stream.

    ``streams`` selects which frame types to keep (1=stdout, 2=stderr); by
    default both. Exec output passes ``(1,)`` to keep stdout only.
    """
    # Each frame: 1 byte stream type, 3 bytes padding, 4 bytes big-endian
    # length, then payload. Tty-less output always uses this format.
    chunks: list[bytes] = []
    offset = 0
    while offset + 8 <= len(raw):
        stream_type = raw[offset]
        length = int.from_bytes(raw[offset + 4 : offset + 8], "big")
        payload = raw[offset + 8 : offset + 8 + length]
        if stream_type in streams:
            chunks.append(payload)
        offset += 8 + length
    if not chunks and raw and raw[0] not in (0, 1, 2):
        # Defensive: a Tty stream has no framing; pass it through.
        return raw.decode("utf-8", errors="replace")
    return b"".join(chunks).decode("utf-8", errors="replace")


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
