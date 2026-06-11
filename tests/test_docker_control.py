# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest

from omlx.proxy import docker_control
from omlx.proxy.docker_control import (
    DockerControlError,
    DockerUnavailableError,
    restart_compose_service,
)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://docker",
    )


@pytest.mark.asyncio
async def test_restart_compose_service_restarts_matching_container():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/containers/json":
            filters = json.loads(request.url.params["filters"])
            assert filters == {"label": ["com.docker.compose.service=vllm"]}
            return httpx.Response(
                200,
                json=[
                    {
                        "Id": "abc123",
                        "Labels": {"com.docker.compose.service": "vllm"},
                    }
                ],
            )
        if request.url.path == "/containers/abc123/restart":
            assert request.method == "POST"
            return httpx.Response(204)
        return httpx.Response(404)

    async with _client(handler) as client:
        container_id = await restart_compose_service("vllm", client=client)

    assert container_id == "abc123"
    assert requests[-1].url.path == "/containers/abc123/restart"


@pytest.mark.asyncio
async def test_restart_compose_service_errors_when_no_container():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/containers/json":
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    async with _client(handler) as client:
        with pytest.raises(DockerControlError, match="No container found"):
            await restart_compose_service("llamacpp", client=client)


@pytest.mark.asyncio
async def test_restart_compose_service_errors_on_restart_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/containers/json":
            return httpx.Response(200, json=[{"Id": "abc123", "Labels": {}}])
        if request.url.path == "/containers/abc123/restart":
            return httpx.Response(500, text="daemon error")
        return httpx.Response(404)

    async with _client(handler) as client:
        with pytest.raises(DockerControlError, match="failed with status 500"):
            await restart_compose_service("vllm", client=client)


@pytest.mark.asyncio
async def test_restart_compose_service_prefers_own_compose_project(monkeypatch):
    monkeypatch.setattr(docker_control.socket, "gethostname", lambda: "selfid")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/containers/json":
            return httpx.Response(
                200,
                json=[
                    {
                        "Id": "other",
                        "Labels": {"com.docker.compose.project": "other-stack"},
                    },
                    {
                        "Id": "mine",
                        "Labels": {"com.docker.compose.project": "omni-stack"},
                    },
                ],
            )
        if request.url.path == "/containers/selfid/json":
            return httpx.Response(
                200,
                json={
                    "Config": {"Labels": {"com.docker.compose.project": "omni-stack"}}
                },
            )
        if request.url.path == "/containers/mine/restart":
            return httpx.Response(204)
        return httpx.Response(404)

    async with _client(handler) as client:
        container_id = await restart_compose_service("vllm", client=client)

    assert container_id == "mine"


@pytest.mark.asyncio
async def test_restart_compose_service_requires_socket(monkeypatch, tmp_path):
    monkeypatch.setenv("OMLX_DOCKER_SOCK", str(tmp_path / "missing.sock"))

    with pytest.raises(DockerUnavailableError, match="not available"):
        await restart_compose_service("vllm")
