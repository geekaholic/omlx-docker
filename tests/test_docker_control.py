# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest

from omlx.proxy import docker_control
from omlx.proxy.docker_control import (
    DockerControlError,
    DockerUnavailableError,
    exec_in_service,
    restart_compose_service,
    service_gpu_memory_bytes,
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


def _mux_frame(stream_type: int, payload: bytes) -> bytes:
    return bytes([stream_type, 0, 0, 0]) + len(payload).to_bytes(4, "big") + payload


def _exec_handler_factory(stdout_frames: bytes, exit_code: int = 0):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/containers/json":
            return httpx.Response(
                200,
                json=[
                    {
                        "Id": "abc123",
                        "Labels": {"com.docker.compose.service": "llamacpp"},
                    }
                ],
            )
        if path == "/containers/abc123/exec":
            body = json.loads(request.content.decode())
            assert body["AttachStdout"] is True
            return httpx.Response(201, json={"Id": "exec1"})
        if path == "/exec/exec1/start":
            return httpx.Response(200, content=stdout_frames)
        if path == "/exec/exec1/json":
            return httpx.Response(200, json={"ExitCode": exit_code})
        return httpx.Response(404)

    return handler


@pytest.mark.asyncio
async def test_exec_in_service_demuxes_stdout_and_reports_exit_code():
    frames = (
        _mux_frame(1, b"hello ")
        + _mux_frame(2, b"warning\n")
        + _mux_frame(1, b"world\n")
    )
    async with _client(_exec_handler_factory(frames)) as client:
        exit_code, output = await exec_in_service(
            "llamacpp", ["echo", "hi"], client=client
        )

    assert exit_code == 0
    assert output == "hello world\n"


@pytest.mark.asyncio
async def test_service_gpu_memory_bytes_sums_compute_processes():
    frames = _mux_frame(1, b"8823\n100\n")
    async with _client(_exec_handler_factory(frames)) as client:
        value = await service_gpu_memory_bytes("llamacpp", client=client)

    assert value == (8823 + 100) * 1024 * 1024


@pytest.mark.asyncio
async def test_service_gpu_memory_bytes_none_when_nvidia_smi_missing():
    frames = _mux_frame(2, b"exec failed\n")
    async with _client(_exec_handler_factory(frames, exit_code=127)) as client:
        value = await service_gpu_memory_bytes("llamacpp", client=client)

    assert value is None


@pytest.mark.asyncio
async def test_service_gpu_memory_bytes_none_without_container():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/containers/json":
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    async with _client(handler) as client:
        value = await service_gpu_memory_bytes("vllm", client=client)

    assert value is None
