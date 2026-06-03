# oMNI

oMNI is a Linux-friendly fork of [oMLX](https://github.com/jundot/omlx) that keeps
the useful web UI, chat surface, Anthropic/OpenAI compatibility layer, and admin
workflow while removing the requirement that inference run through Apple MLX.

The current direction is proxy-first: oMNI runs as a lightweight FastAPI gateway
and admin UI in Docker, then delegates inference to an existing LLM backend such
as vLLM, llama.cpp server, Ollama, or another OpenAI-compatible endpoint.

This repository still contains a lot of upstream oMLX code and the Python
package/CLI is still named `omlx` for now. The fork is being reshaped toward the
oMNI identity incrementally.

## Why oMNI

Upstream oMLX is a strong Mac/Apple-Silicon local inference project built around
MLX, mlx-lm, native macOS app integration, and local model management.

This fork has a different goal:

- Run on Linux and NVIDIA systems, including DGX Spark.
- Run headless in Docker.
- Preserve the oMLX admin/chat UI where it is useful.
- Proxy to multiple inference backends instead of owning one MLX runtime.
- Prefer vLLM or llama.cpp sidecars for production Linux deployments.
- Keep Anthropic-compatible endpoints for tools that expect Claude-style APIs.

In short: oMNI is oMLX without the MLX runtime dependency, adapted for many LLM
inference backends.

## Current Status

Working:

- Dockerized proxy container.
- OpenAI-compatible passthrough for `/v1/chat/completions` and related routes.
- Anthropic Messages API translation at `/v1/messages`.
- Built-in chat UI at `/admin/chat`.
- Admin dashboard compatibility layer at `/admin/dashboard`.
- Backend model discovery from `/v1/models`.
- Proxy backend status and backend metrics panel.
- Ollama backend support when Ollama exposes its OpenAI-compatible API.
- vLLM-compatible Prometheus metrics parsing from `/metrics`.
- Optional vLLM sidecar compose file for NVIDIA hosts.
- Context token scaling for Claude Code style workflows.
- SSE keepalive support for long-running requests.

Still in progress:

- Admin controls for changing backend URL/API key/type from the UI.
- Cleaner proxy-mode UI that hides native MLX-only controls.
- More complete real-backend validation for vLLM, llama.cpp server, and Ollama.
- Formal package/CLI rename from `omlx` to `omni` or another final command name.
- Removing unused macOS/MLX-native code from this fork.

See [LINUX_PROXY_REMAINING_WORK.md](LINUX_PROXY_REMAINING_WORK.md) for the
current implementation backlog.

## Quickstart: Docker Proxy

The default Docker workflow runs oMNI as a proxy and expects an external backend
that provides an OpenAI-compatible `/v1` API.

For Docker Desktop on Mac with Ollama running on the host:

```bash
docker compose -f docker/docker-compose.proxy.yml up --build
```

By default this points at:

```text
http://host.docker.internal:11434/v1
```

Open:

- Admin dashboard: `http://localhost:8080/admin/dashboard`
- Chat UI: `http://localhost:8080/admin/chat`
- OpenAI-compatible API: `http://localhost:8080/v1`
- Anthropic-compatible API: `http://localhost:8080/v1/messages`

To point at a different backend:

```bash
OMLX_BACKEND_URL=http://your-backend:8000/v1 \
docker compose -f docker/docker-compose.proxy.yml up --build
```

If the backend needs an API key:

```bash
OMLX_BACKEND_URL=http://your-backend:8000/v1 \
OMLX_BACKEND_API_KEY=backend-secret \
OMLX_PROXY_API_KEY=proxy-secret \
docker compose -f docker/docker-compose.proxy.yml up --build
```

`OMLX_BACKEND_API_KEY` is sent to the upstream backend. `OMLX_PROXY_API_KEY`
protects the oMNI proxy itself.

## Quickstart: vLLM Sidecar

The Spark compose file runs oMNI plus a vLLM OpenAI server sidecar.

```bash
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
VLLM_SERVED_MODEL_NAME=qwen \
docker compose -f docker/docker-compose.spark.yml up --build
```

The proxy talks to vLLM at `http://vllm:8000/v1` inside the compose network and
publishes oMNI on port `8080`.

Useful environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `VLLM_IMAGE` | `vllm/vllm-openai:latest` | vLLM container image |
| `VLLM_MODEL` | required | Hugging Face model id or container-local path |
| `VLLM_SERVED_MODEL_NAME` | `VLLM_MODEL` | API-visible model name |
| `VLLM_MAX_MODEL_LEN` | `32768` | vLLM context length |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.85` | vLLM GPU memory fraction |
| `VLLM_MAX_NUM_SEQS` | `16` | vLLM max concurrent sequences |
| `VLLM_TOOL_CALL_PARSER` | `hermes` | vLLM auto tool-choice parser |
| `HF_TOKEN` | empty | Hugging Face token for gated models |

The sidecar compose uses `gpus: all` and `ipc: host`, so Docker must be
configured with NVIDIA Container Toolkit on Linux.

## Running Without Docker

The proxy can also run directly from Python:

```bash
pip install -e .
omlx proxy --backend-url http://localhost:8000/v1 --host 0.0.0.0 --port 8080
```

The direct command is useful for development, but Docker is the primary target
for Linux and DGX deployments.

## Configuration

Proxy mode is configured with environment variables or CLI flags.

| Environment Variable | Description |
|---|---|
| `OMLX_BACKEND_URL` | Required backend URL, normally ending in `/v1` |
| `OMLX_BACKEND_API_KEY` | Optional API key for the backend |
| `OMLX_PROXY_API_KEY` | Optional API key required by oMNI clients |
| `OMLX_PROXY_HOST` | Bind host, defaults to `0.0.0.0` |
| `OMLX_PROXY_PORT` | Bind port, defaults to `8080` |
| `OMLX_PROXY_TIMEOUT` | Backend request timeout in seconds, defaults to `600` |
| `OMLX_CONTEXT_SCALING` | Enable reported token scaling |
| `OMLX_TARGET_CONTEXT_SIZE` | Token count target reported to clients |
| `OMLX_ACTUAL_CONTEXT_SIZE` | Actual backend context size |
| `OMLX_SSE_KEEPALIVE_MODE` | Anthropic SSE keepalive mode, defaults to `ping` |
| `OMLX_PROXY_STATE_PATH` | Persistent proxy admin state path |

## API Surface

oMNI exposes an OpenAI-compatible API and an Anthropic-compatible bridge.

| Endpoint | Status | Notes |
|---|---|---|
| `GET /v1/models` | Working | Proxies backend model list |
| `POST /v1/chat/completions` | Working | Passthrough to backend |
| `POST /v1/messages` | Working | Anthropic Messages to OpenAI chat translation |
| `POST /v1/messages/count_tokens` | Approximate | Local estimate, supports context scaling |
| `GET /admin/chat` | Working | Browser chat UI |
| `GET /admin/dashboard` | Working | Proxy-mode admin dashboard |
| `GET /admin/api/proxy/status` | Working | Backend reachability and model count |
| `GET /admin/api/proxy/metrics` | Working | vLLM Prometheus and Ollama probes |

Native oMLX endpoints that depend on MLX model loading, local KV cache
management, quantization, benchmarks, and macOS services are being removed,
stubbed, or hidden in proxy mode.

## Backend Notes

### Ollama

Ollama works through its OpenAI-compatible API:

```bash
OMLX_BACKEND_URL=http://host.docker.internal:11434/v1 \
docker compose -f docker/docker-compose.proxy.yml up --build
```

The admin metrics endpoint also probes Ollama-native `/api/tags` and `/api/ps`
when available, so the dashboard can show available and loaded model counts.

### vLLM

vLLM is the preferred Linux/NVIDIA backend target. oMNI can proxy to an external
vLLM server or run with the provided vLLM sidecar compose file. The dashboard
parses common vLLM Prometheus metrics such as request counts, token counts,
running/waiting requests, and GPU cache usage.

### llama.cpp Server

llama.cpp server is a target backend when launched with OpenAI-compatible
endpoints. Compatibility still needs more real-backend testing, especially
around model listing, streaming behavior, tool calling, and metrics.

## Development

Proxy-focused checks:

```bash
pytest tests/test_proxy.py -q
python -m compileall -q omlx/proxy
node --check omlx/admin/static/js/dashboard.js
docker compose -f docker/docker-compose.proxy.yml build
```

The full upstream test suite still imports MLX/Metal paths. In headless,
sandboxed, virtualized, or Linux environments those tests may fail during
collection before reaching proxy code. Until the fork removes or isolates native
MLX modules, `tests/test_proxy.py` is the main regression suite for the Linux
proxy path.

## Project Layout

| Path | Purpose |
|---|---|
| `omlx/proxy/` | MLX-free proxy gateway and backend adapters |
| `omlx/admin/` | Reused oMLX admin UI assets and proxy compatibility routes |
| `docker/Dockerfile.proxy` | Minimal proxy image |
| `docker/docker-compose.proxy.yml` | Proxy with external backend |
| `docker/docker-compose.spark.yml` | Proxy plus vLLM sidecar |
| `docker/proxy.env.example` | Example environment values |
| `tests/test_proxy.py` | Proxy regression tests |
| `LINUX_PROXY_REMAINING_WORK.md` | Current backlog |

## Attribution

oMNI is forked from oMLX by Jun Kim and keeps substantial oMLX code, UI assets,
API adapters, and design ideas. This fork exists because oMLX built a useful
admin and compatibility layer, and the goal here is to salvage that experience
for non-MLX backends.

Important upstream projects:

- [oMLX](https://github.com/jundot/omlx) - original project and UI foundation.
- [MLX](https://github.com/ml-explore/mlx) and
  [mlx-lm](https://github.com/ml-explore/mlx-lm) - upstream oMLX inference stack.
- [vLLM](https://github.com/vllm-project/vllm) - preferred Linux/NVIDIA backend.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - target local inference backend.
- [Ollama](https://github.com/ollama/ollama) - supported OpenAI-compatible backend.

## License

This fork preserves the upstream Apache 2.0 license. See [LICENSE](LICENSE).
