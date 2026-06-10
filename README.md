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

## Installing the `omni` Tool

`omni` is installed from this repo as a Python console script. The Python
package is named `omlx` (the install output will say `+ omlx==...`), but the
install registers both the `omlx` and `omni` commands in your environment. On
Linux/Spark hosts, the CLI is mainly a Docker Compose launcher, so the
lightweight install path is to install the local package without resolving the
Mac/MLX runtime dependencies:

```bash
cd path/to/omlx-docker
python3 -m pip install -e . --no-deps
rehash  # zsh only; refreshes command lookup after installing console scripts
omni --help
```

If the shell still cannot find `omni`, run it through Python directly:

```bash
cd path/to/omlx-docker
python3 -m omlx.omni_cli --help
python3 -m omlx.omni_cli serve --backend vllm --generate-only
```

With `uv`, use an isolated environment and the same no-dependency install:

```bash
cd path/to/omlx-docker
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e . --no-deps
rehash  # zsh only; refreshes command lookup after installing console scripts
omni --help
```

Use `--no-deps` for the Linux proxy workflow because the upstream project still
declares Mac/MLX-oriented dependencies. The Docker proxy image installs its own
minimal server dependencies. For development tests in this repo, install pytest
into the active environment:

```bash
python3 -m pip install pytest
python3 -m pytest tests/test_omni_cli.py -q
```

## Quickstart: vLLM Sidecar

Use `omni serve --backend vllm` to generate a local vLLM Compose file and launch
oMNI plus a vLLM OpenAI server sidecar on Linux/NVIDIA hosts.

```bash
omni serve --backend vllm \
  --model Qwen/Qwen3-1.7B \
  --served-model-name qwen
```

The command writes two local generated files, then runs Docker Compose with an
explicit env file:

- `docker/docker-compose.vllm.yml` - generated Compose stack, ignored by git.
- `docker/docker-compose.vllm.env` - last-used vLLM launch settings, ignored by git.
- `docker/omni-serve.json` - last-used `omni serve` backend/compose selection, ignored by git.

If `docker/docker-compose.vllm.env` already exists, omitted `omni serve` flags
reuse values from that file. Passing a flag such as `--model`,
`--served-model-name`, or `--context-length` updates only the corresponding env
value. On first run, missing values use the built-in defaults. The proxy talks
to vLLM at `http://vllm:8000/v1` inside the compose network and publishes oMNI
on port `8080`. The compose file bind-mounts the host Hugging Face cache at
`${HOME}/.cache/huggingface`, so already downloaded models are reused.

Restart the vLLM container after changing vLLM launch settings; proxy backend
settings and proxy sampling defaults apply live in the running proxy.

The sidecar compose uses `gpus: all` and `ipc: host`, so Docker must be
configured with NVIDIA Container Toolkit on Linux. Use `omni serve --backend vllm`
for vLLM sidecar launches.

## Quickstart: llama.cpp Sidecar

Use `omni serve --backend llamacpp` to generate a local llama.cpp Compose file
and launch oMNI plus a `llama-server` sidecar. The model is either a Hugging
Face GGUF repo (`owner/repo[:quantTag]`, downloaded by llama.cpp via `-hf` and
cached in the mounted `~/.cache/llama.cpp`) or a local `.gguf` path (passed via
`-m`; relative paths resolve inside the read-only `/models` mount):

```bash
omni serve --backend llamacpp \
  --model ggml-org/Qwen3-1.7B-GGUF:Q8_0 \
  --served-model-name qwen \
  --context-length 16384

# Local GGUF file from the llama.cpp cache / model dir mount
omni serve --backend llamacpp --model qwen3.gguf \
  --llamacpp-model-dir ~/gguf-models
```

Generated files mirror the vLLM workflow:

- `docker/docker-compose.llamacpp.yml` - generated Compose stack, ignored by git.
- `docker/docker-compose.llamacpp.env` - last-used llama.cpp launch settings, ignored by git.

The host Hugging Face cache (`OMNI_HF_HOME`) is mounted into both the vLLM and
llama.cpp containers, and `~/.cache/llama.cpp` persists `-hf` downloads, so
switching backends never re-downloads models. With `--backend llamacpp
--backend-url URL`, the command instead proxies to an external llama.cpp server
and generates no sidecar compose.

## Sidecar Environment Variables

Portable settings carry the `OMNI_` prefix and apply to whichever managed
sidecar backend is selected:

| Variable | Default | Purpose |
|---|---:|---|
| `OMNI_MODEL` | per backend | Model id/path (vLLM: HF id or container path; llama.cpp: GGUF repo `owner/repo[:quant]` or `.gguf` path) |
| `OMNI_SERVED_MODEL_NAME` | `qwen` | API-visible model name |
| `OMNI_CONTEXT_LENGTH` | `8192` | Context length (vLLM `--max-model-len`, llama.cpp `--ctx-size`) |
| `OMNI_MAX_PARALLEL` | `4` | Max concurrent sequences (vLLM `--max-num-seqs`, llama.cpp `--parallel`) |
| `OMNI_BACKEND_PORT` | `8000` | Host port published for the backend container |
| `OMNI_HF_HOME` | `${HOME}/.cache/huggingface` | Host Hugging Face cache to mount into the backend |
| `OMNI_HF_ENDPOINT` | empty | Hugging Face endpoint exposed as `HF_ENDPOINT` |
| `OMNI_HTTP_PROXY` / `OMNI_HTTPS_PROXY` | empty | Proxy env passed to proxy and backend containers |
| `OMNI_NO_PROXY` | empty | No-proxy host list for both containers |
| `OMNI_CA_BUNDLE` | empty | CA bundle path exposed as `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE` (llama.cpp: `CURL_CA_BUNDLE`) |

vLLM-specific settings:

| Variable | Default | Purpose |
|---|---:|---|
| `VLLM_IMAGE` | `vllm/vllm-openai:latest` | vLLM container image |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.80` | vLLM GPU memory fraction |
| `VLLM_GENERATION_CONFIG` | `vllm` | Use vLLM defaults instead of model `generation_config.json` |
| `VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS` | `{"enable_thinking":false}` | Disables Qwen thinking output in chat templates |
| `VLLM_TRUST_REMOTE_CODE` | `true` | Add `--trust-remote-code` |
| `VLLM_ENABLE_AUTO_TOOL_CHOICE` | `false` | Add vLLM auto tool-choice flags when enabled |
| `VLLM_TOOL_CALL_PARSER` | `hermes` | vLLM auto tool-choice parser |
| `VLLM_REASONING_PARSER` | empty | vLLM reasoning parser |
| `VLLM_DTYPE` | empty | Optional `--dtype` override |
| `VLLM_TOKENIZER` | empty | Optional tokenizer id/path |
| `VLLM_TOKENIZER_MODE` | empty | Optional tokenizer mode |
| `VLLM_REVISION` | empty | Optional model revision |
| `VLLM_LOAD_FORMAT` | empty | Optional load format |
| `VLLM_QUANTIZATION` | empty | Optional quantization mode |
| `VLLM_DOWNLOAD_DIR` | empty | Optional vLLM download directory |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | empty | Optional scheduler token budget |
| `VLLM_ENABLE_CHUNKED_PREFILL` | empty | `true`/`false` to force chunked prefill on/off; empty uses vLLM default |
| `VLLM_ENABLE_PREFIX_CACHING` | empty | `true`/`false` to force prefix caching on/off; empty uses vLLM default |
| `VLLM_KV_CACHE_DTYPE` | empty | Optional KV cache dtype |
| `VLLM_CPU_OFFLOAD_GB` | empty | Optional CPU offload GB per GPU |
| `VLLM_SWAP_SPACE` | empty | Optional swap space GB per GPU |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Tensor parallel size |
| `VLLM_PIPELINE_PARALLEL_SIZE` | `1` | Pipeline parallel size |
| `VLLM_UVICORN_LOG_LEVEL` | empty | Optional vLLM API log level |
| `VLLM_DISABLE_LOG_STATS` | `false` | Add `--disable-log-stats` when true |
| `VLLM_EXTRA_ARGS_JSON` | `[]` | Raw extra vLLM args as a JSON array appended last |

llama.cpp-specific settings:

| Variable | Default | Purpose |
|---|---:|---|
| `LLAMACPP_IMAGE` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | llama.cpp server image (use `:server` for CPU, `:server-vulkan` for non-NVIDIA GPUs) |
| `LLAMACPP_N_GPU_LAYERS` | `999` | Layers offloaded to the GPU (`999` = all) |
| `LLAMACPP_FLASH_ATTN` | empty (auto) | `--flash-attn on/off` when set |
| `LLAMACPP_CACHE_TYPE_K` / `LLAMACPP_CACHE_TYPE_V` | empty (f16) | KV cache quantization, e.g. `q8_0` |
| `LLAMACPP_THREADS` | empty (auto) | CPU threads |
| `LLAMACPP_BATCH_SIZE` / `LLAMACPP_UBATCH_SIZE` | empty | Logical/physical batch sizes |
| `LLAMACPP_JINJA` | `true` | `--jinja` chat templates (required for tool calling) |
| `LLAMACPP_REASONING_FORMAT` | empty (auto) | `--reasoning-format` for thinking models |
| `LLAMACPP_CACHE_DIR` | `${HOME}/.cache/llama.cpp` | Host cache for `-hf` GGUF downloads |
| `LLAMACPP_MODEL_DIR` | cache dir | Host dir mounted read-only at `/models` for local `.gguf` files |
| `LLAMACPP_EXTRA_ARGS` | empty | Raw llama-server args appended last (whitespace-split, unlike `VLLM_EXTRA_ARGS_JSON`) |

Proxy defaults shared by both stacks:

| Variable | Default | Purpose |
|---|---:|---|
| `OMLX_SAMPLING_MAX_TOKENS` | `32768` | Proxy default max output tokens when request omits it |
| `OMLX_SAMPLING_TEMPERATURE` | `1.0` | Proxy default temperature when request omits it |
| `OMLX_SAMPLING_TOP_P` | `1.0` | Proxy default top-p when request omits it |
| `OMLX_SAMPLING_TOP_K` | `0` | Proxy default top-k when request omits it |
| `OMLX_SAMPLING_REPETITION_PENALTY` | `1.0` | Proxy default repetition penalty when request omits it |
| `HF_TOKEN` | empty | Hugging Face token for gated models |

The admin settings page edits the same env-backed sidecar launch and proxy
default settings, then regenerates the active stack's compose file. Sampling
defaults such as temperature, top-p, top-k, repetition penalty, and max tokens
are persisted in the generated `.env` file as `OMLX_SAMPLING_*` values and
applied by the oMNI proxy to forwarded chat/completion requests when the client
omits those fields. Backend launch settings still require a backend/container
restart. The generated entrypoints export Hugging Face, proxy, and CA bundle
environment variables only when configured, avoiding empty URL values such as
`HF_ENDPOINT=` inside the backend process.

Note: earlier releases used `VLLM_`-prefixed names for the portable settings
(`VLLM_MODEL`, `VLLM_MAX_MODEL_LEN`, `VLLM_PORT`, ...). Those names are no
longer read; rerun `omni serve` with your flags (or re-save the admin settings)
to regenerate the env file with the `OMNI_` names.

## Managing the Docker Stack

Use `omni status` to view the containers for the active oMNI Compose stack:

```bash
omni status
```

By default, a first `omni serve` with no backend flags launches
`docker/docker-compose.proxy.yml` against the Ollama-compatible
`http://host.docker.internal:11434/v1` backend URL. Explicit backend launches are
remembered in `docker/omni-serve.json`, so future plain `omni serve` invocations
reuse the last backend and compose file. Proxy launch environment is persisted in
`docker/docker-compose.proxy.env`; sidecar launch environments are persisted in
`docker/docker-compose.vllm.env` and `docker/docker-compose.llamacpp.env`.

By default, `omni status` uses the last saved compose file when present. Without
saved state it uses the generated `docker/docker-compose.vllm.yml` when it
exists, otherwise it falls back to `docker/docker-compose.proxy.yml`. To inspect
a specific stack:

```bash
omni status --compose-file docker/docker-compose.proxy.yml
```

View logs for the selected stack, optionally following new output or filtering
to the proxy or managed backend service:

```bash
omni logs
omni logs -f
omni logs --target proxy
omni logs --target backend
omni logs --compose-file docker/docker-compose.proxy.yml
```

Restart the proxy, the managed backend, or both services:

```bash
omni restart --target proxy
omni restart --target backend
omni restart --target both
```

Stop the proxy, the managed backend, or both services. `omni stop` requires an
explicit target and stops containers without removing volumes or networks:

```bash
omni stop --target proxy
omni stop --target backend
omni stop --target both
```

For the vLLM and llama.cpp sidecar stacks, `--target backend` reads logs from,
restarts, or stops the `vllm` or `llamacpp` service. For OpenAI, Ollama, and
external llama.cpp proxy stacks, the backend is not managed by this repo, so use
`--target proxy` or inspect/restart/stop the backend with its own tooling.

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
vLLM server or run with the generated vLLM sidecar compose stack. `omni serve
--backend vllm` owns sidecar generation and Compose launch; Admin edits the same
intended launch settings and regenerates the local env/compose files. The
dashboard parses common vLLM Prometheus metrics such as request counts, token
counts, running/waiting requests, and GPU cache usage.

### llama.cpp Server

llama.cpp runs either as a managed sidecar (`omni serve --backend llamacpp`,
which generates and launches `docker/docker-compose.llamacpp.yml`) or as an
externally managed server (`omni serve --backend llamacpp --backend-url URL`).
The managed sidecar starts `llama-server` with `--jinja` by default so
OpenAI-compatible tool calling works with modern chat templates. Compatibility
still needs more real-backend testing, especially around model listing,
streaming behavior, tool calling, and metrics.

## Development

Proxy-focused checks:

```bash
pytest tests/test_proxy.py tests/test_omni_cli.py -q
python -m compileall -q omlx/proxy
node --check omlx/admin/static/js/dashboard.js
docker compose -f docker/docker-compose.proxy.yml build
```

The full upstream test suite still imports MLX/Metal paths. In headless,
sandboxed, virtualized, or Linux environments those tests may fail during
collection before reaching proxy code. Until the fork removes or isolates native
MLX modules, `tests/test_proxy.py` and `tests/test_omni_cli.py` are the main
regression suites for the Linux proxy path.

## Project Layout

| Path | Purpose |
|---|---|
| `omlx/proxy/` | MLX-free proxy gateway and backend adapters |
| `omlx/admin/` | Reused oMLX admin UI assets and proxy compatibility routes |
| `docker/Dockerfile.proxy` | Minimal proxy image |
| `docker/docker-compose.proxy.yml` | Proxy with external backend |
| `docker/docker-compose.vllm.template.yml` | Default template for generated vLLM compose files |
| `docker/docker-compose.llamacpp.template.yml` | Default template for generated llama.cpp compose files |
| `docker/docker-compose.vllm.yml` | Local generated vLLM sidecar compose file, ignored by git |
| `docker/docker-compose.vllm.env` | Local generated vLLM launch settings, ignored by git |
| `docker/docker-compose.llamacpp.yml` | Local generated llama.cpp sidecar compose file, ignored by git |
| `docker/docker-compose.llamacpp.env` | Local generated llama.cpp launch settings, ignored by git |
| `docker/docker-compose.proxy.env` | Local proxy launch settings, ignored by git |
| `docker/omni-serve.json` | Last `omni serve` backend/compose selection, ignored by git |
| `docker/proxy.env.example` | Example environment values |
| `tests/test_proxy.py` | Proxy regression tests |
| `LINUX_PROXY_REMAINING_WORK.md` | Current backlog |

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
- Optional vLLM and llama.cpp sidecar compose generation for NVIDIA hosts.
- `omni` CLI for generating sidecar stacks and managing Compose containers.
- Admin controls for backend URL/API key/type and sidecar launch settings.
- Env-file persistence for sidecar launch settings and proxy sampling defaults.
- Backend-portable `OMNI_*` settings shared between vLLM and llama.cpp sidecars.
- Context token scaling for Claude Code style workflows.
- SSE keepalive support for long-running requests.

Still in progress:

- Cleaner proxy-mode UI that hides native MLX-only controls.
- More complete real-backend validation for vLLM, llama.cpp server, and Ollama.
- Spark/Linux runbooks with known-good model examples and troubleshooting.
- Formal package/module rename from `omlx` toward `omni` (including the
  remaining `OMLX_*` proxy env vars).
- Removing unused macOS/MLX-native code from this fork.

See [LINUX_PROXY_REMAINING_WORK.md](LINUX_PROXY_REMAINING_WORK.md) for the
current implementation backlog.

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
