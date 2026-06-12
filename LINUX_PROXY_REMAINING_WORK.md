# Linux Proxy Remaining Work

This fork is moving toward a Linux-friendly oMLX UI/proxy that delegates
inference to remote or sidecar OpenAI-compatible backends such as vLLM,
llama.cpp server, or any OpenAI-compatible endpoint (Ollama, for example).

## Current Decisions

- vLLM lifecycle is Compose-owned. `omni serve --backend vllm` generates
  `docker/docker-compose.vllm.yml` plus `docker/docker-compose.vllm.env`, then
  launches the stack with Docker Compose.
- The FastAPI proxy does not directly own or supervise the GPU server process.
- Admin can edit proxy settings live. Admin can also edit intended vLLM launch
  settings and regenerate the env/compose files, but vLLM launch changes require
  a backend/container restart.
- Sampling defaults are proxy request defaults. They are persisted as
  `OMLX_SAMPLING_*` values and injected into forwarded chat/completion requests
  when the client omits those fields.
- Serving stats are proxy-side accounting, not backend scraping: the proxy
  records per-request `usage` (including `prompt_tokens_details.cached_tokens`)
  and wall-clock timing on every request path, mirroring upstream's
  `ServerMetrics` semantics. Backend Prometheus/Ollama metrics remain a
  supplementary panel.
- KV/prefix caching is left to the inference backend (vLLM prefix caching,
  llama.cpp KV cache). The proxy only surfaces the cache metrics backends
  expose; it does not add its own caching layer.

## Completed Linux Proxy Work

- Dockerized MLX-free proxy container.
- OpenAI-compatible passthrough and Anthropic Messages API bridge.
- Proxy admin dashboard and browser chat UI.
- Backend URL/API key/type controls with persisted proxy state. The dedicated
  `ollama` backend type was folded into `openai-compatible`, which defaults to
  the Ollama endpoint; per-backend settings persist keyed by backend type.
- Live application of backend URL/API key/type changes.
- Sidecar backend restart from the admin UI through the Docker Engine API,
  with the sidecar re-reading its generated env file on restart.
- Backend status and metrics endpoints, including vLLM Prometheus and Ollama
  native probes.
- `omni` CLI for `serve`, `status`, `logs`, `restart`, and `stop`.
- vLLM sidecar compose generation for NVIDIA hosts.
- vLLM env-file persistence, admin regeneration, HF cache mounting, and
  advanced launch settings.
- vLLM compose startup sanitation for empty Hugging Face/proxy/cert env vars.
- llama.cpp managed sidecar (template + generated compose/env + `omni serve`
  flags), now launched with `--metrics` for Prometheus observability.
- Dashboard resynced with upstream v0.4.4rc1 (merge of `origin/main`):
  Serving Stats cards, Average Speed, Active Models, cache panel, and API
  Endpoints panels from the upstream redesign.
- Proxy-side serving stats: session/all-time scopes with persistence
  (`OMLX_PROXY_STATS_PATH`, default next to the proxy state file), per-model
  filtering, cached-token counts and cache efficiency, prompt/generation
  tok/s from TTFT and stream timing, and split clear endpoints. The
  chat-completions passthrough injects `stream_options.include_usage` when
  the client didn't request it (the trailing usage-only chunk is stripped
  from the relayed stream; a backend 400 triggers one retry without
  injection).
- Active Models panel in proxy mode: Ollama loaded models with real memory
  size and TTL from `/api/ps`; vLLM/llama.cpp running/waiting request counts
  attributed to the served model; 5s TTL cache on backend metric collection.
- Backend cache observability: the Runtime Cache panel becomes "Backend KV /
  Prefix Cache" in proxy mode, showing vLLM prefix-cache hit rate (v1
  hit/query counters or the v0 hit-rate gauge) and GPU KV usage, or
  llama.cpp KV cache usage/tokens and average throughputs. Backends that
  expose nothing (Ollama) get an explicit note.
- Proxy-mode admin UI cleanup: model downloader/quantizer/uploader/bench
  tabs, MLX cache controls, per-model unload, server restart, and the Engine
  Versions panel are hidden in proxy mode; the sidecar restart card replaces
  native restart.

## Spark Verification Log (June 2026, DGX Spark GB10)

- Proxy test suites run natively on Linux: 266 pass in a plain venv with
  fastapi/uvicorn/httpx/pydantic/jinja2/jsonschema/regex + pytest/pytest-asyncio
  and `PYTHONPATH=.` (no MLX install needed for these files).
- **llama.cpp sidecar (ghcr.io/ggml-org/llama.cpp:server-cuda, b9570)**:
  - Streamed + non-streamed chat OK through the proxy; `usage` injection
    works (backend honors `stream_options.include_usage`; trailing usage
    chunk is stripped when the client didn't ask, relayed when it did).
  - Repeated prompts: llama.cpp reports `cached_tokens` on the second hit
    (22/26 prompt tokens); Serving Stats cached tokens + cache efficiency
    move accordingly; avg prompt/generation tok/s populate.
  - Finding: this build no longer emits `llamacpp:kv_cache_usage_ratio` /
    `llamacpp:kv_cache_tokens`. Fixed by selecting `llamacpp:n_tokens_max`
    as a "Peak Context Tokens" tile and counting throughput metrics toward
    cache-panel availability (commit 2fd4ba7).
- **Active Models memory bar (green bar) now works in proxy mode**
  (commit 769bcc0): per-process GPU memory via `nvidia-smi` run inside the
  sidecar through the Docker exec API (on GB10 unified memory, CUDA
  allocations show up neither in the container cgroup — 922MiB vs the real
  8.8GiB — nor in process RSS); Ollama uses `/api/ps` sizes; hard limit is
  host MemTotal from `/proc/meminfo`; vLLM's `gpu_memory_utilization`
  budget is surfaced as the soft limit. Verified live with all three
  backends (llama.cpp 8.62 GB, vLLM 97.05 GB vs 97.35 GB soft, Ollama
  17.76 GB from `/api/ps`).
- **vLLM sidecar (vllm/vllm-openai:latest, gemma-4-26B-A4B-it)**:
  - Prometheus metric names drifted as predicted: this generation emits
    `vllm:prefix_cache_{hits,queries}_total` (no `gpu_` prefix) and
    `vllm:kv_cache_usage_perc`. Candidates extended (commit 37bacf1),
    ordered so the `*external_prefix_cache*` families are never
    double-counted. Live: 63% prefix hit rate on repeated prompts.
  - `usage.prompt_tokens_details.cached_tokens` requires vLLM's
    `--enable-prompt-tokens-details`; the generated compose now passes it
    (verified: cached_tokens 480/499 on a repeated prompt).
  - Sidecar restart from the admin API works (202 + container restart).
  - Streamed + non-streamed chat, stats accumulation, tok/s all good.
- **Backend switching (CLI vs persisted state)**: launching a stack with
  `omni serve` now overrides stale persisted backend routing in both
  directions (sidecar↔sidecar in 37bacf1, sidecar→standalone/openai in
  dafe959); per-backend profiles are archived for switching back.
- **Ollama (host service 0.14.2, via openai-compatible)**: Active Models
  rows show real size + TTL countdown from `/api/ps`; cache panel takes
  the "not exposed" path; streamed usage honored
  (`stream_options.include_usage`); stats accumulate.
- **Cross-cutting**: Session/All-Time scopes, both Clear endpoints,
  per-model filter, API Endpoints data (host/port/aliases/cli_prefix
  "omni"), dashboard + chat pages render. All-Time stats survived
  multiple proxy container rebuilds/restarts via the `proxy-state`
  volume; clear-alltime zeroes them. (Browser-console check still worth
  an eyeball on a real browser.)
- **Environment gotchas** (documented in `docs/SPARK_RUNBOOK.md`): broken
  DNS sandbox in containers created during a failed first `up`
  (force-recreate fixes); pre-merge generated env files use old
  `VLLM_*` key names and silently fall back to template defaults —
  regenerate with `omni serve` or the admin UI.

## Remaining Implementation Steps

1. Test against real backends. — **Done** (June 2026 on DGX Spark; see
   the verification log above and `docs/SPARK_RUNBOOK.md`).
   - Ollama (via the OpenAI-compatible backend type): verify chat, `/v1/models`, `/admin/api/proxy/metrics`, `/api/tags`, `/api/ps`, and the Active Models size/TTL display.
   - vLLM: verify chat streaming, tool calls, Prometheus `/metrics`, token counters, prefix-cache hit rate movement on repeated prompts, and restart flows on Spark.
   - llama.cpp server: verify OpenAI compatibility, model listing behavior, streaming, tool calling, and that `--metrics` exposes the `llamacpp:*` families the dashboard parses.
   - Prometheus metric names drift across vLLM versions; the candidate lists
     live in `select_prometheus_metrics` (`omlx/proxy/metrics.py`) — `curl
     <backend>/metrics` on the deployed image and extend the candidates if a
     family is missed.

2. Add Spark/Linux runbook coverage. — **Done**: `docs/SPARK_RUNBOOK.md`
   (sidecar quick starts, external Ollama/vLLM, troubleshooting from
   real incidents, unsupported native features).

3. Continue fork cleanup.
   - Isolate or remove native MLX/Metal imports from the Linux proxy path.
   - Decide the final package/module rename strategy from `omlx` toward `omni`.
   - Remove unused macOS app and native-inference code once the proxy surface is stable.

## Deferred

- External caching/router layers (LMCache, KV offload tiers, prompt routers)
  are intentionally not integrated. Cache observability comes only from
  backend-native metrics; revisit if Spark usage shows prefix caching at the
  vLLM/llama.cpp layer is insufficient.
- Serving stats depend on backends emitting `usage` in responses. Backends
  that ignore `stream_options.include_usage` (older llama.cpp builds, some
  Ollama versions) undercount tokens; requests are still counted.

## Current Verification Gaps

- Full pytest collection still imports MLX/Metal paths and fails in headless,
  sandboxed, virtualized, or Linux sessions without an accessible Metal device.
- The proxy-specific and `omni` test suites (`tests/test_proxy*.py`,
  `tests/test_omni_cli.py`, `tests/test_*_compose.py`,
  `tests/test_server_metrics.py`, `tests/test_docker_control.py`,
  `tests/test_integrations.py`) are the reliable Linux-port signal until
  native MLX paths are isolated or removed — 270+ tests, run on the Spark
  in a plain venv with `PYTHONPATH=.`.
- Real-backend manual testing with Ollama, vLLM, and llama.cpp is done
  (see the Spark Verification Log above); re-run it when bumping backend
  images, since both vLLM and llama.cpp have renamed metrics between
  releases.
