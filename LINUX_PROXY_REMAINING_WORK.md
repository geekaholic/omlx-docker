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

## Remaining Implementation Steps

1. Test against real backends.
   - Ollama (via the OpenAI-compatible backend type): verify chat, `/v1/models`, `/admin/api/proxy/metrics`, `/api/tags`, `/api/ps`, and the Active Models size/TTL display.
   - vLLM: verify chat streaming, tool calls, Prometheus `/metrics`, token counters, prefix-cache hit rate movement on repeated prompts, and restart flows on Spark.
   - llama.cpp server: verify OpenAI compatibility, model listing behavior, streaming, tool calling, and that `--metrics` exposes the `llamacpp:*` families the dashboard parses.
   - Prometheus metric names drift across vLLM versions; the candidate lists
     live in `select_prometheus_metrics` (`omlx/proxy/metrics.py`) — `curl
     <backend>/metrics` on the deployed image and extend the candidates if a
     family is missed.

2. Add Spark/Linux runbook coverage.
   - External Ollama through the OpenAI-compatible backend type.
   - External vLLM.
   - vLLM sidecar with known-good model examples.
   - llama.cpp sidecar.
   - Troubleshooting for HF cache mounts, gated models, NVIDIA runtime, vLLM env settings, and backend restarts.
   - Known unsupported native oMLX/MLX features.

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
  `tests/test_server_metrics.py`) are the reliable Linux-port signal until
  native MLX paths are isolated or removed.
- Real backend behavior still needs manual testing with Ollama, vLLM, and
  llama.cpp server.
