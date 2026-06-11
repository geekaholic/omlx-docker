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

## Remaining Implementation Steps

1. Test against real backends.
   - Ollama (via the OpenAI-compatible backend type): verify chat, `/v1/models`, `/admin/api/proxy/metrics`, `/api/tags`, and `/api/ps`.
   - vLLM: verify chat streaming, tool calls, Prometheus `/metrics`, token counters, cache usage, and restart flows on Spark.
   - llama.cpp server: verify OpenAI compatibility, model listing behavior, streaming, tool calling, and any exposed metrics.

2. Finish proxy-mode admin UI cleanup.
   - Hide or remove remaining native MLX-only controls.
   - Keep chat, model aliases/settings, backend status, backend metrics, logs, proxy configuration, and vLLM launch configuration.
   - Replace unsupported native actions with clear proxy-mode messaging or remove them.

3. Decide llama.cpp lifecycle support.
   - Current support assumes an external OpenAI-compatible llama.cpp server.
   - Add a llama.cpp sidecar only if Spark/Linux usage shows it is worth managing from this repo.
   - If added, mirror the vLLM pattern: template + ignored generated compose/env + `omni serve` flags.

4. Add Spark/Linux runbook coverage.
   - External Ollama through the OpenAI-compatible backend type.
   - External vLLM.
   - vLLM sidecar with known-good model examples.
   - llama.cpp server if supported.
   - Troubleshooting for HF cache mounts, gated models, NVIDIA runtime, vLLM env settings, and backend restarts.
   - Known unsupported native oMLX/MLX features.

5. Continue fork cleanup.
   - Isolate or remove native MLX/Metal imports from the Linux proxy path.
   - Decide the final package/module rename strategy from `omlx` toward `omni`.
   - Remove unused macOS app and native-inference code once the proxy surface is stable.

## Current Verification Gaps

- Full pytest collection still imports MLX/Metal paths and fails in headless,
  sandboxed, virtualized, or Linux sessions without an accessible Metal device.
- The proxy-specific and `omni` test suites are the reliable Linux-port signal
  until native MLX paths are isolated or removed.
- Real backend behavior still needs manual testing with Ollama, vLLM, and
  llama.cpp server.
