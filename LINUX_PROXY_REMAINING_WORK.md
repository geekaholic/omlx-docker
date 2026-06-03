# Linux Proxy Remaining Work

This fork is moving toward a Linux-friendly oMLX UI/proxy that delegates
inference to remote or sidecar OpenAI-compatible backends such as vLLM,
llama.cpp server, or Ollama.

## Next Implementation Steps

1. Test against real backends.
   - Ollama: verify chat, `/v1/models`, `/admin/api/proxy/metrics`, `/api/tags`, and `/api/ps`.
   - vLLM: verify chat streaming, tool calls, Prometheus `/metrics`, token counters, and cache usage.
   - llama.cpp server: verify OpenAI compatibility, model listing behavior, streaming, and any exposed metrics.

2. Add admin controls for proxy backend configuration.
   - Backend URL.
   - Backend API key.
   - Backend type selector: OpenAI-compatible, Ollama, vLLM, llama.cpp.
   - Persist settings in proxy state.
   - Mark settings that require container restart versus settings that can hot reload.

3. Add Docker sidecar profiles.
   - Keep the current external-backend proxy compose file as the default.
   - Add an optional vLLM sidecar compose profile for NVIDIA hosts.
   - Add an optional llama.cpp server sidecar profile if it proves useful.
   - Document model volume paths and NVIDIA runtime requirements for DGX Spark.

4. Decide vLLM lifecycle boundaries.
   - Preferred: run vLLM as a Docker sidecar managed by Compose or the host orchestrator.
   - Admin UI can edit intended vLLM launch config and show status.
   - Avoid making the FastAPI proxy directly own GPU server process lifecycle unless there is a clear operational need.

5. Clean up proxy-mode admin UI.
   - Hide or remove native MLX-only controls.
   - Keep chat, model aliases/settings, backend status, backend metrics, logs, and proxy configuration.
   - Replace stubbed actions with clear proxy-mode behavior.

6. Add Linux/DGX runbook coverage.
   - External Ollama.
   - External vLLM.
   - vLLM sidecar.
   - llama.cpp server if supported.
   - Known unsupported native oMLX features.

## Current Verification Gaps

- Full pytest collection still imports MLX/Metal paths and fails in headless or sandboxed Mac sessions without an accessible Metal device.
- The proxy-specific test suite is the reliable Linux-port signal until native MLX paths are isolated or removed.
- Real backend behavior still needs manual testing with Ollama, vLLM, and llama.cpp server.
