# Spark Handoff — Dashboard Resync Verification & Next Steps

Continuation brief for a Claude Code session running on the DGX Spark
(Linux/aarch64, NVIDIA GB10). A prior session on macOS merged upstream
oMLX v0.4.4rc1 into this fork and made the redesigned admin dashboard
functional in proxy mode. Your job: verify it against real backends on
this machine, fix what breaks, and build on it.

Read `CLAUDE.md` and `LINUX_PROXY_REMAINING_WORK.md` first; this file
adds the session context those don't carry.

## State of the branch (`resync-dashboard`)

Commits from the prior session, newest first:

- `b34c277` worklog update (completed/deferred items)
- `26ae876` hide native-only Engine Versions panel in proxy mode
- `f7c6d47` cache panel → "Backend KV / Prefix Cache" in proxy mode;
  llama.cpp sidecar now passes `--metrics`
- `6a5c632` Active Models panel fed from backend state (Ollama
  `/api/ps` size+TTL; vLLM/llama.cpp running/waiting counts); 5s TTL
  cache on backend metric collection
- `a3a0cbb` proxy-side serving stats backed by upstream's MLX-free
  `ServerMetrics`
- `a32f203` merge of `origin/main` (upstream v0.4.4rc1)

All proxy suites pass (`tests/test_proxy.py`, `test_proxy_stats.py`,
`test_proxy_metrics.py`, `test_omni_cli.py`, `test_*_compose.py`,
`test_server_metrics.py`, `test_docker_control.py`,
`test_integrations.py`). The full suite (5,717 tests) passes on macOS
only — full collection imports MLX; on Linux run the proxy suites only.

## Decisions already made (don't relitigate)

1. Serving Stats are **proxy-side accounting**, not backend scraping:
   per-request `usage` + wall-clock timing recorded into a per-app
   `ServerMetrics` instance. Backend Prometheus stays supplementary.
2. KV/prefix caching is **left to the backend** (vLLM prefix caching,
   llama.cpp KV cache). The dashboard only surfaces backend-reported
   cache metrics. LMCache/external caching layers are deferred.
3. Upstream syncs flow through `origin/main`; the fork keeps oMNI
   branding, proxy modules, and the `omni` CLI through merges.

## Implementation map

- `omlx/proxy/stats.py` — usage/timing capture. `track_usage_stream`
  tees the chat-completions SSE passthrough, injects
  `stream_options.include_usage` when the client didn't ask, strips the
  trailing usage-only chunk, and retries once without injection on a
  backend 400. TTFT ≈ prefill duration; first-token→close ≈ generation
  duration; llama.cpp non-streaming `timings` honored.
- `omlx/proxy/app.py` — `ServerMetrics(stats_path=...)` created in
  `create_app` (env `OMLX_PROXY_STATS_PATH`, default next to the proxy
  state file); recording wired into Anthropic (stream + non-stream,
  unscaled tokens), Responses adapter, and raw passthrough paths.
- `omlx/proxy/admin.py` — `/admin/api/stats` (with `model`/`scope`
  params), split `/api/stats/clear` and `/api/stats/clear-alltime`,
  `_active_models_payload` (Ollama vs Prometheus branches).
- `omlx/proxy/metrics.py` — `select_prometheus_metrics` candidate
  tuples (vLLM v1 prefix-cache counters, v0 hit-rate gauge, `llamacpp:*`
  families), `summarize_selected_metrics`,
  `collect_backend_metrics_cached` (5s TTL, per-backend-instance).
- `omlx/admin/templates/dashboard/_status.html` — proxy-mode variant of
  the cache panel (`backendCacheSummary` getters live in
  `omlx/admin/static/js/dashboard.js` next to `proxyMode`).
- `docker/docker-compose.llamacpp.template.yml` and
  `omlx/proxy/llamacpp_compose.py` — `--metrics` flag (the compose is
  rendered from the f-string in llamacpp_compose.py; the docker/
  template file is a synced copy — change both).

## Task 1: verify against real backends (highest value on this box)

For each backend, generate traffic (chat via dashboard Chat tab or
curl, streamed and non-streamed, repeated identical prompts to exercise
prefix caching) and watch `/admin/dashboard`:

- **vLLM sidecar** (`omni serve --backend vllm`): Serving Stats cards
  accumulate; cached tokens / cache efficiency move on repeated prompts
  (vLLM reports `prompt_tokens_details.cached_tokens` when prefix
  caching is on); cache panel shows Prefix Hit Rate + GPU KV usage;
  Active Models shows running/waiting under load; sidecar restart flow
  still works. **Known risk:** vLLM Prometheus metric names drift by
  version — `curl http://<sidecar>:8000/metrics | grep -i prefix` and
  extend the candidate tuples in `select_prometheus_metrics` if the
  panel stays empty. Token-count caveat: confirm streamed `usage` chunks
  arrive (the proxy injects `stream_options.include_usage`).
- **llama.cpp sidecar** (`omni serve --backend llama.cpp`): confirm the
  regenerated compose carries `--metrics` (regenerate from the admin UI
  or delete `docker/docker-compose.llamacpp.yml` first — it's
  git-ignored and may be stale); `curl :8000/metrics` shows
  `llamacpp:*`; cache panel shows KV usage/tokens; stats accumulate;
  verify streamed usage is emitted (older builds ignore
  `stream_options` — the retry path should keep responses working
  either way).
- **Ollama** (openai-compatible type): Active Models rows show real
  size + TTL countdown from `/api/ps`; cache panel shows the "not
  exposed" note; stats accumulate if Ollama emits usage.
- **Cross-cutting:** Session/All-Time toggle, Clear buttons, model
  filter dropdown, API Endpoints copy buttons, browser console clean.
  Restart the proxy container and confirm All-Time stats survive
  (stats file lives next to the proxy state file in the `/data` volume
  — check `docker/docker-compose.proxy.yml` mounts it).

Record results (and fixes) in `LINUX_PROXY_REMAINING_WORK.md`.

## Task 2: build on it (after verification)

In rough priority order from the worklog:

1. Spark/Linux runbook (worklog item 2) — now writable from real
   verified commands.
2. Fork cleanup (worklog item 3): isolate MLX imports off the proxy
   path so the proxy suites collect on Linux without tricks; then the
   `omlx`→`omni` rename decision.
3. Anything verification shakes loose (metric name candidates, usage
   quirks per backend).

## Gotchas

- `pip install -e .` will fail on Linux (MLX deps are Apple-only). The
  proxy runs from `docker/Dockerfile.proxy` / `docker-compose.proxy.yml`
  via `omni serve`; check how that Dockerfile installs deps before
  trying to run tests on the host. Proxy tests need only fastapi/httpx/
  pytest/pytest-asyncio plus the omlx package importable — a venv with
  those and `PYTHONPATH=.` may suffice if full install fails.
- Serving stats depend on backends emitting `usage`; requests without
  it are still counted (zero tokens).
- Non-streaming requests record zero durations by design (don't skew
  tok/s averages); llama.cpp `timings` is the exception.
- `black` formats with target py313; run it through the project venv.
