# oMNI on Linux / DGX Spark — Runbook

Verified on a DGX Spark (NVIDIA GB10, 128 GB unified memory, DGX OS /
Ubuntu, Docker + NVIDIA container runtime) in June 2026. Everything here
was run as-is on that machine; adjust paths and model ids to taste.

oMNI in proxy mode is an MLX-free FastAPI container that fronts any
OpenAI-compatible inference backend (vLLM, llama.cpp server, Ollama,
remote endpoints), adding the admin dashboard, serving stats, an
Anthropic Messages bridge, and client integrations.

## Prerequisites

- Docker with the NVIDIA container toolkit (`docker run --gpus all â€¦`
  works). DGX OS ships this preconfigured.
- The repo checked out (the proxy image builds from `docker/Dockerfile.proxy`).
- The `omni` CLI: any Python ≥3.11 env with the repo on `PYTHONPATH`, or
  an editable install. On Linux a full `pip install -e .` fails (MLX deps
  are Apple-only) — the CLI itself only needs the standard library:

  ```bash
  # from the repo root
  python -m omlx.omni_cli --help     # or: omni --help if installed
  ```

## llama.cpp sidecar (verified)

```bash
omni serve --backend llamacpp \
  --model unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL \
  --served-model-name qwen-3.5-9b \
  --context-length 65535 --max-parallel 4
```

This generates `docker/docker-compose.llamacpp.yml` + `.env`, builds the
proxy image, and starts both containers. The sidecar runs
`ghcr.io/ggml-org/llama.cpp:server-cuda` with `--metrics` enabled and
pulls the GGUF from Hugging Face into `~/.cache/llama.cpp`.

Verify:

```bash
curl -s http://localhost:8000/metrics | grep ^llamacpp   # Prometheus families
curl -s http://localhost:8080/v1/models                  # via the proxy
# Dashboard: http://<spark>:8080/admin/dashboard
```

What to expect on the dashboard:

- Serving Stats accumulate from per-request `usage` (the proxy injects
  `stream_options.include_usage` into streamed requests and strips the
  trailing usage-only chunk when the client didn't ask for it).
- Repeated identical prompts move "Cached Tokens" / "Cache Efficiency"
  (llama.cpp reports `cached_tokens` from its slot prompt cache).
- The Active Models memory bar shows the sidecar's real GPU memory
  (queried via `nvidia-smi` inside the sidecar container) against host
  memory.
- Backend KV / Prefix Cache panel: current server-cuda builds (b9570+)
  no longer export `llamacpp:kv_cache_*`; the panel shows prompt/predict
  tok/s and Peak Context Tokens (`llamacpp:n_tokens_max`) instead.

## vLLM sidecar (verified)

```bash
omni serve --backend vllm \
  --model google/gemma-4-26B-A4B-it \
  --served-model-name gemma-4-26b-A4B \
  --context-length 65535 --max-parallel 4
```

Generates `docker/docker-compose.vllm.yml` + `.env` and starts
`vllm/vllm-openai:latest` plus the proxy. Useful env-file knobs
(editable in the admin UI under Settings → Backend, then Regenerate +
Restart): `VLLM_GPU_MEMORY_UTILIZATION` (default 0.8 — on unified
memory this is a share of system RAM), `VLLM_ENFORCE_EAGER=true`
(faster startup, required on some GB10 stacks), `VLLM_TOOL_CALL_PARSER`,
`VLLM_ENABLE_PREFIX_CACHING` (v1 engine default is on).

A 26B model takes a few minutes to load from a warm HF cache. Verify:

```bash
curl -s http://localhost:8000/metrics | grep -i prefix_cache
curl -s http://localhost:8080/admin/api/stats | jq .active_models.memory_pressure
```

Notes from verification:

- The generated compose passes `--enable-prompt-tokens-details` so vLLM
  reports `usage.prompt_tokens_details.cached_tokens` — without it the
  dashboard's cached-token stats stay at zero even though prefix caching
  works (the Prometheus hit counters still move).
- vLLM preallocates its whole `gpu_memory_utilization` budget, so the
  Active Models memory bar sits near the soft marker by design.
- 2026 vLLM images renamed the cache metrics
  (`vllm:prefix_cache_{hits,queries}_total`, `vllm:kv_cache_usage_perc`);
  the dashboard's candidate lists in `omlx/proxy/metrics.py` cover both
  generations. If a future rename blanks the panel again:
  `curl :8000/metrics | grep -i prefix` and extend the candidates.

## External Ollama (openai-compatible backend type)

Ollama runs on the host; the proxy container reaches it through the
Docker host gateway:

```bash
systemctl start ollama        # host service
omni serve --backend openai \
  --backend-url http://host.docker.internal:11434/v1
```

`docker/docker-compose.proxy.yml` already maps
`host.docker.internal → host-gateway`. The dashboard's Active Models
panel shows each loaded model's real size and TTL countdown from
`/api/ps`; the cache panel shows a "not exposed by this backend" note
(Ollama publishes no cache metrics).

## External vLLM / any OpenAI-compatible endpoint

```bash
omni serve --backend openai --backend-url http://other-box:8000/v1 \
  --backend-api-key sk-...
```

Backend URL/key/type can also be changed live in the admin UI
(Settings → Backend) without restarting the proxy.

## Local model scan (`--scan-models`)

```bash
omni serve --backend vllm --scan-models            # scan the HF cache
omni serve --backend vllm --scan-models --model-dir /path/to/models
```

Mounts the host model caches read-only into the proxy and adds a
"Local Models (scanned)" section to the Models tab listing every
downloaded model (safetensors → vLLM, GGUF → llama.cpp; type, size,
context length). "Use with sidecar" rewrites `OMNI_MODEL` in the
generated env and offers the backend-restart dialog — switching between
already-downloaded models takes one click plus the model load time.
`Rescan` picks up newly downloaded models without restarting anything.

Notes: the llama.cpp stack also scans `LLAMACPP_CACHE_DIR`; GGUF rows
pass the bare repo id (llama.cpp resolves its default quant). The flag
persists in the generated env file — set `OMLX_MODEL_SCAN=false` there
(or in the admin-regenerated env) to turn it back off.

## Operations

```bash
omni status                       # compose ps for the active stack
omni logs --target backend -f     # follow sidecar logs
omni restart --target backend     # restart sidecar (also: admin UI button)
omni stop --target both           # stop the stack
```

Switching backends = `omni stop --target both`, then
`omni serve --backend <other>`. Both sidecar stacks share host port 8000
for the backend and 8080 for the proxy. The proxy reconciles its
persisted backend routing with the launched stack on startup
(`OMLX_SIDECAR_BACKEND` is authoritative), so admin-UI settings saved in
a previous llama.cpp session won't point a fresh vLLM stack at a dead
hostname — per-backend profiles are archived and restored when you
switch back.

## Troubleshooting (all hit during verification)

- **Sidecar can't resolve DNS** (`Temporary failure in name resolution`,
  and `docker exec <ctr> cat /etc/resolv.conf` shows `127.0.0.53`):
  the container was created with a broken network sandbox — usually
  after a failed first `up` (e.g. port conflict with a still-running
  other stack). Fix: `docker compose --env-file docker/docker-compose.vllm.env \
  -f docker/docker-compose.vllm.yml up -d --force-recreate vllm`.
- **Port 8000 already in use**: the other sidecar stack is still up;
  `omni stop --target both` first (compose treats the old service as an
  orphan, it will not stop it for you).
- **HF cache**: both sidecars mount `${OMNI_HF_HOME}` (default
  `~/.cache/huggingface`) into the container; models cached on the host
  are reused. Gated models need `HF_TOKEN` exported in the shell that
  runs `omni serve` (it is passed through to the sidecar).
- **Stale generated compose**: `docker/docker-compose.{vllm,llamacpp}.yml`
  are git-ignored, generated files. If a template change doesn't show up,
  delete them or regenerate from the admin UI (Settings → Backend →
  Regenerate).
- **Every chat request 400s with "maximum context length is N tokens"
  on vLLM**: the proxy's Max Tokens sampling default
  (`OMLX_SAMPLING_MAX_TOKENS` / Settings → Generation → Max Tokens) is
  at or above the backend context window — vLLM enforces
  `prompt + max_tokens ≤ max_model_len`. The proxy now refuses to
  inject such a default (and retries once without it on a
  context-length 400), and the Settings UI flags the value with a
  "Use recommended" fix. Leave Max Tokens empty to use the backend's
  own limit.
- **`nvidia-smi` memory shows `[N/A]`** on GB10: unified memory. Real
  ceiling is host MemTotal (`/proc/meminfo`); per-process GPU usage is
  still reported by `nvidia-smi --query-compute-apps=used_memory ...`.
- **Stats persistence**: serving stats live in the `proxy-state` volume
  (`/data` in the proxy container) next to the proxy state file; they
  survive container restarts. "Clear All-Time" deletes the file.

## Not available in proxy mode (native oMLX/MLX features)

Model downloader/quantizer/uploader tabs, benchmark tab, MLX runtime
cache controls (SSD/hot tier), per-model unload, engine version panel,
speculative decoding (DFlash/MTP), oQ quantization, and the native
memory enforcer. Inference-level features (prefix caching, KV cache,
parallelism) belong to the backend and are configured through its env
file.
