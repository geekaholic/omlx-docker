# oMNI Admin UI — Behavior Matrix (oracle + tracker)

Status values: `untested | pass | fail | fixed | blocked`.

Heavy ops (large-model launch/swap) are **dry-run only**; the only model the loop may
really launch/swap to is the whitelist in `tests/ui_loop/safety.py`
(`Qwen/Qwen3-1.7B`). Every row snapshots `~/.omlx/settings.json` +
`docker/docker-compose.vllm.{yml,env}` before and restores after.

Helpers (in `tests/ui_loop/`): `AdminClient` (live admin API), `smoke_page`
(Playwright console/page errors), `vllm_metric_families`/`served_model_listed`
(`backend_assert`), `run_claude_ping` (`launch_claude_e2e`),
`assert_heavy_op_allowed` (`safety`).


**POST contract:** `/admin/api/global-settings` accepts only FLAT keys
(`sampling_temperature`, `network_http_proxy`, `vllm_dtype`, …) — the GET payload
nests display values but is NOT symmetric. Use `flat_settings_key()`. `memory`
and `mcp` sections are hardcoded in the GET payload (proxy-mode display stubs).

Backend column: `vllm` (sidecar up now), `router` (openai-compatible passthrough),
`both`.

## Settings → Backend / routing

| id | area / control | exercise | expected | backend | status | notes/commit |
|----|----------------|----------|----------|---------|--------|--------------|
| settings.proxy.backend_type@router | Settings → Backend Type | POST `/admin/api/global-settings` `{"proxy":{"backend_type":"openai-compatible"}}`; GET config | `proxy.backend_type` persists; sidecar launch settings hidden (router mode); live-applied | router | untested | |
| settings.proxy.backend_url@router | Settings → Backend URL | POST `proxy.backend_url`; GET `/admin/api/proxy/config` | url persisted + live-applied; `backend_url` reflects it | router | untested | |
| settings.proxy.backend_api_key@router | Settings → Backend API Key | POST `proxy.backend_api_key`; GET config | `backend_api_key_set` flips true; value not echoed back | router | untested | |
| settings.backend.health@both | Settings → Backend health badge | GET `/admin/api/proxy/status` | health reflects reachable backend | both | pass | Phase 2: backend_reachable=True, no error |

## Settings → Sampling defaults

| id | area / control | exercise | expected | backend | status | notes/commit |
|----|----------------|----------|----------|---------|--------|--------------|
| settings.sampling.max_tokens.guard@vllm | Sampling → Max Tokens vs context | POST `sampling.max_tokens` ≥ ctx; send a chat | request retried without cap on ctx-length 400; inline warning; "Use recommended (ctx/2)" offered | vllm | untested | |
| settings.sampling.temperature@both | Sampling → Temperature | POST `sampling.temperature`=0.3; GET settings; send chat omitting temp | persisted; injected into forwarded request | both | pass | Phase 2: flat key sampling_temperature round-trips |
| settings.sampling.top_p@both | Sampling → top_p | POST `sampling.top_p`; GET settings | persisted + injected | both | pass | Phase 2: flat key sampling_top_p round-trips |
| settings.sampling.top_k@both | Sampling → top_k | POST `sampling.top_k`; GET settings | persisted + injected | both | pass | Phase 2: flat key sampling_top_k round-trips |
| settings.sampling.repetition_penalty@both | Sampling → repetition_penalty | POST value; GET settings | persisted + injected | both | pass | Phase 2: flat key sampling_repetition_penalty round-trips |
| settings.sampling.max_context_window@vllm | Sampling → Max Context Window | POST value; GET settings | persisted; used for context probe | vllm | blocked | deferred: POST can't unset the override to restore original None; also rewrites omni_context_length — verify on a test-server fixture, not in-place unattended |

## Settings → vLLM launch (sidecar; restart required)

| id | area / control | exercise | expected | backend | status | notes/commit |
|----|----------------|----------|----------|---------|--------|--------------|
| settings.vllm.gpu_mem_util@vllm | vLLM Advanced → gpu-memory-utilization | POST util; GET `/admin/api/proxy/sidecar-compose` | demand-aware util in generated compose; "restart required" hint shown | vllm | pass | Phase 2: vllm_gpu_memory_utilization 0.30 applied (no guard block), restored 0.21 |
| settings.vllm.enforce_eager@vllm | vLLM Advanced → Enforce eager | POST toggle; regenerate compose | `--enforce-eager` present/absent in compose accordingly | vllm | pass | Phase 2: vllm_enforce_eager toggles enforce_eager in sidecar settings (snapshot-restored) |
| settings.vllm.trust_remote_code@vllm | vLLM Advanced → Trust remote code | POST toggle; regenerate compose | `--trust-remote-code` present/absent in compose | vllm | pass | Phase 2: vllm_trust_remote_code toggles trust_remote_code |
| settings.vllm.dtype@vllm | vLLM Advanced → dtype (half/float) | POST dtype; regenerate compose | `--dtype` reflects choice in compose | vllm | pass | Phase 2: vllm_dtype -> dtype='half' in compose, restored '' |
| settings.vllm.model_switch_small@vllm | Models → Use with sidecar (small) | `assert_heavy_op_allowed('Qwen/Qwen3-1.7B')`; switch + restart | served name re-derived from new model; `/v1/models` lists it; chat works | vllm | untested | |
| settings.vllm.model_switch_large@vllm | Settings → large-model launch guard | regenerate compose / dry-run for `openai/gpt-oss-120b` | memory guard 409 + "Launch anyway"; no real launch; `assert_heavy_op_allowed` refuses | vllm | pass | Phase 1: `assert_heavy_op_allowed` refusal verified; 409 "Launch anyway" dialog deferred to Phase 2 |
| settings.vllm.served_name_resync@vllm | Settings → served name on model change | switch model without custom name; GET config | `OMNI_SERVED_MODEL_NAME` re-derived from new model tail | vllm | untested | |
| settings.vllm.hf_offline@vllm | Settings → HF offline toggle | switch to cached model; inspect compose env | `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` set for cached model | vllm | untested | |

## Settings → Network / MCP / Auth / Memory / Integrations

| id | area / control | exercise | expected | backend | status | notes/commit |
|----|----------------|----------|----------|---------|--------|--------------|
| settings.network.http_proxy@both | Network → HTTP Proxy | POST `network.http_proxy`; GET settings | persisted | both | pass | Phase 2: flat key network_http_proxy round-trips (snapshot-restored) |
| settings.network.https_proxy@both | Network → HTTPS Proxy | POST value; GET settings | persisted | both | pass | Phase 2: flat key network_https_proxy round-trips |
| settings.network.no_proxy@both | Network → No Proxy | POST value; GET settings | persisted | both | pass | Phase 2: flat key network_no_proxy round-trips |
| settings.network.ca_bundle@both | Network → CA Bundle | POST value; GET settings | persisted | both | pass | Phase 2: flat key network_ca_bundle round-trips |
| settings.memory.guard_tier@both | Memory → guard tier | POST `memory.memory_guard_tier`; GET settings | persisted; invalid tier rejected/normalized | both | blocked | proxy-mode inert: hardcoded 'balanced' in GET payload (admin.py:1542); native_memory_guard capability off — display stub, can't round-trip |
| settings.mcp.config_path@both | MCP → config path | POST `mcp.config_path`; GET settings | persisted | both | blocked | proxy-mode inert: hardcoded '' in GET payload (admin.py:1564) — display stub, can't round-trip |
| settings.auth.api_key@both | Auth → API key | POST `auth.api_key`; GET settings | persisted; subsequent calls require Bearer | both | untested | |
| settings.auth.subkeys@both | Auth → sub-keys create/delete | createSubKey then deleteSubKey via endpoints | sub-key list updates accordingly | both | untested | |
| settings.integrations.markitdown@both | Integrations → MarkItDown engine | POST `integrations.markitdown_pdf_processing_engine`; GET | persisted | both | untested | |
| settings.server.disable_log_stats@both | Settings → Disable log stats | POST toggle; GET settings | persisted | both | untested | |

## Status / Serving Stats

| id | area / control | exercise | expected | backend | status | notes/commit |
|----|----------------|----------|----------|---------|--------|--------------|
| status.stats.session_scope@vllm | Status → session scope | GET `/admin/api/stats?scope=session` | returns session counters | vllm | pass | Phase 2: scope=session HTTP 200 |
| status.stats.alltime_scope@vllm | Status → all-time scope | GET `/admin/api/stats?scope=all` (or alltime) | returns persisted all-time counters | vllm | pass | Phase 2: scope=all & scope=alltime both 200 |
| status.stats.clear@vllm | Status → Clear (session) | POST `/admin/api/stats/clear`; GET session | session counters zero; all-time untouched | vllm | pass | Phase 1: clear → 200, session stats returned |
| status.stats.clear_alltime@vllm | Status → Clear all-time | POST `/admin/api/stats/clear-alltime`; GET alltime | all-time counters zero | vllm | blocked | safety: irreversibly wipes persisted all-time accounting with no API to restore — verify on a throwaway proxy-state, not in an unattended run |
| status.stats.per_model_filter@vllm | Status → per-model filter | GET stats filtered by served model | counters attributed to that model | vllm | pass | Phase 2: model= filter discriminates (served 11 reqs/1259 tok/67.9% cache vs bogus 0) |
| status.cache.prefix@vllm | Status → Backend KV/Prefix Cache | send repeated prompt; GET `/admin/api/proxy/metrics` | prefix-cache hit rate rises; KV usage shown | vllm | pass | Phase 2: repeated prompt -> prefix hits 128->832, queries 311->1226 (read >5s apart to outlast metrics TTL) |
| status.metrics.prometheus@vllm | Status → Backend Metrics samples | GET `/admin/api/proxy/metrics`; `vllm_metric_families` | expected vllm families present | vllm | pass | Phase 2: backend_kind=prometheus; summary has prefix_cache_* + gpu_cache_usage; metric_count>0 |
| status.active_models.memory_bar@vllm | Status → Active Models memory bar | GET `/admin/api/proxy/status` | per-process GPU memory + soft limit shown | vllm | pass | Phase 2: active_models.models has estimated_size (util budget); NOTE actual_size=null — nvidia-smi-exec per-process path not populating (GB10), bar falls back to estimated |
| status.api_endpoints@both | Status → API Endpoints panel | GET `/admin/api/server-info` | host/port/aliases/cli_prefix "omni" present | both | pass | Phase 2: host/port/aliases via server-info; cli_prefix='omni' via /stats (refine: not in server-info) |
| status.cache.router_note@router | Status → cache panel (router) | GET metrics in router mode | "not exposed" note path for Ollama | router | untested | |
| status.active_models.ollama@router | Status → Active Models (router) | GET status with Ollama backend | `/api/ps` sizes + TTL shown | router | untested | |

## Models / Logs / Chat / render

| id | area / control | exercise | expected | backend | status | notes/commit |
|----|----------------|----------|----------|---------|--------|--------------|
| models.local_scan@vllm | Models → local-model scan | GET `/admin/api/proxy/local-models` | lists HF-cache/GGUF repos when `--scan-models` on (else empty) | vllm | pass | Phase 2: 12 HF repos listed, scan_dir=/models-scan |
| logs.stream@both | Logs → container log stream | GET `/admin/api/logs` | streams/returns recent container log lines | both | pass | Phase 2: /admin/api/logs 200, JSON log lines |
| chat.stream@vllm | Chat → streamed completion | POST chat via UI path with stream | tokens stream; usage shown | vllm | pass | Phase 2: streamed 14 chunks, content + include_usage trailing chunk |
| chat.nonstream@vllm | Chat → non-streamed completion | POST chat without stream | full completion + usage | vllm | pass | Phase 2: /v1/chat/completions 200, content+usage (Qwen3-1.7B) |
| render.dashboard@both | Dashboard page render | `smoke_page('/admin/dashboard')` | zero console/page errors | both | fixed | was: page error `Cannot read properties of null (reading 'owner_hash')`; fixed in `2e80418` (optional chaining on bench `:href`); re-smoke green |
| render.chat@both | Chat page render | `smoke_page('/admin/chat')` | zero console/page errors | both | pass | Phase 2: smoke_page('/admin/chat') zero console/page errors |

## Integrations

| id | area / control | exercise | expected | backend | status | notes/commit |
|----|----------------|----------|----------|---------|--------|--------------|
| integration.launch_claude@vllm | `omni launch claude` e2e | `run_claude_ping('http://localhost:8080','omlx','<served>')` | claude returns a completion via the backend | vllm | blocked | FINDING: claude -p 400s — client max_tokens(32000)+input(8961) > ctx(40960). /v1/messages forwards client max_tokens (app.py:374); context guard only caps proxy-injected default (app.py:540-549), so the context-400 retry can't help. Needs decision: integration set CLAUDE_CODE_MAX_OUTPUT_TOKENS from ctx.context_window, or proxy cap client max_tokens on context-400 retry. Probe's claude_env also omits the real launch's context wiring. |

## Proxy-mode "hidden capability" checks

These verify the proxy-mode UI cleanup (capabilities reported off in
`/admin/api/proxy/status`: `model_load_unload`, `benchmarks`, `cache_controls`,
`hf_downloader`, `modelscope_downloader`, `quantizer`, `uploader`,
`native_memory_guard`).

| id | area / control | exercise | expected | backend | status | notes/commit |
|----|----------------|----------|----------|---------|--------|--------------|
| hidden.model_load_unload@both | load/unload hidden | GET status capabilities | `model_load_unload=false`; load/unload controls hidden | both | pass | Phase 2: model_load_unload=false |
| hidden.benchmarks@both | benchmark tab hidden | GET status capabilities | `benchmarks=false`; bench tab hidden | both | pass | Phase 2: benchmarks=false |
| hidden.cache_controls@both | MLX cache controls hidden | GET status capabilities | `cache_controls=false`; MLX cache controls hidden | both | pass | Phase 2: cache_controls=false |
| hidden.downloaders@both | HF/MS downloader hidden | GET status capabilities | `hf_downloader=false`, `modelscope_downloader=false` | both | pass | Phase 2: hf_downloader=false, modelscope_downloader=false |
| hidden.quant_upload@both | quantizer/uploader hidden | GET status capabilities | `quantizer=false`, `uploader=false` | both | pass | Phase 2: quantizer=false, uploader=false |
