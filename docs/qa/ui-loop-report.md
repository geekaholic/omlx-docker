# oMNI Admin UI QA Loop — Report

**Date:** 2026-06-15 · **Branch:** `ui-qa-loop` · **Backends in scope:** vLLM sidecar + router/openai-compatible · **Model:** Qwen3-1.7B (live)

## Result

| status | count |
|--------|-------|
| pass | 41 |
| fixed | 2 |
| blocked | 10 |
| untested | 0 |
| **total** | **53** |

All 53 matrix rows resolved. Full detail + per-row evidence in `docs/qa/ui-loop-matrix.md`.

## Fixes applied (one commit each)

- **`2e80418` fix(admin): owner_hash `:href` optional chaining** — real frontend bug found by the Playwright console-error smoke on the very first run: the dashboard threw `Cannot read properties of null (reading 'owner_hash')` because Alpine evaluates a bench `:href` binding even when `x-show` is false and `benchDeviceInfo` is null. Fixed, rebuilt + recreated the proxy image, re-smoked green. Closes the "eyeball on a real browser" gap from `LINUX_PROXY_REMAINING_WORK.md`.
- **`flat-key POST contract`** (harness, not product) — 11 settings initially looked broken; root cause was a test-methodology bug: admin GET nests display values but `POST /admin/api/global-settings` accepts only FLAT keys (`sampling_temperature`, `network_http_proxy`, `vllm_dtype`). Added `flat_settings_key()` + tests. The product was correct; the harness contract was wrong. (This is the matrix oracle earning its keep — it prevented 11 phantom "fixes".)

## Headline finding — needs your decision

**`integration.launch_claude@vllm` — Claude Code over-requests output tokens for small-context backends.**
`claude -p` 400s: client `max_tokens`(32000) + input(8961) > Qwen3-1.7B context(40960). On the Anthropic `/v1/messages` path, `anthropic_to_openai_chat_body` forwards the **client's** `max_tokens` verbatim (`app.py:374`); the context-fit guard only caps the proxy's **own injected** default (`app.py:540-549`), so the documented context-400 retry can't recover. Three valid fix sites:
1. **Integration** — `omni launch claude` sets `CLAUDE_CODE_MAX_OUTPUT_TOKENS` derived from `ctx.context_window` (localized, low-risk; mirrors the existing `CLAUDE_CODE_AUTO_COMPACT_WINDOW` wiring). *Recommended.*
2. **Proxy** — on a context-length 400, retry with `max_tokens` capped to `context − input_tokens` (covers all Anthropic clients, but changes core request handling).
3. **Document** the limitation for small-context backends.

Also: the e2e probe's `claude_env` omits the real `omni launch claude` context wiring — make it faithful regardless of the chosen fix.

## Other observations (low severity)

- **Proxy-mode-inert Settings controls** — `memory.guard_tier`, `mcp.config_path`, `integrations.markitdown`, `server.disable_log_stats`, and `auth.sub_keys` are hardcoded/absent in the proxy-mode GET payload; POSTs return 200 but don't persist. Shared template with native oMLX. **Follow-up:** confirm these are *hidden* in the proxy Settings UI (same cleanup as the downloader/quantizer tabs) — if visible, it's a UX gap.
- **Backend API key echoed in GET** — `/admin/api/global-settings` returns the configured backend API key in plaintext (`admin.py:1523`). Admin-only, but worth masking (`*_set` boolean is already present).
- **Active Models `actual_size` null** — the per-process nvidia-smi-exec memory path isn't populating on GB10 here; the bar falls back to the util budget (`estimated_size`). Renders fine; worth a look vs. the `769bcc0` behavior in the doc.

## Blocked rows (10) — why, and how to clear

**Real finding (1):** `integration.launch_claude@vllm` — see Headline above.

**Proxy-mode inert / display stubs (5):** `memory.guard_tier`, `mcp.config_path`, `integrations.markitdown`, `server.disable_log_stats`, `auth.subkeys` — not backed in proxy mode (see Observations).

**Held back for safety / need a throwaway fixture (3):**
- `auth.api_key` — setting the proxy's own key risks locking the admin API mid-loop.
- `status.stats.clear_alltime` — irreversibly wipes persisted all-time accounting (no restore API).
- `settings.sampling.max_context_window` — POST can't unset the override to restore the original `None` in-place.
These three are fine to test on a disposable proxy-state container.

**Environment (1):** `status.active_models.ollama@router` — no Ollama reachable (`:11434` down on host + container); needs a live Ollama to verify `/api/ps` size+TTL.

## Harness notes for re-runs

- Settings POSTs use **flat keys** (`flat_settings_key()`), not the nested GET shape.
- Backend metric deltas (prefix cache) need reads **>5s apart** to outlast the 5s metric TTL cache.
- Settings/compose-touching rows snapshot `docker/docker-compose.vllm.{yml,env}`; model/backend switches also restore the in-container override via API.
- Template fixes need an **image rebuild** (`docker compose -f docker/docker-compose.vllm.yml build omlx-proxy && up -d --no-deps omlx-proxy`), not just a restart.
- Backend-type switches must be wrapped in try/finally to guarantee restore to vLLM.
