# oMNI Admin UI — Autonomous QA-and-Fix Loop

**Date:** 2026-06-15
**Status:** Design approved (pending spec review)
**Branch for implementation:** `ui-qa-loop`

## Problem

oMNI's admin UI exposes many settings across several areas (Settings, Status/Serving
Stats, Models, Logs, Benchmark, Chat) and three operational shapes (vLLM sidecar,
router / openai-compatible passthrough, and the `omni launch <tool>` client
integrations). Today these are verified by HTTP/pytest plus occasional manual,
real-browser spot checks; `LINUX_PROXY_REMAINING_WORK.md` explicitly flags
"browser-console check still worth an eyeball on a real browser" as an open gap.

We want a loop in which Claude Code walks every UI control, exercises it across the
relevant backends, judges it against a known-good expectation, **fixes it when it's
broken, re-tests, and moves on** — running largely unattended but with hard
guardrails.

## Goals

- Systematically exercise every admin UI control across **vLLM sidecar** and
  **router / openai-compatible** modes.
- Verify the **`omni launch claude`** integration end-to-end: Claude Code talks
  through the proxy to the backend and gets a real completion.
- When a control misbehaves, **auto-fix → re-test → advance**, fully unattended,
  with each fix isolated as one reviewable commit.
- Catch frontend/render/JS/console regressions, not just API behavior.
- Produce a durable QA artifact (the behavior matrix) and a final report.

## Non-Goals (YAGNI)

- llama.cpp sidecar and Ollama-specific paths (`/api/tags`, `/api/ps` TTL display)
  are **out of scope** for this iteration; the matrix schema leaves room to add them.
- No new general-purpose browser-test framework beyond the minimal Playwright smoke
  needed here.
- No actual large-model launches or vLLM model swaps to big models (see Safety).
- Not a replacement for the existing pytest suites; this complements them.

## Definition of "works as intended" (the oracle)

A control passes when its **observable side effect** matches the expectation recorded
in the behavior matrix, derived up front from: the route handlers
(`omlx/proxy/admin.py`, `omlx/admin/routes.py`), the dashboard partials
(`_settings.html`, `_status.html`, `_models.html`, `_logs.html`, `_bench.html`),
`LINUX_PROXY_REMAINING_WORK.md`, and the existing `test_admin_*` / `test_settings` /
`test_proxy*` tests. The loop judges against the matrix, **not** ad-hoc intuition —
this is what makes unattended auto-fix safe.

## Architecture

Four artifacts. State lives in files, not in conversation context, so the loop
survives context compaction and is stop/resume-able.

### 1. Behavior matrix — `docs/qa/ui-loop-matrix.md` (oracle + tracker)

One row per **(UI area × control × backend mode)**:

| column | meaning |
|---|---|
| `id` | stable slug, e.g. `settings.sampling.max_tokens@vllm` |
| `area / control` | human label |
| `exercise` | admin HTTP endpoint + payload **and** browser locator |
| `expected` | observable side effect (persisted `OMLX_*`, regenerated env/compose, injected request field, 409 guard dialog, metric movement, …) |
| `backend` | `vllm` / `router` / `both` |
| `status` | `untested \| pass \| fail \| fixed \| blocked` |
| `notes / commit` | fix commit hash or block reason |

Authored in Phase 0. This is both the spec of intended behavior and the progress
ledger.

**UI areas to enumerate** (from the templates):
- **Settings** (`_settings.html`): backend URL/API key/type; sampling defaults
  (`OMLX_SAMPLING_*`), incl. the Max-Tokens-vs-context guard and "Use recommended
  (ctx/2)"; vLLM launch settings (model, context, gpu-memory-utilization,
  max_num_seqs, demand-aware util hint, unified-memory guard 409 + "Launch anyway");
  HF offline toggle; served-name resync on model change; router-mode hiding of
  sidecar launch settings.
- **Status / Serving Stats** (`_status.html`): session/all-time scopes, both Clear
  endpoints, per-model filter, cached-token / cache-efficiency, prompt/gen tok/s,
  Active Models memory bar, Backend KV/Prefix Cache panel, API Endpoints panel.
- **Models** (`_models.html`): local-model scan listing, "Use with sidecar" switch
  (settings + restart flow) — **whitelisted small models only**.
- **Logs** (`_logs.html`): container-log streaming into the Logs tab.
- **Benchmark** (`_bench.html`, `_bench_accuracy.html`): smoke only.
- **Chat** (`chat.html`): streamed + non-streamed completion, usage display.
- **Integrations**: `omni launch claude` end-to-end.

### 2. Driver harness — `tests/ui_loop/`

- `admin_client.py` — logged-in HTTP session; one helper per admin control plus
  side-effect assertions (read back `~/.omlx/settings.json`, generated
  compose/env, container state via the Docker Engine API).
- `browser_smoke.py` — minimal **Playwright** headless Chromium: load each admin
  page, assert **zero console errors**, capture a screenshot artifact.
- `backend_assert.py` — vLLM `/metrics` (prefix-cache hits/queries, KV usage),
  `/v1/models`; router `/v1/models` and `/api/ps` presence.
- `launch_claude_e2e.py` — replicate `omni launch claude`'s env wiring
  (`omlx/integrations/claude.py`) and run `claude -p "ping"` **non-interactively**
  against the proxy's Anthropic endpoint; assert a real completion returns through
  the active backend.
- `safety.py` — small-model whitelist (e.g. `Qwen/Qwen3-1.7B`); heavy-op gate
  (dry-run / compose-gen only); **snapshot+restore** of `~/.omlx/settings.json` and
  generated compose/env around every row so rows don't poison each other (the
  `omni-serve-verify-isolation` lesson).

### 3. Loop runner (the "loop")

A self-paced **`/loop`** (dynamic mode). Each wake-up:
1. Read matrix → pick next `untested`/`fail` row.
2. Snapshot state (safety) → exercise (HTTP + browser smoke + backend assert).
3. Judge against `expected`.
4. On fail → systematic-debugging → fix → one commit → restart only the affected
   container if required → re-test the same row.
5. Restore state → update `status` → `ScheduleWakeup` for the next row.
6. Repeat until matrix is all green, or `blocked`/budget reached.

### 4. Guardrails

- Dedicated branch `ui-qa-loop`; **one structured commit per fix** (reviewable,
  revertable).
- **Thrash cap**: max N (default 3) re-tries per row; then mark `blocked`, record the
  reason, and advance — never infinite-loop a single control.
- Heavy-op dry-run + small-model whitelist; the loop must never launch/swap to a
  large model (Spark power-cycle history in `LINUX_PROXY_REMAINING_WORK.md`).
- Stop file (`tests/ui_loop/STOP`) and iteration budget for clean termination /
  interruption.
- Snapshot/restore around every row.

## Error handling

- Harness/exercise error (not a product bug, e.g. backend down) → row `blocked`,
  reason recorded, loop continues; surfaced in the report.
- Backend restart needed → restart only the affected container, wait for health,
  bounded retries, else `blocked`.
- Browser-console errors are first-class failures (this is the gap we're closing).

## Testing the harness itself

Phase 1 is a supervised dry-run over ~3 representative rows (one passing, one with a
seeded/known fail, one heavy-op-via-dry-run) to prove the oracle, the fix loop, and
snapshot/restore before any unattended run.

## Rollout

- **Phase 0** — branch + harness + author the matrix.
- **Phase 1** — supervised 3-row dry-run; stop and present results.
- **Phase 2** — *user-initiated* unattended `/loop`: walk the matrix, auto-fix,
  commit, re-test.
- **Phase 3** — report: green summary + fix commits + blocked rows.

The hand-off after Phase 1 to the user starting Phase 2 is explicit and approved.

## Open questions / assumptions

- Assumes the vLLM sidecar and proxy stay up between rows; bringing backends up/down
  is itself a tested flow but bounded by the heavy-op gate.
- `claude -p` non-interactive print mode is the automation entry for the
  integration check; if unavailable in this environment, fall back to a direct
  Anthropic-endpoint HTTP probe using the same env wiring.
