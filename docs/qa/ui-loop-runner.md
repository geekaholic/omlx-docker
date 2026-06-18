# oMNI UI QA Loop — runner prompt

You are running one iteration of the oMNI admin UI QA-and-fix loop. Work on branch
`ui-qa-loop`. Setup each iteration: `source .venv-proxy/bin/activate && export
PYTHONPATH=.` (proxy + vLLM containers must be `Up`; dashboard returns 200).

## Each iteration
1. If `tests/ui_loop/STOP` exists, stop the loop and write the report (see End).
   Otherwise continue.
2. Open `docs/qa/ui-loop-matrix.md`. Pick the FIRST row whose status is `untested`
   or `fail`. If none, go to End.
3. Snapshot state with `SettingsSnapshot([~/.omlx/settings.json,
   docker/docker-compose.vllm.yml, docker/docker-compose.vllm.env])`
   (paths the proxy actually persists; adjust if the live state path differs).
4. Exercise the row's control using the helper named in its `exercise` cell
   (`AdminClient`, `smoke_page`, `vllm_metric_families`/`served_model_listed`,
   `run_claude_ping`). For ANY model launch/swap, call
   `assert_heavy_op_allowed(model)` FIRST — if it raises, switch to
   dry-run / compose-gen only.
5. Judge against the `expected` cell.
   - PASS → set status `pass`, restore snapshot, commit the matrix, schedule the
     next iteration.
   - FAIL → go to Fix.

## Fix (max 3 attempts per row, then mark `blocked`)
1. Use superpowers:systematic-debugging to find the root cause in the relevant
   module (`omlx/proxy/*.py`, `omlx/admin/routes.py`, the dashboard templates).
2. Apply the minimal fix. Add/adjust a unit test under `tests/` that captures the
   bug (source `omlx/<m>.py` → `tests/test_<m>.py`).
3. Restart only the affected container if needed: `docker restart
   docker-omlx-proxy-1` (proxy/admin code or templates) or
   `AdminClient.restart_sidecar()` (backend launch settings). Wait for health
   (dashboard 200 / backend `/v1/models`).
4. Re-exercise the row. On pass → status `fixed`, record the commit hash in notes,
   `git commit` the fix (ONE commit per fix). On the 3rd failure → status
   `blocked` with the reason.
5. Restore the snapshot regardless of outcome.

## End
When no `untested`/`fail` rows remain, or `tests/ui_loop/STOP` is present: write
`docs/qa/ui-loop-report.md` summarizing pass/fixed/blocked counts, every fix commit
hash, and every blocked row with its reason. Commit it.

## Guardrails (never violate)
- Never really launch/swap to a non-whitelisted model (`assert_heavy_op_allowed`).
- One commit per fix. Stay on branch `ui-qa-loop`. Never touch `main`.
- Always restore the snapshot after a row.
- Browser console/page errors are real failures, not harness bugs — never weaken
  `browser_smoke` to make a page "pass".
- If a row needs a real backend the box can't run, mark it `blocked` (reason) and
  move on; do not hang the loop.

## How to drive it with /loop
From the repo root, self-paced:

```
/loop docs/qa/ui-loop-runner.md
```

The loop reads the matrix, does one (or a few) rows per wake-up, updates statuses,
and schedules the next wake-up until the matrix is green/blocked or STOP appears.
Remove `tests/ui_loop/STOP` to begin Phase 2.
