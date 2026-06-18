# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`omlx` is a multi-model LLM/VLM/embedding/reranker inference server for Apple Silicon, built on Apple's MLX (`mlx-lm`, `mlx-vlm`, `mlx-embeddings`). It exposes OpenAI- and Anthropic-compatible HTTP APIs with continuous batching and a tiered (RAM + SSD) KV cache. Apple Silicon only; `requires-python >=3.11`.

## Commands

```bash
# Dev install (extras: mcp, audio, grammar, paroquant, modelscope)
pip install -e ".[dev]"

# Tests — default addopts already excludes `slow` and `integration`
pytest
pytest tests/test_<module>.py -v          # single file
pytest -m slow                            # require model files on disk
pytest -m integration                     # require a running server
pytest -m turboquant                      # TurboQuant KV cache suite

# Lint / format / type (line-length 88)
black .
ruff check .
mypy omlx

# Run the server (entry point: omlx.cli:main)
omlx serve --model-dir ~/models
omlx launch <tool>        # wire up Claude Code / Codex / Copilot / OpenCode / etc.
omlx diagnose menubar

# Build the native macOS app (venvstacks; first cold build 10–20 min)
apps/omlx-mac/Scripts/build.sh release
```

CI (`.github/workflows/ci.yml`) runs `pytest -m "not slow and not integration"` on macos-14 across Python 3.11–3.14.

## Architecture

Request flow, outermost to innermost:

- **`server.py`** — FastAPI app (~190KB). OpenAI endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/models`), Anthropic `/v1/messages`, and the `/admin` dashboard.
- **`api/adapters/`** (`openai.py`, `anthropic.py`, `sse_formatter.py`) — translate wire formats to/from the internal `Request`/`RequestOutput` (`request.py`). API model schemas live in `api/*_models.py`.
- **`engine_pool.py`** — multi-model lifecycle: LRU eviction, per-model TTL, pinning, manual load/unload, and routing a request to the right engine type.
- **`engine/`** — `BatchedEngine` (LLM continuous batching), `VLMBatchedEngine`, `EmbeddingEngine`, `RerankerEngine`, `DFlashEngine` (speculative decoding), `STT/TTS/STSEngine` (audio). Shared core is `engine_core.py` (`AsyncEngineCore`, `EngineConfig`).
- **`scheduler.py`** — FCFS request scheduler wrapping mlx-lm's `BatchGenerator`. The largest file in the tree (~320KB); changes here are high-risk.
- **`cache/`** — tiered KV cache (vLLM-inspired): `PagedCacheManager` (GPU, block-based, copy-on-write, prefix sharing) → in-memory hot tier → `PagedSSDCacheManager` (cold SSD tier in safetensors; survives server restart).
- **`process_memory_enforcer.py`** / **`memory_monitor.py`** — total-process memory ceiling (default system RAM − 8GB) to prevent system-wide OOM, plus TTL enforcement.
- **`model_discovery.py`** — auto-detects model type (LLM / VLM / OCR / embedding / reranker) from `--model-dir` subdirectories.
- **`patches/`** — post-load monkey-patches applied to loaded models, both for specific families (`deepseek_v4/`, `step3p7/`, `qwen3_5_attention`, `qwen3_6_nested_visual`, `mlx_lm_mtp/`, `mlx_vlm_mtp/`, `dflash_lifecycle`, `specprefill`) and for upstream mlx-lm/mlx-vlm fixes.

Supporting subsystems: **`admin/`** (vendored offline web UI — chat, benchmark, HF/ModelScope downloader, i18n, per-model settings), **`integrations/`** (one-click client config), **`mcp/`** (Model Context Protocol), **`eval/`** (benchmark datasets), **`oq.py`** (oQ quantization).

## Conventions & gotchas

- **Git-pinned MLX deps.** `mlx-lm`, `mlx-vlm`, `mlx-embeddings`, `dflash-mlx`, and `mlx-audio` are pinned to specific git commits in `pyproject.toml`. `[tool.uv] override-dependencies` forces the resolver past transitive pins (e.g. mlx-audio → `mlx-lm==0.31.1`). When bumping any pinned MLX commit, update the override block too, or resolution breaks.
- **torch is stubbed.** `omlx/_torch_stub.py` lets the default install run without PyTorch; only the `[grammar]`/xgrammar extra pulls real torch. Don't add torch imports on the default code path.
- **Lazy top-level imports.** `omlx/__init__.py` defers public exports via a `_LAZY` map to keep CLI startup fast — add new public names to `_LAZY`, not to module-level imports.
- **License header.** Every source file starts with `# SPDX-License-Identifier: Apache-2.0`.
- **Test naming.** Source `omlx/<module>.py` → test `tests/test_<module>.py`. New code should ship tests; check whether existing tests are affected when editing.
- **Settings.** Persisted to `~/.omlx/settings.json`; CLI flags take precedence over the file. Env vars include `OMLX_MODEL_DIR`, `OMLX_PORT`.
