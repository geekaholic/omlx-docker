# SPDX-License-Identifier: Apache-2.0
"""Tests for the proxy's opt-in local model scan."""

import json

from omlx.proxy.local_models import scan_local_models


def _make_hf_cache_model(
    hub: "Path",
    org: str,
    name: str,
    *,
    config: dict | None = None,
    weight_files: tuple[str, ...] = ("model.safetensors",),
    weight_bytes: int = 1024,
):
    repo = hub / f"models--{org}--{name}"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123")
    if config is not None:
        (snapshot / "config.json").write_text(json.dumps(config))
    for filename in weight_files:
        target = snapshot / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\0" * weight_bytes)
    return snapshot


def test_scan_hf_cache_safetensors_model(tmp_path):
    hub = tmp_path / "hf" / "hub"
    hub.mkdir(parents=True)
    _make_hf_cache_model(
        hub,
        "qwen",
        "tiny",
        config={
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "max_position_embeddings": 32768,
        },
    )

    rows = scan_local_models(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["repo_id"] == "qwen/tiny"
    assert row["model_format"] == "safetensors"
    assert row["backends"] == ["vllm"]
    assert row["model_type"] == "llm"
    assert row["context_length"] == 32768
    assert row["size_bytes"] > 0


def test_scan_hf_cache_gguf_repo(tmp_path):
    hub = tmp_path / "hf" / "hub"
    hub.mkdir(parents=True)
    _make_hf_cache_model(
        hub,
        "unsloth",
        "tiny-GGUF",
        config=None,
        weight_files=("UD-Q4_K_XL/tiny-UD-Q4_K_XL.gguf",),
    )

    rows = scan_local_models(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["repo_id"] == "unsloth/tiny-GGUF"
    assert row["model_format"] == "gguf"
    assert row["backends"] == ["llama.cpp"]


def test_scan_plain_model_dir_and_loose_gguf(tmp_path):
    plain = tmp_path / "my-model"
    plain.mkdir()
    (plain / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    )
    (plain / "model.safetensors").write_bytes(b"\0" * 64)
    (tmp_path / "loose.gguf").write_bytes(b"\0" * 32)

    rows = scan_local_models(tmp_path)

    by_id = {r["repo_id"]: r for r in rows}
    assert by_id["my-model"]["backends"] == ["vllm"]
    assert by_id["loose"]["backends"] == ["llama.cpp"]


def test_scan_skips_incomplete_and_irrelevant_entries(tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    # Aborted download: refs but no snapshots (seen in the wild).
    aborted = hub / "models--org--aborted"
    (aborted / "refs").mkdir(parents=True)
    (aborted / "refs" / "main").write_text("deadbeef")
    # Random files/dirs are ignored.
    (hub / "CACHEDIR.TAG").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "empty-dir").mkdir()

    assert scan_local_models(tmp_path) == []


def test_scan_missing_root_returns_empty(tmp_path):
    assert scan_local_models(tmp_path / "nope") == []


def test_scan_dedupes_by_repo_id(tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    _make_hf_cache_model(
        hub,
        "qwen",
        "tiny",
        config={"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]},
    )
    # The same cache reachable via a second mount point.
    (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)

    rows = scan_local_models(tmp_path)

    assert [r["repo_id"] for r in rows] == ["qwen/tiny"]
