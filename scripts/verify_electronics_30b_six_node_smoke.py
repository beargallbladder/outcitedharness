#!/usr/bin/env python3
"""Verify and seal six-node BF16 Qwen3-VL-30B LoRA qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


SCHEMA = "harness.electronics-30b-bf16-six-node-mm-smoke-proof.v1"
PREFLIGHT_SCHEMA = "harness.electronics-30b-bf16-six-node-preflight.v1"
ROLES = ("dgx2", "asus1", "dgx3", "asus3", "asus2", "asus4")
OUTPUT_RELATIVE = Path(
    "checkpoints/electronics-teacher-qwen3-vl-30b-bf16-six-node-smoke-v1"
)
TRAINABLE_PARAMETERS = 22_085_120


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, kind: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{kind} must be a regular file: {path}")
    return path


def _json(path: Path, kind: str) -> dict[str, Any]:
    value = json.loads(_regular(path, kind).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{kind} must contain a JSON object")
    return value


def _verify_preflights(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    nodes: list[dict[str, Any]] = []
    fingerprints: dict[str, set[str]] = {
        "config_sha256": set(),
        "model_config_sha256": set(),
        "model_manifest_sha256": set(),
        "dataset_manifest_sha256": set(),
        "dataset_evidence_sha256": set(),
    }
    for rank, role in enumerate(ROLES):
        receipt_path = _regular(
            root
            / "runs"
            / f"electronics-30b-bf16-six-node-rank-{rank}-preflight-v1.json",
            f"rank {rank} preflight",
        )
        receipt = _json(receipt_path, f"rank {rank} preflight")
        if (
            receipt.get("schema") != PREFLIGHT_SCHEMA
            or receipt.get("passed") is not True
            or receipt.get("node_rank") != rank
            or receipt.get("role") != role
            or receipt.get("world_size") != len(ROLES)
            or receipt.get("model_files_verified") != 24
        ):
            raise ValueError(f"rank {rank} preflight identity did not pass")
        expected = receipt.get("evidence_sha256")
        core = {
            key: value
            for key, value in receipt.items()
            if key != "evidence_sha256"
        }
        if hashlib.sha256(_canonical(core)).hexdigest() != expected:
            raise ValueError(f"rank {rank} preflight evidence hash is invalid")
        gates = receipt.get("gates")
        if not isinstance(gates, dict) or not gates or not all(
            value is True for value in gates.values()
        ):
            raise ValueError(f"rank {rank} preflight has an open gate")
        for key in fingerprints:
            value = receipt.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"rank {rank} lacks {key}")
            fingerprints[key].add(value)

        log_path = _regular(
            root
            / "runs"
            / f"electronics-30b-bf16-six-node-rank-{rank}-smoke-v1.log",
            f"rank {rank} training log",
        )
        log = log_path.read_text(errors="replace")
        required = (
            "Using network IB",
            "NCCL_IB_HCA set to rocep1s0f1,roceP2p1s0f1",
            "Total train batch size (w. parallel, distributed & accumulation) = 6",
            "Total optimization steps = 2",
            f"trainable params: {TRAINABLE_PARAMETERS:,}",
            "Training completed.",
        )
        forbidden = (
            "Traceback (most recent call last)",
            "CUDA out of memory",
            "OutOfMemoryError",
            "ChildFailedError",
        )
        if not all(marker in log for marker in required) or any(
            marker in log for marker in forbidden
        ):
            raise ValueError(f"rank {rank} log does not prove clean training")
        nodes.append(
            {
                "rank": rank,
                "role": role,
                "preflight_sha256": _sha256(receipt_path),
                "preflight_evidence_sha256": expected,
                "run_log_sha256": _sha256(log_path),
            }
        )
    inconsistent = {
        key: sorted(values)
        for key, values in fingerprints.items()
        if len(values) != 1
    }
    if inconsistent:
        raise ValueError(f"node artifact fingerprints differ: {inconsistent}")
    return nodes, {
        key: next(iter(values)) for key, values in fingerprints.items()
    }


def _adapter_groups(
    adapter_path: Path,
) -> tuple[list[str], dict[str, dict[str, float | int]]]:
    with safe_open(_regular(adapter_path, "adapter weights"), framework="pt") as file:
        keys = sorted(file.keys())
        groups = {
            "visual": [key for key in keys if ".visual." in key],
            "merger": [key for key in keys if ".merger" in key],
            "language": [key for key in keys if ".language_model." in key],
        }
        if not groups["visual"] or not groups["merger"] or not groups["language"]:
            raise ValueError(
                "adapter lacks visual, merger/projector, or language LoRA tensors"
            )
        summaries: dict[str, dict[str, float | int]] = {}
        for name, group_keys in groups.items():
            b_keys = [key for key in group_keys if ".lora_B." in key]
            if not b_keys:
                raise ValueError(f"{name} adapter has no LoRA-B tensors")
            squared_norm = 0.0
            nonzero = 0
            for key in b_keys:
                tensor = file.get_tensor(key).float()
                squared_norm += float(torch.sum(tensor * tensor))
                nonzero += int(torch.count_nonzero(tensor))
            norm = math.sqrt(squared_norm)
            if not math.isfinite(norm) or norm <= 0 or nonzero <= 0:
                raise ValueError(f"{name} LoRA-B tensors were not updated")
            summaries[name] = {
                "tensors": len(group_keys),
                "lora_b_tensors": len(b_keys),
                "lora_b_nonzero_values": nonzero,
                "lora_b_l2_norm": norm,
            }
    return keys, summaries


def verify(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    nodes, fingerprints = _verify_preflights(root)
    output = root / OUTPUT_RELATIVE
    if output.is_symlink() or not output.is_dir():
        raise ValueError("six-node smoke output is missing or unsafe")

    state_path = output / "trainer_state.json"
    results_path = output / "train_results.json"
    adapter_path = output / "adapter_model.safetensors"
    checkpoint_adapter = output / "checkpoint-2" / "adapter_model.safetensors"
    state = _json(state_path, "trainer state")
    results = _json(results_path, "train results")
    if state.get("global_step") != 2 or state.get("max_steps") != 2:
        raise ValueError("training did not finish exactly two optimizer steps")
    history = [
        row
        for row in state.get("log_history") or []
        if isinstance(row, dict) and "loss" in row and "grad_norm" in row
    ]
    if len(history) != 2 or not all(
        math.isfinite(float(row[field]))
        for row in history
        for field in ("loss", "grad_norm", "learning_rate")
    ):
        raise ValueError("training lacks two finite gradient records")
    for field in ("train_loss", "train_runtime"):
        if not math.isfinite(float(results.get(field, float("nan")))):
            raise ValueError(f"training result {field} is not finite")

    keys, groups = _adapter_groups(adapter_path)
    if _sha256(adapter_path) != _sha256(
        _regular(checkpoint_adapter, "checkpoint adapter")
    ):
        raise ValueError("final and step-two adapters differ")

    core = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "promotion_authorized": False,
        "next_gate": "bf16_base_and_adapter_generation_sanity",
        "nodes": nodes,
        "artifact_fingerprints": fingerprints,
        "training": {
            "world_size": len(ROLES),
            "global_step": 2,
            "trainable_parameters": TRAINABLE_PARAMETERS,
            "losses": [float(row["loss"]) for row in history],
            "gradient_norms": [float(row["grad_norm"]) for row in history],
            "train_loss": float(results["train_loss"]),
            "train_runtime_seconds": float(results["train_runtime"]),
        },
        "adapter": {
            "path": str(adapter_path),
            "bytes": adapter_path.stat().st_size,
            "sha256": _sha256(adapter_path),
            "tensors": len(keys),
            "groups": groups,
        },
        "receipts": {
            "trainer_state_sha256": _sha256(state_path),
            "train_results_sha256": _sha256(results_path),
        },
        "gates": {
            "six_rank_preflight": True,
            "six_rank_clean_exit": True,
            "dual_hca_nccl": True,
            "two_optimizer_steps": True,
            "finite_gradients": True,
            "visual_lora_updated": True,
            "projector_lora_updated": True,
            "language_lora_updated": True,
            "adapter_saved": True,
        },
    }
    core["evidence_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    return core


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise ValueError(f"immutable proof already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _seal(root: Path) -> None:
    output = root / OUTPUT_RELATIVE
    for path in sorted(output.rglob("*"), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(output, 0o555)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--proof-output", required=True, type=Path)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    proof = verify(root)
    _write_new(args.proof_output, proof)
    if args.seal:
        _seal(root)
    print(json.dumps(proof, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
