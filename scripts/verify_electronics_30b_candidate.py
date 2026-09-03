#!/usr/bin/env python3
"""Verify and seal the six-node BF16 Qwen3-VL-30B candidate run."""

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

from verify_electronics_30b_six_node_smoke import (
    ROLES,
    TRAINABLE_PARAMETERS,
    _adapter_groups,
    _canonical,
    _json,
    _regular,
    _sha256,
)


SCHEMA = "harness.electronics-30b-bf16-candidate-proof.v1"
PREFLIGHT_SCHEMA = "harness.electronics-30b-bf16-candidate-preflight.v1"
OUTPUT_RELATIVE = Path(
    "checkpoints/electronics-teacher-qwen3-vl-30b-bf16-candidate-v1"
)
EXPECTED_STEPS = 114


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
            / f"electronics-30b-bf16-candidate-rank-{rank}-preflight-v1.json",
            f"candidate rank {rank} preflight",
        )
        receipt = _json(receipt_path, f"candidate rank {rank} preflight")
        if (
            receipt.get("schema") != PREFLIGHT_SCHEMA
            or receipt.get("passed") is not True
            or receipt.get("node_rank") != rank
            or receipt.get("role") != role
            or receipt.get("world_size") != len(ROLES)
            or receipt.get("model_files_verified") != 24
            or receipt.get("training_rows") != 227
            or receipt.get("validation_rows") != 56
            or receipt.get("expected_optimizer_steps") != EXPECTED_STEPS
        ):
            raise ValueError(f"candidate rank {rank} preflight identity did not pass")
        expected = receipt.get("evidence_sha256")
        core = {
            key: value
            for key, value in receipt.items()
            if key != "evidence_sha256"
        }
        if hashlib.sha256(_canonical(core)).hexdigest() != expected:
            raise ValueError(
                f"candidate rank {rank} preflight evidence hash is invalid"
            )
        gates = receipt.get("gates")
        if not isinstance(gates, dict) or not gates or not all(
            value is True for value in gates.values()
        ):
            raise ValueError(f"candidate rank {rank} preflight has an open gate")
        for key in fingerprints:
            value = receipt.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"candidate rank {rank} lacks {key}")
            fingerprints[key].add(value)

        log_path = _regular(
            root
            / "runs"
            / f"electronics-30b-bf16-candidate-rank-{rank}-train-v1.log",
            f"candidate rank {rank} training log",
        )
        log = log_path.read_text(errors="replace")
        required = (
            "Using network IB",
            "NCCL_IB_HCA set to rocep1s0f1,roceP2p1s0f1",
            "Total train batch size (w. parallel, distributed & accumulation) = 6",
            f"Total optimization steps = {EXPECTED_STEPS}",
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
            raise ValueError(f"candidate rank {rank} log does not prove clean training")
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
        raise ValueError(f"candidate artifact fingerprints differ: {inconsistent}")
    return nodes, {
        key: next(iter(values)) for key, values in fingerprints.items()
    }


def verify(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    nodes, fingerprints = _verify_preflights(root)
    output = root / OUTPUT_RELATIVE
    if output.is_symlink() or not output.is_dir():
        raise ValueError("candidate output is missing or unsafe")

    state_path = output / "trainer_state.json"
    results_path = output / "train_results.json"
    adapter_path = output / "adapter_model.safetensors"
    checkpoint_adapter = (
        output / f"checkpoint-{EXPECTED_STEPS}" / "adapter_model.safetensors"
    )
    state = _json(state_path, "candidate trainer state")
    results = _json(results_path, "candidate train results")
    if (
        state.get("global_step") != EXPECTED_STEPS
        or state.get("max_steps") != EXPECTED_STEPS
    ):
        raise ValueError("candidate did not finish exactly 114 optimizer steps")
    history = state.get("log_history") or []
    gradient_rows = [
        row
        for row in history
        if isinstance(row, dict) and "loss" in row and "grad_norm" in row
    ]
    evaluation_rows = [
        row
        for row in history
        if isinstance(row, dict) and "eval_loss" in row
    ]
    if len(gradient_rows) != EXPECTED_STEPS or not all(
        math.isfinite(float(row[field]))
        for row in gradient_rows
        for field in ("loss", "grad_norm", "learning_rate")
    ):
        raise ValueError("candidate lacks 114 finite gradient records")
    if len(evaluation_rows) != 3 or not all(
        math.isfinite(float(row["eval_loss"])) for row in evaluation_rows
    ):
        raise ValueError("candidate lacks three finite validation records")
    for field in ("train_loss", "train_runtime"):
        if not math.isfinite(float(results.get(field, float("nan")))):
            raise ValueError(f"candidate result {field} is not finite")

    keys, groups = _adapter_groups(adapter_path)
    if _sha256(adapter_path) != _sha256(
        _regular(checkpoint_adapter, "final candidate checkpoint adapter")
    ):
        raise ValueError("final candidate and step-114 adapters differ")
    evaluation_losses = [float(row["eval_loss"]) for row in evaluation_rows]
    core = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "promotion_authorized": False,
        "next_gate": "frozen_base_versus_candidate_extraction_evaluation",
        "nodes": nodes,
        "artifact_fingerprints": fingerprints,
        "training": {
            "world_size": len(ROLES),
            "global_step": EXPECTED_STEPS,
            "trainable_parameters": TRAINABLE_PARAMETERS,
            "first_loss": float(gradient_rows[0]["loss"]),
            "final_loss": float(gradient_rows[-1]["loss"]),
            "first_gradient_norm": float(gradient_rows[0]["grad_norm"]),
            "final_gradient_norm": float(gradient_rows[-1]["grad_norm"]),
            "train_loss": float(results["train_loss"]),
            "train_runtime_seconds": float(results["train_runtime"]),
            "validation_losses": evaluation_losses,
            "best_validation_loss": min(evaluation_losses),
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
            "optimizer_steps_114": True,
            "finite_gradients": True,
            "three_finite_validation_gates": True,
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
        raise ValueError(f"immutable candidate proof already exists: {path}")
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
