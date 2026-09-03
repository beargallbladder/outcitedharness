#!/usr/bin/env python3
"""Verify and seal the six-node Qwen3-VL-30B DPO correction run."""

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

from verify_electronics_30b_six_node_smoke import (
    ROLES,
    TRAINABLE_PARAMETERS,
    _adapter_groups,
    _canonical,
    _json,
    _regular,
    _sha256,
)


SCHEMA = "harness.electronics-30b-bf16-dpo-proof.v1"
PREFLIGHT_SCHEMA = "harness.electronics-30b-bf16-dpo-preflight.v1"
SOURCE_RELATIVE = Path(
    "checkpoints/electronics-teacher-qwen3-vl-30b-bf16-candidate-v1"
)
OUTPUT_RELATIVE = Path(
    "checkpoints/electronics-teacher-qwen3-vl-30b-bf16-candidate-v2-dpo"
)
EXPECTED_STEPS = 74
EXPECTED_ROWS = 221
EXPECTED_VALIDATION_ROWS = 56
SOURCE_ADAPTER_SHA256 = (
    "7a0efbb42b17c303d4435b4e8987907f9d82eb45b9dabef9a7576826fd999abd"
)


def _verify_preflights(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    nodes: list[dict[str, Any]] = []
    fingerprints: dict[str, set[str]] = {
        "config_sha256": set(),
        "model_config_sha256": set(),
        "model_manifest_sha256": set(),
        "dataset_manifest_sha256": set(),
        "dataset_evidence_sha256": set(),
    }
    source_receipts: set[str] = set()
    for rank, role in enumerate(ROLES):
        receipt_path = _regular(
            root
            / "runs"
            / f"electronics-30b-bf16-dpo-rank-{rank}-preflight-v1.json",
            f"DPO rank {rank} preflight",
        )
        receipt = _json(receipt_path, f"DPO rank {rank} preflight")
        if (
            receipt.get("schema") != PREFLIGHT_SCHEMA
            or receipt.get("passed") is not True
            or receipt.get("node_rank") != rank
            or receipt.get("role") != role
            or receipt.get("world_size") != len(ROLES)
            or receipt.get("model_files_verified") != 24
            or receipt.get("training_rows") != EXPECTED_ROWS
            or receipt.get("validation_rows") != EXPECTED_VALIDATION_ROWS
            or receipt.get("expected_optimizer_steps") != EXPECTED_STEPS
        ):
            raise ValueError(f"DPO rank {rank} preflight identity did not pass")
        expected = receipt.get("evidence_sha256")
        core = {
            key: value
            for key, value in receipt.items()
            if key != "evidence_sha256"
        }
        if hashlib.sha256(_canonical(core)).hexdigest() != expected:
            raise ValueError(f"DPO rank {rank} preflight evidence hash is invalid")
        gates = receipt.get("gates")
        if not isinstance(gates, dict) or not gates or not all(
            value is True for value in gates.values()
        ):
            raise ValueError(f"DPO rank {rank} preflight has an open gate")
        adapter_receipts = receipt.get("adapter_receipts")
        if (
            not isinstance(adapter_receipts, dict)
            or adapter_receipts.get("adapter_model.safetensors")
            != SOURCE_ADAPTER_SHA256
        ):
            raise ValueError(f"DPO rank {rank} source adapter is not pinned")
        source_receipts.add(
            hashlib.sha256(_canonical(adapter_receipts)).hexdigest()
        )
        for key in fingerprints:
            value = receipt.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"DPO rank {rank} lacks {key}")
            fingerprints[key].add(value)

        log_path = _regular(
            root
            / "runs"
            / f"electronics-30b-bf16-dpo-rank-{rank}-train-v1.log",
            f"DPO rank {rank} training log",
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
            raise ValueError(f"DPO rank {rank} log does not prove clean training")
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
    if inconsistent or len(source_receipts) != 1:
        raise ValueError(
            f"DPO node artifact fingerprints differ: {inconsistent}"
        )
    resolved = {
        key: next(iter(values)) for key, values in fingerprints.items()
    }
    resolved["source_adapter_receipts_sha256"] = next(iter(source_receipts))
    return nodes, resolved


def _adapter_deltas(before: Path, after: Path) -> dict[str, dict[str, float | int]]:
    group_tokens = {
        "merger": ".merger",
        "visual": ".visual.",
        "language": ".language_model.",
    }
    summaries = {
        name: {
            "changed_tensors": 0,
            "changed_values": 0,
            "delta_l2_norm": 0.0,
        }
        for name in group_tokens
    }
    with (
        safe_open(_regular(before, "source adapter"), framework="pt") as source,
        safe_open(_regular(after, "corrected adapter"), framework="pt") as corrected,
    ):
        source_keys = sorted(source.keys())
        if source_keys != sorted(corrected.keys()):
            raise ValueError("DPO correction changed the adapter tensor contract")
        squared_norms = {name: 0.0 for name in group_tokens}
        for key in source_keys:
            group = next(
                (name for name, token in group_tokens.items() if token in key),
                None,
            )
            if group is None:
                continue
            delta = corrected.get_tensor(key).float() - source.get_tensor(key).float()
            changed = int(torch.count_nonzero(delta))
            if changed:
                summaries[group]["changed_tensors"] += 1
                summaries[group]["changed_values"] += changed
                squared_norms[group] += float(torch.sum(delta * delta))
        for name, squared_norm in squared_norms.items():
            summaries[name]["delta_l2_norm"] = math.sqrt(squared_norm)
            if (
                summaries[name]["changed_tensors"] <= 0
                or summaries[name]["changed_values"] <= 0
                or not math.isfinite(summaries[name]["delta_l2_norm"])
                or summaries[name]["delta_l2_norm"] <= 0
            ):
                raise ValueError(f"DPO did not update the {name} adapter tensors")
    return summaries


def verify(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    nodes, fingerprints = _verify_preflights(root)
    source = root / SOURCE_RELATIVE / "adapter_model.safetensors"
    output = root / OUTPUT_RELATIVE
    if output.is_symlink() or not output.is_dir():
        raise ValueError("DPO output is missing or unsafe")
    if _sha256(_regular(source, "source adapter")) != SOURCE_ADAPTER_SHA256:
        raise ValueError("source candidate adapter changed")

    state_path = output / "trainer_state.json"
    results_path = output / "train_results.json"
    adapter_path = output / "adapter_model.safetensors"
    checkpoint_adapter = (
        output / f"checkpoint-{EXPECTED_STEPS}" / "adapter_model.safetensors"
    )
    state = _json(state_path, "DPO trainer state")
    results = _json(results_path, "DPO train results")
    if (
        state.get("global_step") != EXPECTED_STEPS
        or state.get("max_steps") != EXPECTED_STEPS
    ):
        raise ValueError("DPO did not finish exactly 74 optimizer steps")
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
        raise ValueError("DPO lacks 74 finite gradient records")
    evaluation_fields = (
        "eval_loss",
        "eval_rewards/accuracies",
        "eval_rewards/margins",
    )
    if len(evaluation_rows) != 2 or not all(
        math.isfinite(float(row[field]))
        for row in evaluation_rows
        for field in evaluation_fields
    ):
        raise ValueError("DPO lacks two finite validation records")
    for field in ("train_loss", "train_runtime"):
        if not math.isfinite(float(results.get(field, float("nan")))):
            raise ValueError(f"DPO result {field} is not finite")

    keys, groups = _adapter_groups(adapter_path)
    adapter_sha256 = _sha256(adapter_path)
    if adapter_sha256 == SOURCE_ADAPTER_SHA256:
        raise ValueError("DPO adapter is identical to its source adapter")
    if adapter_sha256 != _sha256(
        _regular(checkpoint_adapter, "final DPO checkpoint adapter")
    ):
        raise ValueError("final DPO and step-74 adapters differ")
    deltas = _adapter_deltas(source, adapter_path)
    evaluation_losses = [float(row["eval_loss"]) for row in evaluation_rows]
    evaluation_accuracies = [
        float(row["eval_rewards/accuracies"]) for row in evaluation_rows
    ]
    evaluation_margins = [
        float(row["eval_rewards/margins"]) for row in evaluation_rows
    ]
    if (
        evaluation_losses[-1] > evaluation_losses[0]
        or evaluation_margins[-1] <= 0
        or evaluation_accuracies[-1] < 0.5
    ):
        raise ValueError("DPO validation preference gate did not pass")

    core = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "promotion_authorized": False,
        "next_gate": "frozen_base_versus_corrected_extraction_evaluation",
        "nodes": nodes,
        "artifact_fingerprints": fingerprints,
        "training": {
            "world_size": len(ROLES),
            "global_step": EXPECTED_STEPS,
            "trainable_parameters": TRAINABLE_PARAMETERS,
            "first_loss": float(gradient_rows[0]["loss"]),
            "final_loss": float(gradient_rows[-1]["loss"]),
            "train_loss": float(results["train_loss"]),
            "train_runtime_seconds": float(results["train_runtime"]),
            "validation_losses": evaluation_losses,
            "validation_preference_accuracies": evaluation_accuracies,
            "validation_preference_margins": evaluation_margins,
        },
        "adapter": {
            "path": str(adapter_path),
            "bytes": adapter_path.stat().st_size,
            "sha256": adapter_sha256,
            "source_sha256": SOURCE_ADAPTER_SHA256,
            "tensors": len(keys),
            "groups": groups,
            "deltas": deltas,
        },
        "receipts": {
            "trainer_state_sha256": _sha256(state_path),
            "train_results_sha256": _sha256(results_path),
        },
        "gates": {
            "six_rank_preflight": True,
            "six_rank_clean_exit": True,
            "dual_hca_nccl": True,
            "optimizer_steps_74": True,
            "finite_gradients": True,
            "two_finite_validation_gates": True,
            "validation_preference_improved": True,
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
        raise ValueError(f"immutable DPO proof already exists: {path}")
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
