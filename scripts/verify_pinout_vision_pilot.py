#!/usr/bin/env python3
"""Verify and seal the bounded Qwen3-VL pinout load/step/save/resume pilot."""

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


SCHEMA = "harness.pinout-vision-pilot-proof.v1"
PREFLIGHT_SCHEMA = "harness.pinout-vision-training-preflight.v1"
RUNS = {
    "pilot": {
        "output": "checkpoints/pinout-rows-qwen3-vl-8b-pilot-v1",
        "preflight": "runs/pinout-vision-qwen3-vl-8b-pilot-preflight-v1.json",
        "log": "runs/pinout-vision-qwen3-vl-8b-pilot-v1.log",
        "global_step": 8,
    },
    "resume": {
        "output": "checkpoints/pinout-rows-qwen3-vl-8b-resume-v1",
        "preflight": "runs/pinout-vision-qwen3-vl-8b-resume-preflight-v1.json",
        "log": "runs/pinout-vision-qwen3-vl-8b-resume-v1.log",
        "global_step": 12,
    },
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
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


def _json_object(path: Path, kind: str) -> dict[str, Any]:
    value = json.loads(_regular(path, kind).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{kind} must contain a JSON object")
    return value


def _preflight(path: Path, mode: str) -> dict[str, Any]:
    value = _json_object(path, f"{mode} preflight")
    if (
        value.get("schema") != PREFLIGHT_SCHEMA
        or value.get("passed") is not True
        or value.get("mode") != mode
    ):
        raise ValueError(f"{mode} preflight did not pass")
    expected = value.get("evidence_sha256")
    core = {
        key: item
        for key, item in value.items()
        if key not in ("created_at", "evidence_sha256")
    }
    if hashlib.sha256(_canonical(core)).hexdigest() != expected:
        raise ValueError(f"{mode} preflight evidence digest is invalid")
    return value


def _finite_history(state: dict[str, Any], expected_step: int) -> dict[str, Any]:
    if state.get("global_step") != expected_step:
        raise ValueError(
            f"trainer state expected step {expected_step}, "
            f"observed {state.get('global_step')}"
        )
    step_rows = [
        row
        for row in state.get("log_history") or []
        if isinstance(row, dict)
        and isinstance(row.get("step"), int)
        and "loss" in row
        and "grad_norm" in row
    ]
    if not step_rows or max(row["step"] for row in step_rows) != expected_step:
        raise ValueError("trainer state lacks complete per-step evidence")
    for row in step_rows:
        for field in ("loss", "grad_norm", "learning_rate"):
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"trainer state contains non-finite {field}")
    return {
        "logged_optimizer_steps": len(step_rows),
        "last_step": max(row["step"] for row in step_rows),
        "maximum_loss": max(float(row["loss"]) for row in step_rows),
        "maximum_gradient_norm": max(float(row["grad_norm"]) for row in step_rows),
    }


def _run(root: Path, mode: str, spec: dict[str, Any]) -> dict[str, Any]:
    output = root / spec["output"]
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"{mode} output is missing or unsafe")
    state_path = output / "trainer_state.json"
    results_path = output / "train_results.json"
    adapter_path = output / "adapter_model.safetensors"
    adapter_config_path = output / "adapter_config.json"
    state = _json_object(state_path, f"{mode} trainer state")
    results = _json_object(results_path, f"{mode} train results")
    adapter_config = _json_object(adapter_config_path, f"{mode} adapter config")
    history = _finite_history(state, int(spec["global_step"]))
    for field in ("train_loss", "train_runtime"):
        if not math.isfinite(float(results.get(field, float("nan")))):
            raise ValueError(f"{mode} result {field} is not finite")
    if adapter_config.get("r") != 16 or adapter_config.get("lora_alpha") != 32:
        raise ValueError(f"{mode} adapter rank or alpha is incorrect")
    _regular(adapter_path, f"{mode} adapter")
    if adapter_path.stat().st_size < 1024 * 1024:
        raise ValueError(f"{mode} adapter artifact is unexpectedly small")
    checkpoint = output / f"checkpoint-{spec['global_step']}"
    checkpoint_adapter = _regular(
        checkpoint / "adapter_model.safetensors",
        f"{mode} checkpoint adapter",
    )
    log_path = _regular(root / spec["log"], f"{mode} run log")
    log_text = log_path.read_text(errors="replace")
    for marker in ("Traceback (most recent call last)", "CUDA out of memory"):
        if marker in log_text:
            raise ValueError(f"{mode} run log contains {marker}")
    if mode == "resume" and "Resuming training from checkpoint" not in log_text:
        raise ValueError("resume log does not prove checkpoint restoration")
    return {
        "global_step": state["global_step"],
        "epoch": float(results["epoch"]),
        "train_loss": float(results["train_loss"]),
        "train_runtime_seconds": float(results["train_runtime"]),
        "history": history,
        "adapter": {
            "path": str(adapter_path),
            "sha256": _sha256(adapter_path),
            "bytes": adapter_path.stat().st_size,
        },
        "checkpoint_adapter": {
            "path": str(checkpoint_adapter),
            "sha256": _sha256(checkpoint_adapter),
            "bytes": checkpoint_adapter.stat().st_size,
        },
        "trainer_state_sha256": _sha256(state_path),
        "train_results_sha256": _sha256(results_path),
        "run_log_sha256": _sha256(log_path),
    }


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    preflights = {
        mode: _preflight(root / spec["preflight"], mode)
        for mode, spec in RUNS.items()
    }
    dataset_digests = {
        value["dataset"]["sha256"] for value in preflights.values()
    }
    if len(dataset_digests) != 1:
        raise ValueError("pilot and resume used different datasets")
    runs = {
        mode: _run(root, mode, spec)
        for mode, spec in RUNS.items()
    }
    if runs["pilot"]["adapter"]["sha256"] == runs["resume"]["adapter"]["sha256"]:
        raise ValueError("resume did not update the adapter artifact")
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "passed": True,
        "root": str(root),
        "dataset_manifest_sha256": next(iter(dataset_digests)),
        "dataset_evidence_sha256": preflights["pilot"]["dataset"][
            "evidence_sha256"
        ],
        "model_config_sha256": preflights["pilot"]["model_config"]["sha256"],
        "image": preflights["pilot"]["image"],
        "pilot": runs["pilot"],
        "resume": runs["resume"],
        "gates": {
            "model_load": True,
            "optimizer_step": True,
            "finite_lora_gradients": True,
            "adapter_only_save": True,
            "resume": True,
        },
        "promotion_authorized": False,
    }
    core["evidence_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **core,
    }


def seal_outputs(root: Path) -> None:
    for spec in RUNS.values():
        output = root / spec["output"]
        for path in sorted(output.rglob("*"), reverse=True):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(output, 0o555)


def write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"immutable proof already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--proof-output", required=True, type=Path)
    parser.add_argument("--seal", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    proof = verify(arguments.root)
    write_new(arguments.proof_output, proof)
    if arguments.seal:
        seal_outputs(arguments.root.resolve(strict=True))
    print(json.dumps(proof, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
