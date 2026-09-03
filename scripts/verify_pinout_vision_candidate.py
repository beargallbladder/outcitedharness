#!/usr/bin/env python3
"""Verify the bounded 1,101-example Qwen3-VL pinout candidate training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "harness.pinout-vision-candidate-proof.v1"
PREFLIGHT_SCHEMA = "harness.pinout-vision-training-preflight.v1"
OUTPUT_RELATIVE = Path("checkpoints/pinout-rows-qwen3-vl-8b-candidate-v1")
PREFLIGHT_RELATIVE = Path(
    "runs/pinout-vision-qwen3-vl-8b-candidate-preflight-v1.json"
)
LOG_RELATIVE = Path("runs/pinout-vision-qwen3-vl-8b-candidate-v1.log")


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


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    preflight_path = root / PREFLIGHT_RELATIVE
    preflight = _json_object(preflight_path, "candidate preflight")
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("passed") is not True
        or preflight.get("mode") != "candidate"
        or preflight.get("config", {}).get("max_samples") != 1101
        or preflight.get("config", {}).get("max_steps") != 1101
    ):
        raise ValueError("candidate preflight did not pass the 1,101-example gate")
    expected_preflight = preflight.get("evidence_sha256")
    preflight_core = {
        key: value
        for key, value in preflight.items()
        if key not in ("created_at", "evidence_sha256")
    }
    if hashlib.sha256(_canonical(preflight_core)).hexdigest() != expected_preflight:
        raise ValueError("candidate preflight evidence digest is invalid")

    output = root / OUTPUT_RELATIVE
    if output.is_symlink() or not output.is_dir():
        raise ValueError("candidate output is missing or unsafe")
    state_path = output / "trainer_state.json"
    results_path = output / "train_results.json"
    adapter_path = output / "adapter_model.safetensors"
    config_path = output / "adapter_config.json"
    state = _json_object(state_path, "candidate trainer state")
    results = _json_object(results_path, "candidate train results")
    adapter_config = _json_object(config_path, "candidate adapter config")
    if state.get("global_step") != 1101:
        raise ValueError("candidate did not complete exactly 1,101 optimizer steps")
    if adapter_config.get("r") != 16 or adapter_config.get("lora_alpha") != 32:
        raise ValueError("candidate adapter rank or alpha is incorrect")
    _regular(adapter_path, "candidate adapter")
    if adapter_path.stat().st_size < 1024 * 1024:
        raise ValueError("candidate adapter is unexpectedly small")

    step_rows = [
        row
        for row in state.get("log_history") or []
        if isinstance(row, dict)
        and "loss" in row
        and "grad_norm" in row
        and isinstance(row.get("step"), int)
    ]
    if len(step_rows) < 100 or max(row["step"] for row in step_rows) < 1100:
        raise ValueError("candidate lacks periodic finite-gradient evidence")
    for row in step_rows:
        for field in ("loss", "grad_norm", "learning_rate"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"candidate contains non-finite {field}")
    for field in ("train_loss", "train_runtime"):
        if not math.isfinite(float(results.get(field, float("nan")))):
            raise ValueError(f"candidate {field} is not finite")

    log_path = _regular(root / LOG_RELATIVE, "candidate run log")
    log_text = log_path.read_text(errors="replace")
    for marker in (
        "Traceback (most recent call last)",
        "CUDA out of memory",
    ):
        if marker.casefold() in log_text.casefold():
            raise ValueError(f"candidate log contains {marker}")
    if re.search(r"(?i)\bnan\b", log_text):
        raise ValueError("candidate log contains NaN")
    if "Training completed" not in log_text:
        raise ValueError("candidate log does not show successful completion")

    core: dict[str, Any] = {
        "schema": SCHEMA,
        "passed": True,
        "promotion_authorized": False,
        "root": str(root),
        "dataset_manifest_sha256": preflight["dataset"]["sha256"],
        "dataset_evidence_sha256": preflight["dataset"]["evidence_sha256"],
        "model_config_sha256": preflight["model_config"]["sha256"],
        "image": preflight["image"],
        "global_step": state["global_step"],
        "epoch": float(results["epoch"]),
        "train_loss": float(results["train_loss"]),
        "train_runtime_seconds": float(results["train_runtime"]),
        "logged_gradient_rows": len(step_rows),
        "maximum_loss": max(float(row["loss"]) for row in step_rows),
        "maximum_gradient_norm": max(
            float(row["grad_norm"]) for row in step_rows
        ),
        "adapter": {
            "path": str(adapter_path),
            "sha256": _sha256(adapter_path),
            "bytes": adapter_path.stat().st_size,
        },
        "trainer_state_sha256": _sha256(state_path),
        "train_results_sha256": _sha256(results_path),
        "run_log_sha256": _sha256(log_path),
        "preflight_sha256": _sha256(preflight_path),
        "preflight_evidence_sha256": expected_preflight,
    }
    core["evidence_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **core,
    }


def write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"immutable output already exists: {path}")
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
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    proof = verify(arguments.root)
    write_new(arguments.output, proof)
    print(json.dumps(proof, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
