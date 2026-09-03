#!/usr/bin/env python3
"""Verify and seal the two-step electronics multimodal LoRA smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "harness.electronics-teacher-mm-smoke-proof.v1"
PREFLIGHT_SCHEMA = "harness.electronics-teacher-training-preflight.v1"
OUTPUT_RELATIVE = Path(
    "checkpoints/electronics-teacher-qwen3-vl-8b-mm-smoke-v1"
)
PREFLIGHT_RELATIVE = Path(
    "runs/electronics-teacher-qwen3-vl-8b-mm-smoke-preflight-v1.json"
)
LOG_RELATIVE = Path(
    "runs/electronics-teacher-qwen3-vl-8b-mm-smoke-v1.log"
)


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


def _safetensor_keys(path: Path) -> list[str]:
    path = _regular(path, "adapter weights")
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if header_size <= 0 or header_size > path.stat().st_size - 8:
            raise ValueError("adapter has an invalid safetensors header")
        header = json.loads(handle.read(header_size))
    if not isinstance(header, dict):
        raise ValueError("adapter safetensors header is malformed")
    return sorted(key for key in header if key != "__metadata__")


def verify(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    preflight_path = root / PREFLIGHT_RELATIVE
    preflight = _json(preflight_path, "preflight receipt")
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("passed") is not True
    ):
        raise ValueError("multimodal preflight did not pass")
    expected = preflight.get("evidence_sha256")
    preflight_core = {
        key: value
        for key, value in preflight.items()
        if key != "evidence_sha256"
    }
    if hashlib.sha256(_canonical(preflight_core)).hexdigest() != expected:
        raise ValueError("preflight evidence SHA-256 is invalid")

    output = root / OUTPUT_RELATIVE
    if output.is_symlink() or not output.is_dir():
        raise ValueError("smoke output is missing or unsafe")
    state_path = output / "trainer_state.json"
    results_path = output / "train_results.json"
    adapter_path = output / "adapter_model.safetensors"
    checkpoint_adapter = output / "checkpoint-2" / "adapter_model.safetensors"
    state = _json(state_path, "trainer state")
    results = _json(results_path, "train results")
    if state.get("global_step") != 2:
        raise ValueError("smoke did not finish two optimizer steps")
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
        raise ValueError("smoke lacks two finite gradient records")
    if not math.isfinite(float(results.get("train_loss", float("nan")))):
        raise ValueError("smoke train loss is not finite")

    keys = _safetensor_keys(adapter_path)
    visual = [key for key in keys if ".visual." in key]
    merger = [key for key in visual if ".merger." in key]
    language = [key for key in keys if ".language_model." in key]
    if not visual or not merger or not language:
        raise ValueError("adapter does not contain vision, merger, and language LoRA")
    if _sha256(adapter_path) != _sha256(checkpoint_adapter):
        raise ValueError("final and checkpoint adapters differ")
    log_path = _regular(root / LOG_RELATIVE, "run log")
    log = log_path.read_text(errors="replace")
    if (
        "trainable params: 52,493,824" not in log
        or "CUDA out of memory" in log
        or "Traceback (most recent call last)" in log
    ):
        raise ValueError("run log does not prove the expected clean training path")

    core = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "promotion_authorized": False,
        "preflight": {
            "path": str(preflight_path),
            "sha256": _sha256(preflight_path),
            "evidence_sha256": expected,
        },
        "training": {
            "global_step": 2,
            "trainable_parameters": 52_493_824,
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
            "visual_tensors": len(visual),
            "merger_tensors": len(merger),
            "language_tensors": len(language),
        },
        "receipts": {
            "trainer_state_sha256": _sha256(state_path),
            "train_results_sha256": _sha256(results_path),
            "run_log_sha256": _sha256(log_path),
        },
        "gates": {
            "load": True,
            "two_optimizer_steps": True,
            "finite_gradients": True,
            "visual_lora_present": True,
            "projector_lora_present": True,
            "language_lora_present": True,
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
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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
    proof = verify(args.root)
    _write_new(args.proof_output, proof)
    if args.seal:
        _seal(args.root.expanduser().resolve(strict=True))
    print(json.dumps(proof, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
