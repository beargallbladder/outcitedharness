#!/usr/bin/env python3
"""Verify the sealed electronics dataset and bounded multimodal smoke recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DATASET_SCHEMA = "harness.electronics-teacher-dataset.v1"
RECEIPT_SCHEMA = "harness.electronics-teacher-training-preflight.v1"
EXPECTED = {
    "model_name_or_path": "/training/models/Qwen3-VL-8B-Instruct",
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "all",
    "lora_rank": 16,
    "lora_alpha": 32,
    "freeze_vision_tower": False,
    "freeze_multi_modal_projector": False,
    "freeze_language_model": False,
    "dataset": "electronics_sft_train",
    "dataset_dir": (
        "/training/datasets/datasheet-electronics-teacher-v3/llamafactory"
    ),
    "template": "qwen3_vl_nothink",
    "cutoff_len": 8192,
    "max_samples": 14,
    "output_dir": (
        "/training/checkpoints/electronics-teacher-qwen3-vl-8b-mm-smoke-v1"
    ),
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 0.00002,
    "max_steps": 2,
    "bf16": True,
}


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


def _inside(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes dataset: {path}") from error


def verify(root: Path, config_path: Path, image: str) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    marker = _regular(root / ".harness-training-owner-v1", "owner marker")
    if "role=dgx2" not in marker.read_text().splitlines():
        raise ValueError("training root is not marked for DGX2")
    config_path = config_path.expanduser().resolve(strict=True)
    _inside(root / "configs", config_path)
    config = yaml.safe_load(_regular(config_path, "config").read_text())
    if not isinstance(config, dict):
        raise ValueError("training config must contain a mapping")
    wrong = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in EXPECTED.items()
        if config.get(key) != expected
    }
    if wrong:
        raise ValueError(f"training config violates bounded recipe: {wrong}")

    dataset = root / "datasets" / "datasheet-electronics-teacher-v3"
    manifest_path = _regular(dataset / "manifest.json", "dataset manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("unsupported electronics teacher dataset")
    evidence = manifest.get("evidence_sha256")
    core = {key: value for key, value in manifest.items() if key != "evidence_sha256"}
    if hashlib.sha256(_canonical(core)).hexdigest() != evidence:
        raise ValueError("dataset evidence SHA-256 is invalid")
    if manifest.get("frozen_evaluation", {}).get("document_overlap") != 0:
        raise ValueError("dataset is not isolated from frozen evaluation")
    if manifest.get("counts", {}).get("sft") != 17:
        raise ValueError("unexpected SFT count")
    if manifest.get("counts", {}).get("dpo") != 16:
        raise ValueError("unexpected DPO count")
    for relative, receipt in manifest["artifacts"].items():
        path = _regular(dataset / relative, "sealed dataset artifact")
        _inside(dataset, path)
        if (
            path.stat().st_size != receipt["bytes"]
            or _sha256(path) != receipt["sha256"]
        ):
            raise ValueError(f"dataset artifact hash mismatch: {relative}")

    info_path = _regular(
        dataset / "llamafactory" / "dataset_info.json",
        "LlamaFactory dataset registry",
    )
    info = json.loads(info_path.read_text())
    registration = info.get("electronics_sft_train")
    if not isinstance(registration, dict):
        raise ValueError("electronics_sft_train is not registered")
    rows_path = _regular(
        dataset
        / "llamafactory"
        / str(registration.get("file_name", "")),
        "LlamaFactory SFT data",
    )
    rows = json.loads(rows_path.read_text())
    if not isinstance(rows, list) or len(rows) != 14:
        raise ValueError("LlamaFactory train split must contain 14 rows")
    for index, row in enumerate(rows):
        messages = row.get("messages")
        images = row.get("images")
        if not isinstance(messages, list) or not isinstance(images, list):
            raise ValueError(f"row {index} lacks messages or images")
        placeholders = sum(
            str(message.get("content") or "").count("<image>")
            for message in messages
            if isinstance(message, dict)
        )
        if placeholders != len(images) or placeholders < 1:
            raise ValueError(f"row {index} image placeholders do not match")
        for relative in images:
            image_path = _regular(
                (rows_path.parent / str(relative)).resolve(),
                "training image",
            )
            _inside(dataset, image_path)

    model_config = _regular(
        root / "models" / "Qwen3-VL-8B-Instruct" / "config.json",
        "model config",
    )
    output = root / "checkpoints" / "electronics-teacher-qwen3-vl-8b-mm-smoke-v1"
    if output.exists() or output.is_symlink():
        raise ValueError(f"immutable output already exists: {output}")
    return {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "mode": "multimodal_lora_smoke",
        "image": image,
        "config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "dataset": {
            "path": str(dataset),
            "manifest_sha256": _sha256(manifest_path),
            "evidence_sha256": evidence,
            "sft": 17,
            "dpo": 16,
            "train_rows": 14,
        },
        "model": {
            "path": str(model_config.parent),
            "config_sha256": _sha256(model_config),
        },
        "gates": {
            "frozen_document_overlap_zero": True,
            "artifact_hashes_verified": True,
            "image_placeholders_verified": True,
            "vision_tower_trainable": True,
            "multimodal_projector_trainable": True,
            "language_model_trainable": True,
            "output_absent": True,
        },
    }


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise ValueError(f"immutable receipt already exists: {path}")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--receipt-output", required=True, type=Path)
    args = parser.parse_args()
    receipt = verify(args.root, args.config, args.image)
    receipt["evidence_sha256"] = hashlib.sha256(
        _canonical(receipt)
    ).hexdigest()
    _write_new(args.receipt_output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
