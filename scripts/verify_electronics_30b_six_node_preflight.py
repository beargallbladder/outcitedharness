#!/usr/bin/env python3
"""Fail closed before six-node BF16 Qwen3-VL-30B LoRA qualification."""

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
MANIFEST_SCHEMA = "harness.training.manifest.v1"
RECEIPT_SCHEMA = "harness.electronics-30b-bf16-six-node-preflight.v1"
IMAGE_ID = "sha256:728b352622d569b1343cb384dd8aa394917203d02631322b02d302d068e38139"
MODEL_NAME = "Qwen3-VL-30B-A3B-Instruct-BF16"
MODEL_CONFIG_SHA256 = (
    "6129b3f54d517a18553cf7525ff4886be6831c5806fc094d632b8ef38d859912"
)
ROLES = ("dgx2", "asus1", "dgx3", "asus3", "asus2", "asus4")
EXPECTED_CONFIG = {
    "model_name_or_path": f"/training/models/{MODEL_NAME}",
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
        "/training/datasets/datasheet-electronics-teacher-v5/llamafactory"
    ),
    "template": "qwen3_vl_nothink",
    "cutoff_len": 8192,
    "max_samples": 14,
    "output_dir": (
        "/training/checkpoints/"
        "electronics-teacher-qwen3-vl-30b-bf16-six-node-smoke-v1"
    ),
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "learning_rate": 0.00002,
    "max_steps": 2,
    "bf16": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "ddp_find_unused_parameters": False,
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
        raise ValueError(f"path escapes expected root: {path}") from error


def _owner_role(root: Path) -> str:
    marker = _regular(root / ".harness-training-owner-v1", "owner marker")
    values: dict[str, str] = {}
    for line in marker.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    role = values.get("role", "")
    if role not in ROLES or values.get("root") != str(root):
        raise ValueError("training owner marker does not match a six-node root")
    return role


def _verify_model_manifest(model: Path, manifest_path: Path) -> int:
    manifest = json.loads(_regular(manifest_path, "model manifest").read_text())
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("algorithm") != "sha256"
        or manifest.get("artifact") != model.name
    ):
        raise ValueError("unsupported or mismatched model manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("model manifest has no file entries")
    expected: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("model manifest entry is not an object")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative or relative.startswith("/"):
            raise ValueError("model manifest contains an invalid path")
        path = (model / relative).resolve(strict=True)
        _inside(model, path)
        _regular(path, "model artifact")
        if relative in expected:
            raise ValueError(f"duplicate model manifest path: {relative}")
        expected.add(relative)
        if (
            path.stat().st_size != entry.get("bytes")
            or _sha256(path) != entry.get("sha256")
        ):
            raise ValueError(f"model artifact hash mismatch: {relative}")
    actual: set[str] = set()
    for path in model.rglob("*"):
        relative = path.relative_to(model)
        if path.is_symlink():
            raise ValueError(f"model directory contains a symlink: {relative}")
        if path.is_file() and not any(part.startswith(".") for part in relative.parts):
            actual.add(relative.as_posix())
    if actual != expected:
        raise ValueError("model directory differs from its sealed manifest")
    return len(entries)


def _verify_dataset(root: Path) -> tuple[Path, str]:
    dataset = root / "datasets" / "datasheet-electronics-teacher-v5"
    manifest_path = _regular(dataset / "manifest.json", "dataset manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("unsupported electronics teacher dataset")
    evidence = manifest.get("evidence_sha256")
    core = {key: value for key, value in manifest.items() if key != "evidence_sha256"}
    if hashlib.sha256(_canonical(core)).hexdigest() != evidence:
        raise ValueError("dataset evidence SHA-256 is invalid")
    if manifest.get("frozen_evaluation", {}).get("document_overlap") != 0:
        raise ValueError("dataset overlaps the frozen evaluation documents")
    if (
        manifest.get("counts", {}).get("sft") != 283
        or manifest.get("counts", {}).get("dpo") != 277
    ):
        raise ValueError("electronics teacher dataset counts changed")
    for relative, receipt in manifest.get("artifacts", {}).items():
        path = (dataset / relative).resolve(strict=True)
        _inside(dataset, path)
        _regular(path, "dataset artifact")
        if (
            path.stat().st_size != receipt.get("bytes")
            or _sha256(path) != receipt.get("sha256")
        ):
            raise ValueError(f"dataset artifact hash mismatch: {relative}")
    rows_path = dataset / "llamafactory" / "electronics_sft_train.json"
    rows = json.loads(_regular(rows_path, "LLaMA Factory train split").read_text())
    if not isinstance(rows, list) or len(rows) != 227:
        raise ValueError("LLaMA Factory train split must contain 227 rows")
    for index, row in enumerate(rows):
        messages = row.get("messages")
        images = row.get("images")
        if not isinstance(messages, list) or not isinstance(images, list):
            raise ValueError(f"dataset row {index} lacks messages or images")
        placeholders = sum(
            str(message.get("content") or "").count("<image>")
            for message in messages
            if isinstance(message, dict)
        )
        if placeholders != len(images) or placeholders < 1:
            raise ValueError(f"dataset row {index} image placeholders do not match")
        for relative in images:
            image = (rows_path.parent / str(relative)).resolve(strict=True)
            _inside(dataset, image)
            _regular(image, "training image")
    return manifest_path, str(evidence)


def verify(
    root: Path,
    config_path: Path,
    model_manifest: Path,
    image: str,
    node_rank: int,
) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    if node_rank not in range(len(ROLES)):
        raise ValueError("node rank must be between 0 and 5")
    role = _owner_role(root)
    if role != ROLES[node_rank]:
        raise ValueError(f"rank {node_rank} must use the {ROLES[node_rank]} root")
    if image != IMAGE_ID:
        raise ValueError("training image is not the pinned LLaMA Factory runtime")

    config_path = config_path.expanduser().resolve(strict=True)
    _inside(root / "configs", config_path)
    config = yaml.safe_load(_regular(config_path, "training config").read_text())
    if not isinstance(config, dict):
        raise ValueError("training config must contain a mapping")
    wrong = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in EXPECTED_CONFIG.items()
        if config.get(key) != expected
    }
    if wrong:
        raise ValueError(f"training config violates the bounded recipe: {wrong}")

    model = (root / "models" / MODEL_NAME).resolve(strict=True)
    _inside(root / "models", model)
    config_json = json.loads(_regular(model / "config.json", "model config").read_text())
    if (
        _sha256(model / "config.json") != MODEL_CONFIG_SHA256
        or config_json.get("model_type") != "qwen3_vl_moe"
        or config_json.get("quantization_config") is not None
    ):
        raise ValueError("model is not the pinned Qwen3-VL-30B BF16 revision")
    model_manifest = model_manifest.expanduser().resolve(strict=True)
    _inside(root / "manifests", model_manifest)
    model_files = _verify_model_manifest(model, model_manifest)
    dataset_manifest, dataset_evidence = _verify_dataset(root)

    output = (
        root
        / "checkpoints"
        / "electronics-teacher-qwen3-vl-30b-bf16-six-node-smoke-v1"
    )
    if output.exists() or output.is_symlink():
        raise ValueError(f"immutable output already exists: {output}")
    return {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "role": role,
        "node_rank": node_rank,
        "world_size": 6,
        "image": image,
        "config_sha256": _sha256(config_path),
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_manifest_sha256": _sha256(model_manifest),
        "model_files_verified": model_files,
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "dataset_evidence_sha256": dataset_evidence,
        "gates": {
            "bf16_training_checkpoint_pinned": True,
            "frozen_document_overlap_zero": True,
            "artifact_hashes_verified": True,
            "image_placeholders_verified": True,
            "vision_tower_trainable": True,
            "multimodal_projector_trainable": True,
            "language_model_trainable": True,
            "six_node_ddp": True,
            "output_absent": True,
        },
    }


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise ValueError(f"immutable receipt already exists: {path}")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--node-rank", required=True, type=int)
    parser.add_argument("--receipt-output", required=True, type=Path)
    args = parser.parse_args()
    receipt = verify(
        args.root,
        args.config,
        args.model_manifest,
        args.image,
        args.node_rank,
    )
    receipt["evidence_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    _write_new(args.receipt_output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
