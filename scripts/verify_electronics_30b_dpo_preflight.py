#!/usr/bin/env python3
"""Fail closed before six-node preference correction of the 30B candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from verify_electronics_30b_six_node_preflight import (
    IMAGE_ID,
    MODEL_CONFIG_SHA256,
    MODEL_NAME,
    ROLES,
    _canonical,
    _inside,
    _owner_role,
    _regular,
    _sha256,
    _verify_dataset,
    _verify_model_manifest,
    _write_new,
)


RECEIPT_SCHEMA = "harness.electronics-30b-bf16-dpo-preflight.v1"
ADAPTER_RELATIVE = Path(
    "checkpoints/electronics-teacher-qwen3-vl-30b-bf16-candidate-v1"
)
OUTPUT_RELATIVE = Path(
    "checkpoints/electronics-teacher-qwen3-vl-30b-bf16-candidate-v2-dpo"
)
ADAPTER_FILES = {
    "adapter_config.json": {
        "bytes": 1093,
        "sha256": (
            "0761eaa529dd129928d703c0a59025408f9e44bd7e11f6c024d0a391c41d908b"
        ),
    },
    "adapter_model.safetensors": {
        "bytes": 88429688,
        "sha256": (
            "7a0efbb42b17c303d4435b4e8987907f9d82eb45b9dabef9a7576826fd999abd"
        ),
    },
}
EXPECTED_CONFIG = {
    "model_name_or_path": f"/training/models/{MODEL_NAME}",
    "adapter_name_or_path": f"/training/{ADAPTER_RELATIVE}",
    "image_max_pixels": 1048576,
    "stage": "dpo",
    "do_train": True,
    "finetuning_type": "lora",
    "create_new_adapter": False,
    "lora_target": "all",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "freeze_vision_tower": False,
    "freeze_multi_modal_projector": False,
    "freeze_language_model": False,
    "pref_beta": 0.1,
    "pref_loss": "sigmoid",
    "pref_ftx": 0.1,
    "dataset": "electronics_dpo_train",
    "eval_dataset": "electronics_dpo_validation",
    "dataset_dir": (
        "/training/datasets/datasheet-electronics-teacher-v5/llamafactory"
    ),
    "template": "qwen3_vl_nothink",
    "cutoff_len": 8192,
    "max_samples": 221,
    "output_dir": f"/training/{OUTPUT_RELATIVE}",
    "eval_strategy": "steps",
    "eval_steps": 37,
    "save_steps": 37,
    "save_total_limit": 2,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "learning_rate": 0.000005,
    "num_train_epochs": 2,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "bf16": True,
    "seed": 1788172801,
    "data_seed": 1788172801,
    "disable_shuffling": False,
    "ddp_find_unused_parameters": False,
}


def _verify_adapter(root: Path) -> tuple[Path, dict[str, str]]:
    adapter = (root / ADAPTER_RELATIVE).resolve(strict=True)
    _inside(root / "checkpoints", adapter)
    receipts: dict[str, str] = {}
    for name, expected in ADAPTER_FILES.items():
        path = _regular(adapter / name, f"candidate v1 {name}")
        if (
            path.stat().st_size != expected["bytes"]
            or _sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"candidate v1 adapter changed: {name}")
        receipts[name] = str(expected["sha256"])
    config = json.loads((adapter / "adapter_config.json").read_text())
    if (
        config.get("r") != 16
        or config.get("lora_alpha") != 32
        or config.get("peft_type") != "LORA"
    ):
        raise ValueError("candidate v1 adapter contract changed")
    return adapter, receipts


def _verify_dpo_split(
    root: Path,
    name: str,
    expected_rows: int,
) -> tuple[int, set[str]]:
    dataset = root / "datasets" / "datasheet-electronics-teacher-v5"
    path = dataset / "llamafactory" / name
    rows = json.loads(_regular(path, f"{name} split").read_text())
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ValueError(f"{name} must contain {expected_rows} rows")
    image_hashes: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{name} row {index} is not an object")
        messages = row.get("conversations")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        images = row.get("images")
        if (
            not isinstance(messages, list)
            or not isinstance(chosen, dict)
            or not isinstance(rejected, dict)
            or not isinstance(images, list)
            or chosen.get("role") != "assistant"
            or rejected.get("role") != "assistant"
            or not str(chosen.get("content") or "").strip()
            or not str(rejected.get("content") or "").strip()
            or chosen.get("content") == rejected.get("content")
        ):
            raise ValueError(f"{name} row {index} is malformed")
        placeholders = sum(
            str(message.get("content") or "").count("<image>")
            for message in messages
            if isinstance(message, dict)
        )
        if placeholders != len(images) or placeholders < 1:
            raise ValueError(f"{name} row {index} image placeholders do not match")
        for relative in images:
            image = (path.parent / str(relative)).resolve(strict=True)
            _inside(dataset, image)
            _regular(image, f"{name} image")
            image_hashes.add(_sha256(image))
    return len(rows), image_hashes


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
    config = yaml.safe_load(_regular(config_path, "DPO config").read_text())
    if not isinstance(config, dict):
        raise ValueError("DPO config must contain a mapping")
    wrong = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in EXPECTED_CONFIG.items()
        if config.get(key) != expected
    }
    if wrong:
        raise ValueError(f"DPO config violates the sealed recipe: {wrong}")

    model = (root / "models" / MODEL_NAME).resolve(strict=True)
    model_config = json.loads(
        _regular(model / "config.json", "model config").read_text()
    )
    if (
        _sha256(model / "config.json") != MODEL_CONFIG_SHA256
        or model_config.get("model_type") != "qwen3_vl_moe"
        or model_config.get("quantization_config") is not None
    ):
        raise ValueError("model is not the pinned Qwen3-VL-30B BF16 revision")
    model_manifest = model_manifest.expanduser().resolve(strict=True)
    _inside(root / "manifests", model_manifest)
    model_files = _verify_model_manifest(model, model_manifest)
    dataset_manifest, dataset_evidence = _verify_dataset(root)
    _, adapter_receipts = _verify_adapter(root)
    training_rows, training_images = _verify_dpo_split(
        root, "electronics_dpo_train.json", 221
    )
    validation_rows, validation_images = _verify_dpo_split(
        root, "electronics_dpo_validation.json", 56
    )

    output = root / OUTPUT_RELATIVE
    if output.exists() or output.is_symlink():
        raise ValueError(f"immutable DPO output already exists: {output}")
    return {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "role": role,
        "node_rank": node_rank,
        "world_size": len(ROLES),
        "image": image,
        "config_sha256": _sha256(config_path),
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_manifest_sha256": _sha256(model_manifest),
        "model_files_verified": model_files,
        "adapter_receipts": adapter_receipts,
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "dataset_evidence_sha256": dataset_evidence,
        "training_rows": training_rows,
        "training_images": len(training_images),
        "validation_rows": validation_rows,
        "validation_images": len(validation_images),
        "expected_optimizer_steps": 74,
        "gates": {
            "bf16_training_checkpoint_pinned": True,
            "candidate_v1_adapter_pinned": True,
            "frozen_document_overlap_zero": True,
            "artifact_hashes_verified": True,
            "preference_pairs_verified": True,
            "training_images_verified": True,
            "validation_images_verified": True,
            "vision_tower_trainable": True,
            "multimodal_projector_trainable": True,
            "language_model_trainable": True,
            "six_node_ddp": True,
            "output_absent": True,
        },
    }


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
