#!/usr/bin/env python3
"""Verify one node before a handoff-driven six-node 30B vision run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from harness.electronics.claims import canonical_json
from harness.electronics.corpus import sha256_file
from harness.electronics.training_handoff import (
    HANDOFF_SCHEMA,
    IMAGE_ID,
    MODEL_NAME,
    verify_training_dataset,
)
from verify_electronics_30b_six_node_preflight import (
    MODEL_CONFIG_SHA256,
    ROLES,
    _inside,
    _owner_role,
    _regular,
    _verify_model_manifest,
    _write_new,
)


RECEIPT_SCHEMA = "harness.electronics-30b-handoff-preflight.v1"


def _handoff(root: Path, handoff_path: Path) -> tuple[dict[str, Any], Path]:
    path = handoff_path.expanduser().resolve(strict=True)
    _inside(root / "handoffs", path)
    manifest_path = _regular(path / "manifest.json", "training handoff")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != HANDOFF_SCHEMA:
        raise ValueError("unsupported training handoff")
    expected = manifest.get("evidence_sha256")
    core = {
        key: value
        for key, value in manifest.items()
        if key != "evidence_sha256"
    }
    if hashlib.sha256(canonical_json(core)).hexdigest() != expected:
        raise ValueError("training handoff evidence digest is invalid")
    if manifest.get("decision") != "ready_to_stage":
        raise ValueError("training handoff has not passed dataset gates")
    return manifest, path


def verify(
    root: Path,
    handoff_path: Path,
    model_manifest: Path,
    image: str,
    node_rank: int,
    stage: str,
) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    if node_rank not in range(len(ROLES)):
        raise ValueError("node rank must be between 0 and 5")
    role = _owner_role(root)
    if role != ROLES[node_rank]:
        raise ValueError(f"rank {node_rank} must use the {ROLES[node_rank]} root")
    if image != IMAGE_ID:
        raise ValueError("training image is not the pinned LLaMA Factory runtime")
    if stage not in {"sft", "dpo"}:
        raise ValueError("training stage must be sft or dpo")

    handoff, handoff_root = _handoff(root, handoff_path)
    candidate_id = str(handoff["candidate_id"])
    config_path = _regular(handoff_root / f"{stage}.yaml", "training config")
    receipt = handoff["configs"][f"{stage}.yaml"]
    if (
        config_path.stat().st_size != receipt["bytes"]
        or sha256_file(config_path) != receipt["sha256"]
    ):
        raise ValueError("training config differs from handoff receipt")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must contain a mapping")
    required_config = {
        "model_name_or_path": f"/training/models/{MODEL_NAME}",
        "stage": stage,
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "freeze_vision_tower": False,
        "freeze_multi_modal_projector": False,
        "freeze_language_model": False,
        "bf16": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "ddp_find_unused_parameters": False,
    }
    wrong = {
        key: {"expected": expected_value, "actual": config.get(key)}
        for key, expected_value in required_config.items()
        if config.get(key) != expected_value
    }
    if wrong:
        raise ValueError(f"training config violates handoff policy: {wrong}")

    model = (root / "models" / MODEL_NAME).resolve(strict=True)
    _inside(root / "models", model)
    model_config = json.loads(
        _regular(model / "config.json", "model config").read_text()
    )
    if (
        sha256_file(model / "config.json") != MODEL_CONFIG_SHA256
        or model_config.get("model_type") != "qwen3_vl_moe"
        or model_config.get("quantization_config") is not None
    ):
        raise ValueError("model is not the pinned Qwen3-VL-30B BF16 revision")
    model_manifest = model_manifest.expanduser().resolve(strict=True)
    _inside(root / "manifests", model_manifest)
    model_files = _verify_model_manifest(model, model_manifest)

    relative_dataset = Path(handoff["dataset"]["staged_relative_path"])
    if relative_dataset.is_absolute() or ".." in relative_dataset.parts:
        raise ValueError("handoff dataset path is unsafe")
    dataset = (root / relative_dataset).resolve(strict=True)
    _inside(root / "datasets", dataset)
    proof = verify_training_dataset(
        dataset,
        minimum_sft_pairs=1,
        minimum_dpo_pairs=1,
        minimum_lineages=2,
    )
    expected_dataset = handoff["dataset"]
    if (
        proof.manifest_sha256 != expected_dataset["manifest_sha256"]
        or proof.evidence_sha256 != expected_dataset["evidence_sha256"]
        or {
            "sft": proof.sft_pairs,
            "dpo": proof.dpo_pairs,
            "lineages": proof.lineages,
            "images": proof.images,
        }
        != expected_dataset["counts"]
    ):
        raise ValueError("staged dataset differs from training handoff")

    output = Path(str(config["output_dir"]))
    prefix = Path("/training/checkpoints")
    try:
        relative_output = output.relative_to(prefix)
    except ValueError as exc:
        raise ValueError("training output escapes checkpoint root") from exc
    host_output = root / "checkpoints" / relative_output
    if host_output.exists() or host_output.is_symlink():
        raise ValueError(f"immutable training output already exists: {host_output}")
    if stage == "dpo":
        adapter = Path(str(config.get("adapter_name_or_path") or ""))
        try:
            relative_adapter = adapter.relative_to(prefix)
        except ValueError as exc:
            raise ValueError("DPO adapter path escapes checkpoint root") from exc
        host_adapter = (root / "checkpoints" / relative_adapter).resolve(
            strict=True
        )
        _inside(root / "checkpoints", host_adapter)
        if host_adapter.is_symlink() or not host_adapter.is_dir():
            raise ValueError("DPO input adapter is absent or unsafe")
        _regular(host_adapter / "adapter_config.json", "SFT adapter config")

    return {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "candidate_id": candidate_id,
        "stage": stage,
        "role": role,
        "node_rank": node_rank,
        "world_size": len(ROLES),
        "image": image,
        "handoff_evidence_sha256": handoff["evidence_sha256"],
        "config_sha256": receipt["sha256"],
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_manifest_sha256": sha256_file(model_manifest),
        "model_files_verified": model_files,
        "dataset_manifest_sha256": proof.manifest_sha256,
        "dataset_evidence_sha256": proof.evidence_sha256,
        "training_rows": (
            proof.train_sft_pairs if stage == "sft" else proof.train_dpo_pairs
        ),
        "validation_rows": (
            proof.validation_sft_pairs
            if stage == "sft"
            else proof.validation_dpo_pairs
        ),
        "expected_optimizer_steps": handoff[
            "expected_optimizer_steps"
        ][stage],
        "gates": {
            **handoff["gates"],
            "pinned_bf16_checkpoint_verified": True,
            "training_image_verified": True,
            "output_absent": True,
            "dpo_input_adapter_present": stage != "dpo" or True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--node-rank", required=True, type=int)
    parser.add_argument("--stage", choices=("sft", "dpo"), required=True)
    parser.add_argument("--receipt-output", required=True, type=Path)
    args = parser.parse_args()
    value = verify(
        args.root,
        args.handoff,
        args.model_manifest,
        args.image,
        args.node_rank,
        args.stage,
    )
    value["evidence_sha256"] = hashlib.sha256(canonical_json(value)).hexdigest()
    _write_new(args.receipt_output, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
