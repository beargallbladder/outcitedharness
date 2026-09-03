"""Fail-closed dataset qualification and reproducible vision-training recipes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Mapping

import yaml

from harness.electronics.claims import canonical_json
from harness.electronics.corpus import sha256_file
from harness.electronics.models import (
    PairDisposition,
    PreferenceTrainingPairCandidate,
    TrainingPairCandidate,
)


DATASET_SCHEMA = "harness.electronics-teacher-dataset.v1"
HANDOFF_SCHEMA = "harness.electronics-30b-training-handoff.v1"
MODEL_NAME = "Qwen3-VL-30B-A3B-Instruct-BF16"
IMAGE_ID = "sha256:728b352622d569b1343cb384dd8aa394917203d02631322b02d302d068e38139"


@dataclass(frozen=True)
class DatasetProof:
    manifest_path: Path
    manifest_sha256: str
    evidence_sha256: str
    sft_pairs: int
    dpo_pairs: int
    train_sft_pairs: int
    validation_sft_pairs: int
    train_dpo_pairs: int
    validation_dpo_pairs: int
    lineages: int
    images: int
    sft_capabilities: dict[str, int]
    dpo_capabilities: dict[str, int]


def _regular(path: Path, kind: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{kind} must be a regular non-symlink file: {path}")
    return path


def _inside(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes dataset root: {path}") from exc


def _canonical_manifest_core(manifest: Mapping[str, Any]) -> bytes:
    return canonical_json(
        {
            key: value
            for key, value in manifest.items()
            if key != "evidence_sha256"
        }
    )


def _jsonl(path: Path, model: type[Any]) -> list[Any]:
    rows = []
    with _regular(path, "canonical training split").open(
        encoding="utf-8"
    ) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid training pair: {exc}"
                ) from exc
    return rows


def _verify_llamafactory_split(
    dataset: Path,
    path: Path,
    *,
    expected: int,
    message_key: str,
) -> set[Path]:
    rows = json.loads(_regular(path, "LLaMA Factory split").read_text())
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(
            f"{path.name} contains {len(rows) if isinstance(rows, list) else 'invalid'} "
            f"rows; expected {expected}"
        )
    images: set[Path] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path.name} row {index} is not an object")
        messages = row.get(message_key)
        image_values = row.get("images")
        if not isinstance(messages, list) or not isinstance(image_values, list):
            raise ValueError(f"{path.name} row {index} lacks messages or images")
        placeholders = sum(
            str(message.get("content") or "").count("<image>")
            for message in messages
            if isinstance(message, dict)
        )
        if placeholders < 1 or placeholders != len(image_values):
            raise ValueError(
                f"{path.name} row {index} image placeholders do not match"
            )
        for relative in image_values:
            image = (path.parent / str(relative)).resolve(strict=True)
            _inside(dataset, image)
            _regular(image, "training image")
            images.add(image)
    return images


def _threshold(
    name: str,
    actual: int,
    minimum: int,
) -> None:
    if minimum < 0:
        raise ValueError(f"{name} minimum cannot be negative")
    if actual < minimum:
        raise ValueError(f"{name} has {actual}; requires at least {minimum}")


def verify_training_dataset(
    dataset: Path,
    *,
    minimum_sft_pairs: int,
    minimum_dpo_pairs: int,
    minimum_lineages: int,
    minimum_sft_capabilities: Mapping[str, int] | None = None,
    minimum_dpo_capabilities: Mapping[str, int] | None = None,
) -> DatasetProof:
    root = dataset.expanduser().resolve(strict=True)
    if dataset.expanduser().is_symlink() or not root.is_dir():
        raise ValueError("training dataset must be a real directory")
    manifest_path = _regular(root / "manifest.json", "dataset manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("unsupported electronics training dataset")
    evidence = manifest.get("evidence_sha256")
    if (
        not isinstance(evidence, str)
        or hashlib.sha256(_canonical_manifest_core(manifest)).hexdigest()
        != evidence
    ):
        raise ValueError("dataset evidence SHA-256 is invalid")
    if manifest.get("frozen_evaluation", {}).get("document_overlap") != 0:
        raise ValueError("dataset overlaps frozen evaluation documents")
    cohorts = manifest.get("frozen_evaluation", {}).get("cohorts")
    if not isinstance(cohorts, list) or not cohorts:
        raise ValueError("dataset has no frozen evaluation cohort receipt")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("dataset manifest has no artifact receipts")
    for relative, receipt in artifacts.items():
        if (
            not isinstance(relative, str)
            or not isinstance(receipt, Mapping)
            or Path(relative).is_absolute()
        ):
            raise ValueError("dataset artifact receipt is malformed")
        path = (root / relative).resolve(strict=True)
        _inside(root, path)
        _regular(path, "dataset artifact")
        if (
            path.stat().st_size != receipt.get("bytes")
            or sha256_file(path) != receipt.get("sha256")
        ):
            raise ValueError(f"dataset artifact hash mismatch: {relative}")

    canonical_root = root / "canonical"
    sft_train = _jsonl(
        canonical_root / "sft-train.jsonl",
        TrainingPairCandidate,
    )
    sft_validation = _jsonl(
        canonical_root / "sft-validation.jsonl",
        TrainingPairCandidate,
    )
    dpo_train = _jsonl(
        canonical_root / "dpo-train.jsonl",
        PreferenceTrainingPairCandidate,
    )
    dpo_validation = _jsonl(
        canonical_root / "dpo-validation.jsonl",
        PreferenceTrainingPairCandidate,
    )
    sft = [*sft_train, *sft_validation]
    dpo = [*dpo_train, *dpo_validation]
    if any(pair.disposition is not PairDisposition.ADMITTED for pair in sft):
        raise ValueError("dataset contains non-admitted SFT pairs")
    sft_ids = [pair.pair_id for pair in sft]
    dpo_ids = [pair.pair_id for pair in dpo]
    if len(sft_ids) != len(set(sft_ids)) or len(dpo_ids) != len(set(dpo_ids)):
        raise ValueError("dataset contains duplicate pair IDs")

    split = manifest.get("split", {}).get("lineages", {})
    train_lineages = set(split.get("train") or [])
    validation_lineages = set(split.get("validation") or [])
    if (
        not train_lineages
        or not validation_lineages
        or train_lineages & validation_lineages
    ):
        raise ValueError("dataset lineage split is absent or overlapping")
    pair_train_lineages = {
        pair.lineage_ids[0] for pair in [*sft_train, *dpo_train]
    }
    pair_validation_lineages = {
        pair.lineage_ids[0]
        for pair in [*sft_validation, *dpo_validation]
    }
    if (
        pair_train_lineages - train_lineages
        or pair_validation_lineages - validation_lineages
        or pair_train_lineages & pair_validation_lineages
    ):
        raise ValueError("canonical pairs violate the sealed lineage split")

    counts = manifest.get("counts") or {}
    split_counts = counts.get("splits") or {}
    expected = {
        "sft": len(sft),
        "dpo": len(dpo),
        "train_sft": len(sft_train),
        "validation_sft": len(sft_validation),
        "train_dpo": len(dpo_train),
        "validation_dpo": len(dpo_validation),
    }
    declared = {
        "sft": counts.get("sft"),
        "dpo": counts.get("dpo"),
        "train_sft": (split_counts.get("train") or {}).get("sft"),
        "validation_sft": (
            split_counts.get("validation") or {}
        ).get("sft"),
        "train_dpo": (split_counts.get("train") or {}).get("dpo"),
        "validation_dpo": (
            split_counts.get("validation") or {}
        ).get("dpo"),
    }
    if expected != declared:
        raise ValueError(
            f"dataset pair counts differ from manifest: "
            f"expected={expected}, declared={declared}"
        )

    lf_root = root / "llamafactory"
    image_paths: set[Path] = set()
    image_paths.update(
        _verify_llamafactory_split(
            root,
            lf_root / "electronics_sft_train.json",
            expected=len(sft_train),
            message_key="messages",
        )
    )
    image_paths.update(
        _verify_llamafactory_split(
            root,
            lf_root / "electronics_sft_validation.json",
            expected=len(sft_validation),
            message_key="messages",
        )
    )
    image_paths.update(
        _verify_llamafactory_split(
            root,
            lf_root / "electronics_dpo_train.json",
            expected=len(dpo_train),
            message_key="conversations",
        )
    )
    image_paths.update(
        _verify_llamafactory_split(
            root,
            lf_root / "electronics_dpo_validation.json",
            expected=len(dpo_validation),
            message_key="conversations",
        )
    )
    if len(image_paths) != counts.get("images"):
        raise ValueError("referenced image count differs from dataset manifest")

    sft_capabilities = Counter(pair.capability.value for pair in sft)
    dpo_capabilities = Counter(pair.capability.value for pair in dpo)
    _threshold("SFT pairs", len(sft), minimum_sft_pairs)
    _threshold("DPO pairs", len(dpo), minimum_dpo_pairs)
    _threshold(
        "document lineages",
        len(train_lineages | validation_lineages),
        minimum_lineages,
    )
    for capability, minimum in (minimum_sft_capabilities or {}).items():
        _threshold(
            f"SFT capability {capability}",
            sft_capabilities[capability],
            minimum,
        )
    for capability, minimum in (minimum_dpo_capabilities or {}).items():
        _threshold(
            f"DPO capability {capability}",
            dpo_capabilities[capability],
            minimum,
        )

    return DatasetProof(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        evidence_sha256=evidence,
        sft_pairs=len(sft),
        dpo_pairs=len(dpo),
        train_sft_pairs=len(sft_train),
        validation_sft_pairs=len(sft_validation),
        train_dpo_pairs=len(dpo_train),
        validation_dpo_pairs=len(dpo_validation),
        lineages=len(train_lineages | validation_lineages),
        images=len(image_paths),
        sft_capabilities=dict(sorted(sft_capabilities.items())),
        dpo_capabilities=dict(sorted(dpo_capabilities.items())),
    )


def _training_config(
    *,
    stage: str,
    dataset_name: str,
    candidate_id: str,
    train_pairs: int,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    if stage not in {"sft", "dpo"}:
        raise ValueError(f"unsupported training stage: {stage}")
    steps_per_epoch = ceil(train_pairs / 6)
    output_suffix = "sft" if stage == "sft" else "dpo"
    config: dict[str, Any] = {
        "model_name_or_path": f"/training/models/{MODEL_NAME}",
        "trust_remote_code": True,
        "image_max_pixels": 1_048_576,
        "stage": stage,
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "freeze_vision_tower": False,
        "freeze_multi_modal_projector": False,
        "freeze_language_model": False,
        "dataset": f"electronics_{stage}_train",
        "eval_dataset": f"electronics_{stage}_validation",
        "dataset_dir": f"/training/datasets/{dataset_name}/llamafactory",
        "template": "qwen3_vl_nothink",
        "cutoff_len": 8192,
        "max_samples": train_pairs,
        "overwrite_cache": True,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 1,
        "output_dir": (
            f"/training/checkpoints/{candidate_id}-{output_suffix}"
        ),
        "logging_steps": 1,
        "eval_strategy": "steps",
        "eval_steps": steps_per_epoch,
        "save_steps": steps_per_epoch,
        "save_total_limit": 2,
        "plot_loss": True,
        "overwrite_output_dir": False,
        "save_only_model": False,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "learning_rate": 0.00002 if stage == "sft" else 0.000005,
        "num_train_epochs": epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "bf16": True,
        "seed": seed,
        "data_seed": seed,
        "disable_shuffling": False,
        "ddp_timeout": 180_000_000,
        "ddp_find_unused_parameters": False,
    }
    if stage == "dpo":
        config.update(
            {
                "adapter_name_or_path": (
                    f"/training/checkpoints/{candidate_id}-sft"
                ),
                "create_new_adapter": False,
                "pref_beta": 0.1,
                "pref_loss": "sigmoid",
                "pref_ftx": 0.1,
            }
        )
    return config


def seal_training_handoff(
    dataset: Path,
    destination: Path,
    *,
    candidate_id: str,
    proof: DatasetProof,
    sft_epochs: int = 3,
    dpo_epochs: int = 2,
) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,95}", candidate_id):
        raise ValueError("candidate_id must be a safe lowercase slug")
    if not 1 <= sft_epochs <= 20 or not 1 <= dpo_epochs <= 20:
        raise ValueError("training epochs must be within 1..20")
    dataset_root = dataset.expanduser().resolve(strict=True)
    output = destination.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"training handoff already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    seed = int(proof.evidence_sha256[:8], 16)
    sft_config = _training_config(
        stage="sft",
        dataset_name=dataset_root.name,
        candidate_id=candidate_id,
        train_pairs=proof.train_sft_pairs,
        epochs=sft_epochs,
        seed=seed,
    )
    dpo_config = _training_config(
        stage="dpo",
        dataset_name=dataset_root.name,
        candidate_id=candidate_id,
        train_pairs=proof.train_dpo_pairs,
        epochs=dpo_epochs,
        seed=seed + 1,
    )
    try:
        config_receipts: dict[str, dict[str, Any]] = {}
        for name, config in (
            ("sft.yaml", sft_config),
            ("dpo.yaml", dpo_config),
        ):
            path = temporary / name
            with path.open("x", encoding="utf-8") as handle:
                yaml.safe_dump(
                    config,
                    handle,
                    sort_keys=False,
                    allow_unicode=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            config_receipts[name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        core = {
            "schema": HANDOFF_SCHEMA,
            "candidate_id": candidate_id,
            "decision": "ready_to_stage",
            "training_image": IMAGE_ID,
            "world_size": 6,
            "roles": ["dgx2", "asus1", "dgx3", "asus3", "asus2", "asus4"],
            "dataset": {
                "source_path": str(dataset_root),
                "staged_relative_path": f"datasets/{dataset_root.name}",
                "manifest_sha256": proof.manifest_sha256,
                "evidence_sha256": proof.evidence_sha256,
                "counts": {
                    "sft": proof.sft_pairs,
                    "dpo": proof.dpo_pairs,
                    "lineages": proof.lineages,
                    "images": proof.images,
                },
                "capabilities": {
                    "sft": proof.sft_capabilities,
                    "dpo": proof.dpo_capabilities,
                },
            },
            "configs": config_receipts,
            "expected_optimizer_steps": {
                "sft": ceil(proof.train_sft_pairs / 6) * sft_epochs,
                "dpo": ceil(proof.train_dpo_pairs / 6) * dpo_epochs,
            },
            "gates": {
                "source_grounded_pairs_only": True,
                "frozen_document_overlap_zero": True,
                "lineage_split_isolated": True,
                "artifact_hashes_verified": True,
                "vision_tower_trainable": True,
                "multimodal_projector_trainable": True,
                "language_model_trainable": True,
                "promotion_requires_frozen_base_candidate_evaluation": True,
            },
        }
        manifest = {
            **core,
            "evidence_sha256": hashlib.sha256(canonical_json(core)).hexdigest(),
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(
                manifest,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for path in temporary.iterdir():
            os.chmod(path, 0o444)
        os.chmod(temporary, 0o555)
        os.rename(temporary, output)
    except BaseException:
        try:
            os.chmod(temporary, 0o755)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


__all__ = [
    "DATASET_SCHEMA",
    "DatasetProof",
    "HANDOFF_SCHEMA",
    "IMAGE_ID",
    "MODEL_NAME",
    "seal_training_handoff",
    "verify_training_dataset",
]
