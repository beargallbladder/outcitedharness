#!/usr/bin/env python3
"""Fail-closed preflight for the DGX2 Qwen3-VL pinout LoRA pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCHEMA = "harness.pinout-vision-training-preflight.v1"
DATASET_SCHEMA = "harness.pinout-vision-row-dataset.v1"
EXPECTED_MODEL = "/training/models/Qwen3-VL-8B-Instruct"
EXPECTED_DATASET_DIR = "/training/datasets/pinout-vision-row-v1/llamafactory"
EXPECTED_DATASET = "pinout_rows_train"
EXPECTED_IMAGE = (
    "sha256:728b352622d569b1343cb384dd8aa394"
    "917203d02631322b02d302d068e38139"
)
MODE = {
    "pilot": {
        "max_steps": 8,
        "max_samples": 32,
        "output_dir": "/training/checkpoints/pinout-rows-qwen3-vl-8b-pilot-v1",
        "resume_from_checkpoint": None,
    },
    "resume": {
        "max_steps": 12,
        "max_samples": 32,
        "output_dir": "/training/checkpoints/pinout-rows-qwen3-vl-8b-resume-v1",
        "resume_from_checkpoint": (
            "/training/checkpoints/pinout-rows-qwen3-vl-8b-pilot-v1/checkpoint-8"
        ),
    },
    "candidate": {
        "max_steps": 1101,
        "max_samples": 1101,
        "output_dir": (
            "/training/checkpoints/pinout-rows-qwen3-vl-8b-candidate-v1"
        ),
        "resume_from_checkpoint": None,
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
    },
    "mm_candidate": {
        "max_steps": 1101,
        "max_samples": 1101,
        "output_dir": (
            "/training/checkpoints/pinout-rows-qwen3-vl-8b-mm-candidate-v1"
        ),
        "resume_from_checkpoint": None,
        "freeze_vision_tower": False,
        "freeze_multi_modal_projector": False,
        "learning_rate": 0.00002,
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


def _under(root: Path, container_path: str) -> Path:
    prefix = "/training/"
    if not container_path.startswith(prefix):
        raise ValueError(f"container path is outside /training: {container_path}")
    path = (root / container_path.removeprefix(prefix)).resolve(strict=False)
    if not path.is_relative_to(root):
        raise ValueError(f"container path escapes training root: {container_path}")
    return path


def _dataset(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_root = root / "datasets" / "pinout-vision-row-v1"
    manifest_path = dataset_root / "manifest.json"
    manifest = _json_object(manifest_path, "dataset manifest")
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("dataset manifest schema is not supported")
    expected_evidence = manifest.get("evidence_sha256")
    core = {
        key: value
        for key, value in manifest.items()
        if key not in ("created_at", "evidence_sha256")
    }
    if hashlib.sha256(_canonical(core)).hexdigest() != expected_evidence:
        raise ValueError("dataset manifest evidence digest is invalid")
    authorization = manifest.get("authorization")
    if (
        not isinstance(authorization, dict)
        or authorization.get("training_authorized") is not True
        or authorization.get("status") != "authorized"
    ):
        raise ValueError("dataset is not authorized for training")
    for evidence in [
        authorization.get("receipt"),
        *(authorization.get("evidence") or []),
    ]:
        if not isinstance(evidence, dict):
            raise ValueError("dataset authorization evidence is malformed")
        relative = Path(str(evidence.get("bundled_path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("dataset authorization evidence path is unsafe")
        path = _regular(dataset_root / relative, "authorization evidence")
        if _sha256(path) != evidence.get("sha256"):
            raise ValueError("dataset authorization evidence hash mismatch")

    counts = manifest.get("counts")
    examples = counts.get("examples") if isinstance(counts, dict) else None
    if (
        not isinstance(examples, dict)
        or int(examples.get("train", 0)) < 1101
        or int(examples.get("validation", 0)) < 1
        or int(examples.get("test", 0)) < 1
    ):
        raise ValueError("dataset split counts do not meet the training floor")
    for relative, expected in manifest.get("artifacts", {}).items():
        path_value = Path(relative)
        if path_value.is_absolute() or ".." in path_value.parts:
            raise ValueError("dataset artifact path is unsafe")
        path = _regular(dataset_root / path_value, "dataset artifact")
        if (
            _sha256(path) != expected.get("sha256")
            or path.stat().st_size != expected.get("bytes")
        ):
            raise ValueError(f"dataset artifact hash mismatch: {relative}")

    image_manifest = dataset_root / "image-manifest.jsonl"
    image_artifacts: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(image_manifest.read_text().splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"image manifest row {line_number} is malformed")
        relative = Path(str(row.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"image manifest row {line_number} is unsafe")
        image = _regular(dataset_root / relative, "dataset image")
        metadata = {
            "sha256": _sha256(image),
            "bytes": image.stat().st_size,
        }
        if metadata != {
            "sha256": row.get("sha256"),
            "bytes": row.get("bytes"),
        }:
            raise ValueError(f"image hash mismatch: {relative}")
        image_artifacts[relative.as_posix()] = metadata
    if hashlib.sha256(_canonical(image_artifacts)).hexdigest() != manifest.get(
        "images_aggregate_sha256"
    ):
        raise ValueError("image aggregate digest mismatch")

    split_lineages: dict[str, set[str]] = {}
    observed_examples: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        canonical_path = dataset_root / "canonical" / f"{split}.jsonl"
        lineages: set[str] = set()
        observed = 0
        for line_number, line in enumerate(
            canonical_path.read_text().splitlines(),
            1,
        ):
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or row.get("split") != split
                or not isinstance(row.get("lineage_id"), str)
            ):
                raise ValueError(
                    f"canonical {split} row {line_number} is malformed"
                )
            lineages.add(row["lineage_id"])
            observed += 1
        split_lineages[split] = lineages
        observed_examples[split] = observed
    if observed_examples != {key: int(value) for key, value in examples.items()}:
        raise ValueError("canonical split counts differ from the manifest")
    if (
        split_lineages["train"] & split_lineages["validation"]
        or split_lineages["train"] & split_lineages["test"]
        or split_lineages["validation"] & split_lineages["test"]
    ):
        raise ValueError("source PDF lineage leaks across dataset splits")
    return manifest, {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "evidence_sha256": expected_evidence,
        "examples": observed_examples,
        "unique_images": len(image_artifacts),
    }


def _config(root: Path, path: Path, mode: str) -> tuple[dict[str, Any], Path]:
    path = _regular(path.resolve(strict=True), "training config")
    if not path.is_relative_to(root / "configs"):
        raise ValueError("training config must be below the DGX2 configs directory")
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("training config must contain a YAML object")
    expected = MODE[mode]
    exact = {
        "model_name_or_path": EXPECTED_MODEL,
        "dataset": EXPECTED_DATASET,
        "dataset_dir": EXPECTED_DATASET_DIR,
        "template": "qwen3_vl_nothink",
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": 16,
        "lora_alpha": 32,
        "freeze_vision_tower": expected.get("freeze_vision_tower", True),
        "freeze_multi_modal_projector": expected.get(
            "freeze_multi_modal_projector",
            True,
        ),
        "freeze_language_model": False,
        "cutoff_len": 4096,
        "max_samples": expected["max_samples"],
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "learning_rate": expected.get("learning_rate", 0.0001),
        "bf16": True,
        "max_steps": expected["max_steps"],
        "output_dir": expected["output_dir"],
    }
    for key, expected_value in exact.items():
        if value.get(key) != expected_value:
            raise ValueError(
                f"training config {key} must be {expected_value!r}, "
                f"observed {value.get(key)!r}"
            )
    if value.get("resume_from_checkpoint") != expected["resume_from_checkpoint"]:
        raise ValueError("training config resume checkpoint is incorrect")
    output_path = _under(root, value["output_dir"])
    if output_path.exists() or output_path.is_symlink():
        raise ValueError(f"immutable training output already exists: {output_path}")
    return value, output_path


def _resume_source(root: Path, config: dict[str, Any], mode: str) -> None:
    if mode != "resume":
        return
    checkpoint = _under(root, config["resume_from_checkpoint"])
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ValueError("pilot checkpoint-8 is missing or unsafe")
    state = _json_object(checkpoint / "trainer_state.json", "pilot trainer state")
    if state.get("global_step") != 8:
        raise ValueError("pilot checkpoint does not contain global step 8")
    _regular(checkpoint / "adapter_model.safetensors", "pilot adapter")


def run_preflight(
    *,
    root: Path,
    config_path: Path,
    mode: str,
    image: str,
    minimum_free_gib: int,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if mode not in MODE:
        raise ValueError(f"unsupported preflight mode: {mode}")
    marker = _regular(root / ".harness-training-owner-v1", "training owner marker")
    if "role=dgx2" not in marker.read_text().splitlines():
        raise ValueError("training root marker is not DGX2")
    if image != EXPECTED_IMAGE:
        raise ValueError(f"training image must be pinned to {EXPECTED_IMAGE}")
    model_config = _regular(
        root / "models" / "Qwen3-VL-8B-Instruct" / "config.json",
        "Qwen3-VL model config",
    )
    config, output_path = _config(root, config_path, mode)
    _resume_source(root, config, mode)
    _manifest, dataset = _dataset(root)
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < minimum_free_gib * 1024**3:
        raise ValueError(
            f"DGX2 free space is below {minimum_free_gib} GiB: "
            f"{free_bytes / 1024**3:.1f} GiB"
        )
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "passed": True,
        "mode": mode,
        "root": str(root),
        "image": image,
        "model_config": {
            "path": str(model_config),
            "sha256": _sha256(model_config),
            "bytes": model_config.stat().st_size,
        },
        "dataset": dataset,
        "config": {
            "path": str(config_path.resolve(strict=True)),
            "sha256": _sha256(config_path),
            "max_steps": config["max_steps"],
            "max_samples": config["max_samples"],
        },
        "output_path": str(output_path),
        "free_gib": round(free_bytes / 1024**3, 3),
        "minimum_free_gib": minimum_free_gib,
    }
    core["evidence_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **core,
    }


def write_new(path: Path, value: dict[str, Any]) -> None:
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
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=sorted(MODE))
    parser.add_argument("--image", default=EXPECTED_IMAGE)
    parser.add_argument("--minimum-free-gib", type=int, default=500)
    parser.add_argument("--receipt-output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.minimum_free_gib < 1:
        raise ValueError("minimum free GiB must be positive")
    receipt = run_preflight(
        root=arguments.root,
        config_path=arguments.config,
        mode=arguments.mode,
        image=arguments.image,
        minimum_free_gib=arguments.minimum_free_gib,
    )
    write_new(arguments.receipt_output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
