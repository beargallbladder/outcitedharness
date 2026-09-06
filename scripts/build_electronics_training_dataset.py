#!/usr/bin/env python3
"""Seal verified electronics teacher shards as portable SFT and DPO data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import unquote, urlparse

from pydantic import BaseModel

from harness.electronics.claims import canonical_json
from harness.electronics.models import (
    PairDisposition,
    PreferenceTrainingPairCandidate,
    TrainingPairCandidate,
)


Pair = TypeVar(
    "Pair",
    TrainingPairCandidate,
    PreferenceTrainingPairCandidate,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                canonical_json(
                    row.model_dump(mode="json", by_alias=True)
                ).decode("utf-8")
                + "\n"
            )


def _jsonl(path: Path, model: type[Pair]) -> list[Pair]:
    rows: list[Pair] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(model.model_validate_json(line))
                except Exception as error:
                    raise ValueError(
                        f"{path}:{line_number}: {error}"
                    ) from error
    return rows


def _verify_bundle(directory: Path) -> tuple[dict[str, Any], Path, Path]:
    directory = directory.resolve(strict=True)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "harness.electronics-frontier-finalization.v1":
        raise ValueError(f"unexpected finalization schema in {manifest_path}")
    sft = directory / "training-pairs.jsonl"
    dpo = directory / "preference-training-pairs.jsonl"
    for name, path in (
        ("training-pairs.jsonl", sft),
        ("preference-training-pairs.jsonl", dpo),
    ):
        expected = manifest["artifacts"][name]["sha256"]
        if _sha256(path) != expected:
            raise ValueError(f"{name} does not match its sealed SHA-256")
    return manifest, sft, dpo


def _unique(pairs: list[Pair]) -> list[Pair]:
    by_id: dict[str, Pair] = {}
    for pair in pairs:
        previous = by_id.get(pair.pair_id)
        if previous is not None and previous != pair:
            raise ValueError(f"pair ID collision: {pair.pair_id}")
        by_id[pair.pair_id] = pair
    return [by_id[pair_id] for pair_id in sorted(by_id)]


_PIN_CAPABILITIES = {"pin_or_ball", "pin_semantics"}


def _null_type_majority(response_json: str) -> bool | None:
    """True/False for a pin pair by row-majority null `type`; None if not pin.

    Round 3 taught why this matters: adding 1,692 majority-null-type pin rows
    (TI power and borderless MCU tables print no Type column) against 293
    typed ones collapsed the candidate's type_accuracy from 1.0 to 0.476 on
    documents that DO print types. The pairs are individually correct — the
    imbalance is the defect — so the remedy is a composition cap, not a
    verification change.
    """

    rows = json.loads(response_json).get("pins")
    if not isinstance(rows, list) or not rows:
        return None
    nulls = sum(1 for row in rows if isinstance(row, dict) and row.get("type") is None)
    return nulls * 2 > len(rows)


def _cap_null_type_pin_pairs(
    pairs: list[Pair],
    *,
    max_null_fraction: float,
    response_of: Any,
) -> tuple[list[Pair], int]:
    """Drop excess majority-null-type pin pairs beyond the requested fraction.

    Selection is deterministic (sorted pair_id order); non-pin pairs and
    typed pin pairs are never dropped.
    """

    null_ids: list[str] = []
    typed_count = 0
    for pair in pairs:
        if pair.capability not in _PIN_CAPABILITIES:
            continue
        majority = _null_type_majority(response_of(pair))
        if majority is True:
            null_ids.append(pair.pair_id)
        elif majority is False:
            typed_count += 1
    # keep_null / (typed + keep_null) <= max_null_fraction
    if max_null_fraction >= 1.0 or not null_ids:
        return pairs, 0
    keep_null = int(
        (max_null_fraction * typed_count) / (1.0 - max_null_fraction)
    )
    if len(null_ids) <= keep_null:
        return pairs, 0
    dropped = set(sorted(null_ids)[keep_null:])
    kept = [pair for pair in pairs if pair.pair_id not in dropped]
    return kept, len(dropped)


def _holdout_documents(cohort: Path) -> set[str]:
    queue = json.loads((cohort.resolve(strict=True) / "work-queue.json").read_text())
    if queue.get("policy", {}).get("evaluation_only") is not True:
        raise ValueError("holdout work queue is not frozen_evaluation")
    if any(row.get("partition") != "frozen_evaluation" for row in queue["work"]):
        raise ValueError("holdout contains a non-evaluation work item")
    return {str(row["document_sha256"]) for row in queue["work"]}


def _split_by_lineage(
    sft: list[TrainingPairCandidate],
    dpo: list[PreferenceTrainingPairCandidate],
    *,
    validation_fraction: float,
    seed: str,
) -> dict[str, set[str]]:
    all_pairs: list[TrainingPairCandidate | PreferenceTrainingPairCandidate] = [
        *sft,
        *dpo,
    ]
    if any(len(pair.lineage_ids) != 1 for pair in all_pairs):
        raise ValueError("electronics page pairs must have one document lineage")
    lineages = sorted({pair.lineage_ids[0] for pair in all_pairs})
    if len(lineages) < 2:
        raise ValueError("at least two document lineages are required")
    ordered = sorted(
        lineages,
        key=lambda value: hashlib.sha256(
            f"{seed}:{value}".encode()
        ).hexdigest(),
    )
    validation_count = max(1, round(len(ordered) * validation_fraction))
    validation_count = min(validation_count, len(ordered) - 1)
    validation = set(ordered[:validation_count])
    return {
        "train": set(ordered) - validation,
        "validation": validation,
    }


def _source_image(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"only file:// source images can be sealed: {uri}")
    return Path(unquote(parsed.path)).resolve(strict=True)


def _portable_images(
    destination: Path,
    sft: list[TrainingPairCandidate],
    dpo: list[PreferenceTrainingPairCandidate],
) -> dict[str, str]:
    names: dict[str, str] = {}
    for pair in [*sft, *dpo]:
        for uri, expected_sha in zip(pair.image_uris, pair.image_sha256):
            source = _source_image(uri)
            actual_sha = _sha256(source)
            if actual_sha != expected_sha:
                raise ValueError(f"image SHA-256 mismatch: {source}")
            suffix = source.suffix.lower() or ".bin"
            name = f"{expected_sha}{suffix}"
            previous = names.get(expected_sha)
            if previous is not None and previous != name:
                raise ValueError(f"image extension collision: {expected_sha}")
            names[expected_sha] = name
            target = destination / "images" / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    return names


def _vision_prompt(prompt: str, images: int) -> str:
    instructions = prompt.split("\n\nPyMuPDF evidence:", 1)[0].rstrip()
    return "\n".join(["<image>"] * images) + "\n" + instructions


def _llamafactory_sft(
    pair: TrainingPairCandidate,
    image_names: dict[str, str],
) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": _vision_prompt(pair.prompt, len(pair.image_sha256)),
            },
            {"role": "assistant", "content": pair.response},
        ],
        "images": [
            f"../images/{image_names[value]}" for value in pair.image_sha256
        ],
    }


def _llamafactory_dpo(
    pair: PreferenceTrainingPairCandidate,
    image_names: dict[str, str],
) -> dict[str, Any]:
    return {
        "conversations": [
            {
                "role": "user",
                "content": _vision_prompt(pair.prompt, len(pair.image_sha256)),
            }
        ],
        "chosen": {"role": "assistant", "content": pair.chosen_response},
        "rejected": {"role": "assistant", "content": pair.rejected_response},
        "images": [
            f"../images/{image_names[value]}" for value in pair.image_sha256
        ],
    }


def _portable_pair(
    pair: Pair,
    image_names: dict[str, str],
) -> Pair:
    uris = tuple(
        f"dataset://electronics/images/{image_names[value]}"
        for value in pair.image_sha256
    )
    return pair.model_copy(update={"image_uris": uris})


def build_dataset(
    bundles: list[Path],
    frozen_cohorts: Path | list[Path],
    destination: Path,
    *,
    validation_fraction: float = 0.2,
    split_seed: str = "electronics-teacher-v1",
    max_null_type_pin_fraction: float = 1.0,
) -> dict[str, Any]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    destination = destination.resolve(strict=False)
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")

    source_receipts: list[dict[str, Any]] = []
    sft: list[TrainingPairCandidate] = []
    dpo: list[PreferenceTrainingPairCandidate] = []
    for bundle in bundles:
        manifest, sft_path, dpo_path = _verify_bundle(bundle)
        sft.extend(_jsonl(sft_path, TrainingPairCandidate))
        dpo.extend(_jsonl(dpo_path, PreferenceTrainingPairCandidate))
        source_receipts.append(
            {
                "path": str(bundle.resolve()),
                "evidence_sha256": manifest["evidence_sha256"],
                "manifest_sha256": _sha256(bundle / "manifest.json"),
            }
        )
    sft = _unique(sft)
    dpo = _unique(dpo)
    sft, sft_null_dropped = _cap_null_type_pin_pairs(
        sft,
        max_null_fraction=max_null_type_pin_fraction,
        response_of=lambda pair: pair.response,
    )
    dpo, dpo_null_dropped = _cap_null_type_pin_pairs(
        dpo,
        max_null_fraction=max_null_type_pin_fraction,
        response_of=lambda pair: pair.chosen_response,
    )
    if not sft or not dpo:
        raise ValueError("both SFT and DPO pairs are required")
    if any(pair.disposition is not PairDisposition.ADMITTED for pair in sft):
        raise ValueError("only admitted SFT pairs may be sealed")
    for pair in sft:
        json.loads(pair.response)
    for pair in dpo:
        json.loads(pair.chosen_response)
        json.loads(pair.rejected_response)

    cohorts = (
        [frozen_cohorts]
        if isinstance(frozen_cohorts, Path)
        else list(frozen_cohorts)
    )
    if not cohorts:
        raise ValueError("at least one frozen cohort is required")
    holdout: set[str] = set()
    cohort_receipts: list[dict[str, Any]] = []
    for cohort in cohorts:
        resolved = cohort.resolve(strict=True)
        holdout.update(_holdout_documents(resolved))
        cohort_receipts.append(
            {
                "path": str(resolved),
                "manifest_sha256": _sha256(resolved / "manifest.json"),
            }
        )
    training_lineages = {
        lineage for pair in [*sft, *dpo] for lineage in pair.lineage_ids
    }
    overlap = training_lineages & holdout
    if overlap:
        raise ValueError(f"training/holdout lineage overlap: {sorted(overlap)}")

    destination.mkdir(parents=True)
    image_names = _portable_images(destination, sft, dpo)
    splits = _split_by_lineage(
        sft,
        dpo,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )
    generated: list[Path] = []
    dataset_info: dict[str, Any] = {}
    counts: dict[str, dict[str, int]] = {}
    for split, lineages in splits.items():
        split_sft = [
            pair for pair in sft if pair.lineage_ids[0] in lineages
        ]
        split_dpo = [
            pair for pair in dpo if pair.lineage_ids[0] in lineages
        ]
        counts[split] = {"sft": len(split_sft), "dpo": len(split_dpo)}
        canonical_sft = destination / "canonical" / f"sft-{split}.jsonl"
        canonical_dpo = destination / "canonical" / f"dpo-{split}.jsonl"
        _write_jsonl(
            canonical_sft,
            [_portable_pair(pair, image_names) for pair in split_sft],
        )
        _write_jsonl(
            canonical_dpo,
            [_portable_pair(pair, image_names) for pair in split_dpo],
        )
        generated.extend((canonical_sft, canonical_dpo))

        lf_sft = destination / "llamafactory" / f"electronics_sft_{split}.json"
        lf_dpo = destination / "llamafactory" / f"electronics_dpo_{split}.json"
        _write_json(
            lf_sft,
            [_llamafactory_sft(pair, image_names) for pair in split_sft],
        )
        _write_json(
            lf_dpo,
            [_llamafactory_dpo(pair, image_names) for pair in split_dpo],
        )
        generated.extend((lf_sft, lf_dpo))
        tags = {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        }
        dataset_info[f"electronics_sft_{split}"] = {
            "file_name": lf_sft.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": tags,
        }
        dataset_info[f"electronics_dpo_{split}"] = {
            "file_name": lf_dpo.name,
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "conversations",
                "images": "images",
                "chosen": "chosen",
                "rejected": "rejected",
            },
            "tags": tags,
        }

    dataset_info_path = destination / "llamafactory" / "dataset_info.json"
    _write_json(dataset_info_path, dataset_info)
    generated.append(dataset_info_path)
    generated.extend(sorted((destination / "images").iterdir()))
    manifest = {
        "schema": "harness.electronics-teacher-dataset.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "local_datasheet_vision_training",
        "sources": source_receipts,
        "frozen_evaluation": {
            "cohorts": cohort_receipts,
            "document_overlap": 0,
        },
        "split": {
            "seed": split_seed,
            "validation_fraction": validation_fraction,
            "lineages": {
                name: sorted(values) for name, values in splits.items()
            },
        },
        "counts": {
            "sft": len(sft),
            "dpo": len(dpo),
            "images": len(image_names),
            "splits": counts,
        },
        "pin_type_balance": {
            "max_null_type_pin_fraction": max_null_type_pin_fraction,
            "sft_null_type_pairs_dropped": sft_null_dropped,
            "dpo_null_type_pairs_dropped": dpo_null_dropped,
        },
        "artifacts": {
            path.relative_to(destination).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(generated)
        },
    }
    manifest["evidence_sha256"] = hashlib.sha256(
        canonical_json(manifest)
    ).hexdigest()
    _write_json(destination / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        action="append",
        required=True,
        type=Path,
        help="verified frontier finalization directory; repeat for each shard",
    )
    parser.add_argument(
        "--frozen-cohort",
        action="append",
        required=True,
        type=Path,
        help="sealed holdout cohort; repeat for every evaluated capability",
    )
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", default="electronics-teacher-v1")
    parser.add_argument(
        "--max-null-type-pin-fraction",
        type=float,
        default=1.0,
        help=(
            "cap majority-null-type pin pairs at this fraction of all pin "
            "pairs (default 1.0 = no cap); round 3's type_accuracy collapse "
            "motivates 0.25 for continual rounds"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_dataset(
        args.bundle,
        args.frozen_cohort,
        args.output_directory,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        max_null_type_pin_fraction=args.max_null_type_pin_fraction,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(json.dumps(manifest["pin_type_balance"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
