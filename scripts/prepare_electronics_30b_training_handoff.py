#!/usr/bin/env python3
"""Assemble verified teacher shards and seal a reusable 30B training handoff."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from harness.electronics.corpus import sha256_file
from harness.electronics.training_handoff import (
    HANDOFF_SCHEMA,
    seal_training_handoff,
    verify_training_dataset,
)


def _capability_threshold(value: str) -> tuple[str, int]:
    name, separator, raw_count = value.partition("=")
    if (
        not separator
        or not name
        or not raw_count.isdigit()
        or int(raw_count) < 0
    ):
        raise argparse.ArgumentTypeError(
            "capability threshold must be NAME=NONNEGATIVE_COUNT"
        )
    return name, int(raw_count)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", type=Path, required=True)
    parser.add_argument(
        "--frozen-cohort",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--dataset-directory", type=Path, required=True)
    parser.add_argument("--handoff-directory", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", default="electronics-teacher-v2")
    parser.add_argument("--minimum-sft-pairs", type=int, default=256)
    parser.add_argument("--minimum-dpo-pairs", type=int, default=192)
    parser.add_argument("--minimum-lineages", type=int, default=100)
    parser.add_argument(
        "--minimum-sft-capability",
        action="append",
        type=_capability_threshold,
        default=[],
    )
    parser.add_argument(
        "--minimum-dpo-capability",
        action="append",
        type=_capability_threshold,
        default=[],
    )
    parser.add_argument("--sft-epochs", type=int, default=3)
    parser.add_argument("--dpo-epochs", type=int, default=2)
    return parser


def _expected_sources(bundles: list[Path]) -> set[tuple[str, str]]:
    values: set[tuple[str, str]] = set()
    for bundle in bundles:
        resolved = bundle.expanduser().resolve(strict=True)
        manifest_path = resolved / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema")
            != "harness.electronics-frontier-finalization.v1"
        ):
            raise ValueError(f"unsupported frontier finalization: {resolved}")
        values.add(
            (
                manifest["evidence_sha256"],
                sha256_file(manifest_path),
            )
        )
    return values


def _verify_dataset_sources(
    dataset: Path,
    expected: set[tuple[str, str]],
) -> None:
    manifest = json.loads(
        (dataset.expanduser().resolve(strict=True) / "manifest.json").read_text()
    )
    actual = {
        (str(row["evidence_sha256"]), str(row["manifest_sha256"]))
        for row in manifest.get("sources") or []
    }
    if actual != expected:
        raise ValueError(
            "dataset source finalizations differ from requested bundles"
        )


def _build_dataset(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("build_electronics_training_dataset.py")),
    ]
    for bundle in args.bundle:
        command.extend(["--bundle", str(bundle)])
    for cohort in args.frozen_cohort:
        command.extend(["--frozen-cohort", str(cohort)])
    command.extend(
        [
            "--output-directory",
            str(args.dataset_directory),
            "--validation-fraction",
            str(args.validation_fraction),
            "--split-seed",
            args.split_seed,
        ]
    )
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"dataset assembly failed ({result.returncode}): "
            f"{result.stderr[-4000:]}"
        )


def main() -> int:
    args = _parser().parse_args()
    expected_sources = _expected_sources(args.bundle)
    dataset = args.dataset_directory.expanduser().resolve()
    if not dataset.exists():
        _build_dataset(args)
    _verify_dataset_sources(dataset, expected_sources)

    proof = verify_training_dataset(
        dataset,
        minimum_sft_pairs=args.minimum_sft_pairs,
        minimum_dpo_pairs=args.minimum_dpo_pairs,
        minimum_lineages=args.minimum_lineages,
        minimum_sft_capabilities=dict(args.minimum_sft_capability),
        minimum_dpo_capabilities=dict(args.minimum_dpo_capability),
    )
    handoff = args.handoff_directory.expanduser().resolve()
    if handoff.exists():
        manifest = json.loads((handoff / "manifest.json").read_text())
        if (
            manifest.get("schema") != HANDOFF_SCHEMA
            or manifest.get("candidate_id") != args.candidate_id
            or manifest.get("dataset", {}).get("evidence_sha256")
            != proof.evidence_sha256
        ):
            raise ValueError("existing handoff conflicts with requested candidate")
    else:
        manifest = seal_training_handoff(
            dataset,
            handoff,
            candidate_id=args.candidate_id,
            proof=proof,
            sft_epochs=args.sft_epochs,
            dpo_epochs=args.dpo_epochs,
        )
    print(
        json.dumps(
            {
                "status": "ready_to_stage",
                "candidate_id": args.candidate_id,
                "dataset": str(dataset),
                "dataset_evidence_sha256": proof.evidence_sha256,
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
                "handoff": str(handoff),
                "handoff_evidence_sha256": manifest["evidence_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
