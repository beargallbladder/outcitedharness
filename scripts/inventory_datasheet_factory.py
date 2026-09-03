#!/usr/bin/env python3
"""Seal the authoritative datasheet/ground-truth/CR corpus join."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harness.electronics.corpus import (
    AssetSource,
    CorpusInputs,
    build_corpus_registry,
    write_new_registry,
)


M5 = Path("/Volumes/M5_4TB")
REFERENCE = M5 / "DigiKey_Reference_Designs"
CR_DROPS = M5 / "exports" / "cr_drops"
DEFAULT_ASSETS = (
    AssetSource(
        "cr_pin_index",
        CR_DROPS
        / "20260818T192445Z_pin_index_consolidated"
        / "pin_tables_consolidated.jsonl",
    ),
    AssetSource(
        "cr_parametric_primary",
        CR_DROPS
        / "20260814_mcu_parametric_knives_m5"
        / "parametric_facts.jsonl",
    ),
    AssetSource(
        "cr_parametric_extended",
        CR_DROPS / "20260814_mcu_parametric_silabs_spc5",
    ),
    AssetSource(
        "cr_opn_decoder",
        CR_DROPS
        / "20260818T235000Z_opn-decoder-wave-v0"
        / "decode_locations.jsonl",
    ),
    AssetSource(
        "cr_ti_applications",
        CR_DROPS
        / "20260808_ti_applications"
        / "ti-applications-49.json",
    ),
    AssetSource(
        "cr_nxp_applications",
        CR_DROPS
        / "20260808_nxp_applications"
        / "nxp-applications-50.json",
    ),
    AssetSource(
        "cr_core_atoms_manifest",
        CR_DROPS / "20260820T213000Z_e3-core-atoms-v0" / "MANIFEST.json",
    ),
    AssetSource(
        "cr_low_power_manifest",
        CR_DROPS / "20260821T184000Z_e1-lowpower-v0" / "MANIFEST.json",
    ),
    AssetSource(
        "designwins_v3_manifest",
        M5
        / "harness-training"
        / "datasets"
        / "designwins-v3-20260829"
        / "manifest.json",
    ),
)


def _asset(value: str) -> AssetSource:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("asset must be NAME=PATH") from exc
    try:
        return AssetSource(name=name, path=Path(raw_path))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-root",
        type=Path,
        default=REFERENCE / "pdf_cache",
    )
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=REFERENCE / "claude_ground_truth",
    )
    parser.add_argument(
        "--validated-root",
        type=Path,
        default=REFERENCE / "MCU_Pinout_Validated",
    )
    parser.add_argument(
        "--source-audit",
        type=Path,
        default=Path(
            "/Users/samkim/Harnessv1/results/"
            "pinout-vision-source-audit-v2-20260901.json"
        ),
    )
    parser.add_argument(
        "--row-dataset-manifest",
        type=Path,
        default=REFERENCE / "training_data_pinout_rows_v1" / "manifest.json",
    )
    parser.add_argument(
        "--asset",
        action="append",
        type=_asset,
        default=[],
        help="Additional NAME=PATH JSON, JSONL, or directory asset.",
    )
    parser.add_argument(
        "--no-default-assets",
        action="store_true",
        help="Inventory only explicitly supplied --asset entries.",
    )
    parser.add_argument(
        "--expected-pdf-files",
        type=int,
        help="Fail closed if the live PDF count differs.",
    )
    parser.add_argument(
        "--hash-workers",
        type=int,
        default=4,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    assets = [
        *(DEFAULT_ASSETS if not args.no_default_assets else ()),
        *args.asset,
    ]
    missing = [str(asset.path) for asset in assets if not asset.path.exists()]
    if missing:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "configured_asset_missing",
                    "paths": missing,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    value = build_corpus_registry(
        CorpusInputs(
            pdf_root=args.pdf_root,
            ground_truth_root=args.ground_truth_root,
            validated_root=args.validated_root,
            source_audit=args.source_audit,
            row_dataset_manifest=args.row_dataset_manifest,
            assets=tuple(assets),
            expected_pdf_files=args.expected_pdf_files,
        ),
        hash_workers=args.hash_workers,
    )
    write_new_registry(args.output, value)
    print(
        json.dumps(
            {
                "status": "sealed",
                "path": str(args.output.resolve()),
                "evidence_sha256": value["evidence_sha256"],
                "counts": value["counts"],
                "assets": {
                    asset["name"]: asset["joins"] for asset in value["assets"]
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
