#!/usr/bin/env python3
"""Build a portable, hash-bound datasheet modality evaluation fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from harness.electronics import validate_physical_pin_truth


SCHEMA = "harness.datasheet-modality-fixture.v1"
PACKAGE = re.compile(
    r"(?i)\b(TFBGA|UFBGA|LFBGA|FBGA|WLCSP|LQFP|VFQFPN|UFQFPN)\s*(\d+)\b"
)
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, kind: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"{kind} must be a regular file: {path}")
    return resolved


def _load_gold(path: Path) -> list[dict[str, Any]]:
    value = json.loads(_regular_file(path, "gold set").read_text())
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("gold set must be a JSON array of objects")
    return value


def _select_package(row: dict[str, Any]) -> tuple[str, int]:
    packages = row.get("packages")
    if not isinstance(packages, list):
        raise ValueError(f"{row.get('stem')}: packages must be a list")
    for raw in packages:
        match = PACKAGE.search(str(raw))
        if match:
            return str(raw), int(match.group(2))
    raise ValueError(f"{row.get('stem')}: no supported exact package")


def build_fixture(
    *,
    gold_set: Path,
    output_root: Path,
    case_ids: list[str],
) -> dict[str, Any]:
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("case ids must be non-empty and unique")
    if any(not IDENTIFIER.fullmatch(case_id) for case_id in case_ids):
        raise ValueError("invalid case id")
    rows = _load_gold(gold_set)
    by_id = {str(row.get("stem") or ""): row for row in rows}
    missing = sorted(set(case_ids) - by_id.keys())
    if missing:
        raise ValueError(f"gold set is missing cases: {missing}")
    if output_root.exists() or output_root.is_symlink():
        raise ValueError(f"output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        (temporary / "pdf").mkdir()
        (temporary / "ground-truth").mkdir()
        cases = []
        for case_id in case_ids:
            row = by_id[case_id]
            package, package_pins = _select_package(row)
            pdf_source = _regular_file(Path(str(row["pdf_path"])), "datasheet")
            truth_source = _regular_file(
                Path(str(row["gt_snapshot"])),
                "ground truth",
            )
            with pdf_source.open("rb") as handle:
                if handle.read(5) != b"%PDF-":
                    raise ValueError(f"{case_id}: datasheet has no PDF signature")
            truth = json.loads(truth_source.read_text())
            pins = truth.get("pins") if isinstance(truth, dict) else None
            expected_rows = int(row["n_pins_gt"])
            if (
                not isinstance(pins, list)
                or len(pins) != expected_rows
                or int(truth.get("n_pins", -1)) != expected_rows
            ):
                raise ValueError(f"{case_id}: ground-truth pin count is inconsistent")
            validate_physical_pin_truth(
                pins,
                package=package,
                expected_package_pins=package_pins,
            )
            pdf_relative = Path("pdf") / f"{case_id}.pdf"
            truth_relative = Path("ground-truth") / f"{case_id}.json"
            pdf_output = temporary / pdf_relative
            truth_output = temporary / truth_relative
            shutil.copyfile(pdf_source, pdf_output)
            shutil.copyfile(truth_source, truth_output)
            os.chmod(pdf_output, 0o444)
            os.chmod(truth_output, 0o444)
            cases.append(
                {
                    "id": case_id,
                    "vendor": str(row.get("vendor") or ""),
                    "bucket": str(row.get("bucket") or ""),
                    "requested_package": package,
                    "expected_package_pins": package_pins,
                    "expected_ground_truth_rows": expected_rows,
                    "pdf": pdf_relative.as_posix(),
                    "pdf_sha256": sha256_file(pdf_output),
                    "ground_truth": truth_relative.as_posix(),
                    "ground_truth_sha256": sha256_file(truth_output),
                    "source_pdf": str(pdf_source),
                    "source_ground_truth": str(truth_source),
                }
            )

        manifest = {
            "schema": SCHEMA,
            "source_gold_set": str(gold_set.resolve()),
            "source_gold_set_sha256": sha256_file(gold_set.resolve()),
            "cases": cases,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(manifest_path, 0o444)
        os.replace(temporary, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-set", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case", action="append", dest="cases", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_fixture(
        gold_set=args.gold_set,
        output_root=args.output_root,
        case_ids=args.cases,
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "cases": len(manifest["cases"]),
                "output": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
