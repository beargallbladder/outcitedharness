"""Package-scoped access to owned electronics ground-truth rows."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from harness.electronics.locator import package_pin_count


PACKAGE_FAMILY_KEYS = (
    "vfqfpn",
    "ufqfpn",
    "wlcsp",
    "tfbga",
    "dsbga",
    "utqfn",
    "tqfn",
    "xqfn",
    "vqfn",
    "wqfn",
    "lqfp",
    "tqfp",
    "uqfn",
    "qfn",
    "bga",
    "wlp",
    "csp",
    "soic",
    "ssop",
    "tssop",
    "pdip",
    "spdip",
    "dfn",
    "lga",
)


def load_ground_truth_records(
    corpus: dict[str, Any],
    ground_truth_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    root = ground_truth_root.expanduser().resolve(strict=True)
    output: dict[str, list[dict[str, Any]]] = {}
    for document in corpus.get("documents") or []:
        records = []
        for item in document.get("ground_truth") or []:
            relative = item.get("path")
            if not isinstance(relative, str):
                continue
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"ground truth is unavailable: {path}")
            expected_sha256 = item.get("sha256")
            if (
                not isinstance(expected_sha256, str)
                or hashlib.sha256(path.read_bytes()).hexdigest()
                != expected_sha256
            ):
                raise ValueError(f"ground truth hash mismatch: {path}")
            value = json.loads(path.read_text(encoding="utf-8"))
            pinout = value.get("pinout") if isinstance(value, dict) else None
            if not isinstance(pinout, dict):
                continue
            rows = pinout.get("pin_functions_summary")
            if not isinstance(rows, list):
                continue
            records.append(
                {
                    "packages": tuple(
                        str(package).strip()
                        for package in pinout.get("packages") or []
                        if str(package).strip()
                    ),
                    "rows": [row for row in rows if isinstance(row, dict)],
                    "vendor": item.get("vendor"),
                    "record_id": item.get("record_id"),
                }
            )
        if records:
            output[document["document_sha256"]] = records
    return output


def package_key(value: str) -> str:
    return "".join(
        character for character in value.casefold() if character.isalnum()
    )


def _package_family(value: str) -> str | None:
    key = package_key(value)
    return next(
        (family for family in PACKAGE_FAMILY_KEYS if family in key),
        None,
    )


def _package_count(value: str) -> int | None:
    count = package_pin_count(value)
    if count is not None:
        return count
    if _package_family(value) is not None:
        match = re.search(r"(?<!\d)(\d{1,4})(?!\d)", value)
        if match:
            return int(match.group(1))
    return None


def rows_for_package(
    records: list[dict[str, Any]],
    package: str | None,
) -> list[dict[str, Any]]:
    if not package:
        return []
    target_key = package_key(package)
    target_count = _package_count(package)
    target_family = _package_family(package)
    rows = []
    for record in records:
        for candidate in record["packages"]:
            candidate_key = package_key(candidate)
            candidate_count = _package_count(candidate)
            candidate_family = _package_family(candidate)
            compatible_count = (
                target_count is None
                or candidate_count is None
                or target_count == candidate_count
            )
            if compatible_count and (
                candidate_key == target_key
                or candidate_key in target_key
                or target_key in candidate_key
                or (
                    target_count is not None
                    and target_count == candidate_count
                    and target_family is not None
                    and target_family == candidate_family
                )
            ):
                rows.extend(record["rows"])
                break
    return rows


__all__ = [
    "load_ground_truth_records",
    "package_key",
    "rows_for_package",
]
