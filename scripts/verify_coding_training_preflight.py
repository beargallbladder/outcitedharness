#!/usr/bin/env python3
"""Fail closed unless a registered coding dataset is complete and untruncated."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml_scalar(path: Path, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}\s*:\s*(.*?)\s*$")
    values = []
    for raw in path.read_text().splitlines():
        match = pattern.fullmatch(raw.split("#", 1)[0].strip())
        if match and match.group(1).strip():
            values.append(match.group(1).strip().strip("\"'"))
    if len(values) != 1:
        raise ValueError(f"config requires exactly one {key}")
    return values[0]


def _inside(root: Path, value: str) -> Path:
    if not value.startswith("/training/"):
        raise ValueError("training path must start with /training/")
    path = (root / value.removeprefix("/training/")).resolve(strict=True)
    if not path.is_relative_to(root) or path.is_symlink():
        raise ValueError("training path escapes the owned root")
    return path


def verify(
    *,
    root: Path,
    config: Path,
    audit: Path,
    database: Path,
    dataset_version_id: str,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    for path in (config, audit, database):
        if path.is_symlink() or not path.is_file():
            raise ValueError("preflight inputs must be regular non-symlink files")
    config = config.resolve(strict=True)
    audit = audit.resolve(strict=True)
    database = database.resolve(strict=True)
    if not config.is_relative_to(root / "configs"):
        raise ValueError("config must be below the owned configs root")
    if not audit.is_relative_to(root) or not database.is_relative_to(
        root / "ledger"
    ):
        raise ValueError("audit or database is outside its owned root")
    if _yaml_scalar(config, "do_train").casefold() != "true":
        raise ValueError("coding training config is not enabled")
    dataset_name = _yaml_scalar(config, "dataset")
    if dataset_name != "coding_sft_train":
        raise ValueError("coding smoke must use the admitted SFT train split")
    dataset_dir = _inside(root, _yaml_scalar(config, "dataset_dir"))
    cutoff = int(_yaml_scalar(config, "cutoff_len"))
    info = json.loads((dataset_dir / "dataset_info.json").read_text())
    spec = info.get(dataset_name)
    if not isinstance(spec, dict) or not isinstance(spec.get("file_name"), str):
        raise ValueError("configured coding dataset is absent")
    dataset = (dataset_dir / spec["file_name"]).resolve(strict=True)
    if not dataset.is_relative_to(dataset_dir) or dataset.is_symlink():
        raise ValueError("configured coding dataset path is unsafe")
    sequence = json.loads(audit.read_text())
    if (
        sequence.get("schema")
        != "harness.coding-sequence-length-audit.v1"
        or sequence.get("dataset_sha256") != _sha256(dataset)
        or sequence.get("cutoff_len") != cutoff
        or sequence.get("truncated_records") != 0
    ):
        raise ValueError("coding sequence audit does not cover the training input")
    dataset_rows = json.loads(dataset.read_text())
    if (
        not isinstance(dataset_rows, list)
        or not dataset_rows
        or sequence.get("records") != len(dataset_rows)
    ):
        raise ValueError("coding sequence audit record count is inconsistent")
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        version = connection.execute(
            """
            SELECT state, manifest_sha256, split_policy_json
            FROM dataset_versions WHERE dataset_version_id = ?
            """,
            (dataset_version_id,),
        ).fetchone()
        if version is None or version["state"] != "eligible":
            raise ValueError("coding dataset version is absent or ineligible")
        policy = json.loads(version["split_policy_json"])
        if (
            policy.get("sequence_audit_sha256") != _sha256(audit)
            or policy.get("sequence_cutoff_len") != cutoff
        ):
            raise ValueError("dataset version does not bind the sequence audit")
        train_members = connection.execute(
            """
            SELECT COUNT(*) FROM dataset_members
            WHERE dataset_version_id = ? AND split = 'train'
            """,
            (dataset_version_id,),
        ).fetchone()[0]
        if train_members != len(dataset_rows):
            raise ValueError("registered train members differ from training rows")
    return {
        "schema": "harness.coding-training-preflight.v1",
        "passed": True,
        "dataset_version_id": dataset_version_id,
        "registry_manifest_sha256": version["manifest_sha256"],
        "dataset_sha256": _sha256(dataset),
        "sequence_audit_sha256": _sha256(audit),
        "records": len(dataset_rows),
        "cutoff_len": cutoff,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sequence-audit", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--dataset-version-id", required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            verify(
                root=arguments.root,
                config=arguments.config,
                audit=arguments.sequence_audit,
                database=arguments.database,
                dataset_version_id=arguments.dataset_version_id,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
