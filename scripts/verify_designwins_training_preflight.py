#!/usr/bin/env python3
"""Fail closed unless a DesignWins training dataset fits its configured cutoff."""

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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml_scalar(path: Path, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}\s*:\s*(.*?)\s*$")
    values: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        match = pattern.fullmatch(line)
        if match:
            value = match.group(1).strip().strip("\"'")
            if value:
                values.append(value)
    if len(values) != 1:
        raise ValueError(f"config requires exactly one {key}")
    return values[0]


def _training_path(root: Path, value: str) -> Path:
    prefix = "/training/"
    if not value.startswith(prefix):
        raise ValueError("dataset_dir must be an absolute /training path")
    path = (root / value.removeprefix(prefix)).resolve(strict=True)
    if not path.is_relative_to(root) or path.is_symlink():
        raise ValueError("training dataset path escapes the owned root")
    return path


def _record_count(path: Path) -> int:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("training dataset JSON must be an array")
        return len(value)
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def verify(
    root: Path,
    config: Path,
    audit: Path,
    *,
    database: Path | None = None,
    dataset_version_id: str | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    config_input = config.expanduser()
    audit_input = audit.expanduser()
    if (
        config_input.is_symlink()
        or audit_input.is_symlink()
        or not config_input.is_file()
        or not audit_input.is_file()
    ):
        raise ValueError("config and audit must be regular non-symlink files")
    config = config_input.resolve(strict=True)
    audit = audit_input.resolve(strict=True)
    if not config.is_relative_to(root / "configs"):
        raise ValueError("config must be below the owned root configs directory")
    if not audit.is_relative_to(root):
        raise ValueError("sequence audit must be below the owned training root")

    dataset_name = _yaml_scalar(config, "dataset")
    dataset_dir = _training_path(root, _yaml_scalar(config, "dataset_dir"))
    cutoff = int(_yaml_scalar(config, "cutoff_len"))
    if cutoff < 256:
        raise ValueError("cutoff_len is invalid")
    info_path = dataset_dir / "dataset_info.json"
    if info_path.is_symlink() or not info_path.is_file():
        raise ValueError("dataset_info.json is missing or unsafe")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    spec = info.get(dataset_name) if isinstance(info, dict) else None
    if not isinstance(spec, dict) or not isinstance(spec.get("file_name"), str):
        raise ValueError("configured dataset is absent from dataset_info.json")
    dataset_input = dataset_dir / spec["file_name"]
    if dataset_input.is_symlink() or not dataset_input.is_file():
        raise ValueError("configured dataset file is missing or unsafe")
    dataset = dataset_input.resolve(strict=True)
    if (
        not dataset.is_file() or not dataset.is_relative_to(dataset_dir)
    ):
        raise ValueError("configured dataset file is missing or unsafe")

    evidence = json.loads(audit.read_text(encoding="utf-8"))
    if evidence.get("schema") != "harness.designwins-sequence-length-audit.v1":
        raise ValueError("unexpected sequence-audit schema")
    dataset_sha256 = _sha256(dataset)
    if evidence.get("dataset_sha256") != dataset_sha256:
        raise ValueError("sequence audit does not bind the configured dataset")
    records = _record_count(dataset)
    if evidence.get("records") != records:
        raise ValueError("sequence audit record count does not match the dataset")
    cutoff_result = evidence.get("cutoffs", {}).get(str(cutoff))
    if not isinstance(cutoff_result, dict):
        raise ValueError("sequence audit does not cover the configured cutoff")
    if cutoff_result.get("truncated_records") != 0:
        raise ValueError(
            f"sequence audit rejects {cutoff_result.get('truncated_records')} "
            f"truncated records at cutoff {cutoff}"
        )
    if cutoff_result.get("truncated_rate") != 0:
        raise ValueError("sequence audit reports a non-zero truncation rate")
    result = {
        "schema": "harness.designwins-training-preflight.v1",
        "passed": True,
        "dataset": dataset_name,
        "dataset_sha256": dataset_sha256,
        "records": records,
        "cutoff_len": cutoff,
        "audit_sha256": _sha256(audit),
    }
    if (database is None) != (dataset_version_id is None):
        raise ValueError("database and dataset version must be provided together")
    if database is not None and dataset_version_id is not None:
        database_input = database.expanduser()
        if database_input.is_symlink() or not database_input.is_file():
            raise ValueError("training registry database is missing or unsafe")
        database_path = database_input.resolve(strict=True)
        if not database_path.is_relative_to(root / "ledger"):
            raise ValueError("training registry database must be below root/ledger")
        with sqlite3.connect(database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT state, manifest_sha256, split_policy_json
                FROM dataset_versions WHERE dataset_version_id = ?
                """,
                (dataset_version_id,),
            ).fetchone()
            if row is None or row["state"] != "eligible":
                raise ValueError("dataset version is absent or ineligible")
            policy = json.loads(row["split_policy_json"])
            if policy.get("sequence_audit_sha256") != result["audit_sha256"]:
                raise ValueError("dataset version does not bind the sequence audit")
            chunk_manifest = dataset_dir.parent / "manifest.json"
            if (
                chunk_manifest.is_symlink()
                or not chunk_manifest.is_file()
                or policy.get("chunk_manifest_sha256")
                != _sha256(chunk_manifest)
            ):
                raise ValueError("dataset version does not bind the chunk manifest")
            train_members = connection.execute(
                """
                SELECT COUNT(*) FROM dataset_members
                WHERE dataset_version_id = ? AND split = 'train'
                """,
                (dataset_version_id,),
            ).fetchone()[0]
            if train_members != records:
                raise ValueError(
                    "registered train member count does not match the dataset"
                )
        result["dataset_version_id"] = dataset_version_id
        result["dataset_manifest_sha256"] = row["manifest_sha256"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sequence-audit", required=True, type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--dataset-version-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify(
        args.root,
        args.config,
        args.sequence_audit,
        database=args.database,
        dataset_version_id=args.dataset_version_id,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
