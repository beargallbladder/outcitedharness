from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_designwins_training_preflight.py"
SPEC = importlib.util.spec_from_file_location("designwins_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def _fixture(tmp_path: Path, *, truncated: int = 0) -> tuple[Path, Path, Path]:
    root = tmp_path / "training"
    configs = root / "configs"
    dataset_dir = root / "datasets" / "designwins-v4" / "llamafactory"
    manifests = root / "manifests"
    configs.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)
    manifests.mkdir()
    dataset = dataset_dir / "train.json"
    dataset.write_text(
        json.dumps(
            [
                {"instruction": "extract one", "input": "", "output": '{"pins":[]}'},
                {"instruction": "extract two", "input": "", "output": '{"pins":[]}'},
            ]
        )
    )
    (dataset_dir / "dataset_info.json").write_text(
        json.dumps({"designwins_v4": {"file_name": "train.json"}})
    )
    config = configs / "train.yaml"
    config.write_text(
        "dataset: designwins_v4\n"
        "dataset_dir: /training/datasets/designwins-v4/llamafactory\n"
        "cutoff_len: 4096\n"
    )
    audit = manifests / "sequence-audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema": "harness.designwins-sequence-length-audit.v1",
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "records": 2,
                "cutoffs": {
                    "4096": {
                        "truncated_records": truncated,
                        "truncated_rate": truncated / 2,
                    }
                },
            }
        )
    )
    return root, config, audit


def test_preflight_accepts_hash_bound_zero_truncation_audit(tmp_path: Path):
    root, config, audit = _fixture(tmp_path)

    result = preflight.verify(root, config, audit)

    assert result["passed"] is True
    assert result["records"] == 2
    assert result["cutoff_len"] == 4096


def test_preflight_rejects_truncation_and_dataset_drift(tmp_path: Path):
    root, config, audit = _fixture(tmp_path, truncated=1)
    with pytest.raises(ValueError, match="truncated records"):
        preflight.verify(root, config, audit)

    root, config, audit = _fixture(tmp_path / "drift")
    dataset = (
        root / "datasets" / "designwins-v4" / "llamafactory" / "train.json"
    )
    dataset.write_text("[]")
    with pytest.raises(ValueError, match="does not bind"):
        preflight.verify(root, config, audit)


def test_preflight_rejects_symlinked_audit(tmp_path: Path):
    root, config, audit = _fixture(tmp_path)
    linked = root / "manifests" / "linked.json"
    linked.symlink_to(audit)

    with pytest.raises(ValueError, match="non-symlink"):
        preflight.verify(root, config, linked)


def test_preflight_requires_registry_binding_and_member_count(tmp_path: Path):
    root, config, audit = _fixture(tmp_path)
    manifest = root / "datasets" / "designwins-v4" / "manifest.json"
    manifest.write_text('{"schema":"chunked"}\n')
    database = root / "ledger" / "learning.db"
    database.parent.mkdir()
    audit_sha256 = hashlib.sha256(audit.read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE dataset_versions (
                dataset_version_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL,
                split_policy_json TEXT NOT NULL
            );
            CREATE TABLE dataset_members (
                dataset_version_id TEXT NOT NULL,
                split TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO dataset_versions VALUES (?, ?, ?, ?)",
            (
                "designwins-v4",
                "eligible",
                "d" * 64,
                json.dumps(
                    {
                        "sequence_audit_sha256": audit_sha256,
                        "chunk_manifest_sha256": manifest_sha256,
                    }
                ),
            ),
        )
        connection.executemany(
            "INSERT INTO dataset_members VALUES (?, 'train')",
            [("designwins-v4",), ("designwins-v4",)],
        )

    result = preflight.verify(
        root,
        config,
        audit,
        database=database,
        dataset_version_id="designwins-v4",
    )

    assert result["dataset_version_id"] == "designwins-v4"
    assert result["dataset_manifest_sha256"] == "d" * 64

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE dataset_versions SET split_policy_json = ?",
            (
                json.dumps(
                    {
                        "sequence_audit_sha256": "0" * 64,
                        "chunk_manifest_sha256": manifest_sha256,
                    }
                ),
            ),
        )
    with pytest.raises(ValueError, match="does not bind the sequence audit"):
        preflight.verify(
            root,
            config,
            audit,
            database=database,
            dataset_version_id="designwins-v4",
        )
