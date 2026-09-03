#!/usr/bin/env python3
"""Register an admitted datasheet-frontier dataset in the durable queue DB."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from harness.storage.db import Store
from harness.training.queue import DatasetMember, DatasetVersionRegistry
from harness.training.split import Split


SCHEMA = "harness.dataset.datasheet-frontier.v1"
POLICY_VERSION = "datasheet-independent-three-way-consensus-v3"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def load_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("dataset manifest must be a regular file")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported dataset manifest")
    expected = value.get("core_sha256")
    core = {key: item for key, item in value.items() if key != "core_sha256"}
    if expected != hashlib.sha256(_canonical(core)).hexdigest():
        raise ValueError("dataset manifest core digest mismatch")
    return value


def load_import_report(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("import report must be a regular file")
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema") != "harness.learning-transfer-import.v1"
    ):
        raise ValueError("unsupported learning import report")
    expected = value.get("result_sha256")
    core = {key: item for key, item in value.items() if key != "result_sha256"}
    if expected != hashlib.sha256(_canonical(core)).hexdigest():
        raise ValueError("learning import report digest mismatch")
    return value


def register(
    *,
    store: Store,
    manifest: dict[str, Any],
    dataset_version_id: str,
    version: str,
    import_report: dict[str, Any] | None = None,
) -> str:
    source_events = manifest.get("source_events")
    counts = manifest.get("counts")
    if (
        not isinstance(source_events, list)
        or not source_events
        or not isinstance(counts, dict)
        or counts.get("train_pairs") != len(source_events)
    ):
        raise ValueError("dataset manifest source-event count is invalid")

    imported_admissions: dict[str, tuple[str, str]] = {}
    if import_report is not None:
        raw_imports = import_report.get("events")
        if not isinstance(raw_imports, list):
            raise ValueError("learning import report events are malformed")
        for imported in raw_imports:
            if not isinstance(imported, dict):
                raise ValueError("learning import report event is malformed")
            event_id = str(imported.get("event_id") or "")
            values = (
                str(imported.get("source_admission_sha256") or ""),
                str(imported.get("destination_admission_sha256") or ""),
            )
            if not event_id or event_id in imported_admissions:
                raise ValueError("learning import report event IDs are invalid")
            if any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in values
            ):
                raise ValueError("learning import admission digests are invalid")
            imported_admissions[event_id] = values

    members: list[DatasetMember] = []
    with store.connect() as connection:
        for source in source_events:
            if not isinstance(source, dict):
                raise ValueError("dataset source event is malformed")
            event_id = source.get("event_id")
            row = connection.execute(
                """
                SELECT
                    e.event_sha256,
                    e.lineage_id,
                    e.source_revision,
                    e.metadata_json,
                    a.artifact_id,
                    a.redacted,
                    ad.admission_id,
                    ad.admission_sha256,
                    ad.policy_version,
                    ad.decision,
                    v.status AS verification_status
                FROM learning_events AS e
                JOIN learning_artifacts AS a
                  ON a.event_id = e.event_id
                 AND a.kind = 'frontier_prediction'
                JOIN learning_admissions AS ad ON ad.event_id = e.event_id
                JOIN learning_verifications AS v
                  ON v.verification_id = ad.verification_id
                WHERE e.event_id = ?
                """,
                (event_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"{event_id}: admitted frontier artifact is missing")
            metadata = json.loads(row["metadata_json"])
            source_admission_sha256 = source.get("admission_sha256")
            accepted_admission_sha256 = {source_admission_sha256}
            imported = imported_admissions.get(str(event_id))
            if imported is not None:
                if imported[0] != source_admission_sha256:
                    raise ValueError(
                        f"{event_id}: import report source admission differs"
                    )
                accepted_admission_sha256.add(imported[1])
            if (
                row["event_sha256"] != source.get("event_sha256")
                or row["admission_id"] != source.get("admission_id")
                or row["admission_sha256"] not in accepted_admission_sha256
                or row["policy_version"] != POLICY_VERSION
                or row["decision"] != "eligible"
                or row["verification_status"] != "pass"
                or not bool(row["redacted"])
                or metadata.get("data_use") != "training"
                or metadata.get("disposition") != "verified"
                or metadata.get("training_eligible") is not True
            ):
                raise ValueError(f"{event_id}: admission evidence is inconsistent")
            members.append(
                DatasetMember(
                    event_id=str(event_id),
                    artifact_id=row["artifact_id"],
                    split=Split.TRAIN,
                    lineage_id=row["lineage_id"],
                    source_document_sha256=row["source_revision"],
                    component_family=str(metadata["case_id"]),
                )
            )

    return DatasetVersionRegistry(store).create(
        dataset_version_id=dataset_version_id,
        name="datasheet-frontier-vision",
        version=version,
        source_revision=str(manifest["core_sha256"]),
        split_policy={
            "kind": "preassigned_admitted_frontier",
            "leakage_keys": [
                "lineage_id",
                "source_document_sha256",
                "component_family",
            ],
            "frozen_fixture_sha256": manifest["frozen_fixture_sha256"],
            "admission_policy": POLICY_VERSION,
        },
        members=members,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--import-report", type=Path)
    parser.add_argument(
        "--dataset-version-id",
        default="datasheet-frontier-vision-v1-20260831",
    )
    parser.add_argument("--version", default="v1-20260831")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    import_report = (
        load_import_report(args.import_report)
        if args.import_report is not None
        else None
    )
    digest = register(
        store=Store(args.database),
        manifest=manifest,
        dataset_version_id=args.dataset_version_id,
        version=args.version,
        import_report=import_report,
    )
    print(
        json.dumps(
            {
                "dataset_version_id": args.dataset_version_id,
                "manifest_sha256": digest,
                "members": manifest["counts"]["train_pairs"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
