#!/usr/bin/env python3
"""Export or import a hash-bound set of admitted learning events."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.storage.db import Store
from harness.training.ledger import (
    ArtifactPayload,
    ArtifactVault,
    LearningLedger,
    VerificationPayload,
)
from harness.training.models import (
    LearningAdmission,
    LearningArtifact,
    LearningEvent,
    LearningVerification,
)


SCHEMA = "harness.learning-transfer-bundle.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.chmod(path, 0o444)


def _write_json(path: Path, value: Any) -> None:
    _write(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _event_model(row: Any) -> LearningEvent:
    return LearningEvent(
        event_id=row["event_id"],
        event_type=row["event_type"],
        source_kind=row["source_kind"],
        source_uri=row["source_uri"],
        source_revision=row["source_revision"],
        task_id=row["task_id"],
        lineage_id=row["lineage_id"],
        authorization_scope=row["authorization_scope"],
        state=row["state"],
        estimated_cost=row["estimated_cost"],
        metadata=json.loads(row["metadata_json"]),
        created_at=row["created_at"],
    )


def export_bundle(
    *,
    store: Store,
    artifact_root: Path,
    destination: Path,
    admission_policy: str,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"destination already exists: {destination}")
    ledger = LearningLedger(store, artifact_root)
    with store.connect() as connection:
        event_rows = connection.execute(
            """
            SELECT e.*
            FROM learning_events AS e
            JOIN learning_admissions AS a ON a.event_id = e.event_id
            WHERE a.decision = 'eligible' AND a.policy_version = ?
            ORDER BY e.event_id
            """,
            (admission_policy,),
        ).fetchall()
    if not event_rows:
        raise ValueError("admission policy selected no learning events")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        events = []
        objects: dict[str, dict[str, Any]] = {}
        for event_row in event_rows:
            capture = ledger.verify_event(event_row["event_id"])
            if len(capture.verifications) != 1:
                raise ValueError("transfer v1 requires exactly one verification per event")
            with store.connect() as connection:
                admission_row = connection.execute(
                    "SELECT * FROM learning_admissions WHERE event_id = ?",
                    (event_row["event_id"],),
                ).fetchone()
            if admission_row is None:
                raise ValueError("eligible transfer event has no admission")
            admission = LearningAdmission(
                admission_id=admission_row["admission_id"],
                event_id=admission_row["event_id"],
                verification_id=admission_row["verification_id"],
                decision=admission_row["decision"],
                policy_version=admission_row["policy_version"],
                reason=admission_row["reason"],
                source_revision=admission_row["source_revision"],
                admission_sha256=admission_row["admission_sha256"],
                created_at=admission_row["created_at"],
            )
            if admission.verification_id != capture.verifications[0].verification_id:
                raise ValueError("admission does not bind the transferred verification")
            for artifact in capture.artifacts:
                source = ledger.vault.path_for(artifact.sha256)
                data = source.read_bytes()
                if _sha256(data) != artifact.sha256 or len(data) != artifact.byte_size:
                    raise ValueError("verified artifact changed during export")
                target = temporary / "objects" / artifact.sha256[:2] / artifact.sha256
                if not target.exists():
                    _write(target, data)
                objects.setdefault(
                    artifact.sha256,
                    {
                        "path": target.relative_to(temporary).as_posix(),
                        "bytes": artifact.byte_size,
                    },
                )
            events.append(
                {
                    "event": _event_model(event_row).model_dump(
                        mode="json",
                        exclude_none=True,
                    ),
                    "event_sha256": capture.event_sha256,
                    "artifacts": [
                        artifact.model_dump(mode="json", exclude_none=True)
                        for artifact in capture.artifacts
                    ],
                    "verifications": [
                        verification.model_dump(mode="json", exclude_none=True)
                        for verification in capture.verifications
                    ],
                    "admission": admission.model_dump(
                        mode="json",
                        exclude_none=True,
                    ),
                }
            )
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "admission_policy": admission_policy,
            "events": events,
            "objects": objects,
        }
        manifest["core_sha256"] = _sha256(_canonical(manifest))
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_bundle(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if (
        path.is_symlink()
        or not path.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        raise ValueError("transfer bundle must be a regular directory")
    value = json.loads(manifest_path.read_text())
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported transfer bundle")
    expected = value.get("core_sha256")
    core = {key: item for key, item in value.items() if key != "core_sha256"}
    if expected != _sha256(_canonical(core)):
        raise ValueError("transfer bundle core digest mismatch")
    return value


def _object_data(
    root: Path,
    objects: dict[str, Any],
    digest: str,
    expected_size: int,
) -> bytes:
    record = objects.get(digest)
    if not isinstance(record, dict) or record.get("bytes") != expected_size:
        raise ValueError("transfer object manifest is inconsistent")
    relative = record.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith("/")
        or ".." in Path(relative).parts
    ):
        raise ValueError("transfer object path is unsafe")
    path = (root / relative).resolve()
    if (
        not path.is_relative_to(root.resolve())
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError("transfer object is not a regular in-bundle file")
    data = path.read_bytes()
    if len(data) != expected_size or _sha256(data) != digest:
        raise ValueError("transfer object digest mismatch")
    return data


def _payload(
    artifact: LearningArtifact,
    data: bytes,
) -> ArtifactPayload:
    textual = artifact.media_type.startswith("text/") or artifact.media_type in {
        "application/json",
        "application/jsonl",
    }
    content: str | bytes = data.decode("utf-8") if textual else data
    payload = ArtifactPayload(
        kind=artifact.kind,
        content=content,
        media_type=artifact.media_type,
        redact=artifact.redacted,
    )
    prepared = ArtifactVault._prepare(payload)
    if _sha256(prepared) != artifact.sha256 or len(prepared) != artifact.byte_size:
        raise ValueError("transfer artifact is not stable under destination policy")
    return payload


def import_bundle(
    *,
    store: Store,
    artifact_root: Path,
    bundle_root: Path,
) -> dict[str, Any]:
    manifest = load_bundle(bundle_root)
    raw_events = manifest.get("events")
    objects = manifest.get("objects")
    if (
        not isinstance(raw_events, list)
        or not raw_events
        or not isinstance(objects, dict)
    ):
        raise ValueError("transfer bundle content is malformed")

    prepared = []
    for raw in raw_events:
        if not isinstance(raw, dict):
            raise ValueError("transfer event is malformed")
        event = LearningEvent.model_validate(raw.get("event"))
        artifacts = [
            LearningArtifact.model_validate(value)
            for value in raw.get("artifacts", [])
        ]
        verifications = [
            LearningVerification.model_validate(value)
            for value in raw.get("verifications", [])
        ]
        admission = LearningAdmission.model_validate(raw.get("admission"))
        if (
            not artifacts
            or len(verifications) != 1
            or admission.event_id != event.event_id
            or admission.verification_id != verifications[0].verification_id
            or admission.policy_version != manifest["admission_policy"]
            or admission.decision != "eligible"
        ):
            raise ValueError("transfer event relationships are inconsistent")
        by_artifact_id = {artifact.artifact_id: artifact for artifact in artifacts}
        output = by_artifact_id.get(verifications[0].output_artifact_id or "")
        if output is None:
            raise ValueError("transfer verification proof artifact is missing")
        payloads = []
        for artifact in artifacts:
            if artifact.event_id != event.event_id:
                raise ValueError("transfer artifact belongs to another event")
            data = _object_data(
                bundle_root,
                objects,
                artifact.sha256,
                artifact.byte_size,
            )
            payloads.append(_payload(artifact, data))
        verification_payload = VerificationPayload(
            kind=verifications[0].kind,
            status=verifications[0].status,
            verifier=verifications[0].verifier,
            output_kind=output.kind,
            command=verifications[0].command,
            metadata=verifications[0].metadata,
        )
        prepared.append(
            (
                event,
                str(raw.get("event_sha256") or ""),
                artifacts,
                verifications[0],
                admission,
                payloads,
                verification_payload,
            )
        )

    ledger = LearningLedger(store, artifact_root)
    imported = []
    for (
        event,
        expected_event_sha256,
        source_artifacts,
        source_verification,
        source_admission,
        payloads,
        verification_payload,
    ) in prepared:
        capture = ledger.capture(event, payloads, [verification_payload])
        if (
            capture.event_sha256 != expected_event_sha256
            or capture.artifacts != tuple(source_artifacts)
            or capture.verifications != (source_verification,)
        ):
            raise ValueError("destination capture differs from transfer source")
        admission = ledger.admit_verified_event(
            event.event_id,
            source_verification.verification_id,
            policy_version=source_admission.policy_version,
            reason=source_admission.reason,
        )
        ledger.verify_event(event.event_id)
        imported.append(
            {
                "event_id": event.event_id,
                "event_sha256": capture.event_sha256,
                "source_admission_sha256": source_admission.admission_sha256,
                "destination_admission_sha256": admission.admission_sha256,
            }
        )
    result: dict[str, Any] = {
        "schema": "harness.learning-transfer-import.v1",
        "bundle_core_sha256": manifest["core_sha256"],
        "events_imported": len(imported),
        "events": imported,
    }
    result["result_sha256"] = _sha256(_canonical(result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--database", required=True, type=Path)
    export.add_argument("--artifact-root", required=True, type=Path)
    export.add_argument("--destination", required=True, type=Path)
    export.add_argument("--admission-policy", required=True)
    ingest = commands.add_parser("import")
    ingest.add_argument("--database", required=True, type=Path)
    ingest.add_argument("--artifact-root", required=True, type=Path)
    ingest.add_argument("--bundle-root", required=True, type=Path)
    ingest.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "export":
        result = export_bundle(
            store=Store(args.database),
            artifact_root=args.artifact_root,
            destination=args.destination,
            admission_policy=args.admission_policy,
        )
        output = {
            "bundle_core_sha256": result["core_sha256"],
            "events_exported": len(result["events"]),
        }
    else:
        output = import_bundle(
            store=Store(args.database),
            artifact_root=args.artifact_root,
            bundle_root=args.bundle_root,
        )
        if args.output is not None:
            if args.output.exists() or args.output.is_symlink():
                raise ValueError("import report already exists")
            _write_json(args.output, output)
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
