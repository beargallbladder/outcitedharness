from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.storage.db import Store
from harness.training.ledger import (
    ArtifactPayload,
    LearningLedger,
    VerificationPayload,
)
from harness.training.models import LearningEvent, SourceKind


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "transfer_learning_bundle.py"
SPEC = importlib.util.spec_from_file_location("learning_bundle_transfer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


def _admitted_event(store: Store, artifact_root: Path) -> str:
    ledger = LearningLedger(store, artifact_root)
    event = LearningEvent(
        event_id="transfer-event-1",
        event_type="datasheet_frontier_vision_comparison",
        source_kind=SourceKind.OTHER,
        source_uri=f"datasheet://sha256/{'a' * 64}/page/1",
        source_revision="a" * 64,
        lineage_id="datasheet-a-page-1",
        authorization_scope="test",
        created_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
        metadata={
            "data_use": "training",
            "disposition": "verified",
        },
    )
    capture = ledger.capture(
        event,
        [
            ArtifactPayload(
                kind="image",
                content=b"\x89PNG\r\n\x1a\nbytes",
                media_type="image/png",
                redact=False,
            ),
            ArtifactPayload(
                kind="proof",
                content='{"elapsed_seconds":3.0870123,"passed":true}',
                media_type="application/json",
            ),
        ],
        [
            VerificationPayload(
                kind="consensus",
                status="pass",
                verifier="test-v1",
                output_kind="proof",
            )
        ],
    )
    ledger.admit_verified_event(
        event.event_id,
        capture.verifications[0].verification_id,
        policy_version="test-transfer-v1",
        reason="verified test event",
    )
    return event.event_id


def test_admitted_learning_bundle_round_trip(tmp_path: Path):
    source_store = Store(tmp_path / "source.db")
    source_artifacts = tmp_path / "source-artifacts"
    event_id = _admitted_event(source_store, source_artifacts)
    bundle = tmp_path / "bundle"
    exported = transfer.export_bundle(
        store=source_store,
        artifact_root=source_artifacts,
        destination=bundle,
        admission_policy="test-transfer-v1",
    )

    destination_store = Store(tmp_path / "destination.db")
    destination_artifacts = tmp_path / "destination-artifacts"
    imported = transfer.import_bundle(
        store=destination_store,
        artifact_root=destination_artifacts,
        bundle_root=bundle,
    )

    assert imported["events_imported"] == 1
    assert imported["bundle_core_sha256"] == exported["core_sha256"]
    assert len(imported["result_sha256"]) == 64
    verified = LearningLedger(
        destination_store,
        destination_artifacts,
    ).verify_event(event_id)
    assert verified.event_sha256 == exported["events"][0]["event_sha256"]
    with destination_store.connect() as connection:
        admission = connection.execute(
            "SELECT * FROM learning_admissions WHERE event_id = ?",
            (event_id,),
        ).fetchone()
    assert admission["policy_version"] == "test-transfer-v1"


def test_bundle_object_tampering_is_rejected_before_import(tmp_path: Path):
    source_store = Store(tmp_path / "source.db")
    source_artifacts = tmp_path / "source-artifacts"
    _admitted_event(source_store, source_artifacts)
    bundle = tmp_path / "bundle"
    manifest = transfer.export_bundle(
        store=source_store,
        artifact_root=source_artifacts,
        destination=bundle,
        admission_policy="test-transfer-v1",
    )
    first = next(iter(manifest["objects"].values()))
    object_path = bundle / first["path"]
    object_path.chmod(0o600)
    object_path.write_bytes(b"tampered")

    destination_store = Store(tmp_path / "destination.db")
    with pytest.raises(ValueError, match="digest mismatch"):
        transfer.import_bundle(
            store=destination_store,
            artifact_root=tmp_path / "destination-artifacts",
            bundle_root=bundle,
        )
    with destination_store.connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM learning_events"
        ).fetchone()[0]
    assert count == 0
