from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

from harness.storage.db import Store
from harness.training.ledger import (
    ArtifactPayload,
    LearningLedger,
    VerificationPayload,
)
from harness.training.models import LearningEvent, SourceKind


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "register_datasheet_frontier_dataset.py"
SPEC = importlib.util.spec_from_file_location("datasheet_registration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
registration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(registration)


def _manifest(core: dict) -> dict:
    return {
        **core,
        "core_sha256": hashlib.sha256(
            registration._canonical(core)
        ).hexdigest(),
    }


def test_registers_only_admitted_frontier_artifacts(tmp_path: Path):
    store = Store(tmp_path / "learning.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    event = LearningEvent(
        event_id="datasheet-frontier-test",
        event_type="datasheet_frontier_vision_comparison",
        source_kind=SourceKind.OTHER,
        source_uri=f"datasheet://sha256/{'a' * 64}/page/1",
        source_revision="a" * 64,
        lineage_id=f"datasheet:{'a' * 64}:LQFP2:page:1",
        authorization_scope="test",
        created_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
        metadata={
            "data_use": "training",
            "disposition": "verified",
            "training_eligible": True,
            "case_id": "part-a",
        },
    )
    capture = ledger.capture(
        event,
        [
            ArtifactPayload(
                kind="frontier_prediction",
                content='{"pins":[{"pin_no":"1","name":"PA0"}]}',
                media_type="application/json",
            )
        ],
        [
            VerificationPayload(
                kind="consensus",
                status="pass",
                verifier=registration.POLICY_VERSION,
                output_kind="frontier_prediction",
            )
        ],
    )
    admission = ledger.admit_verified_event(
        event.event_id,
        capture.verifications[0].verification_id,
        policy_version=registration.POLICY_VERSION,
        reason="test consensus",
    )
    source_admission_sha256 = "c" * 64
    manifest = _manifest(
        {
            "schema": registration.SCHEMA,
            "counts": {"train_pairs": 1},
            "frozen_fixture_sha256": ["b" * 64],
            "source_events": [
                {
                    "event_id": event.event_id,
                    "event_sha256": capture.event_sha256,
                    "admission_id": admission.admission_id,
                    "admission_sha256": source_admission_sha256,
                }
            ],
        }
    )
    import_report_core = {
        "schema": "harness.learning-transfer-import.v1",
        "bundle_core_sha256": "d" * 64,
        "events_imported": 1,
        "events": [
            {
                "event_id": event.event_id,
                "event_sha256": capture.event_sha256,
                "source_admission_sha256": source_admission_sha256,
                "destination_admission_sha256": admission.admission_sha256,
            }
        ],
    }
    import_report = {
        **import_report_core,
        "result_sha256": hashlib.sha256(
            registration._canonical(import_report_core)
        ).hexdigest(),
    }

    digest = registration.register(
        store=store,
        manifest=manifest,
        dataset_version_id="datasheet-frontier-test-v1",
        version="v1",
        import_report=import_report,
    )

    assert len(digest) == 64
    with store.connect() as connection:
        member = connection.execute(
            "SELECT * FROM dataset_members"
        ).fetchone()
    assert member["split"] == "train"
    assert member["source_document_sha256"] == "a" * 64
    assert member["component_family"] == "part-a"


def test_manifest_digest_is_fail_closed(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": registration.SCHEMA,
                "core_sha256": "0" * 64,
                "source_events": [],
            }
        )
    )

    try:
        registration.load_manifest(path)
    except ValueError as error:
        assert "digest mismatch" in str(error)
    else:
        raise AssertionError("tampered manifest was accepted")
