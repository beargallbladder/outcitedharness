from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.gateway.logging import log_turn
from harness.storage.db import Store
from harness.training.ledger import (
    ArtifactIntegrityError,
    ArtifactPayload,
    LearningEventConflictError,
    LearningLedger,
    VerificationPayload,
)
from harness.training.models import LearningEvent, SourceKind


NOW = datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)


def _event(**updates) -> LearningEvent:
    values = {
        "event_id": "gateway-task-1-turn-1",
        "event_type": "frontier_rescue",
        "source_kind": SourceKind.HARNESS,
        "source_uri": "harness://tasks/task-1/turns/1",
        "source_revision": "a" * 40,
        "task_id": None,
        "lineage_id": "repo-a:bug-17",
        "authorization_scope": "operator-opt-in",
        "created_at": NOW,
        "estimated_cost": 0.04,
        "metadata": {"provider": "frontier"},
    }
    values.update(updates)
    return LearningEvent(**values)


def _capture(ledger: LearningLedger, event: LearningEvent | None = None):
    return ledger.capture(
        event or _event(),
        [
            ArtifactPayload(
                kind="prompt",
                content="Contact sam@example.com; api_key=abcdefghijk",
                media_type="text/plain",
            ),
            ArtifactPayload(
                kind="tests",
                content="3 passed",
                media_type="text/plain",
            ),
        ],
        [
            VerificationPayload(
                kind="pytest",
                status="pass",
                verifier="pytest",
                command="pytest -q",
                output_kind="tests",
            )
        ],
    )


def test_learning_ledger_keeps_redacted_content_outside_sqlite(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")

    result = _capture(ledger)
    verified = ledger.verify_event(result.event_id)

    assert verified.event_sha256 == result.event_sha256
    prompt = next(row for row in result.artifacts if row.kind == "prompt")
    content = ledger.vault.path_for(prompt.sha256).read_text()
    assert "[REDACTED_EMAIL]" in content
    assert "[REDACTED_ASSIGNED_SECRET]" in content
    assert "sam@example.com" not in content
    assert oct(os.stat(ledger.vault.path_for(prompt.sha256)).st_mode & 0o777) == "0o600"
    assert "sam@example.com" not in (tmp_path / "harness.db").read_bytes().decode(
        "utf-8", errors="ignore"
    )
    assert oct(os.stat(tmp_path / "harness.db").st_mode & 0o777) == "0o600"
    assert oct(os.stat(tmp_path).st_mode & 0o777) == "0o700"


def test_json_redaction_preserves_valid_json_and_numeric_values(tmp_path: Path):
    ledger = LearningLedger(Store(tmp_path / "harness.db"), tmp_path / "artifacts")
    result = ledger.capture(
        _event(),
        [
            ArtifactPayload(
                kind="response",
                content=json.dumps(
                    {
                        "elapsed_seconds": 3.0870123,
                        "owner": "sam@example.com",
                    }
                ),
                media_type="application/json",
            )
        ],
    )

    artifact = result.artifacts[0]
    value = json.loads(ledger.vault.path_for(artifact.sha256).read_text())
    assert value["elapsed_seconds"] == 3.0870123
    assert value["owner"] == "[REDACTED_EMAIL]"


def test_learning_ledger_is_idempotent_and_conflict_checked(tmp_path: Path):
    ledger = LearningLedger(Store(tmp_path / "harness.db"), tmp_path / "artifacts")
    first = _capture(ledger)
    assert _capture(ledger) == first

    with pytest.raises(LearningEventConflictError):
        _capture(ledger, _event(lineage_id="different"))


def test_learning_tables_are_append_only(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    _capture(ledger)

    with store.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE learning_events SET state = 'eligible' WHERE event_id = ?",
                ("gateway-task-1-turn-1",),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "DELETE FROM learning_artifacts WHERE event_id = ?",
                ("gateway-task-1-turn-1",),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                """
                INSERT OR REPLACE INTO learning_events
                SELECT * FROM learning_events WHERE event_id = ?
                """,
                ("gateway-task-1-turn-1",),
            )


def test_verified_admission_is_separate_and_immutable(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    capture = _capture(
        ledger,
        _event(metadata={"data_use": "training", "disposition": "verified"}),
    )

    admission = ledger.admit_verified_event(
        capture.event_id,
        capture.verifications[0].verification_id,
        policy_version="mechanical-proof-v1",
        reason="passing replay proof",
    )

    assert admission.decision == "eligible"
    assert (
        ledger.admit_verified_event(
            capture.event_id,
            capture.verifications[0].verification_id,
            policy_version="mechanical-proof-v1",
            reason="passing replay proof",
        )
        == admission
    )
    with store.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                """
                UPDATE learning_admissions SET reason = 'changed'
                WHERE admission_id = ?
                """,
                (admission.admission_id,),
            )


def test_verified_admission_rechecks_vault_integrity(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")
    capture = _capture(
        ledger,
        _event(metadata={"data_use": "training", "disposition": "verified"}),
    )
    artifact = capture.artifacts[0]
    ledger.vault.path_for(artifact.sha256).write_text("tampered")

    with pytest.raises(ArtifactIntegrityError):
        ledger.admit_verified_event(
            capture.event_id,
            capture.verifications[0].verification_id,
            policy_version="mechanical-proof-v1",
            reason="passing replay proof",
        )

    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM learning_admissions").fetchone()[0] == 0


def test_learning_ledger_detects_artifact_tampering(tmp_path: Path):
    ledger = LearningLedger(Store(tmp_path / "harness.db"), tmp_path / "artifacts")
    result = _capture(ledger)
    artifact = result.artifacts[0]
    ledger.vault.path_for(artifact.sha256).write_text("tampered")

    with pytest.raises(ArtifactIntegrityError):
        ledger.verify_event(result.event_id)


@pytest.mark.parametrize(
    ("source_kind", "source_uri"),
    [
        (SourceKind.CATEGORYRANK, "categoryrank://records/1"),
        (SourceKind.OTHER, "dataset://category_rank/records/1"),
        (SourceKind.OTHER, "dataset://category%20rank/records/1"),
        (SourceKind.OTHER, "dataset://tapes/v1/1"),
        (SourceKind.OTHER, "file:///owned/tapes/export.jsonl"),
    ],
)
def test_learning_event_rejects_excluded_sources(source_kind, source_uri):
    with pytest.raises(ValidationError, match="capture is disabled"):
        _event(source_kind=source_kind, source_uri=source_uri)


@pytest.mark.parametrize(
    "content",
    [
        "export from Category Rank",
        "process category_rank records",
        "owned /tapes/archive.jsonl",
    ],
)
def test_learning_ledger_rejects_excluded_artifact_bodies(
    tmp_path: Path,
    content: str,
):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")

    with pytest.raises(ValueError, match="artifact capture is disabled"):
        ledger.capture(
            _event(),
            [ArtifactPayload(kind="payload", content=content)],
        )

    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0] == 0


def test_owned_code_scope_allows_code_but_not_dataset_artifacts(tmp_path: Path):
    ledger = LearningLedger(Store(tmp_path / "harness.db"), tmp_path / "artifacts")
    metadata = {
        "content_class": "owned_source_code",
        "owner_attested": True,
        "data_paths_excluded": True,
        "repository_id": "owner/outcited",
        "repository_policy_sha256": "b" * 64,
        "data_use": "training",
        "disposition": "verified",
    }
    event = _event(
        event_id="owned-code-correction-1",
        event_type="coding_frontier_correction",
        source_kind=SourceKind.GIT,
        source_uri="git+https://github.com/owner/outcited",
        authorization_scope="owned_repository_cursor_shadow",
        metadata=metadata,
    )
    capture = ledger.capture(
        event,
        [
            ArtifactPayload(
                kind="coding_patch",
                content=(
                    "diff --git a/app.ts b/app.ts\n"
                    "--- a/app.ts\n"
                    "+++ b/app.ts\n"
                    "@@ -1 +1 @@\n"
                    "-export const label = 'old'\n"
                    "+export const label = 'CategoryRank code only'\n"
                ),
                media_type="text/x-diff",
            ),
            ArtifactPayload(
                kind="coding_replay",
                content=json.dumps({"verdict": "verified_correction"}),
                media_type="application/json",
            ),
        ],
        [
            VerificationPayload(
                kind="mechanical_replay",
                status="pass",
                verifier="harness.shadow.replay.v1",
                output_kind="coding_replay",
            )
        ],
    )
    admission = ledger.admit_verified_event(
        capture.event_id,
        capture.verifications[0].verification_id,
        policy_version="owned-code-mechanical-replay-v1",
        reason="same-parent fail-before and pass-after",
    )
    assert admission.decision == "eligible"

    with pytest.raises(ValueError, match="only permits coding text artifacts"):
        ledger.capture(
            event.model_copy(update={"event_id": "owned-code-correction-2"}),
            [
                ArtifactPayload(
                    kind="dataset_rows",
                    content="CategoryRank records",
                    media_type="application/jsonl",
                )
            ],
        )


def test_owned_code_exception_requires_exact_attestation(tmp_path: Path):
    ledger = LearningLedger(Store(tmp_path / "harness.db"), tmp_path / "artifacts")
    event = _event(
        event_id="unattested-code",
        source_kind=SourceKind.GIT,
        source_uri="git+https://github.com/owner/outcited",
        authorization_scope="owned_repository_cursor_shadow",
        metadata={
            "content_class": "owned_source_code",
            "owner_attested": False,
            "data_paths_excluded": True,
            "repository_id": "owner/outcited",
            "repository_policy_sha256": "b" * 64,
        },
    )
    with pytest.raises(ValueError, match="artifact capture is disabled"):
        ledger.capture(
            event,
            [
                ArtifactPayload(
                    kind="coding_patch",
                    content="CategoryRank code",
                    media_type="text/plain",
                )
            ],
        )


def test_artifact_preflight_leaves_no_partial_vault_objects(tmp_path: Path):
    ledger = LearningLedger(Store(tmp_path / "harness.db"), tmp_path / "artifacts")

    with pytest.raises(ValueError, match="artifact capture is disabled"):
        ledger.capture(
            _event(),
            [
                ArtifactPayload(kind="first", content="safe content"),
                ArtifactPayload(kind="second", content="Category Rank export"),
            ],
        )

    assert list((tmp_path / "artifacts" / "sha256").rglob("*")) == []


def test_binary_text_artifacts_are_secret_scanned(tmp_path: Path):
    ledger = LearningLedger(Store(tmp_path / "harness.db"), tmp_path / "artifacts")

    with pytest.raises(ValueError, match="secret material"):
        ledger.capture(
            _event(),
            [
                ArtifactPayload(
                    kind="payload",
                    content=b"api_key=abcdefghijk",
                    media_type="application/json",
                    redact=False,
                )
            ],
        )


def test_gateway_learning_capture_is_explicit_and_pointer_only(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")

    log_turn(
        store,
        alias="harness-frontier",
        model_key="frontier",
        upstream_model="frontier-model",
        stream=False,
        status=200,
        latency_ms=50,
        input_tokens=10,
        output_tokens=5,
        cost=0.01,
        error=None,
        body={"messages": [{"role": "user", "content": "fix it"}]},
        response={"choices": [{"message": {"content": "fixed"}}]},
        learning_ledger=ledger,
    )

    with store.connect() as conn:
        event = conn.execute("SELECT * FROM learning_events").fetchone()
        assert event is not None
        assert event["authorization_scope"] == "settings.learning_capture_enabled"
        assert '"data_use": "quarantine"' in event["metadata_json"]
        assert '"disposition": "quarantine"' in event["metadata_json"]
        assert conn.execute("SELECT COUNT(*) FROM learning_artifacts").fetchone()[0] == 2
        verification = conn.execute(
            "SELECT * FROM learning_verifications"
        ).fetchone()
        assert verification["status"] == "unknown"
        assert '"proof_scope": "transport_only"' in verification["metadata_json"]
        database_text = (tmp_path / "harness.db").read_bytes().decode(
            "utf-8", errors="ignore"
        )
        assert '"content": "fixed"' not in database_text
    ledger.verify_event(event["event_id"])


def test_gateway_learning_capture_skips_excluded_payloads(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")

    log_turn(
        store,
        alias="harness-orch",
        model_key="orch",
        upstream_model="dispatch",
        stream=False,
        status=200,
        latency_ms=10,
        input_tokens=2,
        output_tokens=2,
        cost=None,
        error=None,
        body={"messages": [{"role": "user", "content": "process Tapes data"}]},
        response={"choices": [{"message": {"content": "no"}}]},
        learning_ledger=ledger,
    )

    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0] == 0
