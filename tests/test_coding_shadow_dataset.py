from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from build_coding_shadow_dataset import (  # noqa: E402
    POLICY_VERSION,
    build,
    load_manifest,
    load_sequence_audit,
    register,
)
from verify_coding_training_preflight import verify as verify_preflight  # noqa: E402
from harness.storage.db import Store  # noqa: E402
from harness.training.ledger import (  # noqa: E402
    ArtifactPayload,
    LearningLedger,
    VerificationPayload,
)
from harness.training.models import LearningEvent, SourceKind  # noqa: E402


def _patch(path: str, before: str, after: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        f"-{before}\n"
        f"+{after}\n"
    )


def _capture(
    ledger: LearningLedger,
    *,
    index: int,
    path: str,
    rejected: bool,
) -> None:
    event_id = f"code-event-{index}"
    revision = f"{index + 1:040x}"
    comparison_digest = hashlib.sha256(
        f"comparison-{index}".encode()
    ).hexdigest()
    state_digest = hashlib.sha256(f"state-{index}".encode()).hexdigest()
    chosen = _patch(path, f"value = {index}", f"value = {index + 1}")
    artifacts = [
        ArtifactPayload(
            kind="coding_prompt",
            content=f"Fix value behavior {index}.",
            media_type="text/plain",
        ),
        ArtifactPayload(
            kind="coding_chosen_patch",
            content=chosen,
            media_type="text/x-diff",
        ),
        ArtifactPayload(
            kind="coding_comparison",
            content=json.dumps(
                {
                    "eligible": True,
                    "chosen": "frontier",
                    "decision": "frontier_correction",
                    "reason": "frontier passed",
                    "teacher_identity_verified": True,
                    "evidence_sha256": comparison_digest,
                    "local": {"verdict": "rejected"},
                    "frontier": {"verdict": "verified_correction"},
                }
            ),
            media_type="application/json",
        ),
        ArtifactPayload(
            kind="coding_chosen_replay",
            content=json.dumps({"verdict": "verified_correction"}),
            media_type="application/json",
        ),
        ArtifactPayload(
            kind="coding_local_attempt",
            content=json.dumps(
                {
                    "status": "completed",
                    "answer": "Tried a local fix.",
                    "patch": (
                        _patch(path, f"value = {index}", "value = 99")
                        if rejected
                        else ""
                    ),
                }
            ),
            media_type="application/json",
        ),
        ArtifactPayload(
            kind="coding_parent_snapshot",
            content=json.dumps({"state_sha256": state_digest}),
            media_type="application/json",
        ),
    ]
    if rejected:
        artifacts.append(
            ArtifactPayload(
                kind="coding_rejected_patch",
                content=_patch(path, f"value = {index}", "value = 99"),
                media_type="text/x-diff",
            )
        )
    event = LearningEvent(
        event_id=event_id,
        event_type="coding_frontier_correction",
        source_kind=SourceKind.GIT,
        source_uri=f"git+https://github.com/owner/repository.git#{revision}",
        source_revision=revision,
        lineage_id="git:owner/repository",
        authorization_scope="owned_repository_cursor_shadow",
        created_at=datetime(2026, 8, 1, tzinfo=timezone.utc)
        + timedelta(days=index),
        metadata={
            "content_class": "owned_source_code",
            "owner_attested": True,
            "data_paths_excluded": True,
            "repository_id": "owner/repository",
            "repository_policy_sha256": "a" * 64,
            "data_use": "training",
            "disposition": "verified",
            "parent_state_sha256": state_digest,
        },
    )
    result = ledger.capture(
        event,
        artifacts,
        [
            VerificationPayload(
                kind="same_parent_fail_before_pass_after",
                status="pass",
                verifier="harness.shadow.replay.v1",
                output_kind="coding_comparison",
                metadata={
                    "comparison_evidence_sha256": comparison_digest,
                    "chosen_replay_evidence_sha256": "f" * 64,
                },
            )
        ],
    )
    ledger.admit_verified_event(
        event_id,
        result.verifications[0].verification_id,
        policy_version=POLICY_VERSION,
        reason="same-parent fail-before and pass-after mechanical proof",
    )


def test_builds_lineage_safe_sft_and_preference_datasets(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    store = Store(training_root / "ledger/learning.db")
    artifact_root = training_root / "ledger/artifacts"
    ledger = LearningLedger(store, artifact_root)
    _capture(ledger, index=1, path="app/value.py", rejected=True)
    _capture(ledger, index=2, path="app/value.py", rejected=False)
    _capture(ledger, index=3, path="lib/other.py", rejected=False)

    destination = training_root / "datasets/cursor-shadow-code-v1"
    manifest = build(
        store=store,
        artifact_root=artifact_root,
        destination=destination,
    )
    loaded = load_manifest(destination / "manifest.json")
    assert loaded == manifest
    membership = {row["event_id"]: row for row in manifest["membership"]}
    assert (
        membership["code-event-1"]["lineage_id"]
        == membership["code-event-2"]["lineage_id"]
    )
    assert (
        membership["code-event-1"]["split"]
        == membership["code-event-2"]["split"]
    )
    assert sum(
        value["sft"] for value in manifest["counts"].values()
    ) == 3
    assert sum(
        value["preference"] for value in manifest["counts"].values()
    ) == 1

    sequence_path = destination / "sequence-audit.json"
    sequence_path.write_text(
        json.dumps(
            {
                "schema": "harness.coding-sequence-length-audit.v1",
                "model_config_sha256": "d" * 64,
                "dataset_sha256": manifest["artifacts"][
                    "llamafactory/coding_sft_train.json"
                ]["sha256"],
                "records": manifest["counts"]["train"]["sft"],
                "cutoff_len": 8192,
                "truncated_records": 0,
            }
        )
    )
    sequence_audit, sequence_audit_sha256 = load_sequence_audit(
        sequence_path,
        manifest,
    )
    digest = register(
        store=store,
        manifest=manifest,
        dataset_version_id="cursor-shadow-code-test-v1",
        version="test-v1",
        sequence_audit=sequence_audit,
        sequence_audit_sha256=sequence_audit_sha256,
    )
    assert len(digest) == 64
    with store.connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM dataset_members "
                "WHERE dataset_version_id = 'cursor-shadow-code-test-v1'"
            ).fetchone()[0]
            == 3
        )
    config = training_root / "configs/smoke.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "\n".join(
            [
                "do_train: true",
                "dataset: coding_sft_train",
                "dataset_dir: /training/datasets/cursor-shadow-code-v1/llamafactory",
                "cutoff_len: 8192",
            ]
        )
        + "\n"
    )
    preflight = verify_preflight(
        root=training_root,
        config=config,
        audit=sequence_path,
        database=store.db_path,
        dataset_version_id="cursor-shadow-code-test-v1",
    )
    assert preflight["passed"]
    assert preflight["records"] == 3


def test_manifest_rejects_dataset_artifact_tampering(tmp_path: Path) -> None:
    store = Store(tmp_path / "learning.db")
    artifact_root = tmp_path / "artifacts"
    ledger = LearningLedger(store, artifact_root)
    _capture(ledger, index=1, path="app/value.py", rejected=True)
    destination = tmp_path / "dataset"
    build(
        store=store,
        artifact_root=artifact_root,
        destination=destination,
    )
    candidate = destination / "canonical/sft/train.jsonl"
    candidate.chmod(0o600)
    candidate.write_text("tampered\n")

    with pytest.raises(ValueError, match="artifact changed"):
        load_manifest(destination / "manifest.json")
