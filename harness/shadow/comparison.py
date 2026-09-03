from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import Field

from harness.shadow.models import Sha256, StrictModel, canonical_json
from harness.shadow.objects import ShadowObjectStore
from harness.shadow.replay import ReplayReport
from harness.shadow.spool import ShadowSpool
from harness.training.ledger import (
    ArtifactPayload,
    CaptureResult,
    LearningLedger,
    VerificationPayload,
)
from harness.training.models import LearningAdmission, LearningEvent, SourceKind
from harness.training.security import assert_no_secrets, redact_text


POLICY_VERSION = "cursor-shadow-mechanical-comparison-v1"


class CandidateDecision(StrictModel):
    kind: Literal["local", "frontier"]
    replay_id: str
    replay_evidence_sha256: Sha256
    patch_sha256: Sha256
    verdict: str
    mechanically_verified: bool


class ComparisonReport(StrictModel):
    version: Literal[1] = 1
    comparison_id: str
    task_id: str
    repository_id: str
    parent_state_sha256: Sha256
    teacher_model: str
    observed_teacher_models: tuple[str, ...]
    teacher_identity_verified: bool
    local_model: str
    local: CandidateDecision
    frontier: CandidateDecision
    decision: Literal[
        "frontier_correction",
        "local_win",
        "verified_equivalent",
        "rejected",
    ]
    chosen: Literal["local", "frontier"] | None
    rejected: Literal["local", "frontier"] | None
    eligible: bool
    reason: str
    created_at: datetime
    evidence_sha256: Sha256


class AdmissionResult(StrictModel):
    comparison: ComparisonReport
    capture: dict
    admission: LearningAdmission


def _verify_replay(report: ReplayReport) -> None:
    unsigned = report.model_dump(mode="json", exclude={"evidence_sha256"})
    actual = hashlib.sha256(canonical_json(unsigned)).hexdigest()
    if actual != report.evidence_sha256:
        raise ValueError("replay evidence digest mismatch")


def _reports(spool: ShadowSpool, task_id: str) -> dict[str, ReplayReport]:
    output = {}
    for value in spool.replays(task_id):
        report = ReplayReport.model_validate(value)
        _verify_replay(report)
        output[report.candidate_kind] = report
    missing = {"local", "frontier"} - set(output)
    if missing:
        raise ValueError(f"comparison requires both replay candidates: {sorted(missing)}")
    return output


def _teacher_identity(
    spool: ShadowSpool,
    task_id: str,
    expected: str,
) -> tuple[bool, tuple[str, ...]]:
    observed = set()
    for event in spool.events(task_id):
        if event.event_type not in {
            "beforeSubmitPrompt",
            "afterAgentResponse",
            "stop",
        }:
            continue
        value = event.payload.get("model_id") or event.payload.get("model")
        if value:
            observed.add(str(value))
    values = tuple(sorted(observed))
    return observed == {expected}, values


def _candidate(report: ReplayReport) -> CandidateDecision:
    return CandidateDecision(
        kind=report.candidate_kind,
        replay_id=report.replay_id,
        replay_evidence_sha256=report.evidence_sha256,
        patch_sha256=report.candidate_patch_sha256,
        verdict=report.verdict,
        mechanically_verified=report.verdict == "verified_correction",
    )


def compare_task(spool: ShadowSpool, task_id: str) -> ComparisonReport:
    task, _state = spool.get_task(task_id)
    attempt = spool.get_attempt(task_id)
    if attempt is None:
        raise ValueError("comparison requires a local shadow attempt")
    reports = _reports(spool, task_id)
    local_report = reports["local"]
    frontier_report = reports["frontier"]
    for report in reports.values():
        if (
            report.task_id != task_id
            or report.repository_id != task.policy.repository_id
            or report.parent_state_sha256 != task.snapshot.state_sha256
            or report.source_revision != task.snapshot.revision
        ):
            raise ValueError("replay report does not bind to the shadow task")
    teacher_verified, observed_teachers = _teacher_identity(
        spool,
        task_id,
        task.policy.teacher_model,
    )
    local_verified = (
        local_report.verdict == "verified_correction"
        and attempt.status == "completed"
    )
    frontier_verified = frontier_report.verdict == "verified_correction"
    rejected: Literal["local", "frontier"] | None = None
    if not teacher_verified:
        decision = "rejected"
        chosen = None
        eligible = False
        reason = (
            f"Cursor teacher identity mismatch: expected "
            f"{task.policy.teacher_model!r}, observed {list(observed_teachers)!r}"
        )
    elif local_verified and frontier_verified:
        if (
            local_report.candidate_patch_sha256
            == frontier_report.candidate_patch_sha256
        ):
            decision = "verified_equivalent"
            reason = "local and frontier produced the same mechanically proven patch"
        else:
            decision = "local_win"
            reason = "local and frontier patches both passed the same-parent replay"
        chosen: Literal["local", "frontier"] | None = "local"
        eligible = True
    elif local_verified:
        decision = "local_win"
        chosen = "local"
        rejected = (
            "frontier" if frontier_report.verdict == "rejected" else None
        )
        eligible = True
        reason = "only the completed local attempt passed fail-before/pass-after replay"
    elif frontier_verified:
        decision = "frontier_correction"
        chosen = "frontier"
        rejected = "local" if local_report.verdict == "rejected" else None
        eligible = True
        reason = "only the Cursor frontier correction passed mechanical replay"
    else:
        decision = "rejected"
        chosen = None
        eligible = False
        reason = "neither candidate produced a completed, mechanically proven correction"

    created_at = datetime.now(timezone.utc)
    identity = canonical_json(
        {
            "task_id": task_id,
            "local_replay": local_report.evidence_sha256,
            "frontier_replay": frontier_report.evidence_sha256,
            "decision": decision,
            "chosen": chosen,
        }
    )
    comparison_id = f"comparison-{hashlib.sha256(identity).hexdigest()[:40]}"
    unsigned = {
        "version": 1,
        "comparison_id": comparison_id,
        "task_id": task_id,
        "repository_id": task.policy.repository_id,
        "parent_state_sha256": task.snapshot.state_sha256,
        "teacher_model": task.policy.teacher_model,
        "observed_teacher_models": observed_teachers,
        "teacher_identity_verified": teacher_verified,
        "local_model": attempt.model,
        "local": _candidate(local_report).model_dump(mode="json"),
        "frontier": _candidate(frontier_report).model_dump(mode="json"),
        "decision": decision,
        "chosen": chosen,
        "rejected": rejected,
        "eligible": eligible,
        "reason": reason,
        "created_at": created_at,
    }
    provisional = ComparisonReport(**unsigned, evidence_sha256="0" * 64)
    report = provisional.model_copy(
        update={
            "evidence_sha256": hashlib.sha256(
                canonical_json(
                    provisional.model_dump(
                        mode="json",
                        exclude={"evidence_sha256"},
                    )
                )
            ).hexdigest()
        }
    )
    spool.record_comparison(
        comparison_id=comparison_id,
        task_id=task_id,
        report=report.model_dump(mode="json", exclude_none=True),
        report_sha256=hashlib.sha256(
            canonical_json(report.model_dump(mode="json", exclude_none=True))
        ).hexdigest(),
        created_at=created_at,
    )
    return report


def _frontier_response(spool: ShadowSpool, task_id: str) -> str:
    responses = []
    for event in spool.events(task_id):
        if event.event_type == "afterAgentResponse":
            value = event.payload.get("text")
            if isinstance(value, str) and value.strip():
                responses.append(value.strip())
    return responses[-1] if responses else ""


def admit_comparison(
    spool: ShadowSpool,
    task_id: str,
    ledger: LearningLedger,
) -> AdmissionResult:
    task, _state = spool.get_task(task_id)
    attempt = spool.get_attempt(task_id)
    if attempt is None:
        raise ValueError("admission requires a local shadow attempt")
    stored = spool.comparison(task_id)
    comparison = (
        ComparisonReport.model_validate(stored)
        if stored is not None
        else compare_task(spool, task_id)
    )
    unsigned = comparison.model_dump(mode="json", exclude={"evidence_sha256"})
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != comparison.evidence_sha256:
        raise ValueError("comparison evidence digest mismatch")
    if not comparison.eligible or comparison.chosen is None:
        raise ValueError("comparison is not eligible for training admission")
    reports = _reports(spool, task_id)
    chosen = reports[comparison.chosen]
    object_store = ShadowObjectStore(spool.root)
    chosen_patch = object_store.read_text(
        chosen.candidate_patch_sha256,
        chosen.candidate_patch_object_path,
    )
    if not chosen_patch.strip():
        raise ValueError("mechanically verified correction patch cannot be empty")
    if redact_text(chosen_patch) != chosen_patch:
        raise ValueError("candidate patch requires redaction and cannot be admitted")
    artifacts = [
        ArtifactPayload(
            kind="coding_prompt",
            content=task.prompt,
            media_type="text/plain",
        ),
        ArtifactPayload(
            kind="coding_chosen_patch",
            content=chosen_patch,
            media_type="text/x-diff",
        ),
        ArtifactPayload(
            kind="coding_comparison",
            content=json.dumps(
                comparison.model_dump(mode="json"),
                sort_keys=True,
            ),
            media_type="application/json",
        ),
        ArtifactPayload(
            kind="coding_chosen_replay",
            content=json.dumps(
                chosen.model_dump(mode="json"),
                sort_keys=True,
            ),
            media_type="application/json",
        ),
        ArtifactPayload(
            kind="coding_local_attempt",
            content=json.dumps(
                attempt.model_dump(mode="json", exclude_none=True),
                sort_keys=True,
            ),
            media_type="application/json",
        ),
        ArtifactPayload(
            kind="coding_parent_snapshot",
            content=json.dumps(
                task.snapshot.model_dump(mode="json", exclude_none=True),
                sort_keys=True,
            ),
            media_type="application/json",
        ),
    ]
    if comparison.rejected is not None:
        rejected = reports[comparison.rejected]
        rejected_patch = object_store.read_text(
            rejected.candidate_patch_sha256,
            rejected.candidate_patch_object_path,
        )
        if rejected_patch and redact_text(rejected_patch) == rejected_patch:
            artifacts.append(
                ArtifactPayload(
                    kind="coding_rejected_patch",
                    content=rejected_patch,
                    media_type="text/x-diff",
                )
            )
        artifacts.append(
            ArtifactPayload(
                kind="coding_rejected_replay",
                content=json.dumps(
                    rejected.model_dump(mode="json"),
                    sort_keys=True,
                ),
                media_type="application/json",
            )
        )
    frontier_response = _frontier_response(spool, task_id)
    if frontier_response:
        artifacts.append(
            ArtifactPayload(
                kind="coding_frontier_response",
                content=frontier_response,
                media_type="text/plain",
            )
        )
    policy_sha256 = hashlib.sha256(canonical_json(task.policy)).hexdigest()
    event_digest = hashlib.sha256(
        canonical_json(
            {
                "comparison": comparison.evidence_sha256,
                "chosen_patch": chosen.candidate_patch_sha256,
                "policy": policy_sha256,
            }
        )
    ).hexdigest()
    event_id = f"code-{event_digest[:40]}"
    event = LearningEvent(
        event_id=event_id,
        event_type=f"coding_{comparison.decision}",
        source_kind=SourceKind.GIT,
        source_uri=(
            f"git+https://github.com/{task.policy.repository_id}.git"
            f"#{task.snapshot.revision}"
        ),
        source_revision=task.snapshot.revision,
        lineage_id=f"git:{task.policy.repository_id}",
        authorization_scope=task.policy.authorization_scope,
        created_at=comparison.created_at,
        metadata={
            "content_class": "owned_source_code",
            "owner_attested": task.policy.owner == "self",
            "data_paths_excluded": True,
            "repository_id": task.policy.repository_id,
            "repository_policy_sha256": policy_sha256,
            "data_use": "training",
            "disposition": "verified",
            "shadow_task_id": task_id,
            "comparison_id": comparison.comparison_id,
            "comparison_decision": comparison.decision,
            "chosen_candidate": comparison.chosen,
            "teacher_model": comparison.teacher_model,
            "local_model": comparison.local_model,
            "parent_state_sha256": comparison.parent_state_sha256,
        },
    )
    capture: CaptureResult = ledger.capture(
        event,
        artifacts,
        [
            VerificationPayload(
                kind="same_parent_fail_before_pass_after",
                status="pass",
                verifier="harness.shadow.replay.v1",
                output_kind="coding_comparison",
                metadata={
                    "comparison_evidence_sha256": comparison.evidence_sha256,
                    "chosen_replay_evidence_sha256": chosen.evidence_sha256,
                },
            )
        ],
    )
    admission = ledger.admit_verified_event(
        capture.event_id,
        capture.verifications[0].verification_id,
        policy_version=POLICY_VERSION,
        reason="same-parent fail-before and pass-after mechanical proof",
    )
    assert_no_secrets(event_id, field="coding learning event ID")
    return AdmissionResult(
        comparison=comparison,
        capture={
            "event_id": capture.event_id,
            "event_sha256": capture.event_sha256,
            "artifact_ids": [artifact.artifact_id for artifact in capture.artifacts],
            "verification_ids": [
                proof.verification_id for proof in capture.verifications
            ],
        },
        admission=admission,
    )
