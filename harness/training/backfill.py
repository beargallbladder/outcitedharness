from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from harness.storage.db import Store
from harness.training.adapters import (
    load_greenfield_git_candidates,
    load_harness_pass_candidates,
)
from harness.training.ledger import (
    ArtifactPayload,
    LearningLedger,
    VerificationPayload,
)
from harness.training.models import (
    GitCandidate,
    LearningEvent,
    SourceKind,
    TextPair,
    is_excluded_learning_source,
)


@dataclass
class BackfillReport:
    source: str
    captured: int = 0
    duplicates: int = 0
    rejected: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    rejection_details: list[dict[str, str]] = field(default_factory=list)

    def reject(self, reason: str, *, record_id: str | None = None) -> None:
        self.rejected += 1
        self.reasons[reason] += 1
        if record_id is not None:
            self.rejection_details.append(
                {"record_id": record_id, "reason": reason}
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "captured": self.captured,
            "duplicates": self.duplicates,
            "rejected": self.rejected,
            "reasons": dict(sorted(self.reasons.items())),
            "rejection_details": self.rejection_details,
        }


def _event_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:32]}"


def backfill_designwins(
    source: Path,
    ledger: LearningLedger,
    *,
    source_snapshot: Path,
    admit_verified: bool = False,
) -> BackfillReport:
    """Capture audited DesignWins pairs, quarantined unless admission is explicit."""

    source = source.expanduser().resolve(strict=True)
    source_snapshot = source_snapshot.expanduser().resolve(strict=True)
    source_hashes: dict[str, str] = {}
    with source_snapshot.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("part"), str):
                raise ValueError(
                    f"invalid DesignWins source record at line {line_number}"
                )
            record_id = f"{record['part']}:text"
            if record_id in source_hashes:
                raise ValueError(f"duplicate DesignWins source record {record_id!r}")
            source_hashes[record_id] = hashlib.sha256(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
    report = BackfillReport(source=str(source))
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                pair = TextPair.model_validate(raw)
                if pair.provenance.source_kind is not SourceKind.DESIGNWINS:
                    raise ValueError("record is not DesignWins provenance")
                if pair.data_use.value != "training":
                    raise ValueError("record is not approved for training")
                source_digest = source_hashes.get(
                    pair.provenance.source_record_id
                )
                if source_digest is None:
                    raise ValueError("source snapshot lacks provenance record")
                if source_digest != pair.provenance.content_sha256:
                    raise ValueError("source snapshot content hash mismatch")
                event_id = _event_id(
                    "designwins" if admit_verified else "designwins-candidate",
                    pair.pair_id,
                    pair.provenance.content_sha256,
                )
                disposition = "verified" if admit_verified else "quarantine"
                before = _event_exists(ledger.store, event_id)
                event_metadata = {
                    "pair_id": pair.pair_id,
                    "data_use": (
                        pair.data_use.value if admit_verified else "quarantine"
                    ),
                    "disposition": disposition,
                    "source_record_sha256": source_digest,
                    **pair.metadata,
                }
                if not admit_verified:
                    event_metadata["quarantine_reason"] = (
                        "datasheet page/package provenance and independent replay "
                        "must be bound before training admission"
                    )
                if before:
                    with ledger.store.connect() as conn:
                        existing = conn.execute(
                            """
                            SELECT metadata_json FROM learning_events
                            WHERE event_id = ?
                            """,
                            (event_id,),
                        ).fetchone()
                    assert existing is not None
                    existing_metadata = json.loads(existing["metadata_json"])
                    for key in ("pair_id", "data_use", "disposition"):
                        if existing_metadata.get(key) != event_metadata[key]:
                            raise ValueError(
                                "existing DesignWins event metadata conflicts "
                                "with verified source"
                            )
                    event_metadata = existing_metadata
                event = LearningEvent(
                    event_id=event_id,
                    event_type="audited_designwins_pair",
                    source_kind=SourceKind.DESIGNWINS,
                    source_uri=pair.provenance.source_uri,
                    source_revision=pair.provenance.revision,
                    lineage_id=pair.provenance.lineage_id,
                    authorization_scope=pair.provenance.license,
                    created_at=pair.provenance.collected_at,
                    metadata=event_metadata,
                )
                capture = ledger.capture(
                    event,
                    [
                        ArtifactPayload(
                            kind="prompt",
                            content=pair.prompt,
                            media_type="text/plain",
                        ),
                        ArtifactPayload(
                            kind="canonical_response",
                            content=pair.response,
                            media_type="application/json",
                        ),
                        ArtifactPayload(
                            kind="provenance",
                            content=json.dumps(
                                pair.provenance.model_dump(mode="json"),
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            media_type="application/json",
                        ),
                    ],
                    [
                        VerificationPayload(
                            kind="audited_ground_truth",
                            status="pass",
                            verifier="harness.training.designwins-audit",
                            output_kind="canonical_response",
                            metadata={
                                "source_line": line_number,
                                "content_sha256": pair.provenance.content_sha256,
                            },
                        )
                    ],
                )
                if admit_verified:
                    ledger.admit_verified_event(
                        event.event_id,
                        capture.verifications[0].verification_id,
                        policy_version="designwins-audit-v1",
                        reason="audited canonical pair with passing ground-truth proof",
                    )
                if before:
                    report.duplicates += 1
                else:
                    report.captured += 1
            except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
                report.reject(f"{type(exc).__name__}: {exc}")
    return report


def backfill_git_repository(
    repository: Path,
    ledger: LearningLedger,
    *,
    approved_repositories: Sequence[Path | str],
    max_commits: int = 500,
    max_patch_bytes: int = 2_000_000,
) -> BackfillReport:
    """Capture owned single-parent repairs as quarantined replay candidates."""

    repository = repository.expanduser().resolve(strict=True)
    report = BackfillReport(source=str(repository))
    if is_excluded_learning_source(
        SourceKind.GIT,
        f"git+{repository.as_uri()}",
        {"path": repository.as_posix()},
    ):
        report.reject("source is excluded by CategoryRank/Tapes policy")
        return report
    approved = {
        Path(path).expanduser().resolve(strict=False)
        for path in approved_repositories
    }
    if repository not in approved:
        report.reject("repository is not in the configured owned-repository allowlist")
        return report
    if max_commits < 1:
        raise ValueError("max_commits must be positive")
    revisions = _git(
        repository,
        "rev-list",
        "--no-merges",
        f"--max-count={max_commits}",
        "HEAD",
    ).splitlines()
    source_uri = f"git+{repository.as_uri()}"
    for revision in revisions:
        try:
            parents = _git(repository, "show", "-s", "--format=%P", revision).split()
            if len(parents) != 1:
                report.reject("commit does not have exactly one parent")
                continue
            message = _git(repository, "show", "-s", "--format=%B", revision).strip()
            patch = _git(
                repository,
                "diff",
                "--no-ext-diff",
                "--unified=3",
                parents[0],
                revision,
                "--",
            )
            if not patch.strip():
                report.reject("commit has no textual patch")
                continue
            if len(patch.encode("utf-8")) > max_patch_bytes:
                report.reject("patch exceeds capture size limit")
                continue
            event = LearningEvent(
                event_id=_event_id("git", source_uri, revision),
                event_type="git_history_candidate",
                source_kind=SourceKind.GIT,
                source_uri=f"{source_uri}?revision={revision}",
                source_revision=revision,
                lineage_id=source_uri,
                authorization_scope="configured-owned-repository",
                created_at=_git_datetime(repository, revision),
                metadata={
                    "repository": repository.name,
                    "parent_revision": parents[0],
                    "disposition": "quarantine",
                    "quarantine_reason": (
                        "requires parent-state replay and mechanical verification"
                    ),
                },
            )
            before = _event_exists(ledger.store, event.event_id)
            ledger.capture(
                event,
                [
                    ArtifactPayload(
                        kind="problem_statement",
                        content=message or f"Repair committed as {revision}",
                    ),
                    ArtifactPayload(
                        kind="patch",
                        content=patch,
                        media_type="text/x-diff",
                    ),
                ],
                [
                    VerificationPayload(
                        kind="parent_state_replay",
                        status="unknown",
                        verifier="harness.training.git-backfill",
                        metadata={"parent_revision": parents[0]},
                    )
                ],
            )
            if before:
                report.duplicates += 1
            else:
                report.captured += 1
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            report.reject(f"{type(exc).__name__}: {exc}")
    return report


def backfill_harness_pass_history(
    database: Path,
    cases_root: Path,
    ledger: LearningLedger,
    *,
    answer_root: Path | None = None,
) -> BackfillReport:
    """Capture complete Harness PASS artifacts as quarantine-only candidates."""

    rejections: list[dict[str, Any]] = []
    pairs = load_harness_pass_candidates(
        database,
        cases_root,
        approved_model_keys=frozenset(),
        answer_root=answer_root,
        strict=False,
        rejections=rejections,
    )
    report = BackfillReport(source=f"harness-pass:{database}")
    for rejection in rejections:
        if rejection.get("duplicate"):
            report.duplicates += 1
        else:
            report.reject(
                str(rejection["reason"]),
                record_id=f"model-result:{rejection['record_id']}",
            )
    for pair in pairs:
        try:
            event = LearningEvent(
                event_id=_event_id(
                    "harness-pass",
                    pair.provenance.source_record_id,
                    pair.provenance.content_sha256,
                ),
                event_type="harness_pass_candidate",
                source_kind=SourceKind.HARNESS,
                source_uri=pair.provenance.source_uri,
                source_revision=pair.provenance.content_sha256,
                lineage_id=pair.provenance.lineage_id,
                authorization_scope=pair.provenance.license,
                created_at=pair.provenance.collected_at,
                metadata={
                    **pair.metadata,
                    "data_use": "quarantine",
                    "disposition": "quarantine",
                    "quarantine_reason": (
                        "legacy PASS output requires independent replay and "
                        "output-license approval"
                    ),
                },
            )
            before = _event_exists(ledger.store, event.event_id)
            ledger.capture(
                event,
                [
                    ArtifactPayload(kind="prompt", content=pair.prompt),
                    ArtifactPayload(kind="candidate_response", content=pair.response),
                    ArtifactPayload(
                        kind="provenance",
                        content=json.dumps(
                            pair.provenance.model_dump(mode="json"),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        media_type="application/json",
                    ),
                ],
                [
                    VerificationPayload(
                        kind="legacy_evaluator_claim",
                        status="unknown",
                        verifier="harness.training.harness-pass-backfill",
                        output_kind="candidate_response",
                        metadata={
                            "claimed_verdict": pair.metadata.get("verdict"),
                            "claimed_evaluator": pair.metadata.get("evaluator"),
                            "proof_scope": "imported_claim_not_replayed",
                        },
                    )
                ],
            )
            if before:
                report.duplicates += 1
            else:
                report.captured += 1
        except (OSError, ValueError, ValidationError) as exc:
            report.reject(
                f"{type(exc).__name__}: {exc}",
                record_id=pair.provenance.source_record_id,
            )
    return report


def backfill_greenfield_history(
    database: Path,
    ledger: LearningLedger,
    *,
    runs_root: Path,
) -> BackfillReport:
    """Capture complete greenfield commits as quarantine-only candidates."""

    rejections: list[dict[str, Any]] = []
    candidates = load_greenfield_git_candidates(
        database,
        runs_root=runs_root,
        strict=False,
        rejections=rejections,
    )
    report = BackfillReport(source=f"greenfield:{database}")
    for rejection in rejections:
        report.reject(
            str(rejection["reason"]),
            record_id=str(rejection["source_record_id"]),
        )
    for candidate in candidates:
        try:
            before = _capture_greenfield_candidate(ledger, candidate)
        except (OSError, ValueError, ValidationError) as exc:
            report.reject(
                f"{type(exc).__name__}: {exc}",
                record_id=candidate.provenance.source_record_id,
            )
            continue
        if before:
            report.duplicates += 1
        else:
            report.captured += 1
    return report


def _capture_greenfield_candidate(
    ledger: LearningLedger,
    candidate: GitCandidate,
) -> bool:
    revision = candidate.provenance.revision
    if revision is None:
        raise ValueError("greenfield candidate lacks a repository revision")
    event = LearningEvent(
        event_id=_event_id(
            "greenfield",
            candidate.provenance.source_record_id,
            revision,
        ),
        event_type="greenfield_commit_candidate",
        source_kind=SourceKind.GIT,
        source_uri=f"git+{candidate.provenance.source_uri}?revision={revision}",
        source_revision=revision,
        lineage_id=candidate.provenance.lineage_id,
        authorization_scope=candidate.provenance.license,
        created_at=candidate.provenance.collected_at,
        metadata={
            "candidate_id": candidate.candidate_id,
            "data_use": "quarantine",
            "disposition": "quarantine",
            "quarantine_reason": candidate.quarantine_reason,
        },
    )
    before = _event_exists(ledger.store, event.event_id)
    ledger.capture(
        event,
        [
            ArtifactPayload(kind="problem_statement", content=candidate.problem),
            ArtifactPayload(
                kind="patch",
                content=candidate.patch,
                media_type="text/x-diff",
            ),
            ArtifactPayload(
                kind="acceptance_contract",
                content=json.dumps(
                    [test.model_dump(mode="json") for test in candidate.tests],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                media_type="application/json",
            ),
        ],
        [
            VerificationPayload(
                kind="acceptance_replay",
                status="unknown",
                verifier="harness.training.greenfield-backfill",
                output_kind="acceptance_contract",
                metadata={"proof_scope": "commands_recorded_not_replayed"},
            )
        ],
    )
    return before


def backfill_cursor_transcript(
    path: Path,
    ledger: LearningLedger,
) -> BackfillReport:
    """Capture revision-bound Cursor envelopes; reject ordinary chat messages."""

    path = path.expanduser().resolve(strict=True)
    report = BackfillReport(source=str(path))
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record_id = f"line:{line_number}"
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Cursor record must be an object")
                repository_uri = _required_text(row, "repository_uri")
                revision = _required_revision(row)
                if row.get("ownership") != "self":
                    raise ValueError("Cursor record lacks explicit self ownership")
                prompt = _required_text(row, "prompt")
                response = _required_text(row, "response")
                proof = row.get("proof")
                if not isinstance(proof, dict):
                    raise ValueError("Cursor record lacks linked proof")
                proof_output = _required_text(proof, "output")
                _require_content_digest(proof_output, proof.get("output_sha256"))
                created_at = _required_timestamp(row)
                event = LearningEvent(
                    event_id=_event_id(
                        "cursor",
                        path.as_uri(),
                        str(line_number),
                        revision,
                        hashlib.sha256(
                            f"{prompt}\0{response}".encode("utf-8")
                        ).hexdigest(),
                    ),
                    event_type="cursor_candidate",
                    source_kind=SourceKind.OTHER,
                    source_uri=f"cursor+{path.as_uri()}?line={line_number}",
                    source_revision=revision,
                    lineage_id=f"cursor:{repository_uri}",
                    authorization_scope="explicit-self-owned-cursor-export",
                    created_at=created_at,
                    metadata={
                        "repository_uri": repository_uri,
                        "data_use": "quarantine",
                        "disposition": "quarantine",
                        "proof_kind": str(proof.get("kind") or "unspecified"),
                    },
                )
                before = _event_exists(ledger.store, event.event_id)
                ledger.capture(
                    event,
                    [
                        ArtifactPayload(kind="prompt", content=prompt),
                        ArtifactPayload(kind="candidate_response", content=response),
                        ArtifactPayload(kind="proof_output", content=proof_output),
                    ],
                    [
                        VerificationPayload(
                            kind="cursor_imported_proof",
                            status="unknown",
                            verifier="harness.training.cursor-backfill",
                            output_kind="proof_output",
                            metadata={
                                "declared_sha256": proof["output_sha256"],
                                "claimed_status": str(
                                    proof.get("status") or "unknown"
                                ),
                                "proof_scope": "digest_checked_not_replayed",
                            },
                        )
                    ],
                )
                if before:
                    report.duplicates += 1
                else:
                    report.captured += 1
            except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
                report.reject(
                    f"{type(exc).__name__}: {exc}",
                    record_id=record_id,
                )
    return report


def backfill_ci_history(
    path: Path,
    ledger: LearningLedger,
) -> BackfillReport:
    """Capture digest-bound CI failure-to-green envelopes as quarantine."""

    path = path.expanduser().resolve(strict=True)
    report = BackfillReport(source=str(path))
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record_id = f"line:{line_number}"
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("CI record must be an object")
                repository_uri = _required_text(row, "repository_uri")
                revision = _required_revision(row)
                if row.get("ownership") != "self":
                    raise ValueError("CI record lacks explicit self ownership")
                patch = _required_text(row, "patch")
                if not patch.lstrip().startswith("diff --git "):
                    raise ValueError("CI patch must be a unified git diff")
                failure = _required_ci_result(row, "failure", expected_success=False)
                success = _required_ci_result(row, "success", expected_success=True)
                if failure["command"] != success["command"]:
                    raise ValueError("CI failure and success must run the same command")
                created_at = _required_timestamp(row)
                event = LearningEvent(
                    event_id=_event_id(
                        "ci",
                        repository_uri,
                        revision,
                        str(line_number),
                        hashlib.sha256(patch.encode("utf-8")).hexdigest(),
                    ),
                    event_type="ci_failure_to_green_candidate",
                    source_kind=SourceKind.GIT,
                    source_uri=f"{repository_uri}?revision={revision}",
                    source_revision=revision,
                    lineage_id=repository_uri,
                    authorization_scope="explicit-self-owned-ci-export",
                    created_at=created_at,
                    metadata={
                        "command": failure["command"],
                        "data_use": "quarantine",
                        "disposition": "quarantine",
                    },
                )
                before = _event_exists(ledger.store, event.event_id)
                ledger.capture(
                    event,
                    [
                        ArtifactPayload(
                            kind="patch",
                            content=patch,
                            media_type="text/x-diff",
                        ),
                        ArtifactPayload(
                            kind="failure_output",
                            content=failure["output"],
                        ),
                        ArtifactPayload(
                            kind="success_output",
                            content=success["output"],
                        ),
                    ],
                    [
                        VerificationPayload(
                            kind="ci_failure_to_green_claim",
                            status="unknown",
                            verifier="harness.training.ci-backfill",
                            output_kind="success_output",
                            metadata={
                                "failure_output_sha256": failure["output_sha256"],
                                "success_output_sha256": success["output_sha256"],
                                "proof_scope": "digests_checked_not_replayed",
                            },
                        )
                    ],
                )
                if before:
                    report.duplicates += 1
                else:
                    report.captured += 1
            except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
                report.reject(
                    f"{type(exc).__name__}: {exc}",
                    record_id=record_id,
                )
    return report


def inventory_harness_learning_gaps(
    store: Store,
    *,
    include_pass_answers: bool = True,
) -> BackfillReport:
    """Identify each legacy row that cannot form a complete learning candidate."""

    report = BackfillReport(source=str(store.db_path))
    with store.connect() as conn:
        frontier_tasks = conn.execute(
            "SELECT task_id FROM tasks WHERE frontier_required = 1 ORDER BY task_id"
        ).fetchall()
        attempts = conn.execute(
            """
            SELECT id FROM attempts
            WHERE task_id IN (
                SELECT task_id FROM tasks WHERE frontier_required = 1
            )
            ORDER BY id
            """
        ).fetchall()
        gateway_frontier = conn.execute(
            """
            SELECT id FROM gateway_turns
            WHERE alias = 'harness-frontier'
            ORDER BY id
            """
        ).fetchall()
        verified_answers = (
            conn.execute(
                """
                SELECT id FROM model_results
                WHERE verdict = 'PASS' AND answer_path IS NOT NULL
                ORDER BY id
                """
            ).fetchall()
            if include_pass_answers
            else []
        )
    for row in frontier_tasks:
        report.reject(
            "frontier task lacks immutable prompt/response artifact pointers",
            record_id=f"task:{row['task_id']}",
        )
    for row in attempts:
        report.reject(
            "attempt retains status but not a complete answer artifact",
            record_id=f"attempt:{row['id']}",
        )
    for row in gateway_frontier:
        report.reject(
            "legacy gateway turn retains telemetry but not content",
            record_id=f"gateway-turn:{row['id']}",
        )
    for row in verified_answers:
        report.reject(
            "verified answer requires the Harness PASS artifact importer",
            record_id=f"model-result:{row['id']}",
        )
    return report


def inventory_cursor_transcript(path: Path) -> BackfillReport:
    """Inventory Cursor messages; no transcript is eligible without proof digests."""

    path = path.expanduser().resolve(strict=True)
    report = BackfillReport(source=str(path))
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                report.reject(
                    "invalid transcript JSON",
                    record_id=f"line:{line_number}",
                )
                continue
            role = row.get("role")
            if role in {"user", "assistant"}:
                report.reject(
                    "transcript message lacks linked repository revision and proof digest",
                    record_id=f"line:{line_number}",
                )
            else:
                report.reject(
                    "transcript row is not a supported learning envelope",
                    record_id=f"line:{line_number}",
                )
    return report


def _required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"record requires non-empty {key}")
    return value.strip()


def _required_revision(row: dict[str, Any]) -> str:
    revision = _required_text(row, "repository_revision")
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("repository_revision must be a full lowercase hash")
    return revision


def _required_timestamp(row: dict[str, Any]) -> datetime:
    raw = _required_text(row, "created_at")
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at must be an ISO-8601 timestamp") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("created_at must include a timezone")
    return value.astimezone(timezone.utc)


def _require_content_digest(content: str, claimed: Any) -> str:
    if not isinstance(claimed, str) or len(claimed) != 64 or any(
        character not in "0123456789abcdef" for character in claimed
    ):
        raise ValueError("output_sha256 must be a lowercase SHA-256")
    actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if actual != claimed:
        raise ValueError("proof output digest mismatch")
    return actual


def _required_ci_result(
    row: dict[str, Any],
    key: str,
    *,
    expected_success: bool,
) -> dict[str, Any]:
    value = row.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"CI record requires {key} result")
    command = _required_text(value, "command")
    output = _required_text(value, "output")
    exit_code = value.get("exit_code")
    if type(exit_code) is not int:
        raise ValueError(f"CI {key} exit_code must be an integer")
    if (exit_code == 0) is not expected_success:
        expected = "zero" if expected_success else "non-zero"
        raise ValueError(f"CI {key} exit_code must be {expected}")
    digest = _require_content_digest(output, value.get("output_sha256"))
    return {
        "command": command,
        "output": output,
        "exit_code": exit_code,
        "output_sha256": digest,
    }


def _event_exists(store: Store, event_id: str) -> bool:
    with store.connect() as conn:
        return (
            conn.execute(
                "SELECT 1 FROM learning_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            is not None
        )


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _git_datetime(repository: Path, revision: str):
    from datetime import datetime

    return datetime.fromisoformat(
        _git(repository, "show", "-s", "--format=%cI", revision).strip()
    )


__all__ = [
    "BackfillReport",
    "backfill_ci_history",
    "backfill_cursor_transcript",
    "backfill_designwins",
    "backfill_greenfield_history",
    "backfill_git_repository",
    "backfill_harness_pass_history",
    "inventory_cursor_transcript",
    "inventory_harness_learning_gaps",
]
