from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import sqlite3
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

from harness.storage.db import Store
from harness.training.models import SourceKind, is_excluded_learning_source
from harness.training.registry import canonical_json
from harness.training.security import assert_no_secrets, assert_value_no_secrets
from harness.training.split import Split, assert_no_lineage_leakage


class QueueError(RuntimeError):
    pass


class InvalidTransitionError(QueueError):
    pass


class JobState(str, Enum):
    ELIGIBLE = "eligible"
    ASSIGNED = "assigned"
    TRAINED = "trained"
    EVALUATED = "evaluated"
    SHADOW = "shadow"
    CANARY = "canary"
    PROMOTED = "promoted"
    REJECTED = "rejected"


TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.ELIGIBLE: frozenset({JobState.ASSIGNED, JobState.REJECTED}),
    JobState.ASSIGNED: frozenset(
        {JobState.ELIGIBLE, JobState.TRAINED, JobState.REJECTED}
    ),
    JobState.TRAINED: frozenset({JobState.EVALUATED, JobState.REJECTED}),
    JobState.EVALUATED: frozenset({JobState.SHADOW, JobState.REJECTED}),
    JobState.SHADOW: frozenset({JobState.CANARY, JobState.REJECTED}),
    JobState.CANARY: frozenset({JobState.PROMOTED, JobState.REJECTED}),
    JobState.PROMOTED: frozenset(),
    JobState.REJECTED: frozenset(),
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _verify_checkpoint_claim(
    checkpoint_uri: str,
    checkpoint_sha256: str,
    *,
    roots: tuple[Path, ...],
) -> Path:
    if len(checkpoint_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in checkpoint_sha256
    ):
        raise ValueError("checkpoint_sha256 must be lowercase SHA-256")
    assert_no_secrets(checkpoint_uri, field="checkpoint_uri")
    parsed = urlparse(checkpoint_uri)
    if (
        parsed.scheme != "file"
        or parsed.netloc
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("checkpoint_uri must be a local credential-free file URI")
    raw_path = Path(unquote(parsed.path))
    if not raw_path.is_absolute():
        raise ValueError("checkpoint must be an absolute non-symlink path")
    path = raw_path.resolve(strict=True)
    allowed = False
    for root in roots:
        resolved_root = root.expanduser().resolve(strict=True)
        try:
            path.relative_to(resolved_root)
            allowed = True
            break
        except ValueError:
            continue
    if not allowed:
        raise ValueError("checkpoint is outside the authoritative training root")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(raw_path, flags)
    except OSError as exc:
        raise ValueError("checkpoint must be an accessible non-symlink file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_mode & 0o002:
            raise ValueError("checkpoint must be a regular non-world-writable file")
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(raw_path, follow_symlinks=False)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        if identity_before != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or identity_before != (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
        ):
            raise ValueError("checkpoint changed while it was being verified")
        if digest.hexdigest() != checkpoint_sha256:
            raise ValueError("checkpoint SHA-256 does not match on-disk bytes")
    finally:
        os.close(descriptor)
    return path


@dataclass(frozen=True)
class PrioritySignals:
    observed_frequency: float
    frontier_cost: float
    local_failure_rate: float
    verification_strength: float
    diversity: float
    expected_gpu_hours: float

    def __post_init__(self) -> None:
        values = (
            self.observed_frequency,
            self.frontier_cost,
            self.local_failure_rate,
            self.verification_strength,
            self.diversity,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("priority signals must be finite and non-negative")
        if not math.isfinite(self.expected_gpu_hours) or self.expected_gpu_hours <= 0:
            raise ValueError("expected_gpu_hours must be finite and positive")
        if self.score <= 0:
            raise ValueError("priority score must be positive")

    @property
    def score(self) -> float:
        return (
            self.observed_frequency
            * self.frontier_cost
            * self.local_failure_rate
            * self.verification_strength
            * self.diversity
            / self.expected_gpu_hours
        )


@dataclass(frozen=True)
class DatasetMember:
    event_id: str
    artifact_id: str
    split: Split
    lineage_id: str
    source_document_sha256: str | None = None
    repository_id: str | None = None
    component_family: str | None = None
    temporal_bucket: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id or not self.artifact_id:
            raise ValueError("dataset member IDs cannot be empty")
        if not self.lineage_id or self.lineage_id != self.lineage_id.strip():
            raise ValueError("lineage_id must be non-empty and canonical")
        if self.source_document_sha256 is not None and (
            len(self.source_document_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_document_sha256
            )
        ):
            raise ValueError("source document digest must be lowercase SHA-256")
        for name, value in (
            ("repository_id", self.repository_id),
            ("component_family", self.component_family),
            ("temporal_bucket", self.temporal_bucket),
        ):
            if value is not None:
                if not value or value != value.strip():
                    raise ValueError(f"{name} must be non-empty and canonical")
                assert_no_secrets(value, field=f"dataset member {name}")


@dataclass(frozen=True)
class ClaimedJob:
    job_id: str
    job_kind: str
    dataset_version_id: str | None
    priority: float
    attempt: int
    assigned_node: str
    lease_expires_at: str
    lease_token: str
    config: dict[str, Any]


@dataclass(frozen=True)
class ExpiredHandler:
    job_id: str
    attempt: int
    assigned_node: str
    handler_pid: int
    handler_pgid: int
    max_attempts: int


class DatasetVersionRegistry:
    def __init__(self, store: Store):
        self.store = store

    def create(
        self,
        *,
        dataset_version_id: str,
        name: str,
        version: str,
        source_revision: str,
        split_policy: Mapping[str, Any],
        members: Sequence[DatasetMember],
    ) -> str:
        if not members:
            raise ValueError("dataset version cannot be empty")
        for field, value in (
            ("dataset_version_id", dataset_version_id),
            ("dataset name", name),
            ("dataset version", version),
        ):
            assert_no_secrets(value, field=field)
        assert_value_no_secrets(dict(split_policy), field="dataset split policy")
        if is_excluded_learning_source(
            SourceKind.OTHER,
            f"dataset://{name}/{version}",
            split_policy,
        ):
            raise ValueError("CategoryRank and Tapes dataset versions are disabled")
        if len(source_revision) not in {40, 64} or any(
            char not in "0123456789abcdef" for char in source_revision
        ):
            raise ValueError("source_revision must be a full lowercase commit hash")
        assert_no_lineage_leakage(
            {
                split: [member for member in members if member.split is split]
                for split in Split
            },
            lineage_key="lineage_id",
        )
        declared_keys = split_policy.get("leakage_keys", ["lineage_id"])
        if not isinstance(declared_keys, (list, tuple)) or not declared_keys:
            raise ValueError("split policy leakage_keys must be a non-empty list")
        allowed_keys = {
            "lineage_id",
            "source_document_sha256",
            "repository_id",
            "component_family",
            "temporal_bucket",
        }
        leakage_keys = set(declared_keys)
        unknown_keys = leakage_keys - allowed_keys
        if unknown_keys:
            raise ValueError(
                f"unsupported split leakage keys: {sorted(unknown_keys)}"
            )
        leakage_keys.add("lineage_id")
        if any(member.source_document_sha256 is not None for member in members):
            leakage_keys.add("source_document_sha256")
        for key in sorted(leakage_keys):
            owners: dict[str, Split] = {}
            for member in members:
                value = getattr(member, key)
                if value is None:
                    raise ValueError(
                        f"dataset member is missing declared leakage key {key}"
                    )
                previous = owners.setdefault(value, member.split)
                if previous is not member.split:
                    label = (
                        "source document"
                        if key == "source_document_sha256"
                        else key
                    )
                    raise ValueError(
                        f"{label} {value} leaks across "
                        f"{previous.value} and {member.split.value}"
                    )
        canonical_members = sorted(
            (
                {
                    "event_id": member.event_id,
                    "artifact_id": member.artifact_id,
                    "split": member.split.value,
                    "lineage_id": member.lineage_id,
                    "source_document_sha256": member.source_document_sha256,
                    **(
                        {"repository_id": member.repository_id}
                        if member.repository_id is not None
                        else {}
                    ),
                    **(
                        {"component_family": member.component_family}
                        if member.component_family is not None
                        else {}
                    ),
                    **(
                        {"temporal_bucket": member.temporal_bucket}
                        if member.temporal_bucket is not None
                        else {}
                    ),
                }
                for member in members
            ),
            key=lambda row: (
                row["split"],
                row["lineage_id"],
                row["event_id"],
                row["artifact_id"],
            ),
        )
        manifest = {
            "schema": "harness.dataset-version.v1",
            "dataset_version_id": dataset_version_id,
            "name": name,
            "version": version,
            "source_revision": source_revision,
            "split_policy": dict(split_policy),
            "members": canonical_members,
        }
        digest = hashlib.sha256(canonical_json(manifest)).hexdigest()
        created_at = _timestamp(_utcnow())
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                """
                SELECT manifest_sha256 FROM dataset_versions
                WHERE dataset_version_id = ?
                """,
                (dataset_version_id,),
            ).fetchone()
            if current:
                if current["manifest_sha256"] != digest:
                    raise QueueError(
                        f"dataset version {dataset_version_id!r} is immutable"
                    )
                return digest
            self._validate_members(conn, members)
            try:
                conn.execute(
                    """
                    INSERT INTO dataset_versions (
                        dataset_version_id, name, version, manifest_sha256,
                        split_policy_json, source_revision, state, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'eligible', ?)
                    """,
                    (
                        dataset_version_id,
                        name,
                        version,
                        digest,
                        json.dumps(dict(split_policy), sort_keys=True),
                        source_revision,
                        created_at,
                    ),
                )
                conn.executemany(
                    """
                    INSERT INTO dataset_members (
                        dataset_version_id, event_id, artifact_id, split,
                        lineage_id, source_document_sha256, repository_id,
                        component_family, temporal_bucket
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            dataset_version_id,
                            member.event_id,
                            member.artifact_id,
                            member.split.value,
                            member.lineage_id,
                            member.source_document_sha256,
                            member.repository_id,
                            member.component_family,
                            member.temporal_bucket,
                        )
                        for member in members
                    ],
                )
            except sqlite3.IntegrityError as exc:
                raise QueueError(
                    f"could not register dataset version {dataset_version_id!r}"
                ) from exc
        return digest

    @staticmethod
    def _validate_members(
        conn: sqlite3.Connection,
        members: Sequence[DatasetMember],
    ) -> None:
        by_identity = {
            (member.event_id, member.artifact_id): member for member in members
        }
        identities = set(by_identity)
        if len(identities) != len(members):
            raise ValueError("dataset contains duplicate event/artifact members")
        for event_id, artifact_id in identities:
            row = conn.execute(
                """
                SELECT
                    a.redacted,
                    e.lineage_id,
                    e.source_kind,
                    e.source_uri,
                    e.metadata_json,
                    ad.decision,
                    ad.source_revision,
                    v.status AS verification_status
                FROM learning_artifacts AS a
                JOIN learning_events AS e ON e.event_id = a.event_id
                JOIN learning_admissions AS ad ON ad.event_id = e.event_id
                JOIN learning_verifications AS v
                  ON v.verification_id = ad.verification_id
                WHERE a.artifact_id = ? AND a.event_id = ?
                """,
                (artifact_id, event_id),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"artifact {artifact_id!r} lacks verified admission "
                    f"for event {event_id!r}"
                )
            member = by_identity[(event_id, artifact_id)]
            if row["decision"] != "eligible" or row["verification_status"] != "pass":
                raise ValueError(f"event {event_id!r} is not eligible")
            if not row["source_revision"]:
                raise ValueError(f"event {event_id!r} lacks immutable revision")
            if row["lineage_id"] != member.lineage_id:
                raise ValueError(f"event {event_id!r} lineage does not match member")
            metadata = json.loads(row["metadata_json"])
            if (
                metadata.get("data_use") != "training"
                or metadata.get("disposition") != "verified"
            ):
                raise ValueError(f"event {event_id!r} is not approved for training")
            if is_excluded_learning_source(
                SourceKind(row["source_kind"]),
                row["source_uri"],
                metadata,
            ):
                raise ValueError("CategoryRank and Tapes dataset admission is disabled")
            if not bool(row["redacted"]):
                raise ValueError(f"artifact {artifact_id!r} is not redacted")


class TrainingQueue:
    def __init__(self, store: Store):
        self.store = store

    def enqueue(
        self,
        *,
        job_id: str,
        job_kind: str,
        signals: PrioritySignals,
        dataset_version_id: str | None = None,
        config: Mapping[str, Any] | None = None,
        max_attempts: int = 3,
        initial_state: JobState = JobState.ELIGIBLE,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        assert_no_secrets(job_id, field="training job_id")
        assert_no_secrets(job_kind, field="training job_kind")
        if is_excluded_learning_source(
            SourceKind.OTHER,
            f"job://{job_kind}/{job_id}",
            config,
        ):
            raise ValueError("CategoryRank and Tapes training jobs are disabled")
        if initial_state is not JobState.ELIGIBLE:
            raise ValueError("new jobs must enter through eligible admission")
        if dataset_version_id is None:
            raise QueueError("training jobs require an eligible dataset version")
        config_value = dict(config or {})
        assert_value_no_secrets(config_value, field="training job config")
        experiment_sha256 = hashlib.sha256(
            canonical_json(
                {
                    "dataset_version_id": dataset_version_id,
                    "job_kind": job_kind,
                    "config": config_value,
                }
            )
        ).hexdigest()
        now = _timestamp(_utcnow())
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            dataset = conn.execute(
                """
                SELECT state FROM dataset_versions
                WHERE dataset_version_id = ?
                """,
                (dataset_version_id,),
            ).fetchone()
            if dataset is None or dataset["state"] != "eligible":
                raise QueueError("job requires an eligible dataset version")
            counts = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(
                        CASE WHEN ad.decision = 'eligible'
                                  AND v.status = 'pass'
                             THEN 1 ELSE 0 END
                    ) AS admitted,
                    SUM(CASE WHEN dm.split = 'train' THEN 1 ELSE 0 END)
                        AS train_members
                FROM dataset_members AS dm
                LEFT JOIN learning_admissions AS ad
                  ON ad.event_id = dm.event_id
                LEFT JOIN learning_verifications AS v
                  ON v.verification_id = ad.verification_id
                WHERE dm.dataset_version_id = ?
                """,
                (dataset_version_id,),
            ).fetchone()
            if (
                counts is None
                or int(counts["total"] or 0) == 0
                or int(counts["total"]) != int(counts["admitted"] or 0)
                or int(counts["train_members"] or 0) == 0
            ):
                raise QueueError(
                    "dataset contains members without verified admission"
                )
            duplicate = conn.execute(
                """
                SELECT job_id FROM training_jobs
                WHERE dataset_version_id = ?
                  AND job_kind = ?
                  AND experiment_sha256 = ?
                """,
                (dataset_version_id, job_kind, experiment_sha256),
            ).fetchone()
            if duplicate is not None:
                raise QueueError(
                    "an identical dataset/job/config experiment already exists "
                    f"as {duplicate['job_id']!r}; create a new dataset version "
                    "or change the declared experiment config"
                )
            try:
                conn.execute(
                    """
                    INSERT INTO training_jobs (
                        job_id, job_kind, dataset_version_id, state, priority,
                        observed_frequency, frontier_cost, local_failure_rate,
                        verification_strength, diversity, expected_gpu_hours,
                        max_attempts, config_json, experiment_sha256,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        job_kind,
                        dataset_version_id,
                        initial_state.value,
                        signals.score,
                        signals.observed_frequency,
                        signals.frontier_cost,
                        signals.local_failure_rate,
                        signals.verification_strength,
                        signals.diversity,
                        signals.expected_gpu_hours,
                        max_attempts,
                        json.dumps(config_value, sort_keys=True),
                        experiment_sha256,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise QueueError(f"job {job_id!r} already exists or is invalid") from exc

    def claim(
        self,
        node: str,
        *,
        lease_seconds: int = 1800,
        now: datetime | None = None,
        allowed_job_kinds: frozenset[str] | None = None,
    ) -> ClaimedJob | None:
        if not node.strip():
            raise ValueError("node cannot be empty")
        if lease_seconds < 30:
            raise ValueError("lease_seconds must be at least 30")
        current = now or _utcnow()
        now_text = _timestamp(current)
        lease = _timestamp(current + timedelta(seconds=lease_seconds))
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._recover_expired(conn, current)
            kind_clause = ""
            parameters: list[Any] = []
            if allowed_job_kinds is not None:
                if not allowed_job_kinds:
                    return None
                placeholders = ", ".join("?" for _ in allowed_job_kinds)
                kind_clause = f" AND job_kind IN ({placeholders})"
                parameters.extend(sorted(allowed_job_kinds))
            row = conn.execute(
                f"""
                SELECT * FROM training_jobs
                WHERE state = 'eligible' AND attempt < max_attempts
                    {kind_clause}
                ORDER BY
                    CASE job_kind
                        WHEN 'production_failure_replay' THEN 0
                        WHEN 'frozen_evaluation' THEN 1
                        WHEN 'main_model_lora' THEN 2
                        WHEN 'main_model_qlora' THEN 2
                        WHEN 'electronics_text' THEN 3
                        WHEN 'electronics_vision' THEN 3
                        WHEN 'ablation' THEN 4
                        WHEN 'hyperparameter_sweep' THEN 4
                        WHEN 'specforge_draft' THEN 5
                        ELSE 6
                    END,
                    priority DESC, created_at, job_id
                LIMIT 1
                """,
                parameters,
            ).fetchone()
            if row is None:
                return None
            attempt = int(row["attempt"]) + 1
            lease_token = secrets.token_hex(32)
            cursor = conn.execute(
                """
                UPDATE training_jobs
                SET state = 'assigned', assigned_node = ?, attempt = ?,
                    lease_expires_at = ?, lease_token = ?,
                    error = NULL, updated_at = ?
                WHERE job_id = ? AND state = 'eligible'
                """,
                (node, attempt, lease, lease_token, now_text, row["job_id"]),
            )
            if cursor.rowcount != 1:
                raise QueueError("selected training job could not be claimed")
            return ClaimedJob(
                job_id=row["job_id"],
                job_kind=row["job_kind"],
                dataset_version_id=row["dataset_version_id"],
                priority=float(row["priority"]),
                attempt=attempt,
                assigned_node=node,
                lease_expires_at=lease,
                lease_token=lease_token,
                config=json.loads(row["config_json"]),
            )

    def attach_handler(
        self,
        job_id: str,
        *,
        node: str,
        attempt: int,
        lease_token: str,
        pid: int,
        pgid: int,
        now: datetime | None = None,
    ) -> None:
        if pid <= 1 or pgid <= 1:
            raise ValueError("handler pid and process group must be positive")
        current = now or _utcnow()
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE training_jobs
                SET handler_pid = ?, handler_pgid = ?, handler_started_at = ?,
                    updated_at = ?
                WHERE job_id = ? AND state = 'assigned'
                  AND assigned_node = ? AND attempt = ? AND lease_token = ?
                  AND lease_expires_at > ?
                  AND handler_pid IS NULL AND handler_pgid IS NULL
                  AND handler_started_at IS NULL
                """,
                (
                    pid,
                    pgid,
                    _timestamp(current),
                    _timestamp(current),
                    job_id,
                    node,
                    attempt,
                    lease_token,
                    _timestamp(current),
                ),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(
                    "handler attachment requires the current unexpired lease"
                )

    def detach_handler(
        self,
        job_id: str,
        *,
        node: str,
        attempt: int,
        lease_token: str,
        pid: int,
        pgid: int,
        now: datetime | None = None,
    ) -> None:
        current = now or _utcnow()
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE training_jobs
                SET handler_pid = NULL, handler_pgid = NULL,
                    handler_started_at = NULL, updated_at = ?
                WHERE job_id = ? AND state = 'assigned'
                  AND assigned_node = ? AND attempt = ? AND lease_token = ?
                  AND lease_expires_at > ?
                  AND handler_pid = ? AND handler_pgid = ?
                """,
                (
                    _timestamp(current),
                    job_id,
                    node,
                    attempt,
                    lease_token,
                    _timestamp(current),
                    pid,
                    pgid,
                ),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(
                    "handler detachment requires the current unexpired lease"
                )

    def expired_handlers(
        self,
        node: str,
        *,
        now: datetime | None = None,
    ) -> tuple[ExpiredHandler, ...]:
        if not node.strip():
            raise ValueError("node cannot be empty")
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id, attempt, assigned_node, handler_pid,
                       handler_pgid, max_attempts
                FROM training_jobs
                WHERE state = 'assigned'
                  AND assigned_node = ?
                  AND lease_expires_at <= ?
                  AND handler_pid IS NOT NULL
                  AND handler_pgid IS NOT NULL
                ORDER BY job_id
                """,
                (node, _timestamp(now or _utcnow())),
            ).fetchall()
        return tuple(
            ExpiredHandler(
                job_id=row["job_id"],
                attempt=int(row["attempt"]),
                assigned_node=row["assigned_node"],
                handler_pid=int(row["handler_pid"]),
                handler_pgid=int(row["handler_pgid"]),
                max_attempts=int(row["max_attempts"]),
            )
            for row in rows
        )

    def release_expired_handler(
        self,
        handler: ExpiredHandler,
        *,
        now: datetime | None = None,
    ) -> JobState:
        current = now or _utcnow()
        target = (
            JobState.ELIGIBLE
            if handler.attempt < handler.max_attempts
            else JobState.REJECTED
        )
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE training_jobs
                SET state = ?, assigned_node = NULL,
                    lease_expires_at = NULL, lease_token = NULL,
                    handler_pid = NULL, handler_pgid = NULL,
                    handler_started_at = NULL,
                    error = 'expired handler process reaped', updated_at = ?
                WHERE job_id = ? AND state = 'assigned'
                  AND assigned_node = ? AND attempt = ?
                  AND handler_pid = ? AND handler_pgid = ?
                  AND lease_expires_at <= ?
                """,
                (
                    target.value,
                    _timestamp(current),
                    handler.job_id,
                    handler.assigned_node,
                    handler.attempt,
                    handler.handler_pid,
                    handler.handler_pgid,
                    _timestamp(current),
                ),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(
                    "expired handler ownership changed before release"
                )
        return target

    def renew(
        self,
        job_id: str,
        node: str,
        attempt: int,
        lease_token: str,
        *,
        lease_seconds: int = 1800,
        now: datetime | None = None,
    ) -> str:
        if lease_seconds < 30:
            raise ValueError("lease_seconds must be at least 30")
        current = now or _utcnow()
        lease = _timestamp(current + timedelta(seconds=lease_seconds))
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE training_jobs
                SET lease_expires_at = ?, updated_at = ?
                WHERE job_id = ? AND state = 'assigned'
                  AND assigned_node = ? AND attempt = ? AND lease_token = ?
                  AND lease_expires_at > ?
                """,
                (
                    lease,
                    _timestamp(current),
                    job_id,
                    node,
                    attempt,
                    lease_token,
                    _timestamp(current),
                ),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(
                    "only the assigned worker can renew a job lease"
                )
        return lease

    def complete(
        self,
        job_id: str,
        *,
        node: str,
        attempt: int,
        lease_token: str,
        checkpoint_uri: str,
        checkpoint_sha256: str,
        now: datetime | None = None,
    ) -> None:
        current = now or _utcnow()
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            ownership = conn.execute(
                """
                SELECT dataset_version_id FROM training_jobs
                WHERE job_id = ? AND state = 'assigned'
                  AND assigned_node = ? AND attempt = ? AND lease_token = ?
                  AND lease_expires_at > ?
                  AND handler_pid IS NULL AND handler_pgid IS NULL
                """,
                (
                    job_id,
                    node,
                    attempt,
                    lease_token,
                    _timestamp(current),
                ),
            ).fetchone()
            if ownership is None:
                raise InvalidTransitionError(
                    "job completion requires the current unexpired lease"
                )
            _verify_checkpoint_claim(
                checkpoint_uri,
                checkpoint_sha256,
                roots=(
                    self.store.db_path.parent,
                    self.store.db_path.parent.parent,
                ),
            )
            invalid_member = conn.execute(
                """
                SELECT 1
                FROM dataset_members AS dm
                LEFT JOIN learning_admissions AS ad
                  ON ad.event_id = dm.event_id
                LEFT JOIN learning_verifications AS v
                  ON v.verification_id = ad.verification_id
                WHERE dm.dataset_version_id = ?
                  AND (ad.decision IS NOT 'eligible' OR v.status IS NOT 'pass')
                LIMIT 1
                """,
                (ownership["dataset_version_id"],),
            ).fetchone()
            if invalid_member is not None:
                raise InvalidTransitionError(
                    "job dataset lost verified admission"
                )
            cursor = conn.execute(
                """
                UPDATE training_jobs
                SET state = 'trained', checkpoint_uri = ?,
                    checkpoint_sha256 = ?, assigned_node = NULL,
                    lease_expires_at = NULL, lease_token = NULL,
                    handler_pid = NULL, handler_pgid = NULL,
                    handler_started_at = NULL,
                    error = NULL, updated_at = ?
                WHERE job_id = ? AND state = 'assigned'
                  AND assigned_node = ? AND attempt = ? AND lease_token = ?
                  AND lease_expires_at > ?
                  AND handler_pid IS NULL AND handler_pgid IS NULL
                """,
                (
                    checkpoint_uri,
                    checkpoint_sha256,
                    _timestamp(current),
                    job_id,
                    node,
                    attempt,
                    lease_token,
                    _timestamp(current),
                ),
            )
            if cursor.rowcount != 1:
                raise InvalidTransitionError(
                    "job completion requires the current unexpired lease"
                )

    def transition(
        self,
        job_id: str,
        target: JobState,
        *,
        expected: JobState | None = None,
        checkpoint_uri: str | None = None,
        checkpoint_sha256: str | None = None,
        error: str | None = None,
        now: datetime | None = None,
    ) -> None:
        if target is JobState.TRAINED:
            raise InvalidTransitionError("assigned jobs require lease-owned completion")
        if checkpoint_uri is not None or checkpoint_sha256 is not None:
            raise ValueError("checkpoint evidence is accepted only by complete()")
        if error is not None:
            assert_no_secrets(error, field="training job error")
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT state FROM training_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            current = JobState(row["state"])
            if expected is not None and current is not expected:
                raise InvalidTransitionError(
                    f"expected {expected.value}, found {current.value}"
                )
            if target not in TRANSITIONS[current]:
                raise InvalidTransitionError(
                    f"cannot move job from {current.value} to {target.value}"
                )
            if current is JobState.ASSIGNED:
                raise InvalidTransitionError(
                    "assigned jobs require lease-owned completion or failure"
                )
            if target is JobState.EVALUATED:
                evidence = conn.execute(
                    """
                    SELECT decision FROM evaluation_results
                    WHERE job_id = ?
                      AND json_extract(metrics_json, '$.stage') = 'offline'
                    """,
                    (job_id,),
                ).fetchone()
                if evidence is None or evidence["decision"] != "shadow":
                    raise InvalidTransitionError(
                        "evaluated state requires passing offline evidence"
                    )
            if target is JobState.SHADOW:
                evidence = conn.execute(
                    """
                    SELECT decision FROM evaluation_results
                    WHERE job_id = ?
                      AND json_extract(metrics_json, '$.stage') = 'offline'
                    """,
                    (job_id,),
                ).fetchone()
                if evidence is None or evidence["decision"] != "shadow":
                    raise InvalidTransitionError(
                        "shadow state requires a passing offline decision"
                    )
            if target is JobState.CANARY:
                evidence = conn.execute(
                    """
                    SELECT decision FROM evaluation_results
                    WHERE job_id = ?
                      AND json_extract(metrics_json, '$.stage') = 'shadow'
                    """,
                    (job_id,),
                ).fetchone()
                if evidence is None or evidence["decision"] != "canary":
                    raise InvalidTransitionError(
                        "canary state requires a passing shadow decision"
                    )
            if target is JobState.PROMOTED:
                evidence = conn.execute(
                    """
                    SELECT decision FROM evaluation_results
                    WHERE job_id = ?
                      AND json_extract(metrics_json, '$.stage') = 'canary'
                    """,
                    (job_id,),
                ).fetchone()
                if evidence is None or evidence["decision"] != "promote":
                    raise InvalidTransitionError(
                        "promotion requires a passing canary decision"
                    )
            conn.execute(
                """
                UPDATE training_jobs
                SET state = ?, checkpoint_uri = COALESCE(?, checkpoint_uri),
                    checkpoint_sha256 = COALESCE(?, checkpoint_sha256),
                    error = ?, lease_expires_at = NULL, lease_token = NULL,
                    handler_pid = NULL, handler_pgid = NULL,
                    handler_started_at = NULL,
                    assigned_node = CASE WHEN ? = 'assigned' THEN assigned_node ELSE NULL END,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    target.value,
                    None,
                    None,
                    error,
                    target.value,
                    _timestamp(now or _utcnow()),
                    job_id,
                ),
            )

    def fail(
        self,
        job_id: str,
        error: str,
        *,
        node: str,
        attempt: int,
        lease_token: str,
        terminal: bool = False,
        now: datetime | None = None,
    ) -> JobState:
        assert_no_secrets(error, field="training job error")
        timestamp = _timestamp(now or _utcnow())
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT state, attempt, max_attempts FROM training_jobs
                WHERE job_id = ? AND state = 'assigned'
                  AND assigned_node = ? AND attempt = ? AND lease_token = ?
                  AND lease_expires_at > ?
                """,
                (job_id, node, attempt, lease_token, timestamp),
            ).fetchone()
            if row is None:
                raise InvalidTransitionError(
                    "job failure requires the current unexpired lease"
                )
            target = (
                JobState.ELIGIBLE
                if not terminal
                and int(row["attempt"]) < int(row["max_attempts"])
                else JobState.REJECTED
            )
            conn.execute(
                """
                UPDATE training_jobs
                SET state = ?, error = ?, assigned_node = NULL,
                    lease_expires_at = NULL, lease_token = NULL,
                    handler_pid = NULL, handler_pgid = NULL,
                    handler_started_at = NULL, updated_at = ?
                WHERE job_id = ? AND state = 'assigned'
                  AND assigned_node = ? AND attempt = ? AND lease_token = ?
                """,
                (
                    target.value,
                    error,
                    timestamp,
                    job_id,
                    node,
                    attempt,
                    lease_token,
                ),
            )
            return target

    def recover_expired(self, *, now: datetime | None = None) -> int:
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._recover_expired(conn, now or _utcnow())

    @staticmethod
    def _recover_expired(conn: sqlite3.Connection, now: datetime) -> int:
        rows = conn.execute(
            """
            SELECT job_id, attempt, max_attempts FROM training_jobs
            WHERE state = 'assigned' AND lease_expires_at <= ?
              AND handler_pid IS NULL AND handler_pgid IS NULL
            """,
            (_timestamp(now),),
        ).fetchall()
        for row in rows:
            target = (
                JobState.ELIGIBLE.value
                if int(row["attempt"]) < int(row["max_attempts"])
                else JobState.REJECTED.value
            )
            conn.execute(
                """
                UPDATE training_jobs
                SET state = ?, assigned_node = NULL, lease_expires_at = NULL,
                    lease_token = NULL,
                    handler_pid = NULL, handler_pgid = NULL,
                    handler_started_at = NULL,
                    error = 'worker lease expired', updated_at = ?
                WHERE job_id = ?
                """,
                (target, _timestamp(now), row["job_id"]),
            )
        return len(rows)
