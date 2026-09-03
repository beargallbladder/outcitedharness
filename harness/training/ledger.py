from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from harness.storage.db import Store
from harness.training.models import (
    LearningArtifact,
    LearningAdmission,
    LearningEvent,
    LearningVerification,
    SourceKind,
    is_authorized_owned_code_learning,
    is_excluded_learning_source,
)
from harness.training.registry import canonical_json
from harness.training.security import assert_no_secrets, redact_text


class LearningLedgerError(RuntimeError):
    pass


class LearningEventConflictError(LearningLedgerError):
    pass


class ArtifactIntegrityError(LearningLedgerError):
    pass


def _redact_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_redact_json_value(item) for item in value]
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            redacted_key = redact_text(str(key))
            if redacted_key in redacted:
                raise ValueError("JSON redaction produced duplicate object keys")
            redacted[redacted_key] = _redact_json_value(item)
        return redacted
    return value


@dataclass(frozen=True)
class ArtifactPayload:
    kind: str
    content: str | bytes
    media_type: str = "text/plain"
    redact: bool = True


@dataclass(frozen=True)
class VerificationPayload:
    kind: str
    status: str
    verifier: str
    output_kind: str | None = None
    command: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CaptureResult:
    event_id: str
    event_sha256: str
    artifacts: tuple[LearningArtifact, ...]
    verifications: tuple[LearningVerification, ...]


class ArtifactVault:
    """Private content-addressed storage for data that must not enter SQLite."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.objects = self.root / "sha256"
        self.objects.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        os.chmod(self.objects, 0o700)

    @staticmethod
    def _prepare(payload: ArtifactPayload) -> bytes:
        if isinstance(payload.content, str):
            content = payload.content
            if payload.redact:
                if payload.media_type == "application/json":
                    try:
                        value = json.loads(content)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "JSON artifacts must contain valid JSON"
                        ) from exc
                    content = json.dumps(
                        _redact_json_value(value),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                elif payload.media_type == "application/jsonl":
                    rows = []
                    try:
                        for line in content.splitlines():
                            if line.strip():
                                rows.append(_redact_json_value(json.loads(line)))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "JSONL artifacts must contain valid JSON lines"
                        ) from exc
                    content = "".join(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                        for row in rows
                    )
                else:
                    content = redact_text(content)
            assert_no_secrets(content, field=f"artifact {payload.kind}")
            return content.encode("utf-8")
        if payload.redact:
            raise ValueError("binary artifacts cannot claim text redaction")
        content = bytes(payload.content)
        if payload.media_type.startswith("text/") or payload.media_type in {
            "application/json",
            "application/jsonl",
        }:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("text artifacts must contain valid UTF-8") from exc
            assert_no_secrets(text, field=f"artifact {payload.kind}")
        return content

    def put(self, payload: ArtifactPayload) -> tuple[str, int]:
        data = self._prepare(payload)
        digest = hashlib.sha256(data).hexdigest()
        directory = self.objects / digest[:2]
        directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
        destination = directory / digest
        if destination.exists():
            self._verify_path(destination, digest, len(data))
            return digest, len(data)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{digest}.", suffix=".tmp", dir=directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                self._verify_path(destination, digest, len(data))
            directory_descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)
        return digest, len(data)

    def path_for(self, digest: str) -> Path:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("artifact digest must be lowercase SHA-256")
        return self.objects / digest[:2] / digest

    def verify(self, artifact: LearningArtifact) -> None:
        self._verify_path(
            self.path_for(artifact.sha256),
            artifact.sha256,
            artifact.byte_size,
        )

    @staticmethod
    def _verify_path(path: Path, expected_digest: str, expected_size: int) -> None:
        if path.is_symlink() or not path.is_file():
            raise ArtifactIntegrityError(f"artifact is not a regular file: {path}")
        stat = path.stat()
        if stat.st_size != expected_size:
            raise ArtifactIntegrityError(f"artifact size mismatch: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise ArtifactIntegrityError(f"artifact checksum mismatch: {path}")
        if stat.st_mode & 0o077:
            raise ArtifactIntegrityError(f"artifact permissions are too broad: {path}")


class LearningLedger:
    def __init__(self, store: Store, artifact_root: Path):
        self.store = store
        self.vault = ArtifactVault(artifact_root)

    def capture(
        self,
        event: LearningEvent,
        artifacts: Sequence[ArtifactPayload],
        verifications: Sequence[VerificationPayload] = (),
    ) -> CaptureResult:
        if not artifacts:
            raise ValueError("learning event requires at least one external artifact")
        kinds = [payload.kind for payload in artifacts]
        if len(kinds) != len(set(kinds)):
            raise ValueError("artifact kinds must be unique within an event")

        owned_code = is_authorized_owned_code_learning(
            event.source_kind,
            event.authorization_scope,
            event.metadata,
        )
        for payload in artifacts:
            if owned_code and (
                not payload.kind.startswith("coding_")
                or payload.media_type
                not in {
                    "application/json",
                    "text/plain",
                    "text/x-diff",
                }
            ):
                raise ValueError(
                    "owned-code authorization only permits coding text artifacts"
                )
            content = (
                payload.content
                if isinstance(payload.content, str)
                else bytes(payload.content).decode("utf-8", errors="ignore")
            )
            if is_excluded_learning_source(
                SourceKind.OTHER,
                f"artifact://capture/{payload.kind}",
                content,
            ) and not owned_code:
                raise ValueError(
                    "CategoryRank and Tapes artifact capture is disabled"
                )
            self.vault._prepare(payload)

        pointers: list[LearningArtifact] = []
        by_kind: dict[str, LearningArtifact] = {}
        for payload in artifacts:
            digest, size = self.vault.put(payload)
            kind_digest = hashlib.sha256(payload.kind.encode("utf-8")).hexdigest()[:12]
            pointer = LearningArtifact(
                artifact_id=f"artifact-{event.event_id}-{kind_digest}-{digest[:16]}",
                event_id=event.event_id,
                kind=payload.kind,
                uri=f"artifact://sha256/{digest}",
                sha256=digest,
                byte_size=size,
                media_type=payload.media_type,
                redacted=payload.redact,
                created_at=event.created_at,
            )
            pointers.append(pointer)
            by_kind[payload.kind] = pointer

        proof_rows: list[LearningVerification] = []
        for index, payload in enumerate(verifications):
            output = by_kind.get(payload.output_kind or "")
            if payload.output_kind is not None and output is None:
                raise ValueError(
                    f"verification references unknown artifact kind {payload.output_kind!r}"
                )
            identity = canonical_json(
                {
                    "event_id": event.event_id,
                    "index": index,
                    "kind": payload.kind,
                    "status": payload.status,
                    "verifier": payload.verifier,
                    "output_artifact_id": output.artifact_id if output else None,
                }
            )
            proof_rows.append(
                LearningVerification(
                    verification_id=(
                        f"verification-{event.event_id}-"
                        f"{hashlib.sha256(identity).hexdigest()[:20]}"
                    ),
                    event_id=event.event_id,
                    kind=payload.kind,
                    status=payload.status,
                    verifier=payload.verifier,
                    command=payload.command,
                    output_artifact_id=output.artifact_id if output else None,
                    metadata=dict(payload.metadata or {}),
                    created_at=event.created_at,
                )
            )

        pointers.sort(key=lambda row: row.artifact_id)
        proof_rows.sort(key=lambda row: row.verification_id)
        event_payload = {
            "event": event.model_dump(mode="json", exclude_none=True),
            "artifacts": [
                pointer.model_dump(mode="json", exclude_none=True)
                for pointer in pointers
            ],
            "verifications": [
                proof.model_dump(mode="json", exclude_none=True)
                for proof in proof_rows
            ],
        }
        event_sha256 = hashlib.sha256(canonical_json(event_payload)).hexdigest()
        self._insert_bundle(event, event_sha256, pointers, proof_rows)
        return CaptureResult(
            event_id=event.event_id,
            event_sha256=event_sha256,
            artifacts=tuple(pointers),
            verifications=tuple(proof_rows),
        )

    def _insert_bundle(
        self,
        event: LearningEvent,
        event_sha256: str,
        artifacts: Sequence[LearningArtifact],
        verifications: Sequence[LearningVerification],
    ) -> None:
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT event_sha256 FROM learning_events WHERE event_id = ?",
                (event.event_id,),
            ).fetchone()
            if current:
                if current["event_sha256"] != event_sha256:
                    raise LearningEventConflictError(
                        f"learning event {event.event_id!r} already has different content"
                    )
                return
            try:
                conn.execute(
                    """
                    INSERT INTO learning_events (
                        event_id, event_type, source_kind, source_uri,
                        source_revision, task_id, lineage_id, authorization_scope,
                        state, estimated_cost, metadata_json, event_sha256, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.event_type,
                        event.source_kind.value,
                        event.source_uri,
                        event.source_revision,
                        event.task_id,
                        event.lineage_id,
                        event.authorization_scope,
                        event.state.value,
                        event.estimated_cost,
                        json.dumps(event.metadata, sort_keys=True),
                        event_sha256,
                        event.created_at.isoformat(),
                    ),
                )
                conn.executemany(
                    """
                    INSERT INTO learning_artifacts (
                        artifact_id, event_id, kind, uri, sha256, byte_size,
                        media_type, redacted, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            row.artifact_id,
                            row.event_id,
                            row.kind,
                            row.uri,
                            row.sha256,
                            row.byte_size,
                            row.media_type,
                            int(row.redacted),
                            row.created_at.isoformat(),
                        )
                        for row in artifacts
                    ],
                )
                conn.executemany(
                    """
                    INSERT INTO learning_verifications (
                        verification_id, event_id, kind, status, verifier,
                        command, output_artifact_id, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            row.verification_id,
                            row.event_id,
                            row.kind,
                            row.status,
                            row.verifier,
                            row.command,
                            row.output_artifact_id,
                            json.dumps(row.metadata, sort_keys=True),
                            row.created_at.isoformat(),
                        )
                        for row in verifications
                    ],
                )
            except sqlite3.IntegrityError as exc:
                raise LearningLedgerError(
                    f"could not append learning event {event.event_id!r}"
                ) from exc

    def admit_verified_event(
        self,
        event_id: str,
        verification_id: str,
        *,
        policy_version: str,
        reason: str,
    ) -> LearningAdmission:
        """Append a fail-closed training admission for one verified event."""

        verified = self.verify_event(event_id)
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM learning_admissions WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            event = conn.execute(
                "SELECT * FROM learning_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if event is None:
                raise KeyError(event_id)
            if event["event_sha256"] != verified.event_sha256:
                raise ArtifactIntegrityError(
                    f"learning event changed during admission: {event_id}"
                )
            verification = conn.execute(
                """
                SELECT * FROM learning_verifications
                WHERE verification_id = ? AND event_id = ?
                """,
                (verification_id, event_id),
            ).fetchone()
            if verification is None:
                raise ValueError("admission verification does not belong to event")
            metadata = json.loads(event["metadata_json"])
            source_kind = SourceKind(event["source_kind"])
            if is_excluded_learning_source(
                source_kind,
                event["source_uri"],
                metadata,
            ) and not is_authorized_owned_code_learning(
                source_kind,
                event["authorization_scope"],
                metadata,
            ):
                raise ValueError("CategoryRank and Tapes admission is disabled")
            verification_metadata = json.loads(verification["metadata_json"])
            source_revision = event["source_revision"]
            if source_revision is None and source_kind is SourceKind.DESIGNWINS:
                source_revision = verification_metadata.get("content_sha256")
            if source_revision is None:
                raise ValueError("training admission requires an immutable revision")
            if metadata.get("data_use") != "training":
                raise ValueError("event is not approved for training")
            if metadata.get("disposition") != "verified":
                raise ValueError("event disposition is not verified")
            if verification["status"] != "pass":
                raise ValueError("training admission requires a passing verification")
            if verification["output_artifact_id"] is None:
                raise ValueError("training admission requires a proof artifact")
            unsafe_artifact = conn.execute(
                """
                SELECT artifact_id FROM learning_artifacts
                WHERE event_id = ?
                  AND (
                      media_type LIKE 'text/%'
                      OR media_type IN ('application/json', 'application/jsonl')
                  )
                  AND redacted != 1
                LIMIT 1
                """,
                (event_id,),
            ).fetchone()
            if unsafe_artifact is not None:
                raise ValueError(
                    "text training artifacts must pass redaction before admission"
                )
            created_at = datetime.now(timezone.utc)
            admission_payload = {
                "event_id": event_id,
                "verification_id": verification_id,
                "decision": "eligible",
                "policy_version": policy_version,
                "reason": reason,
                "source_revision": source_revision,
                "created_at": created_at.isoformat(),
            }
            admission_sha256 = hashlib.sha256(
                canonical_json(admission_payload)
            ).hexdigest()
            admission = LearningAdmission(
                admission_id=f"admission-{event_id}",
                admission_sha256=admission_sha256,
                **admission_payload,
            )
            if existing is not None:
                if (
                    existing["verification_id"] != verification_id
                    or existing["policy_version"] != policy_version
                    or existing["reason"] != reason
                    or existing["source_revision"] != source_revision
                    or existing["decision"] != "eligible"
                ):
                    raise LearningEventConflictError(
                        f"event {event_id!r} already has a different admission"
                    )
                return LearningAdmission(
                    admission_id=existing["admission_id"],
                    event_id=existing["event_id"],
                    verification_id=existing["verification_id"],
                    decision=existing["decision"],
                    policy_version=existing["policy_version"],
                    reason=existing["reason"],
                    source_revision=existing["source_revision"],
                    admission_sha256=existing["admission_sha256"],
                    created_at=existing["created_at"],
                )
            conn.execute(
                """
                INSERT INTO learning_admissions (
                    admission_id, event_id, verification_id, decision,
                    policy_version, reason, source_revision,
                    admission_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    admission.admission_id,
                    admission.event_id,
                    admission.verification_id,
                    admission.decision,
                    admission.policy_version,
                    admission.reason,
                    admission.source_revision,
                    admission.admission_sha256,
                    admission.created_at.isoformat(),
                ),
            )
            return admission

    def verify_event(self, event_id: str) -> CaptureResult:
        with self.store.connect() as conn:
            event_row = conn.execute(
                "SELECT * FROM learning_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if event_row is None:
                raise KeyError(event_id)
            artifact_rows = conn.execute(
                "SELECT * FROM learning_artifacts WHERE event_id = ? ORDER BY artifact_id",
                (event_id,),
            ).fetchall()
            verification_rows = conn.execute(
                """
                SELECT * FROM learning_verifications
                WHERE event_id = ? ORDER BY verification_id
                """,
                (event_id,),
            ).fetchall()

        artifacts = tuple(
            LearningArtifact(
                artifact_id=row["artifact_id"],
                event_id=row["event_id"],
                kind=row["kind"],
                uri=row["uri"],
                sha256=row["sha256"],
                byte_size=row["byte_size"],
                media_type=row["media_type"],
                redacted=bool(row["redacted"]),
                created_at=row["created_at"],
            )
            for row in artifact_rows
        )
        for artifact in artifacts:
            self.vault.verify(artifact)
        verifications = tuple(
            LearningVerification(
                verification_id=row["verification_id"],
                event_id=row["event_id"],
                kind=row["kind"],
                status=row["status"],
                verifier=row["verifier"],
                command=row["command"],
                output_artifact_id=row["output_artifact_id"],
                metadata=json.loads(row["metadata_json"]),
                created_at=row["created_at"],
            )
            for row in verification_rows
        )
        event = LearningEvent(
            event_id=event_row["event_id"],
            event_type=event_row["event_type"],
            source_kind=event_row["source_kind"],
            source_uri=event_row["source_uri"],
            source_revision=event_row["source_revision"],
            task_id=event_row["task_id"],
            lineage_id=event_row["lineage_id"],
            authorization_scope=event_row["authorization_scope"],
            state=event_row["state"],
            estimated_cost=event_row["estimated_cost"],
            metadata=json.loads(event_row["metadata_json"]),
            created_at=event_row["created_at"],
        )
        canonical = {
            "event": event.model_dump(mode="json", exclude_none=True),
            "artifacts": [
                row.model_dump(mode="json", exclude_none=True) for row in artifacts
            ],
            "verifications": [
                row.model_dump(mode="json", exclude_none=True)
                for row in verifications
            ],
        }
        actual = hashlib.sha256(canonical_json(canonical)).hexdigest()
        if actual != event_row["event_sha256"]:
            raise ArtifactIntegrityError(
                f"learning event metadata checksum mismatch: {event_id}"
            )
        return CaptureResult(
            event_id=event_id,
            event_sha256=actual,
            artifacts=artifacts,
            verifications=verifications,
        )
