"""Durable operational state for the continuous datasheet-learning factory."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import stat
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from harness.electronics.claims import canonical_json
from harness.electronics.corpus import sha256_file, verify_corpus_registry


class FactoryStateError(RuntimeError):
    """Raised when durable factory state would become inconsistent."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("factory timestamps must include a timezone")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("stored factory timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _secure_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise FactoryStateError(f"expected a regular non-symlink file: {path}")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FactoryStateError(f"immutable output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            payload = (
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ChunkLease:
    chunk_id: str
    node: str
    attempt: int
    lease_token: str
    lease_expires_at: str
    queue_path: Path
    queue_sha256: str
    offset: int
    item_count: int
    output_directory: Path


@dataclass(frozen=True)
class DiscoveryReport:
    discovered_paths: int
    new_observations: int
    waiting_for_stability: int
    ready: int
    duplicates: int
    quarantined: int
    unchanged: int
    unsafe_paths: int


class ElectronicsFactoryState:
    """SQLite-backed leases plus append-only transition evidence.

    Mutable rows are operational projections. ``factory_events`` is immutable
    evidence of every state transition and can be replayed independently.
    """

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or not self.root.is_dir():
            raise FactoryStateError("factory state root must be a real directory")
        os.chmod(self.root, 0o700)
        self.database = self.root / "factory.sqlite3"
        if self.database.is_symlink():
            raise FactoryStateError("factory database cannot be a symlink")
        if not self.database.exists():
            descriptor = os.open(
                self.database,
                os.O_RDWR | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            os.close(descriptor)
        info = self.database.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
            raise FactoryStateError("factory database must be an owned regular file")
        os.chmod(self.database, 0o600)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS factory_source_observations (
                    observation_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    device INTEGER NOT NULL,
                    inode INTEGER NOT NULL,
                    byte_size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    stable_after_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK (status IN (
                            'observed', 'ready', 'duplicate',
                            'quarantined', 'superseded'
                        )),
                    sha256 TEXT,
                    canonical_observation_id TEXT,
                    reason TEXT,
                    updated_at TEXT NOT NULL,
                    UNIQUE (
                        source_path, device, inode, byte_size, mtime_ns
                    ),
                    FOREIGN KEY (canonical_observation_id)
                        REFERENCES factory_source_observations(observation_id)
                );

                CREATE INDEX IF NOT EXISTS idx_factory_sources_status
                    ON factory_source_observations(status, stable_after_at);
                CREATE INDEX IF NOT EXISTS idx_factory_sources_path
                    ON factory_source_observations(source_path, first_seen_at);
                CREATE INDEX IF NOT EXISTS idx_factory_sources_sha
                    ON factory_source_observations(sha256, status);

                CREATE TABLE IF NOT EXISTS factory_source_batches (
                    observation_id TEXT PRIMARY KEY,
                    cohort_id TEXT NOT NULL,
                    snapshot_path TEXT NOT NULL,
                    snapshot_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (observation_id)
                        REFERENCES factory_source_observations(observation_id)
                );

                CREATE INDEX IF NOT EXISTS idx_factory_source_batches_cohort
                    ON factory_source_batches(cohort_id, observation_id);

                CREATE TABLE IF NOT EXISTS factory_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    queue_path TEXT NOT NULL,
                    queue_sha256 TEXT NOT NULL,
                    queue_evidence_sha256 TEXT NOT NULL,
                    offset INTEGER NOT NULL,
                    item_count INTEGER NOT NULL,
                    output_directory TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL
                        CHECK (status IN (
                            'queued', 'leased', 'completed', 'failed'
                        )),
                    attempt INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    available_at TEXT NOT NULL,
                    lease_node TEXT,
                    lease_token TEXT,
                    lease_expires_at TEXT,
                    output_manifest_sha256 TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (queue_sha256, offset, item_count)
                );

                CREATE INDEX IF NOT EXISTS idx_factory_chunks_queue
                    ON factory_chunks(status, available_at, offset);
                CREATE INDEX IF NOT EXISTS idx_factory_chunks_lease
                    ON factory_chunks(status, lease_expires_at);

                CREATE TABLE IF NOT EXISTS factory_frontier_runs (
                    run_id TEXT PRIMARY KEY,
                    prepared_bundle TEXT NOT NULL,
                    prepared_evidence_sha256 TEXT NOT NULL,
                    submission_state TEXT NOT NULL,
                    lifecycle_root TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL
                        CHECK (status IN (
                            'submitted', 'processing', 'ended',
                            'retrieved', 'reconciled', 'verified',
                            'finalized', 'failed'
                        )),
                    request_count INTEGER NOT NULL,
                    estimated_maximum_usd REAL NOT NULL,
                    latest_status_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS factory_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_factory_events_subject
                    ON factory_events(subject_id, created_at);

                CREATE TRIGGER IF NOT EXISTS factory_events_no_update
                BEFORE UPDATE ON factory_events
                BEGIN
                    SELECT RAISE(ABORT, 'factory events are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS factory_events_no_delete
                BEFORE DELETE ON factory_events
                BEGIN
                    SELECT RAISE(ABORT, 'factory events are immutable');
                END;
                """
            )
        for sidecar in (
            self.database.with_name(self.database.name + "-wal"),
            self.database.with_name(self.database.name + "-shm"),
        ):
            if sidecar.exists():
                os.chmod(sidecar, 0o600)

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        event_type: str,
        subject_id: str,
        payload: Mapping[str, Any],
        *,
        now: datetime,
    ) -> None:
        encoded = canonical_json(payload)
        connection.execute(
            """
            INSERT INTO factory_events (
                event_id, event_type, subject_id, payload_json,
                payload_sha256, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                f"factory-event-{uuid.uuid4().hex}",
                event_type,
                subject_id,
                encoded.decode("utf-8"),
                _sha256_bytes(encoded),
                _timestamp(now),
            ),
        )

    @staticmethod
    def _file_identity(path: Path) -> tuple[int, int, int, int]:
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise FactoryStateError(f"source is not a regular file: {path}")
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

    @staticmethod
    def _observation_id(
        path: Path,
        identity: tuple[int, int, int, int],
    ) -> str:
        payload = canonical_json(
            {
                "path": str(path),
                "device": identity[0],
                "inode": identity[1],
                "byte_size": identity[2],
                "mtime_ns": identity[3],
            }
        )
        return f"source-{_sha256_bytes(payload)[:32]}"

    @staticmethod
    def _inspect_pdf(
        path: Path,
        expected: tuple[int, int, int, int],
    ) -> str:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            if identity != expected or not stat.S_ISREG(before.st_mode):
                raise FactoryStateError("source changed before hashing")
            if before.st_size < 128:
                raise FactoryStateError("PDF is implausibly small")
            digest = hashlib.sha256()
            prefix = b""
            tail = b""
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    if len(prefix) < 1024:
                        prefix = (prefix + chunk)[:1024]
                    tail = (tail + chunk)[-8192:]
                    digest.update(chunk)
            after = os.fstat(descriptor)
            current = os.stat(path, follow_symlinks=False)
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            path_identity = (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
            )
            if expected != after_identity or expected != path_identity:
                raise FactoryStateError("source changed while hashing")
            if b"%PDF-" not in prefix:
                raise FactoryStateError("file has no PDF header")
            if b"%%EOF" not in tail:
                raise FactoryStateError("file has no terminal PDF marker")
            return digest.hexdigest()
        finally:
            os.close(descriptor)

    def discover_pdfs(
        self,
        roots: Sequence[Path],
        *,
        stability_seconds: int = 120,
        now: datetime | None = None,
    ) -> DiscoveryReport:
        if stability_seconds < 0 or stability_seconds > 86_400:
            raise ValueError("stability_seconds must be within 0..86400")
        observed_at = now or _utcnow()
        candidates: list[tuple[Path, tuple[int, int, int, int]]] = []
        unsafe_paths = 0
        for raw_root in roots:
            root = raw_root.expanduser().resolve(strict=True)
            if raw_root.expanduser().is_symlink() or not root.is_dir():
                raise FactoryStateError(
                    f"download root must be a real directory: {raw_root}"
                )
            for path in sorted(root.rglob("*")):
                if path.suffix.casefold() != ".pdf":
                    continue
                if path.is_symlink():
                    unsafe_paths += 1
                    continue
                try:
                    resolved = path.resolve(strict=True)
                    candidates.append((resolved, self._file_identity(resolved)))
                except (OSError, FactoryStateError):
                    unsafe_paths += 1

        counters: Counter[str] = Counter()
        eligible: list[
            tuple[str, Path, tuple[int, int, int, int]]
        ] = []
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for path, identity in candidates:
                observation_id = self._observation_id(path, identity)
                current = connection.execute(
                    """
                    SELECT * FROM factory_source_observations
                    WHERE observation_id = ?
                    """,
                    (observation_id,),
                ).fetchone()
                if current is None:
                    connection.execute(
                        """
                        UPDATE factory_source_observations
                        SET status = 'superseded', updated_at = ?
                        WHERE source_path = ? AND status = 'observed'
                        """,
                        (_timestamp(observed_at), str(path)),
                    )
                    stable_after = observed_at + timedelta(
                        seconds=stability_seconds
                    )
                    connection.execute(
                        """
                        INSERT INTO factory_source_observations (
                            observation_id, source_path, device, inode,
                            byte_size, mtime_ns, first_seen_at,
                            stable_after_at, last_seen_at, status, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'observed', ?)
                        """,
                        (
                            observation_id,
                            str(path),
                            *identity,
                            _timestamp(observed_at),
                            _timestamp(stable_after),
                            _timestamp(observed_at),
                            _timestamp(observed_at),
                        ),
                    )
                    self._append_event(
                        connection,
                        "source_observed",
                        observation_id,
                        {
                            "path": str(path),
                            "byte_size": identity[2],
                            "mtime_ns": identity[3],
                            "stable_after_at": _timestamp(stable_after),
                        },
                        now=observed_at,
                    )
                    counters["new_observations"] += 1
                    current_status = "observed"
                    stable_after_at = stable_after
                else:
                    current_status = str(current["status"])
                    stable_after_at = _parse_timestamp(
                        str(current["stable_after_at"])
                    )
                    connection.execute(
                        """
                        UPDATE factory_source_observations
                        SET last_seen_at = ?, updated_at = ?
                        WHERE observation_id = ?
                        """,
                        (
                            _timestamp(observed_at),
                            _timestamp(observed_at),
                            observation_id,
                        ),
                    )
                    counters["unchanged"] += 1
                if (
                    current_status == "observed"
                    and observed_at >= stable_after_at
                ):
                    eligible.append((observation_id, path, identity))
                elif current_status == "observed":
                    counters["waiting_for_stability"] += 1

        for observation_id, path, identity in eligible:
            try:
                digest = self._inspect_pdf(path, identity)
                error = None
            except (OSError, FactoryStateError) as exc:
                digest = None
                error = f"{type(exc).__name__}: {exc}"
            finalized_at = now or _utcnow()
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = connection.execute(
                    """
                    SELECT status FROM factory_source_observations
                    WHERE observation_id = ?
                    """,
                    (observation_id,),
                ).fetchone()
                if current is None or current["status"] != "observed":
                    continue
                if digest is None:
                    status = "quarantined"
                    canonical_id = None
                    counters["quarantined"] += 1
                else:
                    duplicate = connection.execute(
                        """
                        SELECT observation_id
                        FROM factory_source_observations
                        WHERE sha256 = ? AND status = 'ready'
                        ORDER BY first_seen_at, observation_id
                        LIMIT 1
                        """,
                        (digest,),
                    ).fetchone()
                    status = "duplicate" if duplicate else "ready"
                    canonical_id = (
                        str(duplicate["observation_id"])
                        if duplicate is not None
                        else None
                    )
                    counters[
                        "duplicates" if duplicate else "ready"
                    ] += 1
                connection.execute(
                    """
                    UPDATE factory_source_observations
                    SET status = ?, sha256 = ?, canonical_observation_id = ?,
                        reason = ?, updated_at = ?
                    WHERE observation_id = ?
                    """,
                    (
                        status,
                        digest,
                        canonical_id,
                        error,
                        _timestamp(finalized_at),
                        observation_id,
                    ),
                )
                self._append_event(
                    connection,
                    f"source_{status}",
                    observation_id,
                    {
                        "path": str(path),
                        "sha256": digest,
                        "canonical_observation_id": canonical_id,
                        "reason": error,
                    },
                    now=finalized_at,
                )

        return DiscoveryReport(
            discovered_paths=len(candidates),
            new_observations=counters["new_observations"],
            waiting_for_stability=counters["waiting_for_stability"],
            ready=counters["ready"],
            duplicates=counters["duplicates"],
            quarantined=counters["quarantined"],
            unchanged=counters["unchanged"],
            unsafe_paths=unsafe_paths,
        )

    def seal_ready_source_snapshot(self, destination: Path) -> dict[str, Any]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT observation_id, source_path, byte_size, mtime_ns, sha256
                FROM factory_source_observations
                WHERE status = 'ready'
                ORDER BY sha256, source_path
                """
            ).fetchall()
        documents = [dict(row) for row in rows]
        core = {
            "schema": "harness.electronics-incremental-source-snapshot.v1",
            "purpose": "stable_deduplicated_datasheet_intake",
            "counts": {"documents": len(documents)},
            "documents": documents,
        }
        manifest = {
            "created_at": _timestamp(_utcnow()),
            **core,
            "evidence_sha256": _sha256_bytes(canonical_json(core)),
        }
        _write_new_json(destination, manifest)
        return manifest

    def seal_unassigned_source_snapshot(
        self,
        destination: Path,
        *,
        cohort_id: str,
        maximum_documents: int = 5000,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Seal each new canonical PDF into exactly one processing cohort."""

        if (
            not cohort_id
            or cohort_id != cohort_id.strip()
            or len(cohort_id) > 160
        ):
            raise ValueError("cohort_id must be non-empty, trimmed, and bounded")
        if not 1 <= maximum_documents <= 100_000:
            raise ValueError("maximum_documents must be within 1..100000")
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT source.*
                FROM factory_source_observations AS source
                LEFT JOIN factory_source_batches AS assigned
                    ON assigned.observation_id = source.observation_id
                WHERE source.status = 'ready'
                    AND assigned.observation_id IS NULL
                ORDER BY source.first_seen_at, source.observation_id
                LIMIT ?
                """,
                (maximum_documents,),
            ).fetchall()
        if not rows:
            return None
        documents = [
            {
                "observation_id": row["observation_id"],
                "source_path": row["source_path"],
                "byte_size": int(row["byte_size"]),
                "mtime_ns": int(row["mtime_ns"]),
                "sha256": row["sha256"],
            }
            for row in rows
        ]
        core = {
            "schema": "harness.electronics-incremental-source-snapshot.v1",
            "purpose": "stable_deduplicated_datasheet_intake",
            "cohort_id": cohort_id,
            "counts": {"documents": len(documents)},
            "documents": documents,
        }
        sealed_at = now or _utcnow()
        manifest = {
            "created_at": _timestamp(sealed_at),
            **core,
            "evidence_sha256": _sha256_bytes(canonical_json(core)),
        }
        _write_new_json(destination, manifest)
        snapshot_path = destination.expanduser().resolve(strict=True)
        snapshot_sha = sha256_file(snapshot_path)
        try:
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                for row in rows:
                    current = connection.execute(
                        """
                        SELECT cohort_id, snapshot_sha256
                        FROM factory_source_batches
                        WHERE observation_id = ?
                        """,
                        (row["observation_id"],),
                    ).fetchone()
                    if current is not None:
                        if (
                            current["cohort_id"] != cohort_id
                            or current["snapshot_sha256"] != snapshot_sha
                        ):
                            raise FactoryStateError(
                                "source was concurrently assigned to another cohort"
                            )
                        continue
                    connection.execute(
                        """
                        INSERT INTO factory_source_batches (
                            observation_id, cohort_id, snapshot_path,
                            snapshot_sha256, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            row["observation_id"],
                            cohort_id,
                            str(snapshot_path),
                            snapshot_sha,
                            _timestamp(sealed_at),
                        ),
                    )
                self._append_event(
                    connection,
                    "source_cohort_sealed",
                    cohort_id,
                    {
                        "snapshot_path": str(snapshot_path),
                        "snapshot_sha256": snapshot_sha,
                        "documents": len(documents),
                    },
                    now=sealed_at,
                )
        except BaseException:
            try:
                os.chmod(snapshot_path, 0o600)
                snapshot_path.unlink()
            except OSError:
                pass
            raise
        return manifest

    def seed_corpus_registry(
        self,
        registry_path: Path,
        *,
        now: datetime | None = None,
    ) -> dict[str, int]:
        """Seed already-sealed PDFs so discovery hashes only genuinely new files."""

        path = registry_path.expanduser().resolve(strict=True)
        _secure_regular_file(path)
        registry = json.loads(path.read_text(encoding="utf-8"))
        verify_corpus_registry(registry)
        root = Path(registry["sources"]["pdf_root"]).resolve(strict=True)
        if root.is_symlink() or not root.is_dir():
            raise FactoryStateError("sealed corpus PDF root is unavailable")
        registry_sha = sha256_file(path)
        seeded_at = now or _utcnow()
        inserted = 0
        existing = 0
        missing = 0
        unsafe = 0
        canonical_by_digest: dict[str, str] = {}
        baseline_cohort = f"baseline-{registry_sha[:24]}"
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for document in registry["documents"]:
                digest = str(document["document_sha256"])
                for relative in document.get("paths") or []:
                    candidate = (root / str(relative)).resolve()
                    try:
                        candidate.relative_to(root)
                    except ValueError:
                        unsafe += 1
                        continue
                    if (
                        candidate.is_symlink()
                        or not candidate.exists()
                        or not candidate.is_file()
                    ):
                        missing += 1
                        continue
                    try:
                        identity = self._file_identity(candidate)
                    except (OSError, FactoryStateError):
                        unsafe += 1
                        continue
                    if identity[2] != int(document["byte_size"]):
                        unsafe += 1
                        continue
                    observation_id = self._observation_id(candidate, identity)
                    current = connection.execute(
                        """
                        SELECT status, sha256
                        FROM factory_source_observations
                        WHERE observation_id = ?
                        """,
                        (observation_id,),
                    ).fetchone()
                    if current is not None:
                        if current["sha256"] not in {None, digest}:
                            raise FactoryStateError(
                                "seeded corpus conflicts with source observation"
                            )
                        existing += 1
                        canonical_by_digest.setdefault(digest, observation_id)
                    else:
                        canonical = canonical_by_digest.get(digest)
                        status = "duplicate" if canonical else "ready"
                        connection.execute(
                            """
                            INSERT INTO factory_source_observations (
                                observation_id, source_path, device, inode,
                                byte_size, mtime_ns, first_seen_at,
                                stable_after_at, last_seen_at, status, sha256,
                                canonical_observation_id, reason, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                observation_id,
                                str(candidate),
                                *identity,
                                _timestamp(seeded_at),
                                _timestamp(seeded_at),
                                _timestamp(seeded_at),
                                status,
                                digest,
                                canonical,
                                f"seeded from sealed registry {registry_sha}",
                                _timestamp(seeded_at),
                            ),
                        )
                        self._append_event(
                            connection,
                            "source_seeded_from_registry",
                            observation_id,
                            {
                                "path": str(candidate),
                                "sha256": digest,
                                "registry_sha256": registry_sha,
                                "status": status,
                                "canonical_observation_id": canonical,
                            },
                            now=seeded_at,
                        )
                        canonical_by_digest.setdefault(digest, observation_id)
                        inserted += 1
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO factory_source_batches (
                            observation_id, cohort_id, snapshot_path,
                            snapshot_sha256, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            observation_id,
                            baseline_cohort,
                            str(path),
                            registry_sha,
                            _timestamp(seeded_at),
                        ),
                    )
        return {
            "inserted": inserted,
            "existing": existing,
            "missing": missing,
            "unsafe": unsafe,
        }

    @staticmethod
    def _load_queue(path: Path) -> tuple[dict[str, Any], str]:
        queue_path = path.expanduser().resolve(strict=True)
        _secure_regular_file(queue_path)
        payload = queue_path.read_bytes()
        value = json.loads(payload)
        if (
            not isinstance(value, dict)
            or value.get("schema")
            != "harness.electronics-structural-local-work.v1"
        ):
            raise FactoryStateError("unsupported structural work queue")
        core = {
            key: item
            for key, item in value.items()
            if key not in {"created_at", "evidence_sha256"}
        }
        if _sha256_bytes(canonical_json(core)) != value.get("evidence_sha256"):
            raise FactoryStateError("structural work queue evidence is invalid")
        work = value.get("work")
        if not isinstance(work, list) or not work:
            raise FactoryStateError("structural work queue is empty")
        return value, _sha256_bytes(payload)

    def register_queue(
        self,
        queue_path: Path,
        output_root: Path,
        *,
        chunk_size: int = 10,
        max_attempts: int = 3,
        start_offset: int = 0,
        now: datetime | None = None,
    ) -> list[str]:
        if not 1 <= chunk_size <= 1000:
            raise ValueError("chunk_size must be within 1..1000")
        if not 1 <= max_attempts <= 20:
            raise ValueError("max_attempts must be within 1..20")
        queue, queue_sha = self._load_queue(queue_path)
        resolved_queue = queue_path.expanduser().resolve(strict=True)
        resolved_output = output_root.expanduser().resolve()
        if resolved_output.is_symlink():
            raise FactoryStateError("chunk output root cannot be a symlink")
        total = len(queue["work"])
        if not 0 <= start_offset < total:
            raise ValueError("start_offset must select an item in the queue")
        registered: list[str] = []
        created_at = now or _utcnow()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for offset in range(start_offset, total, chunk_size):
                count = min(chunk_size, total - offset)
                identity = canonical_json(
                    {
                        "queue_sha256": queue_sha,
                        "offset": offset,
                        "item_count": count,
                    }
                )
                chunk_id = (
                    f"chunk-{queue_sha[:12]}-{offset:06d}-{count:04d}-"
                    f"{_sha256_bytes(identity)[:12]}"
                )
                output = resolved_output / chunk_id
                current = connection.execute(
                    "SELECT * FROM factory_chunks WHERE chunk_id = ?",
                    (chunk_id,),
                ).fetchone()
                if current is not None:
                    if (
                        current["queue_sha256"] != queue_sha
                        or int(current["offset"]) != offset
                        or int(current["item_count"]) != count
                        or Path(str(current["output_directory"])) != output
                    ):
                        raise FactoryStateError(
                            f"chunk identity conflict: {chunk_id}"
                        )
                    registered.append(chunk_id)
                    continue
                connection.execute(
                    """
                    INSERT INTO factory_chunks (
                        chunk_id, queue_path, queue_sha256,
                        queue_evidence_sha256, offset, item_count,
                        output_directory, status, max_attempts,
                        available_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        str(resolved_queue),
                        queue_sha,
                        queue["evidence_sha256"],
                        offset,
                        count,
                        str(output),
                        max_attempts,
                        _timestamp(created_at),
                        _timestamp(created_at),
                        _timestamp(created_at),
                    ),
                )
                self._append_event(
                    connection,
                    "chunk_registered",
                    chunk_id,
                    {
                        "queue_sha256": queue_sha,
                        "offset": offset,
                        "item_count": count,
                        "output_directory": str(output),
                    },
                    now=created_at,
                )
                registered.append(chunk_id)
        return registered

    def _recover_expired_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        now: datetime,
    ) -> list[str]:
        rows = connection.execute(
            """
            SELECT * FROM factory_chunks
            WHERE status = 'leased' AND lease_expires_at <= ?
            ORDER BY chunk_id
            """,
            (_timestamp(now),),
        ).fetchall()
        recovered: list[str] = []
        for row in rows:
            terminal = int(row["attempt"]) >= int(row["max_attempts"])
            status = "failed" if terminal else "queued"
            connection.execute(
                """
                UPDATE factory_chunks
                SET status = ?, available_at = ?, lease_node = NULL,
                    lease_token = NULL, lease_expires_at = NULL,
                    error = ?, updated_at = ?
                WHERE chunk_id = ? AND status = 'leased'
                """,
                (
                    status,
                    _timestamp(now),
                    "lease expired",
                    _timestamp(now),
                    row["chunk_id"],
                ),
            )
            self._append_event(
                connection,
                "chunk_lease_expired",
                str(row["chunk_id"]),
                {
                    "attempt": int(row["attempt"]),
                    "previous_node": row["lease_node"],
                    "new_status": status,
                },
                now=now,
            )
            recovered.append(str(row["chunk_id"]))
        return recovered

    def recover_expired(self, *, now: datetime | None = None) -> list[str]:
        recovered_at = now or _utcnow()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._recover_expired_in_transaction(
                connection,
                now=recovered_at,
            )

    def claim_chunk(
        self,
        node: str,
        *,
        lease_seconds: int = 1800,
        now: datetime | None = None,
    ) -> ChunkLease | None:
        if not node or node != node.strip():
            raise ValueError("node must be a non-empty canonical name")
        if not 30 <= lease_seconds <= 86_400:
            raise ValueError("lease_seconds must be within 30..86400")
        claimed_at = now or _utcnow()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._recover_expired_in_transaction(connection, now=claimed_at)
            row = connection.execute(
                """
                SELECT * FROM factory_chunks
                WHERE status = 'queued' AND available_at <= ?
                    AND attempt < max_attempts
                ORDER BY created_at, offset, chunk_id
                LIMIT 1
                """,
                (_timestamp(claimed_at),),
            ).fetchone()
            if row is None:
                return None
            token = secrets.token_urlsafe(32)
            attempt = int(row["attempt"]) + 1
            expires = claimed_at + timedelta(seconds=lease_seconds)
            connection.execute(
                """
                UPDATE factory_chunks
                SET status = 'leased', attempt = ?, lease_node = ?,
                    lease_token = ?, lease_expires_at = ?, error = NULL,
                    updated_at = ?
                WHERE chunk_id = ? AND status = 'queued'
                """,
                (
                    attempt,
                    node,
                    token,
                    _timestamp(expires),
                    _timestamp(claimed_at),
                    row["chunk_id"],
                ),
            )
            self._append_event(
                connection,
                "chunk_claimed",
                str(row["chunk_id"]),
                {
                    "node": node,
                    "attempt": attempt,
                    "lease_expires_at": _timestamp(expires),
                },
                now=claimed_at,
            )
            return ChunkLease(
                chunk_id=str(row["chunk_id"]),
                node=node,
                attempt=attempt,
                lease_token=token,
                lease_expires_at=_timestamp(expires),
                queue_path=Path(str(row["queue_path"])),
                queue_sha256=str(row["queue_sha256"]),
                offset=int(row["offset"]),
                item_count=int(row["item_count"]),
                output_directory=Path(str(row["output_directory"])),
            )

    @staticmethod
    def _assert_active_lease(
        row: sqlite3.Row | None,
        lease: ChunkLease,
    ) -> None:
        if row is None:
            raise FactoryStateError(f"unknown chunk: {lease.chunk_id}")
        if (
            row["status"] != "leased"
            or row["lease_node"] != lease.node
            or int(row["attempt"]) != lease.attempt
            or not secrets.compare_digest(
                str(row["lease_token"] or ""),
                lease.lease_token,
            )
        ):
            raise FactoryStateError("chunk lease is stale or belongs to another node")

    def renew_chunk(
        self,
        lease: ChunkLease,
        *,
        lease_seconds: int = 1800,
        now: datetime | None = None,
    ) -> str:
        if not 30 <= lease_seconds <= 86_400:
            raise ValueError("lease_seconds must be within 30..86400")
        renewed_at = now or _utcnow()
        expires = renewed_at + timedelta(seconds=lease_seconds)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM factory_chunks WHERE chunk_id = ?",
                (lease.chunk_id,),
            ).fetchone()
            self._assert_active_lease(row, lease)
            connection.execute(
                """
                UPDATE factory_chunks
                SET lease_expires_at = ?, updated_at = ?
                WHERE chunk_id = ?
                """,
                (
                    _timestamp(expires),
                    _timestamp(renewed_at),
                    lease.chunk_id,
                ),
            )
            self._append_event(
                connection,
                "chunk_lease_renewed",
                lease.chunk_id,
                {
                    "node": lease.node,
                    "attempt": lease.attempt,
                    "lease_expires_at": _timestamp(expires),
                },
                now=renewed_at,
            )
        return _timestamp(expires)

    @staticmethod
    def _verify_chunk_output(lease: ChunkLease) -> str:
        output = lease.output_directory.resolve(strict=True)
        if lease.output_directory.is_symlink() or not output.is_dir():
            raise FactoryStateError("chunk output must be a real directory")
        manifest_path = output / "manifest.json"
        _secure_regular_file(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema")
            != "harness.electronics-structural-local-extraction.v1"
        ):
            raise FactoryStateError("chunk output has an unsupported schema")
        sources = manifest.get("sources") or {}
        selection = manifest.get("selection") or {}
        if sources.get("structural_queue_sha256") != lease.queue_sha256:
            raise FactoryStateError("chunk output used a different queue")
        if (
            int(selection.get("offset", -1)) != lease.offset
            or int(selection.get("work_items", -1)) != lease.item_count
        ):
            raise FactoryStateError("chunk output selection does not match lease")
        return sha256_file(manifest_path)

    def complete_chunk(
        self,
        lease: ChunkLease,
        *,
        now: datetime | None = None,
    ) -> str:
        manifest_sha = self._verify_chunk_output(lease)
        completed_at = now or _utcnow()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM factory_chunks WHERE chunk_id = ?",
                (lease.chunk_id,),
            ).fetchone()
            if row is not None and row["status"] == "completed":
                if row["output_manifest_sha256"] != manifest_sha:
                    raise FactoryStateError("completed chunk output changed")
                return manifest_sha
            self._assert_active_lease(row, lease)
            connection.execute(
                """
                UPDATE factory_chunks
                SET status = 'completed', output_manifest_sha256 = ?,
                    lease_node = NULL, lease_token = NULL,
                    lease_expires_at = NULL, error = NULL, updated_at = ?
                WHERE chunk_id = ?
                """,
                (manifest_sha, _timestamp(completed_at), lease.chunk_id),
            )
            self._append_event(
                connection,
                "chunk_completed",
                lease.chunk_id,
                {
                    "node": lease.node,
                    "attempt": lease.attempt,
                    "manifest_sha256": manifest_sha,
                },
                now=completed_at,
            )
        return manifest_sha

    def fail_chunk(
        self,
        lease: ChunkLease,
        error: str,
        *,
        terminal: bool = False,
        retry_delay_seconds: int = 30,
        now: datetime | None = None,
    ) -> str:
        if not error or error != error.strip():
            raise ValueError("error must be non-empty and trimmed")
        if not 0 <= retry_delay_seconds <= 86_400:
            raise ValueError("retry delay must be within 0..86400")
        failed_at = now or _utcnow()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM factory_chunks WHERE chunk_id = ?",
                (lease.chunk_id,),
            ).fetchone()
            self._assert_active_lease(row, lease)
            exhausted = int(row["attempt"]) >= int(row["max_attempts"])
            status = "failed" if terminal or exhausted else "queued"
            available = failed_at + timedelta(seconds=retry_delay_seconds)
            connection.execute(
                """
                UPDATE factory_chunks
                SET status = ?, available_at = ?, lease_node = NULL,
                    lease_token = NULL, lease_expires_at = NULL,
                    error = ?, updated_at = ?
                WHERE chunk_id = ?
                """,
                (
                    status,
                    _timestamp(available),
                    error[:4000],
                    _timestamp(failed_at),
                    lease.chunk_id,
                ),
            )
            self._append_event(
                connection,
                "chunk_failed",
                lease.chunk_id,
                {
                    "node": lease.node,
                    "attempt": lease.attempt,
                    "new_status": status,
                    "error": error[:4000],
                },
                now=failed_at,
            )
        return status

    def register_frontier_run(
        self,
        *,
        run_id: str,
        prepared_bundle: Path,
        submission_state: Path,
        lifecycle_root: Path,
        now: datetime | None = None,
    ) -> bool:
        prepared = prepared_bundle.expanduser().resolve(strict=True)
        state = submission_state.expanduser().resolve(strict=True)
        manifest_path = prepared / "manifest.json"
        _secure_regular_file(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema")
            != "harness.electronics-frontier-batch.v1"
        ):
            raise FactoryStateError("unsupported prepared frontier bundle")
        if not (state / "submission-intent.json").is_file():
            raise FactoryStateError("frontier submission state is incomplete")
        created_at = now or _utcnow()
        lifecycle = lifecycle_root.expanduser().resolve()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM factory_frontier_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if current is not None:
                if (
                    current["prepared_evidence_sha256"]
                    != manifest["evidence_sha256"]
                    or Path(str(current["submission_state"])) != state
                    or Path(str(current["lifecycle_root"])) != lifecycle
                ):
                    raise FactoryStateError(
                        f"frontier run identity conflict: {run_id}"
                    )
                return False
            connection.execute(
                """
                INSERT INTO factory_frontier_runs (
                    run_id, prepared_bundle, prepared_evidence_sha256,
                    submission_state, lifecycle_root, status, request_count,
                    estimated_maximum_usd, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'submitted', ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(prepared),
                    manifest["evidence_sha256"],
                    str(state),
                    str(lifecycle),
                    int(manifest["counts"]["requests"]),
                    float(manifest["pricing"]["estimated_maximum_usd"]),
                    _timestamp(created_at),
                    _timestamp(created_at),
                ),
            )
            self._append_event(
                connection,
                "frontier_run_registered",
                run_id,
                {
                    "prepared_evidence_sha256": manifest["evidence_sha256"],
                    "request_count": int(manifest["counts"]["requests"]),
                    "estimated_maximum_usd": float(
                        manifest["pricing"]["estimated_maximum_usd"]
                    ),
                },
                now=created_at,
            )
        return True

    def record_frontier_status(
        self,
        run_id: str,
        status_payload: Mapping[str, Any],
        *,
        stage: str | None = None,
        error: str | None = None,
        now: datetime | None = None,
    ) -> None:
        recorded_at = now or _utcnow()
        batches = status_payload.get("batches") or []
        processing = {
            str(batch.get("processing_status"))
            for batch in batches
            if isinstance(batch, Mapping)
        }
        inferred = (
            "ended"
            if processing and processing == {"ended"}
            else "processing"
        )
        new_status = stage or inferred
        allowed = {
            "submitted",
            "processing",
            "ended",
            "retrieved",
            "reconciled",
            "verified",
            "finalized",
            "failed",
        }
        if new_status not in allowed:
            raise ValueError(f"unsupported frontier run stage: {new_status}")
        encoded = canonical_json(dict(status_payload)).decode("utf-8")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT status FROM factory_frontier_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if current is None:
                raise FactoryStateError(f"unknown frontier run: {run_id}")
            connection.execute(
                """
                UPDATE factory_frontier_runs
                SET status = ?, latest_status_json = ?, error = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    new_status,
                    encoded,
                    error,
                    _timestamp(recorded_at),
                    run_id,
                ),
            )
            self._append_event(
                connection,
                "frontier_status_recorded",
                run_id,
                {
                    "previous_status": current["status"],
                    "new_status": new_status,
                    "error": error,
                },
                now=recorded_at,
            )

    def status(self) -> dict[str, Any]:
        with self.connect() as connection:
            source_counts = {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM factory_source_observations
                    GROUP BY status
                    ORDER BY status
                    """
                )
            }
            chunk_counts = {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM factory_chunks
                    GROUP BY status
                    ORDER BY status
                    """
                )
            }
            active = [
                {
                    "chunk_id": row["chunk_id"],
                    "node": row["lease_node"],
                    "attempt": int(row["attempt"]),
                    "offset": int(row["offset"]),
                    "item_count": int(row["item_count"]),
                    "lease_expires_at": row["lease_expires_at"],
                }
                for row in connection.execute(
                    """
                    SELECT * FROM factory_chunks
                    WHERE status = 'leased'
                    ORDER BY lease_node, chunk_id
                    """
                )
            ]
            frontier = [
                {
                    "run_id": row["run_id"],
                    "status": row["status"],
                    "request_count": int(row["request_count"]),
                    "estimated_maximum_usd": float(
                        row["estimated_maximum_usd"]
                    ),
                    "updated_at": row["updated_at"],
                    "error": row["error"],
                }
                for row in connection.execute(
                    """
                    SELECT * FROM factory_frontier_runs
                    ORDER BY created_at, run_id
                    """
                )
            ]
            unassigned_sources = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM factory_source_observations AS source
                    LEFT JOIN factory_source_batches AS assigned
                        ON assigned.observation_id = source.observation_id
                    WHERE source.status = 'ready'
                        AND assigned.observation_id IS NULL
                    """
                ).fetchone()[0]
            )
            source_cohorts = int(
                connection.execute(
                    """
                    SELECT COUNT(DISTINCT cohort_id)
                    FROM factory_source_batches
                    """
                ).fetchone()[0]
            )
            events = int(
                connection.execute(
                    "SELECT COUNT(*) FROM factory_events"
                ).fetchone()[0]
            )
        return {
            "schema": "harness.electronics-factory-status.v1",
            "created_at": _timestamp(_utcnow()),
            "sources": source_counts,
            "unassigned_ready_sources": unassigned_sources,
            "source_cohorts": source_cohorts,
            "chunks": chunk_counts,
            "active_leases": active,
            "frontier_runs": frontier,
            "events": events,
        }

    def completed_bundle_paths(self, queue_sha256: str) -> list[Path]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT output_directory FROM factory_chunks
                WHERE queue_sha256 = ? AND status = 'completed'
                ORDER BY offset
                """,
                (queue_sha256,),
            ).fetchall()
        return [Path(str(row["output_directory"])) for row in rows]

    def queue_chunk_summary(self, queue_sha256: str) -> dict[str, Any]:
        with self.connect() as connection:
            counts = {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM factory_chunks
                    WHERE queue_sha256 = ?
                    GROUP BY status
                    ORDER BY status
                    """,
                    (queue_sha256,),
                )
            }
            items = {
                str(row["status"]): int(row["items"])
                for row in connection.execute(
                    """
                    SELECT status, SUM(item_count) AS items
                    FROM factory_chunks
                    WHERE queue_sha256 = ?
                    GROUP BY status
                    ORDER BY status
                    """,
                    (queue_sha256,),
                )
            }
        return {
            "queue_sha256": queue_sha256,
            "chunks": counts,
            "items": items,
            "registered_chunks": sum(counts.values()),
            "registered_items": sum(items.values()),
        }

