from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from harness.config import GCIRefreshPolicy, Settings
from harness.gci.models import RepoSnapshot
from harness.gci.scanner import (
    build_snapshot,
    canonical_approved_root,
    repo_id,
)

AUTOMATION_SCHEMA_VERSION = 1


class RefreshClient(Protocol):
    def manifest(self, repo_id: str) -> dict: ...

    def submit(self, snapshot: RepoSnapshot, *, refresh: bool) -> str: ...

    def wait_job(self, job_id: str, *, timeout: float = 300.0) -> dict: ...


class AutomationBusyError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalGitSource:
    path: Path
    owner: str = "self"
    type: str = "local_git"


@dataclass(frozen=True)
class RepositoryProbe:
    fingerprint: str
    head: str
    dirty: bool
    last_commit_at: float


@dataclass(frozen=True)
class RefreshOutcome:
    repo_id: str
    root: str
    status: str
    changed_documents: int = 0
    deleted_documents: int = 0
    job_id: str = ""
    error: str = ""


@dataclass(frozen=True)
class AutomationRun:
    started_at: float
    completed_at: float
    outcomes: tuple[RefreshOutcome, ...]
    locked: bool = False


class AutomationState:
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def _init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS repositories (
                    repo_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_host TEXT NOT NULL,
                    repo_root TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    last_head TEXT NOT NULL,
                    last_checked REAL NOT NULL,
                    last_changed REAL NOT NULL,
                    last_indexed REAL,
                    next_due REAL NOT NULL,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    last_status TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_refresh_source_root
                    ON repositories(source_host, repo_root);
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at REAL NOT NULL,
                    completed_at REAL NOT NULL,
                    outcomes_json TEXT NOT NULL
                );
                """
            )
            row = conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if row and int(row["value"]) != AUTOMATION_SCHEMA_VERSION:
                raise RuntimeError(
                    "unsupported GCI refresh state schema "
                    f"{row['value']}; expected {AUTOMATION_SCHEMA_VERSION}"
                )
            conn.execute(
                """
                INSERT OR REPLACE INTO meta(key, value)
                VALUES ('schema_version', ?)
                """,
                (str(AUTOMATION_SCHEMA_VERSION),),
            )

    def get(self, rid: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM repositories WHERE repo_id=?",
                (rid,),
            ).fetchone()
        return dict(row) if row else None

    def rows(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM repositories ORDER BY repo_root"
            ).fetchall()
        return [dict(row) for row in rows]

    def success(
        self,
        *,
        rid: str,
        host: str,
        root: Path,
        probe: RepositoryProbe,
        checked_at: float,
        changed_at: float,
        indexed_at: float | None,
        next_due: float,
        status: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO repositories(
                    repo_id, source_type, source_host, repo_root, fingerprint,
                    last_head, last_checked, last_changed, last_indexed,
                    next_due, failure_count, last_error, last_status
                ) VALUES (?, 'local_git', ?, ?, ?, ?, ?, ?, ?, ?, 0, '', ?)
                ON CONFLICT(repo_id) DO UPDATE SET
                    fingerprint=excluded.fingerprint,
                    last_head=excluded.last_head,
                    last_checked=excluded.last_checked,
                    last_changed=excluded.last_changed,
                    last_indexed=COALESCE(excluded.last_indexed, repositories.last_indexed),
                    next_due=excluded.next_due,
                    failure_count=0,
                    last_error='',
                    last_status=excluded.last_status
                """,
                (
                    rid,
                    host,
                    str(root),
                    probe.fingerprint,
                    probe.head,
                    checked_at,
                    changed_at,
                    indexed_at,
                    next_due,
                    status,
                ),
            )

    def failure(
        self,
        *,
        rid: str,
        host: str,
        root: Path,
        now: float,
        next_due: float,
        error: str,
    ) -> None:
        current = self.get(rid)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO repositories(
                    repo_id, source_type, source_host, repo_root, fingerprint,
                    last_head, last_checked, last_changed, last_indexed,
                    next_due, failure_count, last_error, last_status
                ) VALUES (?, 'local_git', ?, ?, '', '', ?, ?, NULL, ?, 1, ?, 'failed')
                ON CONFLICT(repo_id) DO UPDATE SET
                    last_checked=excluded.last_checked,
                    next_due=excluded.next_due,
                    failure_count=repositories.failure_count + 1,
                    last_error=excluded.last_error,
                    last_status='failed'
                """,
                (
                    rid,
                    host,
                    str(root),
                    now,
                    float((current or {}).get("last_changed") or now),
                    next_due,
                    error[:1000],
                ),
            )

    def record_run(self, run: AutomationRun) -> None:
        payload = [
            {
                "repo_id": row.repo_id,
                "root": row.root,
                "status": row.status,
                "changed_documents": row.changed_documents,
                "deleted_documents": row.deleted_documents,
                "job_id": row.job_id,
                "error": row.error,
            }
            for row in run.outcomes
        ]
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(started_at, completed_at, outcomes_json)
                VALUES (?, ?, ?)
                """,
                (run.started_at, run.completed_at, json.dumps(payload, sort_keys=True)),
            )


def _git(root: Path, *argv: str, binary: bool = False) -> str | bytes:
    proc = subprocess.run(
        ["git", "-C", str(root), *argv],
        capture_output=True,
        text=not binary,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (
            proc.stderr.decode(errors="replace")
            if binary and isinstance(proc.stderr, bytes)
            else str(proc.stderr or "")
        )
        raise RuntimeError(f"git {' '.join(argv)} failed: {stderr.strip()[:300]}")
    return proc.stdout


def _status_paths(status: bytes) -> list[str]:
    paths: list[str] = []
    for index, token in enumerate(status.split(b"\0")):
        if not token:
            continue
        raw = token[3:] if len(token) >= 3 and token[2:3] == b" " else token
        try:
            value = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        if value and value not in paths:
            paths.append(value)
        # In porcelain -z output, the second rename token is a bare path.
        if index and value and value not in paths:
            paths.append(value)
    return paths


def probe_local_git(root: Path) -> RepositoryProbe:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"repository root is missing: {root}")
    head = str(_git(root, "rev-parse", "HEAD")).strip()
    commit_raw = str(_git(root, "log", "-1", "--format=%ct", "HEAD")).strip()
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "-z",
        binary=True,
    )
    assert isinstance(status, bytes)
    digest = hashlib.sha256()
    digest.update(head.encode())
    digest.update(b"\0")
    digest.update(status)
    for rel in sorted(_status_paths(status)):
        path = (root / rel).resolve(strict=False)
        if root != path and root not in path.parents:
            continue
        try:
            stat = path.stat()
        except OSError:
            digest.update(f"\0{rel}\0missing".encode())
            continue
        digest.update(
            f"\0{rel}\0{stat.st_size}\0{stat.st_mtime_ns}".encode(
                "utf-8",
                errors="replace",
            )
        )
    return RepositoryProbe(
        fingerprint=digest.hexdigest(),
        head=head,
        dirty=bool(status),
        last_commit_at=float(commit_raw or 0),
    )


def refresh_repository(
    client: RefreshClient,
    root: Path,
    *,
    approved_roots: list[str],
    source_host: str | None = None,
    wait: bool = True,
) -> RefreshOutcome:
    root = canonical_approved_root(root, approved_roots)
    host = source_host or socket.gethostname()
    rid = repo_id(host, root)
    previous = client.manifest(rid)
    snapshot = build_snapshot(
        root,
        approved_roots=approved_roots,
        previous_files=previous.get("files") or {},
        source_host=host,
    )
    if snapshot.state_hash == previous.get("state_hash"):
        return RefreshOutcome(rid, str(root), "unchanged")
    job_id = client.submit(
        snapshot,
        refresh=bool(previous.get("state_hash")),
    )
    status = "queued"
    if wait:
        job = client.wait_job(job_id)
        status = str(job.get("state") or "failed")
        if status != "complete":
            raise RuntimeError(
                f"GCI job {job_id} ended in {status}: {job.get('error') or ''}"
            )
    return RefreshOutcome(
        rid,
        str(root),
        status,
        changed_documents=len(snapshot.documents),
        deleted_documents=len(snapshot.deleted),
        job_id=job_id,
    )


def _next_interval(
    policy: GCIRefreshPolicy,
    *,
    now: float,
    last_changed: float,
) -> int:
    stale_at = policy.stale_after_days * 86_400
    if now - last_changed >= stale_at:
        return policy.stale_interval_seconds
    return policy.active_interval_seconds


def _jitter(rid: str, maximum: int) -> int:
    if maximum <= 0:
        return 0
    return int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) % (maximum + 1)


@contextmanager
def automation_lock(state_path: Path) -> Iterator[None]:
    lock_path = state_path.expanduser().with_suffix(
        state_path.expanduser().suffix + ".lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AutomationBusyError("another GCI refresh pass is running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_automation(
    settings: Settings,
    client: RefreshClient,
    *,
    now: float | None = None,
    source_host: str | None = None,
    probe: Callable[[Path], RepositoryProbe] = probe_local_git,
    roots: list[str] | None = None,
    force: bool = False,
) -> AutomationRun:
    policy = settings.gci_refresh
    clock = float(now if now is not None else time.time())
    started = clock
    host = source_host or socket.gethostname()
    state = AutomationState(policy.state_path)
    selected = roots if roots is not None else settings.code_index_repos
    sources = tuple(
        LocalGitSource(Path(raw_root).expanduser().resolve())
        for raw_root in selected
    )
    outcomes: list[RefreshOutcome] = []
    with automation_lock(policy.state_path):
        for source in sources:
            root = source.path
            rid = repo_id(host, root)
            current = state.get(rid)
            if not force and current and clock < float(current["next_due"]):
                outcomes.append(RefreshOutcome(rid, str(root), "deferred"))
                continue
            try:
                observed = probe(root)
                changed = current is None or observed.fingerprint != current["fingerprint"]
                previous_changed = float(
                    (current or {}).get("last_changed")
                    or (clock if observed.dirty else observed.last_commit_at)
                    or clock
                )
                changed_at = (
                    clock
                    if current is not None and changed
                    else previous_changed
                )
                if not changed and current is not None:
                    interval = _next_interval(
                        policy,
                        now=clock,
                        last_changed=changed_at,
                    )
                    state.success(
                        rid=rid,
                        host=host,
                        root=root,
                        probe=observed,
                        checked_at=clock,
                        changed_at=changed_at,
                        indexed_at=None,
                        next_due=clock
                        + interval
                        + _jitter(rid, policy.jitter_seconds),
                        status="unchanged",
                    )
                    outcomes.append(RefreshOutcome(rid, str(root), "unchanged"))
                    continue
                outcome = refresh_repository(
                    client,
                    root,
                    approved_roots=settings.code_index_repos,
                    source_host=host,
                    wait=True,
                )
                interval = _next_interval(
                    policy,
                    now=clock,
                    last_changed=changed_at,
                )
                state.success(
                    rid=rid,
                    host=host,
                    root=root,
                    probe=observed,
                    checked_at=clock,
                    changed_at=changed_at,
                    indexed_at=clock,
                    next_due=clock
                    + interval
                    + _jitter(rid, policy.jitter_seconds),
                    status=outcome.status,
                )
                outcomes.append(outcome)
            except Exception as exc:
                failures = int((current or {}).get("failure_count") or 0) + 1
                retry = min(
                    policy.failure_retry_seconds * (2 ** min(failures - 1, 6)),
                    policy.stale_interval_seconds,
                )
                message = f"{type(exc).__name__}: {exc}"
                state.failure(
                    rid=rid,
                    host=host,
                    root=root,
                    now=clock,
                    next_due=clock + retry,
                    error=message,
                )
                outcomes.append(
                    RefreshOutcome(
                        rid,
                        str(root),
                        "failed",
                        error=message[:1000],
                    )
                )
    completed = float(now if now is not None else time.time())
    run = AutomationRun(started, completed, tuple(outcomes))
    state.record_run(run)
    return run


def refresh_after_publication(
    settings: Settings,
    client: RefreshClient,
    destination: Path,
) -> RefreshOutcome | None:
    root = destination.expanduser().resolve()
    approved = {
        Path(item).expanduser().resolve()
        for item in settings.code_index_repos
    }
    if root not in approved:
        return None
    return refresh_repository(
        client,
        root,
        approved_roots=settings.code_index_repos,
        wait=False,
    )


def notify_publication(
    settings: Settings,
    destination: Path,
) -> RefreshOutcome | None:
    """Queue an approved published repository without delaying publication."""
    if not settings.gci_enabled:
        return None
    token = os.environ.get(settings.gci_token_env, "")
    if not token:
        return None
    from harness.gci.client import GCIClient

    client = GCIClient(
        settings.gci_url,
        token=token,
        timeout=settings.gci_timeout_s,
    )
    return refresh_after_publication(settings, client, destination)
