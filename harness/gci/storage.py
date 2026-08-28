from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
import time
import uuid
from pathlib import Path
from typing import Iterable

from harness.gci.models import GCIHit, IndexStats, PreparedDocument, RepoSnapshot


SCHEMA_VERSION = 1
GENERATION_RETENTION = 2
FORBIDDEN_STORAGE_ROOTS = (
    Path("/home/samkim/semantic_search"),
    Path("/data/categoryrank"),
)


class GCIStorageError(RuntimeError):
    pass


def _now() -> float:
    return time.time()


def _pack_vector(values: Iterable[float]) -> bytes:
    rows = [float(value) for value in values]
    return struct.pack(f"{len(rows)}f", *rows)


def _unpack_vector(blob: bytes) -> tuple[float, ...]:
    count = len(blob) // 4
    return struct.unpack(f"{count}f", blob)


def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
    av = tuple(a)
    bv = tuple(b)
    if not av or len(av) != len(bv):
        return 0.0
    dot = sum(x * y for x, y in zip(av, bv))
    an = sum(x * x for x in av)
    bn = sum(x * x for x in bv)
    if an <= 0 or bn <= 0:
        return 0.0
    return dot / math.sqrt(an * bn)


def assert_isolated_db(path: Path) -> Path:
    lexical = Path(os.path.abspath(path.expanduser()))
    resolved = path.expanduser().resolve()
    for forbidden in FORBIDDEN_STORAGE_ROOTS:
        if (
            lexical == forbidden
            or forbidden in lexical.parents
            or resolved == forbidden
            or forbidden in resolved.parents
        ):
            raise GCIStorageError(f"GCI storage cannot use CategoryRank path: {resolved}")
    return resolved


class GCIStore:
    def __init__(self, path: Path):
        self.path = assert_isolated_db(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _init(self) -> None:
        conn = self.connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS repos (
                    repo_id TEXT PRIMARY KEY,
                    source_host TEXT NOT NULL,
                    repo_root TEXT NOT NULL,
                    remote TEXT,
                    branch TEXT NOT NULL,
                    head TEXT NOT NULL,
                    dirty INTEGER NOT NULL,
                    state_hash TEXT NOT NULL,
                    current_generation INTEGER,
                    last_indexed REAL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_repos_source_root
                    ON repos(source_host, repo_root);
                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_id TEXT NOT NULL REFERENCES repos(repo_id) ON DELETE CASCADE,
                    state_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    completed_at REAL
                );
                CREATE TABLE IF NOT EXISTS files (
                    repo_id TEXT NOT NULL,
                    generation INTEGER NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
                    path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    content TEXT NOT NULL,
                    language TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    PRIMARY KEY (repo_id, generation, path)
                );
                CREATE TABLE IF NOT EXISTS slices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_id TEXT NOT NULL,
                    generation INTEGER NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
                    path TEXT NOT NULL,
                    symbol TEXT,
                    symbol_type TEXT,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    embedding BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_slices_repo_generation
                    ON slices(repo_id, generation);
                CREATE INDEX IF NOT EXISTS idx_slices_repo_path
                    ON slices(repo_id, generation, path);
                CREATE TABLE IF NOT EXISTS symbols (
                    repo_id TEXT NOT NULL,
                    generation INTEGER NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
                    path TEXT NOT NULL,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    line INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_symbols_name
                    ON symbols(name, repo_id, generation);
                CREATE TABLE IF NOT EXISTS imports (
                    repo_id TEXT NOT NULL,
                    generation INTEGER NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
                    path TEXT NOT NULL,
                    module TEXT NOT NULL,
                    line INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_imports_module
                    ON imports(module, repo_id, generation);
                CREATE VIRTUAL TABLE IF NOT EXISTS gci_fts USING fts5(
                    text,
                    repo_id UNINDEXED,
                    generation UNINDEXED,
                    path UNINDEXED,
                    symbol UNINDEXED,
                    slice_id UNINDEXED
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    submitted_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    error TEXT,
                    stats_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS counters (
                    key TEXT PRIMARY KEY,
                    value REAL NOT NULL
                );
                INSERT OR IGNORE INTO settings(key, value) VALUES ('paused', 'false');
                """
            )
            row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row and int(row["value"]) != SCHEMA_VERSION:
                raise GCIStorageError(
                    f"unsupported GCI schema {row['value']}; expected {SCHEMA_VERSION}"
                )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            conn.commit()
        finally:
            conn.close()

    def paused(self) -> bool:
        conn = self.connect()
        try:
            row = conn.execute("SELECT value FROM settings WHERE key='paused'").fetchone()
            return bool(row and row["value"] == "true")
        finally:
            conn.close()

    def set_paused(self, paused: bool) -> None:
        conn = self.connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO settings(key, value) VALUES ('paused', ?)",
                ("true" if paused else "false",),
            )
            conn.commit()
        finally:
            conn.close()

    def create_job(self, repo_id: str) -> str:
        job_id = uuid.uuid4().hex
        conn = self.connect()
        try:
            conn.execute(
                "INSERT INTO jobs(job_id, repo_id, state, submitted_at) VALUES (?, ?, 'queued', ?)",
                (job_id, repo_id, _now()),
            )
            conn.commit()
            return job_id
        finally:
            conn.close()

    def fail_incomplete_jobs(self, reason: str) -> int:
        conn = self.connect()
        try:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET state='failed', completed_at=?, error=?
                WHERE state IN ('queued', 'running')
                """,
                (_now(), reason),
            )
            conn.commit()
            return int(cursor.rowcount)
        finally:
            conn.close()

    def update_job(
        self,
        job_id: str,
        state: str,
        *,
        error: str | None = None,
        stats: IndexStats | None = None,
    ) -> None:
        conn = self.connect()
        try:
            started = _now() if state == "running" else None
            completed = _now() if state in {"complete", "failed", "cancelled"} else None
            conn.execute(
                """
                UPDATE jobs
                SET state = ?,
                    started_at = COALESCE(started_at, ?),
                    completed_at = COALESCE(?, completed_at),
                    error = ?,
                    stats_json = ?
                WHERE job_id = ?
                """,
                (
                    state,
                    started,
                    completed,
                    error,
                    json.dumps(vars(stats) if stats else {}, sort_keys=True),
                    job_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def job(self, job_id: str) -> dict | None:
        conn = self.connect()
        try:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row:
                return None
            out = dict(row)
            out["stats"] = json.loads(out.pop("stats_json") or "{}")
            return out
        finally:
            conn.close()

    def repo_manifest(self, repo_id: str) -> dict:
        conn = self.connect()
        try:
            repo = conn.execute("SELECT * FROM repos WHERE repo_id=?", (repo_id,)).fetchone()
            if not repo or repo["current_generation"] is None:
                return {"repo_id": repo_id, "state_hash": None, "files": {}}
            rows = conn.execute(
                "SELECT path, content_hash FROM files WHERE repo_id=? AND generation=?",
                (repo_id, repo["current_generation"]),
            ).fetchall()
            return {
                "repo_id": repo_id,
                "state_hash": repo["state_hash"],
                "files": {row["path"]: row["content_hash"] for row in rows},
            }
        finally:
            conn.close()

    def commit_generation(
        self,
        snapshot: RepoSnapshot,
        prepared: Iterable[PreparedDocument],
        stats: IndexStats,
    ) -> int:
        changed = {row.document.path: row for row in prepared}
        expected = dict(snapshot.file_hashes)
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT current_generation FROM repos WHERE repo_id=?", (snapshot.repo_id,)
            ).fetchone()
            previous = current["current_generation"] if current else None
            conn.execute(
                """
                INSERT INTO repos(
                    repo_id, source_host, repo_root, remote, branch, head, dirty,
                    state_hash, current_generation, last_indexed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                ON CONFLICT(repo_id) DO UPDATE SET
                    source_host=excluded.source_host,
                    repo_root=excluded.repo_root,
                    remote=excluded.remote,
                    branch=excluded.branch,
                    head=excluded.head,
                    dirty=excluded.dirty
                """,
                (
                    snapshot.repo_id,
                    snapshot.source_host,
                    snapshot.repo_root,
                    snapshot.remote,
                    snapshot.branch,
                    snapshot.head,
                    int(snapshot.dirty),
                    snapshot.state_hash,
                ),
            )
            cursor = conn.execute(
                "INSERT INTO generations(repo_id, state_hash, status, created_at) VALUES (?, ?, 'building', ?)",
                (snapshot.repo_id, snapshot.state_hash, _now()),
            )
            generation = int(cursor.lastrowid)
            if previous is not None:
                conn.execute(
                    """
                    INSERT INTO files(repo_id, generation, path, content_hash, content, language, size)
                    SELECT repo_id, ?, path, content_hash, content, language, size
                    FROM files WHERE repo_id=? AND generation=?
                    """,
                    (generation, snapshot.repo_id, previous),
                )
                conn.execute(
                    """
                    INSERT INTO slices(
                        repo_id, generation, path, symbol, symbol_type, start_line,
                        end_line, text, text_hash, embedding
                    )
                    SELECT repo_id, ?, path, symbol, symbol_type, start_line,
                           end_line, text, text_hash, embedding
                    FROM slices WHERE repo_id=? AND generation=?
                    """,
                    (generation, snapshot.repo_id, previous),
                )
                conn.execute(
                    """
                    INSERT INTO symbols(repo_id, generation, path, name, kind, line)
                    SELECT repo_id, ?, path, name, kind, line
                    FROM symbols WHERE repo_id=? AND generation=?
                    """,
                    (generation, snapshot.repo_id, previous),
                )
                conn.execute(
                    """
                    INSERT INTO imports(repo_id, generation, path, module, line)
                    SELECT repo_id, ?, path, module, line
                    FROM imports WHERE repo_id=? AND generation=?
                    """,
                    (generation, snapshot.repo_id, previous),
                )
            affected = set(changed) | set(snapshot.deleted)
            for path in affected:
                for table in ("files", "slices", "symbols", "imports"):
                    conn.execute(
                        f"DELETE FROM {table} WHERE repo_id=? AND generation=? AND path=?",
                        (snapshot.repo_id, generation, path),
                    )
            for row in changed.values():
                document = row.document
                conn.execute(
                    """
                    INSERT INTO files(repo_id, generation, path, content_hash, content, language, size)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.repo_id,
                        generation,
                        document.path,
                        document.content_hash,
                        document.content,
                        document.language,
                        len(document.content.encode("utf-8")),
                    ),
                )
                if len(row.slices) != len(row.embeddings):
                    raise GCIStorageError(f"slice/vector mismatch for {document.path}")
                for item, vector in zip(row.slices, row.embeddings):
                    conn.execute(
                        """
                        INSERT INTO slices(
                            repo_id, generation, path, symbol, symbol_type, start_line,
                            end_line, text, text_hash, embedding
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            snapshot.repo_id,
                            generation,
                            item.path,
                            item.symbol,
                            item.symbol_type,
                            item.start_line,
                            item.end_line,
                            item.text,
                            item.text_hash,
                            _pack_vector(vector),
                        ),
                    )
                conn.executemany(
                    "INSERT INTO symbols(repo_id, generation, path, name, kind, line) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (snapshot.repo_id, generation, item.path, item.name, item.kind, item.line)
                        for item in row.symbols
                    ],
                )
                conn.executemany(
                    "INSERT INTO imports(repo_id, generation, path, module, line) VALUES (?, ?, ?, ?, ?)",
                    [
                        (snapshot.repo_id, generation, item.path, item.module, item.line)
                        for item in row.imports
                    ],
                )
            actual = {
                row["path"]: row["content_hash"]
                for row in conn.execute(
                    "SELECT path, content_hash FROM files WHERE repo_id=? AND generation=?",
                    (snapshot.repo_id, generation),
                )
            }
            if actual != expected:
                raise GCIStorageError("resulting generation does not match submitted manifest")
            conn.execute(
                "DELETE FROM gci_fts WHERE repo_id=? AND generation=?",
                (snapshot.repo_id, generation),
            )
            slice_rows = conn.execute(
                "SELECT id, path, symbol, text FROM slices WHERE repo_id=? AND generation=?",
                (snapshot.repo_id, generation),
            ).fetchall()
            conn.executemany(
                "INSERT INTO gci_fts(text, repo_id, generation, path, symbol, slice_id) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        item["text"],
                        snapshot.repo_id,
                        generation,
                        item["path"],
                        item["symbol"],
                        item["id"],
                    )
                    for item in slice_rows
                ],
            )
            conn.execute(
                "UPDATE generations SET status='complete', completed_at=? WHERE id=?",
                (_now(), generation),
            )
            conn.execute(
                """
                UPDATE repos
                SET state_hash=?, current_generation=?, last_indexed=?
                WHERE repo_id=?
                """,
                (snapshot.state_hash, generation, _now(), snapshot.repo_id),
            )
            for key, value in (
                ("index_jobs_complete", 1),
                ("encoder_calls", stats.encoder_calls),
                ("encoder_backoffs", stats.backoffs),
                ("encoder_latency_ms_total", sum(stats.latency_ms)),
                ("encoder_vectors", stats.embedded),
            ):
                conn.execute(
                    """
                    INSERT INTO counters(key, value) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=value+excluded.value
                    """,
                    (key, value),
                )
            stale = [
                row["id"]
                for row in conn.execute(
                    """
                    SELECT id FROM generations
                    WHERE repo_id=?
                    ORDER BY id DESC
                    LIMIT -1 OFFSET ?
                    """,
                    (snapshot.repo_id, GENERATION_RETENTION),
                )
            ]
            for stale_generation in stale:
                conn.execute(
                    "DELETE FROM gci_fts WHERE repo_id=? AND generation=?",
                    (snapshot.repo_id, stale_generation),
                )
                conn.execute(
                    "DELETE FROM generations WHERE id=?",
                    (stale_generation,),
                )
            conn.commit()
            return generation
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def repos(self) -> list[dict]:
        conn = self.connect()
        try:
            return [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT r.*, COUNT(f.path) AS file_count
                    FROM repos r
                    LEFT JOIN files f
                      ON f.repo_id=r.repo_id AND f.generation=r.current_generation
                    GROUP BY r.repo_id
                    ORDER BY r.repo_id
                    """
                )
            ]
        finally:
            conn.close()

    def semantic_search(
        self,
        vector: Iterable[float],
        *,
        limit: int = 8,
        repo_root: str | None = None,
        source_host: str | None = None,
    ) -> list[GCIHit]:
        conn = self.connect()
        try:
            where = ["s.generation=r.current_generation"]
            params: list[object] = []
            if repo_root is not None:
                where.append("r.repo_root=?")
                params.append(repo_root)
            if source_host is not None:
                where.append("r.source_host=?")
                params.append(source_host)
            rows = conn.execute(
                f"""
                SELECT r.repo_id, r.source_host, r.repo_root, r.head, r.state_hash,
                       s.path, s.symbol, s.symbol_type, s.start_line, s.end_line,
                       s.text, s.embedding
                FROM slices s JOIN repos r ON r.repo_id=s.repo_id
                WHERE {' AND '.join(where)}
                """,
                params,
            ).fetchall()
            scored = [
                (
                    _cosine(vector, _unpack_vector(row["embedding"])),
                    row,
                )
                for row in rows
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            return [self._hit(row, score, "semantic") for score, row in scored[:limit]]
        finally:
            conn.close()

    def exact_search(
        self,
        query: str,
        *,
        limit: int = 20,
        repo_root: str | None = None,
    ) -> list[GCIHit]:
        conn = self.connect()
        try:
            tokens = re.findall(r"[A-Za-z0-9_]+", query)
            lexical = " AND ".join(f'"{token}"' for token in tokens[:12])
            params: list[object] = [lexical]
            where = [
                "s.generation=r.current_generation",
                "gci_fts MATCH ?",
                "CAST(gci_fts.slice_id AS INTEGER)=s.id",
            ]
            if repo_root is not None:
                where.append("r.repo_root=?")
                params.append(repo_root)
            rows = []
            if lexical:
                rows = conn.execute(
                    f"""
                    SELECT r.repo_id, r.source_host, r.repo_root, r.head, r.state_hash,
                           s.path, s.symbol, s.symbol_type, s.start_line, s.end_line,
                           s.text
                    FROM gci_fts
                    JOIN slices s ON CAST(gci_fts.slice_id AS INTEGER)=s.id
                    JOIN repos r ON r.repo_id=s.repo_id
                    WHERE {' AND '.join(where)}
                    ORDER BY rank, r.repo_id, s.path, s.start_line
                    LIMIT ?
                    """,
                    [*params, limit],
                ).fetchall()
            if not rows:
                like_params: list[object] = [f"%{query}%"]
                like_where = ["s.generation=r.current_generation", "s.text LIKE ?"]
                if repo_root is not None:
                    like_where.append("r.repo_root=?")
                    like_params.append(repo_root)
                rows = conn.execute(
                    f"""
                    SELECT r.repo_id, r.source_host, r.repo_root, r.head, r.state_hash,
                           s.path, s.symbol, s.symbol_type, s.start_line, s.end_line,
                           s.text
                    FROM slices s JOIN repos r ON r.repo_id=s.repo_id
                    WHERE {' AND '.join(like_where)}
                    ORDER BY r.repo_id, s.path, s.start_line
                    LIMIT ?
                    """,
                    [*like_params, limit],
                ).fetchall()
            return [
                self._hit(
                    row,
                    1.0 if query.lower() in str(row["text"]).lower() else 0.8,
                    "exact",
                )
                for row in rows
            ]
        finally:
            conn.close()

    def symbol_search(
        self,
        query: str,
        *,
        limit: int = 20,
        repo_root: str | None = None,
    ) -> list[GCIHit]:
        conn = self.connect()
        try:
            params: list[object] = [query, f"%{query}%"]
            where = ["y.generation=r.current_generation", "(y.name=? OR y.name LIKE ?)"]
            if repo_root is not None:
                where.append("r.repo_root=?")
                params.append(repo_root)
            rows = conn.execute(
                f"""
                SELECT r.repo_id, r.source_host, r.repo_root, r.head, r.state_hash,
                       y.path, y.name AS symbol, y.kind AS symbol_type,
                       y.line AS start_line, y.line AS end_line,
                       COALESCE((
                           SELECT text FROM slices s
                           WHERE s.repo_id=y.repo_id AND s.generation=y.generation
                             AND s.path=y.path AND s.start_line<=y.line
                           ORDER BY s.start_line DESC LIMIT 1
                       ), '') AS text
                FROM symbols y JOIN repos r ON r.repo_id=y.repo_id
                WHERE {' AND '.join(where)}
                ORDER BY CASE WHEN y.name=? THEN 0 ELSE 1 END, y.name
                LIMIT ?
                """,
                [*params, query, limit],
            ).fetchall()
            return [self._hit(row, 1.0, "symbol") for row in rows]
        finally:
            conn.close()

    @staticmethod
    def _hit(row: sqlite3.Row, score: float, match_type: str) -> GCIHit:
        return GCIHit(
            repo_id=str(row["repo_id"]),
            source_host=str(row["source_host"]),
            repo_root=str(row["repo_root"]),
            revision=str(row["head"]),
            state_hash=str(row["state_hash"]),
            path=str(row["path"]),
            symbol=row["symbol"],
            symbol_type=row["symbol_type"],
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            score=float(score),
            match_type=match_type,
            text=str(row["text"])[:800],
        )

    def metrics(self) -> dict:
        conn = self.connect()
        try:
            counters = {
                row["key"]: row["value"]
                for row in conn.execute("SELECT key, value FROM counters")
            }
            encoder_calls = counters.get("encoder_calls", 0)
            return {
                "repos": conn.execute("SELECT COUNT(*) FROM repos").fetchone()[0],
                "files": conn.execute(
                    "SELECT COUNT(*) FROM files f JOIN repos r ON r.repo_id=f.repo_id AND r.current_generation=f.generation"
                ).fetchone()[0],
                "slices": conn.execute(
                    "SELECT COUNT(*) FROM slices s JOIN repos r ON r.repo_id=s.repo_id AND r.current_generation=s.generation"
                ).fetchone()[0],
                "jobs": {
                    row["state"]: row["count"]
                    for row in conn.execute(
                        "SELECT state, COUNT(*) AS count FROM jobs GROUP BY state"
                    )
                },
                "paused": self.paused(),
                "db_bytes": self.path.stat().st_size if self.path.exists() else 0,
                "encoder": {
                    "calls": int(encoder_calls),
                    "vectors": int(counters.get("encoder_vectors", 0)),
                    "backoffs": int(counters.get("encoder_backoffs", 0)),
                    "average_latency_ms": (
                        counters.get("encoder_latency_ms_total", 0) / encoder_calls
                        if encoder_calls
                        else None
                    ),
                },
            }
        finally:
            conn.close()
