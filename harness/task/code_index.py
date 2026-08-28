"""M5 code index. Spark :8800 encodes; vectors stay local. Not the CR category index."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from harness.config import find_project_root
from harness.task.search import embed_texts

log = logging.getLogger("harness.index")

INCLUDE = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".next",
    "dist",
    "build",
    ".cursor",
    "results",
    "models",
}
MAX_FILE_BYTES = 200_000
MAX_CHUNK_CHARS = 2400
MIN_CHUNK_CHARS = 40
BATCH = 16
DEFAULT_REPOS = (
    "/Users/samkim/Harnessv1",
    "/Users/samkim/locationlocationlocation",
)

_SPLIT = re.compile(
    r"^(def |class |export |function |async function |const [A-Z][A-Za-z0-9_]*\s*=)",
    re.M,
)

EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class CodeHit:
    repo: str
    path: str
    start_line: int
    end_line: int
    score: float
    text: str


def default_index_path(root: Path | None = None) -> Path:
    return (root or find_project_root()) / "results" / "code_index.sqlite"


def pack_vector(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *[float(v) for v in values])


def unpack_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / math.sqrt(na * nb)


def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def chunk_source(text: str) -> list[tuple[int, int, str]]:
    lines = text.splitlines()
    chunks: list[tuple[int, int, str]] = []
    buf: list[str] = []
    start = 1
    for index, line in enumerate(lines, 1):
        hit_split = bool(buf) and bool(_SPLIT.match(line))
        too_big = bool(buf) and len("\n".join(buf)) + len(line) + 1 > MAX_CHUNK_CHARS
        if hit_split or too_big:
            body = "\n".join(buf)
            if len(body) >= MIN_CHUNK_CHARS:
                chunks.append((start, index - 1, body))
            buf = [line]
            start = index
        else:
            buf.append(line)
    if buf:
        body = "\n".join(buf)
        if len(body) >= MIN_CHUNK_CHARS:
            chunks.append((start, start + len(buf) - 1, body))
    return chunks


def iter_source_files(repo: Path) -> Iterable[Path]:
    root = repo.resolve()
    if not root.is_dir():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in INCLUDE:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            repo TEXT NOT NULL,
            path TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            PRIMARY KEY (repo, path)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo TEXT NOT NULL,
            path TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            text_hash TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_repo_path ON chunks(repo, path)")
    return conn


def index_repos(
    repos: list[Path],
    db_path: Path,
    *,
    embed: EmbedFn | None = None,
) -> dict[str, int]:
    embed_fn = embed or (lambda texts: embed_texts(texts))
    conn = connect(db_path)
    stats = {"files": 0, "unchanged": 0, "chunks": 0, "embedded": 0}
    pending_text: list[str] = []
    pending_meta: list[tuple[str, str, int, int, str]] = []

    def flush() -> None:
        if not pending_text:
            return
        vectors = embed_fn(list(pending_text))
        for meta, vec, body in zip(pending_meta, vectors, pending_text):
            repo, rel, start, end, digest = meta
            conn.execute(
                """
                INSERT INTO chunks (repo, path, start_line, end_line, text_hash, text, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (repo, rel, start, end, digest, body, pack_vector(vec)),
            )
            stats["embedded"] += 1
        pending_text.clear()
        pending_meta.clear()

    wanted: dict[str, set[str]] = {}
    for repo in repos:
        root = repo.resolve()
        name = root.name
        wanted.setdefault(name, set())
        if not root.is_dir():
            log.warning("code_index skip missing repo %s", root)
            continue
        for path in iter_source_files(root):
            rel = str(path.relative_to(root))
            wanted[name].add(rel)
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            digest = file_hash(text)
            row = conn.execute(
                "SELECT file_hash FROM files WHERE repo = ? AND path = ?",
                (name, rel),
            ).fetchone()
            stats["files"] += 1
            if row and row[0] == digest:
                stats["unchanged"] += 1
                continue
            conn.execute("DELETE FROM chunks WHERE repo = ? AND path = ?", (name, rel))
            conn.execute(
                "INSERT OR REPLACE INTO files (repo, path, file_hash) VALUES (?, ?, ?)",
                (name, rel, digest),
            )
            for start, end, body in chunk_source(text):
                pending_text.append(body)
                pending_meta.append((name, rel, start, end, file_hash(body)))
                stats["chunks"] += 1
                if len(pending_text) >= BATCH:
                    flush()
        stale = [
            path
            for (path,) in conn.execute("SELECT path FROM files WHERE repo = ?", (name,))
            if path not in wanted[name]
        ]
        for path in stale:
            conn.execute("DELETE FROM files WHERE repo = ? AND path = ?", (name, path))
            conn.execute("DELETE FROM chunks WHERE repo = ? AND path = ?", (name, path))
    flush()
    conn.commit()
    conn.close()
    log.info(
        "code_index files=%s unchanged=%s chunks=%s embedded=%s",
        stats["files"],
        stats["unchanged"],
        stats["chunks"],
        stats["embedded"],
    )
    return stats


def query_index(
    text: str,
    db_path: Path,
    *,
    limit: int = 8,
    embed: EmbedFn | None = None,
) -> list[CodeHit]:
    if not db_path.exists() or not (text or "").strip():
        return []
    embed_fn = embed or (lambda texts: embed_texts(texts))
    query_vec = embed_fn([text.strip()])[0]
    conn = connect(db_path)
    rows = conn.execute(
        "SELECT repo, path, start_line, end_line, text, embedding FROM chunks"
    ).fetchall()
    conn.close()
    scored: list[CodeHit] = []
    for repo, path, start, end, body, blob in rows:
        scored.append(
            CodeHit(
                repo=repo,
                path=path,
                start_line=int(start),
                end_line=int(end),
                score=cosine(query_vec, unpack_vector(blob)),
                text=body,
            )
        )
    scored.sort(key=lambda hit: hit.score, reverse=True)
    return scored[:limit]


def gather_paths_for_intent(
    intent: str,
    db_path: Path | None = None,
    *,
    limit: int = 6,
    embed: EmbedFn | None = None,
) -> list[str]:
    """Unique workspace-relative paths for Cline read_file. Skips generic names."""
    skip = {"readme.md", "package.json", "license", "pyproject.toml"}
    hits = query_index(intent, db_path or default_index_path(), limit=limit * 3, embed=embed)
    out: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        rel = hit.path
        if Path(rel).name.lower() in skip:
            continue
        if rel in seen:
            continue
        seen.add(rel)
        out.append(rel)
        if len(out) >= limit:
            break
    return out
