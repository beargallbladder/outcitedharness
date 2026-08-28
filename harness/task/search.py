from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Mode = Literal["auto", "grep", "ast", "semantic", "hybrid"]

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_STRUCT = re.compile(r"\b(class|def|function|interface|type|import)\b", re.I)


@dataclass
class SearchHit:
    path: str
    line: int
    text: str
    backend: str


@dataclass
class SearchResult:
    query: str
    mode: str
    backend: str
    hits: list[SearchHit]
    detail: str = ""


def choose_backend(query: str, mode: Mode) -> str:
    if mode == "auto":
        if _IDENT.fullmatch(query.strip()):
            return "grep"
        if _STRUCT.search(query):
            return "ast"
        return "hybrid"
    return mode


def search_code(query: str, repo: Path, mode: Mode = "auto", limit: int = 40) -> SearchResult:
    backend = choose_backend(query, mode)
    if backend == "semantic":
        return semantic_search(query, limit=limit)
    if backend == "hybrid":
        grep = _ripgrep(query, repo, mode, "grep", limit)
        semantic = semantic_search(query, limit=max(4, limit // 2))
        hits = list(grep.hits[: max(1, limit // 2)]) + list(semantic.hits)
        return SearchResult(
            query=query,
            mode=mode,
            backend="hybrid",
            hits=hits[:limit],
            detail="; ".join(part for part in (grep.detail, semantic.detail) if part),
        )
    if backend == "ast":
        if shutil.which("ast-grep"):
            return _ast_grep(query, repo, mode, limit)
        return SearchResult(
            query=query,
            mode=mode,
            backend="ast",
            hits=[],
            detail="ast-grep unavailable",
        )
    return _ripgrep(query, repo, mode, backend, limit)


def embed_texts(
    texts: list[str],
    base_url: str | None = None,
    model: str = "bge-m3-cr-tapes-v1",
) -> list[list[float]]:
    """Encode text on Spark :8800. Does not write to the CR category index."""
    if not texts:
        return []
    root = (base_url or embedder_base_url()).rstrip("/")
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        f"{root}/v1/embeddings",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8") or "{}")
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) != len(texts):
        raise RuntimeError("embedder returned unexpected embeddings payload")
    ordered = sorted(rows, key=lambda row: int(row.get("index") or 0))
    out: list[list[float]] = []
    for row in ordered:
        vec = row.get("embedding") if isinstance(row, dict) else None
        if not isinstance(vec, list):
            raise RuntimeError("embedder row missing embedding")
        out.append([float(x) for x in vec])
    return out


def embedder_base_url() -> str:
    return (
        os.environ.get("HARNESS_EMBED_URL")
        or "http://100.81.201.24:8800"
    ).rstrip("/")


def semantic_search(query: str, limit: int = 8, base_url: str | None = None) -> SearchResult:
    """Query Spark :8800. This is the CR category index, not a workspace file index."""
    root = (base_url or embedder_base_url()).rstrip("/")
    payload = json.dumps({"query": query, "top_k": limit}).encode("utf-8")
    request = urllib.request.Request(
        f"{root}/semantic-search",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8") or "{}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return SearchResult(
            query=query,
            mode="semantic",
            backend="semantic",
            hits=[],
            detail=f"semantic error: {type(exc).__name__}",
        )
    hits: list[SearchHit] = []
    rows = data.get("results") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        hits.append(
            SearchHit(
                path=str(row.get("source_artifact") or row.get("kim_slug") or ""),
                line=int(row.get("rank") or 0),
                text=str(row.get("text") or ""),
                backend="semantic",
            )
        )
        if len(hits) >= limit:
            break
    return SearchResult(
        query=query,
        mode="semantic",
        backend="semantic",
        hits=hits,
        detail="spark :8800 category index",
    )


_CATEGORY_INTENT = re.compile(
    r"\b(category|keyword|kim_slug|encoder|cpc|brand domain|semantic search)\b",
    re.I,
)


def embedder_thread_block(intent: str) -> str:
    """Attach labeled CR hits. Never present them as workspace files."""
    if not _CATEGORY_INTENT.search(intent or ""):
        return ""
    result = semantic_search(intent, limit=5)
    if not result.hits:
        return result.detail and f"CR EMBEDDER: {result.detail}" or ""
    lines = [
        "CR EMBEDDER HITS (spark :8800 category index, not workspace source):",
    ]
    for hit in result.hits:
        lines.append(f"- {hit.text} artifact={hit.path}")
    return "\n".join(lines)


def _ripgrep(query: str, repo: Path, mode: str, backend: str, limit: int) -> SearchResult:
    root = repo.resolve()
    rg = shutil.which("rg")
    if rg:
        proc = subprocess.run(
            [rg, "--line-number", "--no-heading", "--color", "never", "-m", str(limit), query, str(root)],
            capture_output=True,
            text=True,
            check=False,
        )
        hits = _parse_rg(proc.stdout, backend)
        return SearchResult(query=query, mode=mode, backend=backend, hits=hits[:limit], detail="")
    hits = _python_scan(query, root, backend, limit)
    return SearchResult(
        query=query,
        mode=mode,
        backend=backend,
        hits=hits,
        detail="rg unavailable; used python scan",
    )


def _ast_grep(query: str, repo: Path, mode: str, limit: int) -> SearchResult:
    proc = subprocess.run(
        ["ast-grep", "run", "--pattern", query, "--json=stream", str(repo.resolve())],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return SearchResult(
            query=query,
            mode=mode,
            backend="ast",
            hits=[],
            detail=(proc.stderr or proc.stdout or "ast-grep failed")[:200],
        )
    hits: list[SearchHit] = []
    for line in proc.stdout.splitlines():
        if len(hits) >= limit:
            break
        if not line.strip():
            continue
        hits.append(SearchHit(path="(ast-grep)", line=0, text=line[:200], backend="ast"))
    return SearchResult(query=query, mode=mode, backend="ast", hits=hits)


def _parse_rg(stdout: str, backend: str) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for raw in stdout.splitlines():
        path, _, rest = raw.partition(":")
        line_s, _, text = rest.partition(":")
        try:
            line = int(line_s)
        except ValueError:
            continue
        hits.append(SearchHit(path=path, line=line, text=text, backend=backend))
    return hits


def _python_scan(query: str, root: Path, backend: str, limit: int) -> list[SearchHit]:
    hits: list[SearchHit] = []
    skip = {".git", ".venv", "node_modules", "__pycache__"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip for part in path.parts):
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for idx, line in enumerate(lines, 1):
            if query in line:
                hits.append(
                    SearchHit(path=str(path.relative_to(root)), line=idx, text=line.strip(), backend=backend)
                )
                if len(hits) >= limit:
                    return hits
    return hits
