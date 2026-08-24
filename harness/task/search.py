from __future__ import annotations

import re
import shutil
import subprocess
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
        return SearchResult(
            query=query,
            mode=mode,
            backend="semantic",
            hits=[],
            detail="semantic unavailable (BGE-M3 worker not wired into search yet)",
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
