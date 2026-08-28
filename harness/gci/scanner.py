from __future__ import annotations

import hashlib
import socket
import subprocess
from pathlib import Path

from harness.gci.indexer import snapshot_state_hash
from harness.gci.models import CodeDocument, RepoSnapshot
from harness.gci.storage import GCIStorageError
from harness.task.code_index import INCLUDE, MAX_FILE_BYTES, SKIP_DIRS


LANGUAGES = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
}


def _git(root: Path, *argv: str, binary: bool = False):
    proc = subprocess.run(
        ["git", "-C", str(root), *argv],
        capture_output=True,
        text=not binary,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace") if binary else proc.stderr
        raise GCIStorageError(f"git {' '.join(argv)} failed: {stderr.strip()[:300]}")
    return proc.stdout


def _git_optional(root: Path, *argv: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def canonical_approved_root(root: Path, approved_roots: list[str]) -> Path:
    resolved = root.expanduser().resolve()
    approved = {Path(item).expanduser().resolve() for item in approved_roots}
    if resolved not in approved:
        raise GCIStorageError(f"repository is not approved for GCI: {resolved}")
    if not (resolved / ".git").exists():
        _git(resolved, "rev-parse", "--show-toplevel")
    return resolved


def repo_id(source_host: str, root: Path) -> str:
    value = f"{source_host}\0{root.resolve()}".encode("utf-8", errors="replace")
    return hashlib.sha256(value).hexdigest()[:24]


def _eligible(rel: str) -> bool:
    path = Path(rel)
    return (
        path.suffix.lower() in INCLUDE
        and not any(part in SKIP_DIRS or part == "coverage" for part in path.parts)
    )


def build_snapshot(
    root: Path,
    *,
    approved_roots: list[str],
    previous_files: dict[str, str] | None = None,
    source_host: str | None = None,
) -> RepoSnapshot:
    root = canonical_approved_root(root, approved_roots)
    source_host = source_host or socket.gethostname()
    head_before = str(_git(root, "rev-parse", "HEAD")).strip()
    status_before = _git(root, "status", "--porcelain=v1", "-z", binary=True)
    raw_files = _git(
        root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        binary=True,
    )
    file_hashes: dict[str, str] = {}
    contents: dict[str, tuple[str, str]] = {}
    observed: dict[str, tuple[int, int]] = {}
    for raw in raw_files.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="strict")
        if not _eligible(rel):
            continue
        candidate = root / rel
        path = candidate.resolve()
        if candidate.is_symlink() or (root != path and root not in path.parents):
            raise GCIStorageError(f"unsafe indexed path: {rel}")
        stat = path.stat()
        if stat.st_size > MAX_FILE_BYTES:
            continue
        content = path.read_text(errors="replace")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        normalized = Path(rel).as_posix()
        file_hashes[normalized] = digest
        contents[normalized] = (content, LANGUAGES[Path(rel).suffix.lower()])
        observed[normalized] = (stat.st_size, stat.st_mtime_ns)
    head_after = str(_git(root, "rev-parse", "HEAD")).strip()
    status_after = _git(root, "status", "--porcelain=v1", "-z", binary=True)
    if head_before != head_after or status_before != status_after:
        raise GCIStorageError("repository changed while GCI snapshot was being assembled")
    for rel, before in observed.items():
        stat = (root / rel).stat()
        if (stat.st_size, stat.st_mtime_ns) != before:
            raise GCIStorageError(f"file changed while GCI snapshot was being assembled: {rel}")
    previous = previous_files or {}
    documents = tuple(
        CodeDocument(
            path=path,
            content=contents[path][0],
            content_hash=digest,
            language=contents[path][1],
        )
        for path, digest in sorted(file_hashes.items())
        if previous.get(path) != digest
    )
    remote = _git_optional(root, "remote", "get-url", "origin")
    branch = str(_git(root, "branch", "--show-current")).strip() or "(detached)"
    rid = repo_id(source_host, root)
    return RepoSnapshot(
        repo_id=rid,
        source_host=source_host,
        repo_root=str(root),
        remote=remote or None,
        branch=branch,
        head=head_after,
        dirty=bool(status_after),
        state_hash=snapshot_state_hash(head_after, file_hashes),
        file_hashes=file_hashes,
        documents=documents,
        deleted=tuple(sorted(set(previous) - set(file_hashes))),
    )
