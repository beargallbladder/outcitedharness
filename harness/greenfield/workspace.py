from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from harness.greenfield.manifest import safe_project_name


IGNORED_STATE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "coverage",
}


class WorkspaceError(RuntimeError):
    pass


def _git(root: Path, *argv: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise WorkspaceError(
            f"git {' '.join(argv)} failed: {(proc.stderr or proc.stdout).strip()[:500]}"
        )
    return proc.stdout.strip()


def full_tree_state_hash(root: Path) -> str:
    root = root.resolve()
    digest = hashlib.sha256()
    files: list[Path] = []
    for current, directories, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        retained = []
        for name in sorted(directories):
            path = current_path / name
            if name in IGNORED_STATE_DIRS:
                continue
            if path.is_symlink():
                raise WorkspaceError(
                    f"greenfield state cannot contain symlink: {path.relative_to(root)}"
                )
            retained.append(name)
        directories[:] = retained
        for name in sorted(names):
            path = current_path / name
            if path.is_symlink():
                raise WorkspaceError(
                    f"greenfield state cannot contain symlink: {path.relative_to(root)}"
                )
            files.append(path)
    for path in sorted(files):
        relative = path.relative_to(root)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_mode & 0o111).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class GreenfieldWorkspaceLease:
    run_id: str
    run_root: Path
    repo_root: Path

    @classmethod
    def acquire(cls, runs_root: Path, run_id: str) -> GreenfieldWorkspaceLease:
        safe_project_name(run_id)
        base = runs_root.expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        run_root = base / run_id
        repo_root = run_root / "repo"
        if run_root.exists():
            if (
                run_root.is_symlink()
                or repo_root.is_symlink()
                or not repo_root.is_dir()
                or run_root.resolve().parent != base
                or repo_root.resolve().parent != run_root.resolve()
            ):
                raise WorkspaceError("existing greenfield run directory is incomplete")
            return cls(run_id, run_root, repo_root)
        run_root.mkdir(mode=0o700)
        repo_root.mkdir(mode=0o700)
        _git(repo_root, "init", "-b", "main")
        _git(repo_root, "config", "user.name", "Harness Greenfield")
        _git(repo_root, "config", "user.email", "greenfield@harness.local")
        return cls(run_id, run_root, repo_root)

    def head(self) -> str | None:
        proc = subprocess.run(
            ["git", "-C", str(self.repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else None

    def dirty_paths(self) -> list[str]:
        output = _git(
            self.repo_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        return [line[3:] for line in output.splitlines() if len(line) >= 4]

    def commit(self, message: str, expected_state_hash: str) -> str:
        actual = full_tree_state_hash(self.repo_root)
        if actual != expected_state_hash:
            raise WorkspaceError("workspace changed after verification; refusing commit")
        _git(self.repo_root, "add", "--all")
        staged = _git(self.repo_root, "diff", "--cached", "--name-only")
        if not staged:
            raise WorkspaceError("verified milestone produced no committable change")
        _git(self.repo_root, "commit", "-m", message)
        return _git(self.repo_root, "rev-parse", "HEAD")

    def assert_clean(self) -> None:
        if self.dirty_paths():
            raise WorkspaceError("greenfield repository is not clean")
