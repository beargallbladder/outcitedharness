from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from harness.gci.scanner import build_snapshot
from harness.gci.storage import GCIStorageError


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    (root / "tracked.py").write_text("def tracked():\n    return 'tracked source'\n")
    (root / ".gitignore").write_text("ignored.py\nnode_modules/\n")
    _git(root, "add", "tracked.py", ".gitignore")
    _git(root, "commit", "-m", "seed")
    return root


def test_scanner_includes_dirty_and_untracked_but_not_ignored(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "tracked.py").write_text("def tracked():\n    return 'dirty source'\n")
    (root / "new.py").write_text("def new():\n    return 'untracked source'\n")
    (root / "ignored.py").write_text("SECRET = 'ignored'\n")
    snapshot = build_snapshot(
        root,
        approved_roots=[str(root)],
        previous_files={},
        source_host="m5",
    )
    assert snapshot.dirty
    assert set(snapshot.file_hashes) == {"new.py", "tracked.py"}
    assert len(snapshot.documents) == 2
    assert snapshot.source_host == "m5"


def test_scanner_emits_only_delta_and_deletions(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "delete.py").write_text("def deleted():\n    return True\n")
    first = build_snapshot(root, approved_roots=[str(root)], source_host="m5")
    previous = first.file_hashes
    (root / "delete.py").unlink()
    (root / "tracked.py").write_text("def tracked():\n    return 'changed'\n")
    second = build_snapshot(
        root,
        approved_roots=[str(root)],
        previous_files=previous,
        source_host="m5",
    )
    assert [row.path for row in second.documents] == ["tracked.py"]
    assert second.deleted == ("delete.py",)


def test_scanner_rejects_unapproved_root_and_symlink(tmp_path: Path):
    root = _repo(tmp_path)
    with pytest.raises(GCIStorageError, match="not approved"):
        build_snapshot(root, approved_roots=[], source_host="m5")
    outside = tmp_path / "outside.py"
    outside.write_text("def outside():\n    return True\n")
    os.symlink(outside, root / "escape.py")
    with pytest.raises(GCIStorageError, match="unsafe indexed path"):
        build_snapshot(root, approved_roots=[str(root)], source_host="m5")
