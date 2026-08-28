from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import harness.checkpoints as checkpoint_module
from harness.checkpoints import (
    CheckpointError,
    CheckpointStore,
    RollbackConflict,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "checkpoint@test")
    _git(root, "config", "user.name", "Checkpoint Test")
    (root / "foo.py").write_text("FOO = 'committed'\n")
    (root / "bar.py").write_text("BAR = 'committed'\n")
    (root / "unrelated.py").write_text("UNRELATED = 'committed'\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "seed")
    return root


def test_rollback_restores_dirty_untracked_deleted_and_created_files(tmp_path: Path):
    root = _repo(tmp_path)
    store = CheckpointStore(tmp_path / "checkpoints")
    (root / "foo.py").write_text("FOO = 'user dirty'\n")
    os.chmod(root / "foo.py", 0o744)
    (root / "existing_untracked.py").write_text("USER_UNTRACKED = True\n")
    (root / "unrelated.py").write_text("UNRELATED = 'keep user work'\n")

    baseline = store.capture_baseline(
        task_id="task-1",
        run_id="run-1",
        repo_root=root,
        intent="edit task files",
        paths=["foo.py", "bar.py", "existing_untracked.py", "new/deep/baz.py"],
        before_iteration=1,
    )
    assert baseline.files["foo.py"].tracked is True
    assert baseline.files["foo.py"].dirty is True
    assert baseline.files["foo.py"].status == "M"
    assert baseline.files["existing_untracked.py"].tracked is False
    assert baseline.files["existing_untracked.py"].untracked is True
    assert baseline.files["existing_untracked.py"].status == "??"
    assert baseline.files["new/deep/baz.py"].exists is False
    assert baseline.head

    (root / "foo.py").write_text("FOO = 'agent'\n")
    os.chmod(root / "foo.py", 0o600)
    (root / "bar.py").unlink()
    (root / "existing_untracked.py").write_text("USER_UNTRACKED = False\n")
    (root / "new/deep").mkdir(parents=True)
    (root / "new/deep/baz.py").write_text("BAZ = 'agent created'\n")
    store.record_checkpoint(
        task_id="task-1",
        run_id="run-1",
        number=1,
    )

    preview = store.rollback_preview("task-1", "run-1")
    assert preview.conflicts == ()
    assert set(preview.restore) == {"foo.py", "bar.py", "existing_untracked.py"}
    assert preview.remove == ("new/deep/baz.py",)

    result = store.rollback("task-1", "run-1")
    assert (root / "foo.py").read_text() == "FOO = 'user dirty'\n"
    assert (root / "foo.py").stat().st_mode & 0o777 == 0o744
    assert (root / "bar.py").read_text() == "BAR = 'committed'\n"
    assert (root / "existing_untracked.py").read_text() == "USER_UNTRACKED = True\n"
    assert not (root / "new/deep/baz.py").exists()
    assert not (root / "new").exists()
    assert (root / "unrelated.py").read_text() == "UNRELATED = 'keep user work'\n"
    assert Path(result.audit_path).exists()


def test_rollback_conflict_refuses_all_paths(tmp_path: Path):
    root = _repo(tmp_path)
    store = CheckpointStore(tmp_path / "checkpoints")
    store.capture_baseline(
        task_id="task-2",
        run_id="run-2",
        repo_root=root,
        intent="change two files",
        paths=["foo.py", "bar.py"],
        before_iteration=1,
    )
    (root / "foo.py").write_text("FOO = 'agent'\n")
    (root / "bar.py").write_text("BAR = 'agent'\n")
    store.record_checkpoint(
        task_id="task-2",
        run_id="run-2",
        number=1,
    )
    (root / "foo.py").write_text("FOO = 'user after agent'\n")

    with pytest.raises(RollbackConflict) as caught:
        store.rollback("task-2", "run-2")
    assert caught.value.paths == ["foo.py"]
    assert (root / "foo.py").read_text() == "FOO = 'user after agent'\n"
    assert (root / "bar.py").read_text() == "BAR = 'agent'\n"


def test_checkpoint_rejects_symlink_escape_and_oversized_files(tmp_path: Path):
    root = _repo(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = True\n")
    (root / "link.py").symlink_to(outside)
    (root / "large.py").write_bytes(b"x" * 65)
    store = CheckpointStore(tmp_path / "checkpoints", max_file_bytes=64)

    with pytest.raises(CheckpointError, match="symlink"):
        store.capture_baseline(
            task_id="task-3",
            run_id="run-3",
            repo_root=root,
            intent="unsafe",
            paths=["link.py"],
            before_iteration=1,
        )
    with pytest.raises(CheckpointError, match="exceeds"):
        store.capture_baseline(
            task_id="task-3",
            run_id="run-3",
            repo_root=root,
            intent="too large",
            paths=["large.py"],
            before_iteration=1,
        )
    with pytest.raises(CheckpointError, match="unsafe"):
        store.capture_baseline(
            task_id="task-3",
            run_id="run-3",
            repo_root=root,
            intent="escape",
            paths=["../outside.py"],
            before_iteration=1,
        )
    assert outside.read_text() == "SECRET = True\n"


def test_content_blobs_are_deduplicated_across_checkpoints(tmp_path: Path):
    root = _repo(tmp_path)
    store = CheckpointStore(tmp_path / "checkpoints")
    store.capture_baseline(
        task_id="task-4",
        run_id="run-4",
        repo_root=root,
        intent="no content change",
        paths=["foo.py"],
        before_iteration=1,
    )
    store.record_checkpoint(
        task_id="task-4",
        run_id="run-4",
        number=1,
    )
    blobs = list((tmp_path / "checkpoints/task-4/run-4/blobs").iterdir())
    assert len(blobs) == 1


def test_checkpoint_state_hash_tracks_untracked_creation_content_and_deletion(
    tmp_path: Path,
):
    root = _repo(tmp_path)
    store = CheckpointStore(tmp_path / "checkpoints")
    store.capture_baseline(
        task_id="task-5",
        run_id="run-5",
        repo_root=root,
        intent="create generated file",
        paths=["generated.py"],
        before_iteration=1,
    )
    (root / "generated.py").write_text("VALUE = 1\n")
    first = store.record_checkpoint(task_id="task-5", run_id="run-5", number=1)
    (root / "generated.py").write_text("VALUE = 2\n")
    second = store.record_checkpoint(task_id="task-5", run_id="run-5", number=2)
    (root / "generated.py").unlink()
    third = store.record_checkpoint(task_id="task-5", run_id="run-5", number=3)

    assert len({first.active_diff_hash, second.active_diff_hash, third.active_diff_hash}) == 3
    assert first.files["generated.py"].exists is True
    assert third.files["generated.py"].exists is False


def test_rollback_io_failure_compensates_already_restored_paths(
    tmp_path: Path,
    monkeypatch,
):
    root = _repo(tmp_path)
    store = CheckpointStore(tmp_path / "checkpoints")
    store.capture_baseline(
        task_id="task-6",
        run_id="run-6",
        repo_root=root,
        intent="change two files",
        paths=["foo.py", "bar.py"],
        before_iteration=1,
    )
    (root / "foo.py").write_text("FOO = 'agent'\n")
    (root / "bar.py").write_text("BAR = 'agent'\n")
    store.record_checkpoint(task_id="task-6", run_id="run-6", number=1)

    original_atomic_write = checkpoint_module._atomic_write

    def fail_on_foo(path: Path, data: bytes, mode: int | None = None):
        if path == root / "foo.py" and data == b"FOO = 'committed'\n":
            raise OSError("simulated disk failure")
        return original_atomic_write(path, data, mode)

    monkeypatch.setattr(checkpoint_module, "_atomic_write", fail_on_foo)
    with pytest.raises(CheckpointError, match="simulated disk failure"):
        store.rollback("task-6", "run-6")

    assert (root / "foo.py").read_text() == "FOO = 'agent'\n"
    assert (root / "bar.py").read_text() == "BAR = 'agent'\n"
    attempts = list(
        (tmp_path / "checkpoints/task-6/run-6/rollback-attempts").glob("*.json")
    )
    assert len(attempts) == 1


def test_rollback_audit_failure_compensates_files_and_created_directories(
    tmp_path: Path,
    monkeypatch,
):
    root = _repo(tmp_path)
    store = CheckpointStore(tmp_path / "checkpoints")
    store.capture_baseline(
        task_id="task-7",
        run_id="run-7",
        repo_root=root,
        intent="change and create",
        paths=["foo.py", "new/deep/created.py"],
        before_iteration=1,
    )
    (root / "foo.py").write_text("FOO = 'agent'\n")
    (root / "new/deep").mkdir(parents=True)
    (root / "new/deep/created.py").write_text("CREATED = True\n")
    store.record_checkpoint(task_id="task-7", run_id="run-7", number=1)

    original_atomic_write = checkpoint_module._atomic_write

    def fail_on_audit(path: Path, data: bytes, mode: int | None = None):
        if path.name == "rollback.json":
            raise OSError("simulated audit failure")
        return original_atomic_write(path, data, mode)

    monkeypatch.setattr(checkpoint_module, "_atomic_write", fail_on_audit)
    with pytest.raises(CheckpointError, match="simulated audit failure"):
        store.rollback("task-7", "run-7")

    assert (root / "foo.py").read_text() == "FOO = 'agent'\n"
    assert (root / "new/deep/created.py").read_text() == "CREATED = True\n"
