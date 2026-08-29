from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from harness.greenfield.models import GreenfieldManifest
from harness.greenfield.package_policy import (
    PackageAction,
    PackagePolicyError,
    validate_package_action,
)
from harness.greenfield.publish import PublishError, publish_verified_run
from harness.greenfield.workspace import (
    GreenfieldWorkspaceLease,
    WorkspaceError,
    full_tree_state_hash,
)


def _manifest(destination: Path, dependencies: tuple[str, ...] = ()) -> GreenfieldManifest:
    return GreenfieldManifest(
        run_id="gf-test",
        project_name="sample",
        stack="python",
        runtime="python>=3.11",
        package_manager="uv",
        approved_dependencies=dependencies,
        destination=str(destination),
        destination_fingerprint="fingerprint",
        spec_hash="spec",
        plan_hash="plan",
        discovery_hash="discovery",
    )


def test_workspace_lease_is_isolated_and_commit_checks_exact_state(tmp_path: Path):
    lease = GreenfieldWorkspaceLease.acquire(tmp_path / "runs", "gf-test")
    assert lease.repo_root == (tmp_path / "runs" / "gf-test" / "repo").resolve()
    (lease.repo_root / "app.py").write_text("VALUE = 1\n")
    verified = full_tree_state_hash(lease.repo_root)
    (lease.repo_root / "app.py").write_text("VALUE = 2\n")
    with pytest.raises(WorkspaceError, match="changed after verification"):
        lease.commit("harness(m0): test", verified)
    current = full_tree_state_hash(lease.repo_root)
    commit = lease.commit("harness(m0): test", current)
    assert len(commit) == 40
    lease.assert_clean()


def test_tree_hash_includes_untracked_and_rejects_symlinks(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "one.py").write_text("one\n")
    before = full_tree_state_hash(root)
    (root / "new.py").write_text("new\n")
    assert full_tree_state_hash(root) != before
    (root / "escape.py").symlink_to(tmp_path / "outside.py")
    with pytest.raises(WorkspaceError, match="symlink"):
        full_tree_state_hash(root)


def test_package_policy_reads_manifest_and_rejects_arbitrary_actions(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr("harness.greenfield.package_policy.shutil.which", lambda _name: "/bin/tool")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        """[project]
name = "sample"
version = "0.1.0"
dependencies = ["fastapi"]

[project.optional-dependencies]
dev = ["pytest", "ruff"]
"""
    )
    valid = PackageAction("uv", ("uv", "sync", "--extra", "dev"), root)
    validate_package_action(valid, repo_root=root, approved_dependencies=("fastapi",))
    with pytest.raises(PackagePolicyError, match="unapproved"):
        validate_package_action(valid, repo_root=root, approved_dependencies=())
    with pytest.raises(PackagePolicyError):
        validate_package_action(
            PackageAction("uv", ("uv", "add", "requests"), root),
            repo_root=root,
            approved_dependencies=("requests",),
        )


def test_publish_refuses_noncomplete_run():
    from types import SimpleNamespace

    with pytest.raises(PublishError, match="complete"):
        publish_verified_run(SimpleNamespace(status="running", final_state_hash=None))
