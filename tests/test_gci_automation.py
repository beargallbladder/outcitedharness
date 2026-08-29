from __future__ import annotations

import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from harness.config import GCIRefreshPolicy, Settings, load_config
from harness.gci.automation import (
    AutomationBusyError,
    AutomationState,
    RepositoryProbe,
    automation_lock,
    probe_local_git,
    refresh_after_publication,
    refresh_repository,
    run_automation,
)


class FakeClient:
    def __init__(self):
        self.manifest_calls = 0
        self.submit_calls = 0
        self.snapshots = []
        self.files: dict[str, dict[str, str]] = {}
        self.states: dict[str, str] = {}

    def manifest(self, repo_id: str) -> dict:
        self.manifest_calls += 1
        return {
            "files": dict(self.files.get(repo_id, {})),
            "state_hash": self.states.get(repo_id),
        }

    def submit(self, snapshot, *, refresh: bool) -> str:
        self.submit_calls += 1
        self.snapshots.append((snapshot, refresh))
        self.files[snapshot.repo_id] = dict(snapshot.file_hashes)
        self.states[snapshot.repo_id] = snapshot.state_hash
        return f"job-{self.submit_calls}"

    def wait_job(self, job_id: str, *, timeout: float = 300.0) -> dict:
        return {"job_id": job_id, "state": "complete"}


def _repo(tmp_path: Path, name: str = "repo") -> Path:
    root = tmp_path / name
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Test"],
        check=True,
    )
    (root / "app.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(root), "add", "app.py"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-qm", "initial"],
        check=True,
    )
    return root


def _settings(tmp_path: Path, roots: list[Path]) -> Settings:
    return Settings(
        results_dir=tmp_path / "results",
        db_path=tmp_path / "harness.sqlite",
        code_index_repos=[str(root) for root in roots],
        gci_refresh=GCIRefreshPolicy(
            enabled=True,
            active_interval_seconds=300,
            stale_after_days=30,
            stale_interval_seconds=86_400,
            failure_retry_seconds=900,
            jitter_seconds=0,
            state_path=tmp_path / "refresh.sqlite",
        ),
    )


def test_repository_sources_are_owned_local_allowlist(tmp_path: Path):
    root = _repo(tmp_path)
    settings = _settings(tmp_path, [root])
    assert len(settings.gci_repository_sources) == 1
    source = settings.gci_repository_sources[0]
    assert source.type == "local_git"
    assert source.owner == "self"
    assert source.path == str(root)


def test_project_refresh_policy_loads_from_yaml():
    policy = load_config().settings.gci_refresh
    assert policy.enabled is True
    assert policy.active_interval_seconds == 300
    assert policy.stale_after_days == 30
    assert policy.stale_interval_seconds == 86_400
    assert policy.state_path == Path("~/.harness/gci-refresh.sqlite").expanduser()


def test_probe_detects_repeated_edits_while_git_status_stays_dirty(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "app.py").write_text("VALUE = 2\n")
    first = probe_local_git(root)
    (root / "app.py").write_text("VALUE = 3\n")
    second = probe_local_git(root)
    assert first.dirty and second.dirty
    assert first.fingerprint != second.fingerprint


def test_unchanged_due_pass_makes_no_gci_request(tmp_path: Path):
    root = _repo(tmp_path)
    settings = _settings(tmp_path, [root])
    client = FakeClient()
    now = time.time()

    first = run_automation(
        settings,
        client,
        now=now,
        source_host="test-host",
    )
    assert first.outcomes[0].status == "complete"
    assert client.manifest_calls == 1
    assert client.submit_calls == 1

    second = run_automation(
        settings,
        client,
        now=now + 300,
        source_host="test-host",
    )
    assert second.outcomes[0].status == "unchanged"
    assert client.manifest_calls == 1
    assert client.submit_calls == 1


def test_changed_file_and_deletion_submit_only_delta(tmp_path: Path):
    root = _repo(tmp_path)
    (root / "delete.py").write_text("DELETE_ME = True\n")
    subprocess.run(["git", "-C", str(root), "add", "delete.py"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-qm", "add deletion fixture"],
        check=True,
    )
    settings = _settings(tmp_path, [root])
    client = FakeClient()
    now = time.time()
    run_automation(settings, client, now=now, source_host="test-host")

    (root / "app.py").write_text("VALUE = 2\n")
    (root / "delete.py").unlink()
    changed = run_automation(
        settings,
        client,
        now=now + 300,
        source_host="test-host",
    )
    snapshot = client.snapshots[-1][0]
    assert changed.outcomes[0].changed_documents == 1
    assert [document.path for document in snapshot.documents] == ["app.py"]
    assert snapshot.deleted == ("delete.py",)


def test_stale_repository_backs_off_but_remains_registered(tmp_path: Path):
    root = _repo(tmp_path)
    settings = _settings(tmp_path, [root])
    client = FakeClient()
    clock = 4_000_000.0
    stale_probe = lambda _root: RepositoryProbe(
        fingerprint="stable",
        head="abc",
        dirty=False,
        last_commit_at=1.0,
    )
    run_automation(
        settings,
        client,
        now=clock,
        source_host="test-host",
        probe=stale_probe,
    )
    row = AutomationState(settings.gci_refresh.state_path).rows()[0]
    assert row["next_due"] == clock + 86_400
    assert row["last_status"] == "complete"


def test_failure_isolated_and_new_configured_repo_is_discovered(tmp_path: Path):
    first = _repo(tmp_path, "first")
    second = _repo(tmp_path, "second")
    missing = tmp_path / "missing"
    settings = _settings(tmp_path, [missing, first])
    client = FakeClient()
    now = time.time()

    run = run_automation(
        settings,
        client,
        now=now,
        source_host="test-host",
    )
    assert [row.status for row in run.outcomes] == ["failed", "complete"]
    missing_row = next(
        row
        for row in AutomationState(settings.gci_refresh.state_path).rows()
        if row["repo_root"] == str(missing)
    )
    assert missing_row["failure_count"] == 1
    assert missing_row["next_due"] == now + 900

    settings.code_index_repos.append(str(second))
    added = run_automation(
        settings,
        client,
        now=now + 300,
        source_host="test-host",
    )
    assert any(row.root == str(second) and row.status == "complete" for row in added.outcomes)


def test_lock_rejects_overlapping_pass(tmp_path: Path):
    state_path = tmp_path / "refresh.sqlite"
    with automation_lock(state_path):
        with pytest.raises(AutomationBusyError):
            with automation_lock(state_path):
                pass


def test_state_schema_is_versioned_and_rejects_future_database(tmp_path: Path):
    path = tmp_path / "future.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', '99')"
        )
    with pytest.raises(RuntimeError, match="unsupported GCI refresh state schema"):
        AutomationState(path)


def test_publication_refresh_requires_exact_approved_root(tmp_path: Path):
    root = _repo(tmp_path)
    settings = _settings(tmp_path, [root])
    client = FakeClient()
    assert refresh_after_publication(settings, client, tmp_path / "other") is None
    outcome = refresh_after_publication(settings, client, root)
    assert outcome is not None
    assert outcome.status == "queued"
