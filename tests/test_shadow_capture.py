from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.shadow.hook import capture_hook_event
from harness.shadow.__main__ import hook_command
from harness.shadow.models import (
    HookRecord,
    ShadowAttempt,
    ShadowPolicy,
    canonical_json,
)
from harness.shadow.objects import ShadowObjectStore
from harness.shadow.policy import (
    canonical_relative_path,
    load_policy,
    path_allowed,
    sanitize_payload,
    source_allowed,
)
from harness.shadow.repository import capture_repository_snapshot
from harness.shadow.spool import ShadowSpool
from harness.shadow.workspace import (
    ShadowWorkspace,
    materialize_snapshot,
    transition_to_snapshot,
)


def _run(root: Path, *argv: str) -> str:
    result = subprocess.run(
        list(argv),
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _run(root, "git", "init", "-q")
    (root / "app.py").write_text("value = 1\n")
    _run(root, "git", "add", "app.py")
    _run(
        root,
        "git",
        "-c",
        "user.name=Harness Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "base",
    )
    return root


def _policy(root: Path) -> ShadowPolicy:
    value = {
        "version": 1,
        "enabled": True,
        "repository_id": "owner/example",
        "owner": "self",
        "data_use": "shadow_learning",
        "authorization_scope": "owned_repository_cursor_shadow",
        "teacher_model": "gpt-5.6-sol-max-fast",
        "local_model_key": "asus2_qwen",
        "allowed_paths": ["."],
        "excluded_paths": [
            ".env",
            ".git",
            ".harness-shadow",
            ".harness-shadow.json",
            "**/.git/**",
        ],
        "max_agent_turns": 4,
    }
    (root / ".harness-shadow.json").write_text(json.dumps(value))
    return ShadowPolicy.model_validate(value)


def test_policy_is_explicit_and_paths_are_repository_scoped(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    assert load_policy(root) is None
    policy = _policy(root)
    assert load_policy(root) == policy
    assert path_allowed(policy, "src/app.py")
    assert not path_allowed(policy, ".env")
    assert not path_allowed(policy, ".git/config")
    assert canonical_relative_path(root, "src/new.py") == "src/new.py"
    with pytest.raises(ValueError, match="escapes"):
        canonical_relative_path(root, "../outside")


def test_owned_code_scope_allows_product_names_but_general_scope_does_not(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    policy = _policy(root)
    prompt = "Repair the CategoryRank application code without reading datasets."

    assert source_allowed(policy, prompt)
    assert not source_allowed(
        policy.model_copy(update={"authorization_scope": "general"}),
        prompt,
    )


def test_payload_sanitizer_removes_secret_fields_without_losing_usage() -> None:
    value = sanitize_payload(
        {
            "authorization": "Bearer not-for-storage",
            "api_key": "not-for-storage",
            "usage": {"completion_tokens": 17},
        }
    )
    assert value["authorization"] == "redacted"
    assert value["api_key"] == "redacted"
    assert value["usage"]["completion_tokens"] == 17


def test_snapshot_is_hash_bound_and_excludes_denied_files(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    policy = _policy(root)
    (root / "app.py").write_text("value = 2\n")
    (root / "new.py").write_text("new_value = 3\n")
    (root / ".env").write_text("PASSWORD=never-capture-this\n")
    store = ShadowObjectStore(tmp_path / "spool")

    snapshot = capture_repository_snapshot(root, policy, store)

    assert snapshot.dirty_patch_sha256 == hashlib.sha256(
        snapshot.dirty_patch.encode()
    ).hexdigest()
    assert "value = 2" in snapshot.dirty_patch
    assert [item.path for item in snapshot.untracked_files] == ["new.py"]
    assert snapshot.omitted_path_count == 2
    assert ".env" not in canonical_json(snapshot).decode()


def test_snapshot_omits_secret_bearing_learning_material(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    policy = _policy(root)
    fake_secret = "ghp_" + ("a" * 32)
    (root / "app.py").write_text(f"token = '{fake_secret}'\n")
    snapshot = capture_repository_snapshot(
        root,
        policy,
        ShadowObjectStore(tmp_path / "spool"),
    )
    assert snapshot.dirty_patch == ""
    assert snapshot.omitted_path_count == 2
    assert fake_secret not in canonical_json(snapshot).decode()


def test_prompt_hook_enqueues_immutable_task_and_links_later_event(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    _policy(root)
    spool_root = tmp_path / "spool"
    payload = {
        "session_id": "session-1",
        "generation_id": "generation-1",
        "prompt": "Change value to two.",
        "authorization": "Bearer not-for-storage",
    }

    task_id = capture_hook_event(
        "beforeSubmitPrompt",
        payload,
        repository_root=root,
        spool_root=spool_root,
    )
    assert task_id
    assert capture_hook_event(
        "afterFileEdit",
        {
            "session_id": "session-1",
            "file_path": str(root / "app.py"),
            "edits": [{"old_string": "value = 1", "new_string": "value = 2"}],
        },
        repository_root=root,
        spool_root=spool_root,
    ) == task_id
    assert (
        capture_hook_event(
            "beforeReadFile",
            {
                "session_id": "session-1",
                "file_path": str(root / ".env"),
                "content": "PASSWORD=do-not-store",
            },
            repository_root=root,
            spool_root=spool_root,
        )
        is None
    )
    assert capture_hook_event(
        "beforeReadFile",
        {
            "session_id": "session-1",
            "file_path": str(root / "app.py"),
            "content": "value = 1\n",
            "transcript_path": "/private/cursor/transcript.jsonl",
        },
        repository_root=root,
        spool_root=spool_root,
    ) == task_id
    assert capture_hook_event(
        "stop",
        {"session_id": "session-1", "status": "completed", "loop_count": 1},
        repository_root=root,
        spool_root=spool_root,
    ) == task_id

    spool = ShadowSpool(spool_root)
    task, state = spool.get_task(task_id)
    assert task.prompt == "Change value to two."
    assert state["status"] == "queued"
    assert "not-for-storage" not in spool.database.read_bytes().decode(
        errors="ignore"
    )
    with sqlite3.connect(spool.database) as connection:
        payloads = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT payload_json FROM shadow_hook_events ORDER BY created_at"
            )
        ]
        assert len(payloads) == 4
        read_payload = next(row for row in payloads if "content_sha256" in row)
        assert "content" not in read_payload
        assert "transcript_path" not in read_payload
        edit_payload = next(row for row in payloads if "edits_sha256" in row)
        assert "edits" not in edit_payload
        assert any("repository_state" in row for row in payloads)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE shadow_hook_events SET event_type = 'changed'"
            )


def test_spool_claim_retry_and_terminal_attempt(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _policy(root)
    spool_root = tmp_path / "spool"
    task_id = capture_hook_event(
        "beforeSubmitPrompt",
        {
            "session_id": "session-1",
            "generation_id": "generation-1",
            "prompt": "<TASK_KIND>shadow</TASK_KIND>\nExplain app.py.",
        },
        repository_root=root,
        spool_root=spool_root,
    )
    assert task_id
    spool = ShadowSpool(spool_root)
    lease = spool.claim()
    assert lease and lease.task.task_id == task_id
    assert spool.fail(lease, "temporary endpoint outage", retry_delay_seconds=0) == "queued"
    retry = spool.claim()
    assert retry and retry.attempt == 2
    attempt = ShadowAttempt(
        attempt_id=f"attempt-{task_id}-2",
        task_id=task_id,
        status="completed",
        model="qwen-local",
        model_endpoint_sha256="1" * 64,
        answer="The file defines one value.",
        created_at=datetime.now(timezone.utc),
    )
    spool.complete(retry, attempt)
    assert spool.status() == {"completed": 1}
    assert spool.get_attempt(task_id) == attempt


def test_conversational_prompt_is_not_sent_to_code_shadow(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _policy(root)
    spool_root = tmp_path / "spool"

    task_id = capture_hook_event(
        "beforeSubmitPrompt",
        {
            "session_id": "session-conversation",
            "generation_id": "generation-conversation",
            "prompt": "Why is this taking so long?",
        },
        repository_root=root,
        spool_root=spool_root,
    )

    assert task_id is None
    spool = ShadowSpool(spool_root)
    assert spool.status() == {}
    with spool.connect() as connection:
        row = connection.execute(
            "SELECT payload_json FROM shadow_hook_events"
        ).fetchone()
    assert json.loads(row["payload_json"])["shadow_disposition"] == (
        "ignored_non_actionable"
    )


def test_snapshot_materializes_only_inside_disposable_workspace(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    policy = _policy(root)
    (root / "app.py").write_text("value = 2\n")
    (root / "new.py").write_text("new_value = 3\n")
    store = ShadowObjectStore(tmp_path / "spool")
    snapshot = capture_repository_snapshot(root, policy, store)

    isolated = materialize_snapshot(
        snapshot,
        policy,
        work_root=tmp_path / "work",
        object_store=store,
        workspace_id="attempt-one",
    )
    workspace = ShadowWorkspace(isolated, policy)
    assert "value = 2" in workspace.read_file("app.py")
    assert "new_value = 3" in workspace.read_file("new.py")
    assert workspace.diff() == ""
    assert (
        workspace.apply_patch(
            """diff --git a/app.py b/app.py
index 7c02ec3..7fe2e53 100644
--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-value = 2
+value = 4
"""
        )
        == "PATCH_APPLIED"
    )
    assert "value = 4" in workspace.diff()
    assert "new_value = 3" not in workspace.diff()
    assert (root / "app.py").read_text() == "value = 2\n"


def test_hook_record_digest_is_enforced(tmp_path: Path) -> None:
    spool = ShadowSpool(tmp_path / "spool")
    with pytest.raises(ValueError, match="digest"):
        spool.append_hook(
            HookRecord(
                event_id="hook-one",
                correlation_id="cursor:one",
                event_type="afterFileEdit",
                payload={"path": "app.py"},
                payload_sha256="0" * 64,
                created_at=datetime.now(timezone.utc),
            )
        )


def test_snapshot_transition_derives_only_frontier_delta(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    policy = _policy(root)
    (root / "app.py").write_text("value = 2\n")
    (root / "new.py").write_text("new_value = 3\n")
    store = ShadowObjectStore(tmp_path / "spool")
    parent = capture_repository_snapshot(root, policy, store)
    (root / "app.py").write_text("value = 4\n")
    (root / "new.py").write_text("new_value = 5\n")
    final = capture_repository_snapshot(root, policy, store)
    isolated = materialize_snapshot(
        parent,
        policy,
        work_root=tmp_path / "work",
        object_store=store,
        workspace_id="frontier-transition",
    )

    transition_to_snapshot(
        isolated,
        policy,
        parent=parent,
        final=final,
        object_store=store,
    )

    delta = ShadowWorkspace(isolated, policy).diff()
    assert "-value = 2" in delta
    assert "+value = 4" in delta
    assert "-new_value = 3" in delta
    assert "+new_value = 5" in delta


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ("beforeSubmitPrompt", {"continue": True}),
        ("beforeReadFile", {"permission": "allow"}),
        ("afterFileEdit", {}),
    ],
)
def test_hook_cli_returns_event_specific_observer_contract(
    event: str,
    expected: dict[str, object],
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("harness.shadow.__main__.read_hook_input", lambda: {})
    monkeypatch.setattr(
        "harness.shadow.__main__.capture_hook_event",
        lambda *args, **kwargs: None,
    )
    assert (
        hook_command(
            Namespace(
                event=event,
                spool=str(tmp_path / "spool"),
                repository_root=str(tmp_path),
            )
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == expected
