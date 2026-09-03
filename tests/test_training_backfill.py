from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from harness.storage.db import Store
from harness.training.backfill import (
    backfill_ci_history,
    backfill_cursor_transcript,
    backfill_designwins,
    backfill_greenfield_history,
    backfill_git_repository,
    backfill_harness_pass_history,
    inventory_cursor_transcript,
    inventory_harness_learning_gaps,
)
from harness.training.ledger import LearningLedger
from harness.training.models import DataUse, SourceKind, SourceProvenance, TextPair


NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)


def test_designwins_backfill_is_quarantined_by_default_and_explicitly_admitted(
    tmp_path: Path,
):
    source = tmp_path / "designwins.jsonl"
    snapshot = tmp_path / "text_pairs.jsonl"
    source_record = {
        "part": "part-1",
        "prompt": "extract pins",
        "target": '{"pins":[]}',
    }
    source_digest = hashlib.sha256(
        json.dumps(
            source_record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    snapshot.write_text(json.dumps(source_record) + "\n")
    provenance = SourceProvenance(
        source_kind=SourceKind.DESIGNWINS,
        source_uri="dataset://designwins/part-1",
        source_record_id="part-1:text",
        collected_at=NOW,
        content_sha256=source_digest,
        lineage_id="designwins:part-1",
        license="internal-owned",
        data_use=DataUse.TRAINING,
    )
    pair = TextPair(
        pair_id="part-1",
        prompt="extract pins",
        response='{"pins":[]}',
        provenance=provenance,
        metadata={"part": "part-1"},
    )
    source.write_text(json.dumps(pair.model_dump(mode="json")) + "\n")
    ledger = LearningLedger(Store(tmp_path / "ledger.db"), tmp_path / "artifacts")

    first = backfill_designwins(source, ledger, source_snapshot=snapshot)
    second = backfill_designwins(source, ledger, source_snapshot=snapshot)
    admitted = backfill_designwins(
        source,
        ledger,
        source_snapshot=snapshot,
        admit_verified=True,
    )

    assert first.captured == 1
    assert first.rejected == 0
    assert second.duplicates == 1
    assert admitted.captured == 1
    with ledger.store.connect() as conn:
        events = conn.execute(
            "SELECT * FROM learning_events ORDER BY event_id"
        ).fetchall()
        admissions = conn.execute("SELECT * FROM learning_admissions").fetchall()
    dispositions = {
        json.loads(event["metadata_json"])["disposition"] for event in events
    }
    assert dispositions == {"quarantine", "verified"}
    assert len(admissions) == 1
    assert admissions[0]["decision"] == "eligible"


def test_git_backfill_captures_single_parent_commit_as_quarantine(tmp_path: Path):
    repository = tmp_path / "owned-repo"
    repository.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Harness Test",
        "GIT_AUTHOR_EMAIL": "harness@example.invalid",
        "GIT_COMMITTER_NAME": "Harness Test",
        "GIT_COMMITTER_EMAIL": "harness@example.invalid",
    }
    subprocess.run(
        ["git", "init", "-b", "main", str(repository)],
        check=True,
        capture_output=True,
        env=env,
    )
    tracked = repository / "value.txt"
    tracked.write_text("bad\n")
    subprocess.run(["git", "-C", str(repository), "add", "value.txt"], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-m", "initial"],
        check=True,
        capture_output=True,
        env=env,
    )
    tracked.write_text("fixed\n")
    subprocess.run(["git", "-C", str(repository), "add", "value.txt"], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-m", "fix value"],
        check=True,
        capture_output=True,
        env=env,
    )
    store = Store(tmp_path / "ledger.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")

    report = backfill_git_repository(
        repository,
        ledger,
        approved_repositories=[repository],
    )

    assert report.captured == 1
    assert report.rejected == 1
    with store.connect() as conn:
        row = conn.execute("SELECT metadata_json FROM learning_events").fetchone()
    assert '"disposition": "quarantine"' in row["metadata_json"]


def test_excluded_git_source_is_rejected_before_reading_history(tmp_path: Path):
    repository = tmp_path / "CategoryRank"
    repository.mkdir()
    ledger = LearningLedger(Store(tmp_path / "ledger.db"), tmp_path / "artifacts")

    report = backfill_git_repository(
        repository,
        ledger,
        approved_repositories=[repository],
    )

    assert report.captured == 0
    assert report.rejected == 1
    assert "CategoryRank/Tapes" in next(iter(report.reasons))


def test_git_backfill_rejects_repository_outside_allowlist(tmp_path: Path):
    repository = tmp_path / "unlisted-repo"
    repository.mkdir()
    ledger = LearningLedger(Store(tmp_path / "ledger.db"), tmp_path / "artifacts")

    report = backfill_git_repository(
        repository,
        ledger,
        approved_repositories=[],
    )

    assert report.captured == 0
    assert report.rejected == 1
    assert "allowlist" in next(iter(report.reasons))


def test_git_backfill_uses_canonical_spaced_exclusion(tmp_path: Path):
    repository = tmp_path / "category rank"
    repository.mkdir()
    ledger = LearningLedger(Store(tmp_path / "ledger.db"), tmp_path / "artifacts")

    report = backfill_git_repository(
        repository,
        ledger,
        approved_repositories=[repository],
    )

    assert report.captured == 0
    assert report.rejected == 1
    assert "CategoryRank/Tapes" in next(iter(report.reasons))


def test_legacy_and_cursor_inventory_fail_closed(tmp_path: Path):
    store = Store(tmp_path / "harness.db")
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO tasks (
                task_id, intent, status, created_at, frontier_required
            ) VALUES ('task-1', 'fix it', 'done', ?, 1)
            """,
            (NOW.isoformat(),),
        )
        conn.execute(
            """
            INSERT INTO attempts (
                task_id, attempt, worker, started_at, result
            ) VALUES ('task-1', 1, 'frontier_senior', ?, 'success')
            """,
            (NOW.isoformat(),),
        )
        conn.execute(
            """
            INSERT INTO gateway_turns (
                task_id, started_at, alias
            ) VALUES ('task-1', ?, 'harness-frontier')
            """,
            (NOW.isoformat(),),
        )
    legacy = inventory_harness_learning_gaps(store)
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"role": "user", "message": {"content": "fix it"}}) + "\n"
    )

    cursor = inventory_cursor_transcript(transcript)

    assert legacy.captured == 0
    assert legacy.rejected == 3
    assert cursor.rejected == 1
    assert legacy.rejection_details[0]["record_id"]
    assert cursor.rejection_details == [
        {
            "record_id": "line:1",
            "reason": (
                "transcript message lacks linked repository revision and proof digest"
            ),
        }
    ]


def test_harness_pass_backfill_captures_complete_rows_and_rejects_missing_cases(
    tmp_path: Path,
):
    cases = tmp_path / "cases" / "case-1"
    cases.mkdir(parents=True)
    (cases / "case.yaml").write_text("id: case-1\n")
    (cases / "prompt.md").write_text("Fix the verified behavior.")
    results = tmp_path / "results"
    results.mkdir()
    answer = results / "answer.txt"
    answer.write_text("The verified answer.")
    store = Store(results / "harness.db")
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, mode, started_at)
            VALUES ('run-1', 'tournament', ?)
            """,
            (NOW.isoformat(),),
        )
        conn.executemany(
            """
            INSERT INTO model_results (
                run_id, case_id, model_key, provider, model, started_at,
                answer_path, verdict, evaluator
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PASS', 'command')
            """,
            [
                (
                    "run-1",
                    "case-1",
                    "frontier",
                    "anthropic",
                    "frontier-model",
                    NOW.isoformat(),
                    str(answer),
                ),
                (
                    "run-1",
                    "missing-case",
                    "frontier",
                    "anthropic",
                    "frontier-model",
                    NOW.isoformat(),
                    str(answer),
                ),
            ],
        )
    ledger = LearningLedger(store, tmp_path / "artifacts")

    report = backfill_harness_pass_history(
        store.db_path,
        tmp_path / "cases",
        ledger,
        answer_root=results,
    )

    assert report.captured == 1
    assert report.rejected == 1
    assert report.rejection_details[0]["record_id"].startswith("model-result:")
    with store.connect() as conn:
        event = conn.execute(
            "SELECT * FROM learning_events WHERE event_type = 'harness_pass_candidate'"
        ).fetchone()
        proof = conn.execute(
            """
            SELECT * FROM learning_verifications
            WHERE event_id = ?
            """,
            (event["event_id"],),
        ).fetchone()
        assert json.loads(event["metadata_json"])["disposition"] == "quarantine"
        assert proof["status"] == "unknown"


def test_greenfield_backfill_writes_quarantine_candidate(
    tmp_path: Path,
    monkeypatch,
):
    runs_root = tmp_path / "runs"
    repository = runs_root / "run-1" / "repo"
    repository.mkdir(parents=True)
    store = Store(tmp_path / "harness.db")
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO greenfield_runs (
                run_id, intent, project_name, stack, destination,
                destination_fingerprint, workspace_root, status,
                discovery_json, spec_json, plan_json, spec_hash, plan_hash,
                created_at, updated_at
            ) VALUES (
                'run-1', 'Build parser', 'parser', 'python', '/tmp/parser',
                'dest', ?, 'complete', '{}', '{}', '{}', 'spec', 'plan', ?, ?
            )
            """,
            (str(repository), NOW.isoformat(), NOW.isoformat()),
        )
        conn.execute(
            """
            INSERT INTO greenfield_milestones (
                run_id, ordinal, milestone_id, title, objective,
                acceptance_json, state, starting_commit,
                verified_state_hash, commit_sha, updated_at
            ) VALUES (
                'run-1', 1, 'm1', 'Parser', 'Implement parser',
                ?, 'complete', ?, ?, ?, ?
            )
            """,
            (
                json.dumps({"acceptance_commands": ["pytest -q"]}),
                "a" * 40,
                "c" * 64,
                "b" * 40,
                NOW.isoformat(),
            ),
        )

    def fake_git(_repository: Path, *arguments: str) -> str:
        if arguments[0] == "rev-parse":
            return "b" * 40 + "\n"
        return (
            "diff --git a/parser.py b/parser.py\n"
            "--- a/parser.py\n"
            "+++ b/parser.py\n"
            "@@ -1 +1 @@\n"
            "-pass\n"
            "+return 1\n"
        )

    monkeypatch.setattr("harness.training.adapters._git", fake_git)
    ledger = LearningLedger(store, tmp_path / "artifacts")

    first = backfill_greenfield_history(
        store.db_path,
        ledger,
        runs_root=runs_root,
    )
    second = backfill_greenfield_history(
        store.db_path,
        ledger,
        runs_root=runs_root,
    )

    assert first.captured == 1
    assert second.duplicates == 1
    with store.connect() as conn:
        event = conn.execute(
            """
            SELECT * FROM learning_events
            WHERE event_type = 'greenfield_commit_candidate'
            """
        ).fetchone()
    assert json.loads(event["metadata_json"])["disposition"] == "quarantine"


def test_cursor_backfill_requires_revision_ownership_and_proof_digest(tmp_path: Path):
    proof_output = "pytest: 5 passed"
    transcript = tmp_path / "cursor.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "repository_uri": "file:///owned/repo",
                "repository_revision": "a" * 40,
                "ownership": "self",
                "created_at": NOW.isoformat(),
                "prompt": "Fix the parser.",
                "response": "Apply this patch.",
                "proof": {
                    "kind": "pytest",
                    "status": "pass",
                    "output": proof_output,
                    "output_sha256": hashlib.sha256(
                        proof_output.encode()
                    ).hexdigest(),
                },
            }
        )
        + "\n"
        + json.dumps({"role": "assistant", "message": {"content": "unbound"}})
        + "\n"
    )
    store = Store(tmp_path / "ledger.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")

    first = backfill_cursor_transcript(transcript, ledger)
    second = backfill_cursor_transcript(transcript, ledger)

    assert first.captured == 1
    assert first.rejected == 1
    assert second.duplicates == 1
    assert second.rejected == 1
    with store.connect() as conn:
        event = conn.execute(
            "SELECT * FROM learning_events WHERE event_type = 'cursor_candidate'"
        ).fetchone()
        proof = conn.execute(
            "SELECT * FROM learning_verifications WHERE event_id = ?",
            (event["event_id"],),
        ).fetchone()
    assert json.loads(event["metadata_json"])["disposition"] == "quarantine"
    assert proof["status"] == "unknown"


def test_ci_backfill_requires_digest_bound_failure_to_green(tmp_path: Path):
    failure_output = "1 failed"
    success_output = "5 passed"
    history = tmp_path / "ci.jsonl"
    history.write_text(
        json.dumps(
            {
                "repository_uri": "git+file:///owned/repo",
                "repository_revision": "b" * 40,
                "ownership": "self",
                "created_at": NOW.isoformat(),
                "patch": (
                    "diff --git a/app.py b/app.py\n"
                    "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-pass\n+return 1\n"
                ),
                "failure": {
                    "command": "pytest -q",
                    "exit_code": 1,
                    "output": failure_output,
                    "output_sha256": hashlib.sha256(
                        failure_output.encode()
                    ).hexdigest(),
                },
                "success": {
                    "command": "pytest -q",
                    "exit_code": 0,
                    "output": success_output,
                    "output_sha256": hashlib.sha256(
                        success_output.encode()
                    ).hexdigest(),
                },
            }
        )
        + "\n"
    )
    store = Store(tmp_path / "ledger.db")
    ledger = LearningLedger(store, tmp_path / "artifacts")

    first = backfill_ci_history(history, ledger)
    second = backfill_ci_history(history, ledger)

    assert first.captured == 1
    assert second.duplicates == 1
    with store.connect() as conn:
        event = conn.execute(
            """
            SELECT * FROM learning_events
            WHERE event_type = 'ci_failure_to_green_candidate'
            """
        ).fetchone()
    assert json.loads(event["metadata_json"])["disposition"] == "quarantine"
