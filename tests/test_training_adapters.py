from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from harness.training.adapters import (
    AdapterValidationError,
    load_greenfield_git_candidates,
    load_harness_pass_candidates,
)
from harness.training.models import DataUse


def _write_case(root: Path, case_id: str = "case-1") -> Path:
    case = root / "nested" / case_id
    case.mkdir(parents=True)
    (case / "case.yaml").write_text(
        "id: case-1\n"
        "title: Test case\n"
        "evaluation:\n"
        "  type: keyword_rubric\n"
        "  required: [done]\n"
    )
    (case / "prompt.md").write_text("Complete the verified task.")
    return case


def _harness_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE model_results (
            id INTEGER PRIMARY KEY,
            run_id TEXT,
            case_id TEXT,
            model_key TEXT,
            provider TEXT,
            model TEXT,
            started_at TEXT,
            answer_path TEXT,
            evaluator TEXT,
            verdict TEXT,
            error TEXT
        )
        """
    )
    return connection


def test_harness_adapter_joins_real_pass_rows_and_gates_model_reuse(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases"
    _write_case(cases)
    results = tmp_path / "results"
    local_answer = results / "local.txt"
    cloud_answer = results / "cloud.txt"
    results.mkdir()
    local_answer.write_text("done by local model")
    cloud_answer.write_text("done by cloud model")
    database = results / "harness.db"
    connection = _harness_db(database)
    connection.executemany(
        """
        INSERT INTO model_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                1,
                "run-1",
                "case-1",
                "dgx_qwen",
                "openai_compatible",
                "qwen3-coder-next",
                "2026-08-29T12:00:00Z",
                str(local_answer),
                "command",
                "PASS",
                None,
            ),
            (
                2,
                "run-2",
                "case-1",
                "frontier",
                "anthropic",
                "claude",
                "2026-08-29T12:01:00Z",
                str(cloud_answer),
                "command",
                "PASS",
                None,
            ),
        ],
    )
    connection.commit()
    connection.close()

    destination = tmp_path / "pairs.jsonl"
    rows = load_harness_pass_candidates(
        database,
        cases,
        approved_model_keys=frozenset({"dgx_qwen"}),
        destination=destination,
    )

    by_model = {row.metadata["model_key"]: row for row in rows}
    assert by_model["dgx_qwen"].data_use is DataUse.TRAINING
    assert by_model["frontier"].data_use is DataUse.QUARANTINE
    assert by_model["frontier"].metadata["quarantine_reason"]
    assert len(destination.read_text().splitlines()) == 2


def _greenfield_db(path: Path, repo: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE greenfield_runs (
            run_id TEXT PRIMARY KEY,
            workspace_root TEXT
        );
        CREATE TABLE greenfield_milestones (
            run_id TEXT,
            ordinal INTEGER,
            milestone_id TEXT,
            objective TEXT,
            acceptance_json TEXT,
            state TEXT,
            starting_commit TEXT,
            verified_state_hash TEXT,
            commit_sha TEXT,
            updated_at TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO greenfield_runs VALUES (?, ?)",
        ("run-1", str(repo)),
    )
    connection.execute(
        "INSERT INTO greenfield_milestones VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "run-1",
            1,
            "m1",
            "Implement the parser",
            json.dumps({"acceptance_commands": ["pytest -q"]}),
            "complete",
            "a" * 40,
            "c" * 64,
            "b" * 40,
            "2026-08-29T12:00:00Z",
        ),
    )
    connection.commit()
    connection.close()


def test_greenfield_adapter_emits_only_quarantined_real_diffs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    repo = runs / "run-1" / "repo"
    repo.mkdir(parents=True)
    database = tmp_path / "harness.db"
    _greenfield_db(database, repo)

    def fake_git(_repo: Path, *arguments: str) -> str:
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
    destination = tmp_path / "git.jsonl"
    rows = load_greenfield_git_candidates(
        database,
        runs_root=runs,
        destination=destination,
    )

    assert len(rows) == 1
    assert rows[0].data_use is DataUse.QUARANTINE
    assert rows[0].approved_for_training is False
    assert rows[0].tests[0].status == "unknown"
    assert json.loads(destination.read_text())["candidate_id"] == rows[0].candidate_id


def test_greenfield_adapter_rejects_repo_outside_approved_runs_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    database = tmp_path / "harness.db"
    _greenfield_db(database, outside)
    monkeypatch.setattr(
        "harness.training.adapters._git",
        lambda *_args: pytest.fail("git must not run for an escaped repository"),
    )

    with pytest.raises(AdapterValidationError, match="escapes"):
        load_greenfield_git_candidates(database, runs_root=runs)
