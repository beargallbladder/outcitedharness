import json
from pathlib import Path

from harness.gateway.logging import _worker_for_alias, log_turn
from harness.orch_loop import LoopState, WorkingFile, save_loop_state
from harness.rescue import missing_sections
from harness.storage.db import Store
from harness.task.context import ContextManager
from harness.task.models import AttemptRecord, Decision
from harness.task.service import TaskService


def _svc(tmp_path: Path) -> TaskService:
    return TaskService(Store(tmp_path / "t.db"))


def test_record_matches_evidence_shape(tmp_path: Path):
    svc = _svc(tmp_path)
    task = svc.start("fix geocode bug")
    rec = svc.record(
        AttemptRecord(
            task_id=task.task_id,
            attempt=0,
            worker="primary_coder",
            result="success",
            files_changed=["services/engine/geocode.py"],
            commands=["pytest tests/test_geocode.py"],
            tests_passed=17,
            tests_failed=0,
            tool_calls=4,
        )
    )
    payload = rec.to_evidence_json()
    assert payload == {
        "task_id": task.task_id,
        "worker": "primary_coder",
        "attempt": 1,
        "files_changed": ["services/engine/geocode.py"],
        "commands": ["pytest tests/test_geocode.py"],
        "tests": {"passed": 17, "failed": 0},
        "result": "success",
        "ttft_ms": None,
        "tokens_per_sec": None,
        "tool_calls": 4,
    }
    assert json.dumps(payload)
    shown = svc.get(task.task_id)
    assert shown.status == "success"


def test_frontier_claim_is_atomic_and_capped(tmp_path: Path):
    svc = _svc(tmp_path)
    task = svc.start("repair a failed local answer")
    assert svc.claim_frontier(task.task_id, 1) is True
    assert svc.claim_frontier(task.task_id, 1) is False
    saved = svc.get(task.task_id)
    assert saved.frontier_required is True
    assert saved.frontier_calls == 1
    assert saved.stage == "frontier_rescue"


def test_packet_is_rescue_shaped_not_a_transcript(tmp_path: Path):
    svc = _svc(tmp_path)
    task = svc.start("fix geocode bug", plan="engine geocodes ingest rows")
    svc.add_decision(
        Decision(task_id=task.task_id, actor="fallback_reasoner", text="use nominatim cache", accepted=True)
    )
    svc.record(
        AttemptRecord(
            task_id=task.task_id,
            attempt=0,
            worker="primary_coder",
            result="fail",
            files_changed=["services/engine/geocode.py"],
            tests_passed=16,
            tests_failed=1,
        )
    )
    packet = svc.packet(task.task_id, "primary_coder")
    text = packet.to_markdown()
    assert missing_sections(text) == []
    assert "15 previous explanations" not in text
    assert "services/engine/geocode.py" in text
    assert "nominatim cache" in text
    assert packet.worker == "primary_coder"


def test_context_adapters_project_nested_working_set(tmp_path: Path):
    svc = _svc(tmp_path)
    task = svc.start("fix nested loop context")
    state = LoopState(intent=task.intent, failed_tests="pytest failed")
    state.working_set.files_changed = ["src/current.py"]
    state.working_set.files_read = {
        "src/current.py": WorkingFile("VALUE = 2\n", "hash-current"),
        "tests/test_current.py": WorkingFile("def test_value(): ...\n", "hash-test"),
    }
    state.working_set.current_diff = "diff --git a/src/current.py b/src/current.py"
    save_loop_state(svc, task.task_id, state)

    direct = ContextManager.from_loop(task.intent, state)
    restored = svc.context_from_task(task)

    for context in (direct, restored):
        assert context.files == ["src/current.py", "tests/test_current.py"]
        assert context.diff == state.working_set.current_diff
        assert context.failed_tests == "pytest failed"


def test_worker_for_alias_maps_correctly():
    """Test that alias mapping returns correct worker roles."""
    assert _worker_for_alias("harness-local") == "primary_coder"
    assert _worker_for_alias("harness-auto") == "primary_coder"
    assert _worker_for_alias("harness-m5") == "fallback_reasoner"
    assert _worker_for_alias("harness-frontier") == "frontier_senior"
    assert _worker_for_alias("harness-dgx2") == "dgx2_coder"
    assert _worker_for_alias("harness-asus") == "asus_coder"
    assert _worker_for_alias("harness-dgx3") == "dgx3_coder"
    assert _worker_for_alias("harness-orch") == "fallback_reasoner"
    assert _worker_for_alias("harness-researcher") == "researcher"
    assert _worker_for_alias("unknown-alias") == "primary_coder"


def test_log_turn_records_attempt(tmp_path: Path):
    """Test that log_turn creates a task and records an attempt."""
    store = Store(tmp_path / "test.db")
    
    # First call should create a task and record attempt 1
    log_turn(
        store,
        alias="harness-local",
        model_key="dgx_qwen",
        upstream_model="qwen2.5-72b",
        stream=False,
        status=200,
        latency_ms=150.5,
        input_tokens=100,
        output_tokens=50,
        cost=0.001,
        error=None,
        body={"messages": [{"role": "user", "content": "test"}]},
    )
    
    # Verify task was created
    with store.connect() as conn:
        task_row = conn.execute("SELECT * FROM tasks").fetchone()
        assert task_row is not None
        assert task_row["intent"] == "gateway session"
        assert task_row["status"] == "open"  # Gateway turns do not close the session
        
        # Verify attempt was recorded
        attempt_row = conn.execute("SELECT * FROM attempts").fetchone()
        assert attempt_row is not None
        assert attempt_row["task_id"] == task_row["task_id"]
        assert attempt_row["attempt"] == 1
        assert attempt_row["worker"] == "primary_coder"
        assert attempt_row["result"] == "success"
        assert attempt_row["input_tokens"] == 100
        assert attempt_row["output_tokens"] == 50
        assert attempt_row["ttft_ms"] == 150.5
        turn_row = conn.execute("SELECT * FROM gateway_turns").fetchone()
        assert turn_row["task_id"] == task_row["task_id"]


def test_log_turn_increments_attempt(tmp_path: Path):
    """Test that subsequent log_turn calls increment the attempt number."""
    store = Store(tmp_path / "test2.db")
    
    log_turn(
        store,
        alias="harness-m5",
        model_key="m5_qwen",
        upstream_model="qwen2.5-32b",
        stream=False,
        status=200,
        latency_ms=200.0,
        input_tokens=200,
        output_tokens=100,
        cost=0.002,
        error=None,
        body={"messages": [{"role": "user", "content": "test1"}]},
    )
    
    log_turn(
        store,
        alias="harness-m5",
        model_key="m5_qwen",
        upstream_model="qwen2.5-32b",
        stream=False,
        status=200,
        latency_ms=180.0,
        input_tokens=150,
        output_tokens=75,
        cost=0.0015,
        error=None,
        body={"messages": [{"role": "user", "content": "test2"}]},
    )
    
    with store.connect() as conn:
        attempts = conn.execute("SELECT attempt, worker FROM attempts ORDER BY attempt").fetchall()
        assert len(attempts) == 2
        assert attempts[0]["attempt"] == 1
        assert attempts[0]["worker"] == "fallback_reasoner"
        assert attempts[1]["attempt"] == 2
        assert attempts[1]["worker"] == "fallback_reasoner"


def test_log_turn_does_not_attach_to_unrelated_task(tmp_path: Path):
    store = Store(tmp_path / "isolate.db")
    other = TaskService(store).start("fix geocode bug")
    log_turn(
        store,
        alias="harness-local",
        model_key="dgx_qwen",
        upstream_model="qwen3-coder-next",
        stream=False,
        status=200,
        latency_ms=10,
        input_tokens=1,
        output_tokens=1,
        cost=None,
        error=None,
        body={"messages": []},
    )
    with store.connect() as conn:
        tasks = conn.execute("SELECT task_id, intent FROM tasks ORDER BY created_at").fetchall()
        assert len(tasks) == 2
        assert tasks[0]["task_id"] == other.task_id
        assert tasks[1]["intent"] == "gateway session"
        stolen = conn.execute(
            "SELECT COUNT(*) FROM attempts WHERE task_id = ?", (other.task_id,)
        ).fetchone()[0]
        assert stolen == 0


def test_log_turn_opens_new_session_after_idle(tmp_path: Path):
    store = Store(tmp_path / "idle.db")
    log_turn(
        store,
        alias="harness-local",
        model_key="dgx_qwen",
        upstream_model="x",
        stream=False,
        status=200,
        latency_ms=10,
        input_tokens=1,
        output_tokens=1,
        cost=None,
        error=None,
        body={"messages": []},
    )
    with store.connect() as conn:
        first = conn.execute("SELECT task_id FROM tasks").fetchone()["task_id"]
        conn.execute(
            "UPDATE tasks SET created_at = '2020-01-01T00:00:00+00:00' WHERE task_id = ?",
            (first,),
        )
        conn.execute(
            "UPDATE attempts SET started_at = '2020-01-01T00:00:00+00:00', "
            "finished_at = '2020-01-01T00:00:00+00:00' WHERE task_id = ?",
            (first,),
        )
    log_turn(
        store,
        alias="harness-local",
        model_key="dgx_qwen",
        upstream_model="x",
        stream=False,
        status=200,
        latency_ms=10,
        input_tokens=1,
        output_tokens=1,
        cost=None,
        error=None,
        body={"messages": []},
    )
    with store.connect() as conn:
        ids = [r["task_id"] for r in conn.execute("SELECT task_id FROM tasks ORDER BY created_at")]
        assert len(ids) == 2
        assert ids[0] == first
        assert ids[1] != first


def test_log_turn_records_failure(tmp_path: Path):
    """Test that log_turn records failed attempts for error status codes."""
    store = Store(tmp_path / "test3.db")
    
    log_turn(
        store,
        alias="harness-frontier",
        model_key="claude-sonnet",
        upstream_model="claude-sonnet-4-6",
        stream=False,
        status=502,
        latency_ms=5000.0,
        input_tokens=100,
        output_tokens=None,
        cost=None,
        error="Connection refused",
        body={"messages": [{"role": "user", "content": "test"}]},
    )
    
    with store.connect() as conn:
        attempt_row = conn.execute("SELECT result FROM attempts").fetchone()
        assert attempt_row is not None
        assert attempt_row["result"] == "failed"


def test_latest_session_returns_none_when_no_open_session(tmp_path: Path):
    """Test that latest_session returns None when no gateway session exists."""
    store = Store(tmp_path / "no_session.db")
    svc = TaskService(store)
    assert svc.latest_session() is None


def test_latest_session_returns_open_session(tmp_path: Path):
    """Test that latest_session returns the most recent open gateway session."""
    store = Store(tmp_path / "session.db")
    svc = TaskService(store)
    
    # Create a non-gateway session
    svc.start("other task")
    
    # Create gateway sessions
    svc.start("gateway session")
    later = svc.start("gateway session")

    task = svc.latest_session()
    assert task is not None
    assert task.intent == "gateway session"
    assert task.task_id == later.task_id


def test_task_current_cli_no_session(tmp_path: Path, monkeypatch):
    from typer.testing import CliRunner

    from harness.cli import app
    from harness.config import AppConfig, Settings

    db = tmp_path / "cli.db"
    Store(db)

    def fake_cfg():
        return AppConfig(
            root=tmp_path,
            settings=Settings(results_dir=tmp_path / "results", db_path=db),
            models={},
            pricing={},
        )

    monkeypatch.setattr("harness.cli._cfg", fake_cfg)
    result = CliRunner().invoke(app, ["task", "current"])
    assert result.exit_code == 0
    assert "no open gateway session" in result.stdout
