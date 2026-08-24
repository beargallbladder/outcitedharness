import json
from pathlib import Path

from harness.rescue import missing_sections
from harness.storage.db import Store
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
