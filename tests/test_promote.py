from harness.cases.loader import load_case
from harness.promote import promote_task
from harness.storage.db import Store, utcnow
from harness.task.models import Evidence
from harness.task.service import TaskService


def test_verified_frontier_task_promotes_to_human_case(tmp_path):
    store = Store(tmp_path / "h.db")
    svc = TaskService(store)
    task = svc.start("fix the demonstrated bug")
    assert svc.claim_frontier(task.task_id, 1)
    svc.add_evidence(
        Evidence(
            task_id=task.task_id,
            kind="critic_grade",
            payload={"verdict": "reject"},
        )
    )
    svc.finish(task.task_id, True, "frontier_verified")

    answer = tmp_path / "frontier-answer.md"
    answer.write_text("Apply the verified correction.\n")
    store.create_run("rescue-test", "rescue")
    store.insert_model_result(
        {
            "run_id": "rescue-test",
            "case_id": task.task_id,
            "model_key": "frontier",
            "provider": "anthropic",
            "model": "frontier",
            "tier": 4,
            "started_at": utcnow(),
            "latency_ms": 10,
            "answer_path": str(answer),
            "verdict": "PARTIAL",
            "evaluator": "rescue",
            "evaluation_detail": {},
        }
    )

    case_dir = promote_task(store, task.task_id, tmp_path / "learned")
    case = load_case(case_dir)
    assert case.historical.local_failed is True
    assert case.historical.frontier_succeeded is True
    assert case.evaluation.type == "human"
    assert (case_dir / "expected" / "answer.md").read_text() == answer.read_text()
