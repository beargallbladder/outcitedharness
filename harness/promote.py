from __future__ import annotations

import json
from pathlib import Path

import yaml

from harness.storage.db import Store
from harness.task.service import TaskService


def promote_task(store: Store, task_id: str, pack: Path) -> Path:
    """Promote a verified frontier rescue into a human-reviewed regression case."""
    svc = TaskService(store)
    task = svc.get(task_id)
    if task.final_outcome != "frontier_verified":
        raise ValueError("only a locally verified frontier rescue can be promoted")

    rescue_rows = [
        row
        for row in store.results_for_case(task_id)
        if row["answer_path"] and row["verdict"] in {"PASS", "PARTIAL"}
    ]
    if not rescue_rows:
        raise ValueError("verified task has no stored frontier answer")
    answer_path = Path(rescue_rows[0]["answer_path"])
    if not answer_path.exists():
        raise ValueError(f"frontier answer is missing: {answer_path}")

    case_id = f"learned_{task_id}"
    case_dir = pack.resolve() / case_id
    if case_dir.exists():
        raise FileExistsError(case_dir)
    inputs = case_dir / "inputs"
    expected = case_dir / "expected"
    inputs.mkdir(parents=True)
    expected.mkdir()

    evidence = [
        {
            "kind": item.kind,
            "attempt": item.attempt,
            "payload": item.payload,
            "created_at": item.created_at,
        }
        for item in svc.evidence(task_id)
    ]
    (case_dir / "prompt.md").write_text(task.intent.strip() + "\n")
    (inputs / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (expected / "answer.md").write_text(answer_path.read_text())
    spec = {
        "id": case_id,
        "title": f"Learned rescue: {task.intent[:80]}",
        "category": "coding",
        "tags": ["learned", "frontier_rescue", "human_review_required"],
        "input_files": ["inputs/evidence.json"],
        "evaluation": {"type": "human"},
        "historical": {
            "local_failed": True,
            "frontier_succeeded": True,
            "notes": f"Promoted from task {task_id}; local critic verified the rescue.",
        },
        "reference_answer": {"file": "expected/answer.md"},
    }
    (case_dir / "case.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return case_dir
