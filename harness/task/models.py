from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Task:
    task_id: str
    intent: str
    status: str
    created_at: str
    plan: str = ""
    hypothesis: str = ""
    intervened: bool = False
    frontier_required: bool = False
    stage: str = "new"
    frontier_calls: int = 0
    updated_at: str = ""
    final_outcome: str = ""


@dataclass
class WorkPacket:
    task_id: str
    worker: str
    intent: str
    plan: str = ""
    files: list[str] = field(default_factory=list)
    diff: str = ""
    failed_tests: str = ""
    hypothesis: str = ""
    decisions: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        files = "\n".join(f"- {p}" for p in self.files) or "- (none)"
        decisions = "\n".join(f"- {d}" for d in self.decisions) or "- (none)"
        return (
            f"# TASK\n{self.intent}\n\n"
            f"# RELEVANT ARCHITECTURE\n{self.plan or '(none)'}\n\n"
            f"# FILES\n{files}\n\n"
            f"# OBSERVED FAILURE\n{self.failed_tests or '(none)'}\n\n"
            f"# ATTEMPTS\nworker={self.worker} task={self.task_id}\n\n"
            f"# TEST EVIDENCE\n{self.failed_tests or '(none)'}\n\n"
            f"# FOREMAN HYPOTHESIS\n{self.hypothesis or '(none)'}\n\n"
            f"# QUESTION\nroot cause + next execution plan for the Spark coder\n\n"
            f"# ACCEPTED DECISIONS\n{decisions}\n\n"
            f"# ACTIVE DIFF\n{self.diff or '(none)'}\n"
        )


@dataclass
class AttemptRecord:
    task_id: str
    attempt: int
    worker: str
    result: str
    files_changed: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    tests_passed: int | None = None
    tests_failed: int | None = None
    ttft_ms: float | None = None
    tokens_per_sec: float | None = None
    tool_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost: float | None = None
    started_at: str = ""
    finished_at: str = ""

    def to_evidence_json(self) -> dict[str, Any]:
        tests: dict[str, int] | None = None
        if self.tests_passed is not None or self.tests_failed is not None:
            tests = {
                "passed": int(self.tests_passed or 0),
                "failed": int(self.tests_failed or 0),
            }
        return {
            "task_id": self.task_id,
            "worker": self.worker,
            "attempt": self.attempt,
            "files_changed": list(self.files_changed),
            "commands": list(self.commands),
            "tests": tests,
            "result": self.result,
            "ttft_ms": self.ttft_ms,
            "tokens_per_sec": self.tokens_per_sec,
            "tool_calls": self.tool_calls,
        }


@dataclass
class Evidence:
    task_id: str
    kind: str
    payload: dict[str, Any]
    attempt: int | None = None
    created_at: str = ""


@dataclass
class Decision:
    task_id: str
    actor: str
    text: str
    accepted: bool
    created_at: str = ""


def as_json(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "to_evidence_json"):
        return obj.to_evidence_json()
    return asdict(obj)
