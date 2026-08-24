from __future__ import annotations

from dataclasses import dataclass, field

from harness.task.models import WorkPacket


@dataclass
class ContextManager:
    """Structured working set. Not a chat transcript."""

    intent: str = ""
    plan: str = ""
    files: list[str] = field(default_factory=list)
    diff: str = ""
    failed_tests: str = ""
    hypothesis: str = ""
    decisions: list[str] = field(default_factory=list)

    def packet(self, task_id: str, worker: str) -> WorkPacket:
        return WorkPacket(
            task_id=task_id,
            worker=worker,
            intent=self.intent,
            plan=self.plan,
            files=list(self.files),
            diff=self.diff,
            failed_tests=self.failed_tests,
            hypothesis=self.hypothesis,
            decisions=list(self.decisions),
        )
