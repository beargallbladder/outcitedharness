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

    @classmethod
    def from_loop(cls, intent: str, state) -> "ContextManager":
        working_set = getattr(state, "working_set", None)
        changed = list(getattr(working_set, "files_changed", None) or [])
        read = list((getattr(working_set, "files_read", None) or {}).keys())
        files = list(dict.fromkeys([*changed, *read]))
        return cls(
            intent=intent,
            files=files,
            diff=str(getattr(working_set, "current_diff", "") or ""),
            failed_tests=str(getattr(state, "failed_tests", "") or ""),
        )

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
