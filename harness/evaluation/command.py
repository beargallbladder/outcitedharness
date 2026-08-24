from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from harness.cases.schema import Case
from harness.evaluation.base import EvalResult


def evaluate_command(case: Case, answer: str) -> EvalResult:
    command = case.evaluation.command
    if not command:
        return EvalResult(
            verdict="ERROR",
            evaluator="command",
            reason="evaluation.command is required",
        )
    argv = command if isinstance(command, list) else ["bash", "-lc", command]

    with tempfile.TemporaryDirectory(prefix="harness-eval-") as tmp:
        answer_path = Path(tmp) / "answer.txt"
        answer_path.write_text(answer)
        env = os.environ.copy()
        env["HARNESS_ANSWER_PATH"] = str(answer_path)
        env["HARNESS_CASE_DIR"] = str(case.path)
        try:
            completed = subprocess.run(
                argv,
                cwd=case.path,
                env=env,
                input=answer,
                text=True,
                capture_output=True,
                timeout=case.evaluation.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return EvalResult(
                verdict="ERROR",
                evaluator="command",
                reason=f"command timed out after {case.evaluation.timeout_s}s",
            )
        except OSError as exc:
            return EvalResult(
                verdict="ERROR",
                evaluator="command",
                reason=f"failed to run command: {exc}",
            )

    passed = completed.returncode == 0
    return EvalResult(
        verdict="PASS" if passed else "FAIL",
        evaluator="command",
        reason="command exited 0" if passed else f"command exited {completed.returncode}",
        detail={
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        },
    )
