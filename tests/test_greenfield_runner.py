from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.greenfield.runner import (
    GreenfieldRunnerError,
    execute_headless_call,
)


def _call(name: str, arguments: dict) -> dict:
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_headless_runner_writes_and_reads_only_inside_workspace(tmp_path: Path):
    written = json.loads(
        execute_headless_call(
            tmp_path,
            _call(
                "write_to_file",
                {"path": "src/app.py", "content": "VALUE = 1\n"},
            ),
        )
    )
    assert written["success"] is True
    assert tmp_path.joinpath("src/app.py").read_text() == "VALUE = 1\n"

    read = json.loads(
        execute_headless_call(
            tmp_path,
            _call("read_file", {"path": "src/app.py"}),
        )
    )
    assert read["result"] == "VALUE = 1\n"

    replaced = json.loads(
        execute_headless_call(
            tmp_path,
            _call(
                "replace_in_file",
                {
                    "path": "src/app.py",
                    "old_string": "VALUE = 1",
                    "new_string": "VALUE = 2",
                },
            ),
        )
    )
    assert replaced["success"] is True
    assert tmp_path.joinpath("src/app.py").read_text() == "VALUE = 2\n"

    escaped = json.loads(
        execute_headless_call(
            tmp_path,
            _call("write_to_file", {"path": "../escape.py", "content": "bad"}),
        )
    )
    assert escaped["success"] is False
    assert not tmp_path.parent.joinpath("escape.py").exists()


def test_headless_runner_rejects_unknown_tools(tmp_path: Path):
    with pytest.raises(GreenfieldRunnerError, match="unsupported headless tool"):
        execute_headless_call(tmp_path, _call("browser_action", {}))


def test_headless_runner_command_execution_is_allowlisted(tmp_path: Path):
    unsafe = json.loads(
        execute_headless_call(
            tmp_path,
            _call("execute_command", {"command": "rm -rf /"}),
        )
    )
    assert unsafe["success"] is False
    assert "allowlist" in unsafe["error"]


@pytest.mark.parametrize("phase", ["expand", "repair", "verify"])
def test_headless_runner_reconstructs_failed_verify_on_resume(
    monkeypatch,
    phase: str,
):
    from harness.greenfield.runner import _initial_messages

    active = SimpleNamespace(ordinal=1, state="active", task_id="task-1")
    controller = SimpleNamespace(
        service=SimpleNamespace(
            get=lambda _run_id: SimpleNamespace(milestones=[active])
        ),
        tasks=object(),
    )
    state = SimpleNamespace(
        phase=phase,
        last_cmd=".venv/bin/pytest -q",
        result_cmd=".venv/bin/pytest -q",
        stdout_tail="",
        stderr_tail="collection failed",
        last_exit=2,
        timed_out=False,
    )
    monkeypatch.setattr(
        "harness.greenfield.runner.load_loop_state",
        lambda _tasks, _task_id: state,
    )
    messages = _initial_messages(controller, "gf_test", "Continue")
    assert messages[1]["tool_calls"][0]["function"]["name"] == "execute_command"
    result = json.loads(messages[2]["content"])
    assert result["exit_code"] == 2
    assert result["success"] is False


def test_headless_runner_does_not_mislabel_stale_result_on_verify(monkeypatch):
    from harness.greenfield.runner import _initial_messages

    active = SimpleNamespace(ordinal=1, state="active", task_id="task-1")
    controller = SimpleNamespace(
        service=SimpleNamespace(
            get=lambda _run_id: SimpleNamespace(milestones=[active])
        ),
        tasks=object(),
    )
    state = SimpleNamespace(
        phase="verify",
        last_cmd=".venv/bin/pytest -q",
        result_cmd=".venv/bin/ruff check .",
        stdout_tail="old lint failure",
        stderr_tail="",
        last_exit=1,
        timed_out=False,
    )
    monkeypatch.setattr(
        "harness.greenfield.runner.load_loop_state",
        lambda _tasks, _task_id: state,
    )
    assert _initial_messages(controller, "gf_test", "Continue") == [
        {"role": "user", "content": "Continue"}
    ]


def test_headless_runner_applies_contract_owned_ruff_safe_fix(
    tmp_path: Path,
    monkeypatch,
):
    from harness.greenfield.runner import _try_deterministic_lint_repair

    target = tmp_path / "tests.py"
    target.write_text("from pathlib import Path\nimport json\n")
    state = SimpleNamespace(
        phase="repair",
        last_cmd=".venv/bin/ruff check .",
        result_cmd=".venv/bin/ruff check .",
        stdout_tail="Found 1 error. [*] 1 fixable with --fix.",
        stderr_tail="",
        iteration=2,
        verify_index=1,
        active_diff_hash="old",
        verification_results=[object()],
        last_exit=1,
        working_set=SimpleNamespace(
            files_changed=[],
            refresh_pending=[],
            refresh_diff_pending=False,
        ),
    )

    def safe_fix(*_args, **_kwargs):
        target.write_text("import json\nfrom pathlib import Path\n")
        return SimpleNamespace(returncode=0, stdout="fixed", stderr="")

    monkeypatch.setattr("harness.greenfield.runner.subprocess.run", safe_fix)
    assert _try_deterministic_lint_repair(tmp_path, state)
    assert state.phase == "verify"
    assert state.verify_index == 0
    assert state.verification_results == []
    assert state.result_cmd is None
    assert state.working_set.refresh_pending == ["tests.py"]
