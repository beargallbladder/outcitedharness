from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.config import AppConfig, ModelConfig, Settings
from harness.dispatch import AcceptSpec, DispatchReport, Packet, Shot
from harness.orch_loop import (
    MAX_CYCLES,
    LoopState,
    WorkingFile,
    command_allowed,
    commands_named_in_intent,
    complete_working_set_refresh,
    load_loop_state,
    parse_argv,
    parse_command_outcome,
    prepare_failure_expansion,
    remember_write,
    select_verify_command,
    working_set_diff_hash,
)
from harness.providers.base import ChatResult
from harness.storage.db import Store
from harness.task.service import TaskService

FIX = "fix the failing unit test in tests/test_x.py"


def _model(key: str = "m5_qwen") -> ModelConfig:
    return ModelConfig(
        key=key,
        tier=1,
        display_name=key,
        short_name=key,
        provider="openai_compatible",
        base_url=f"http://127.0.0.1/{key}",
        model=key,
    )


def _cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        root=tmp_path,
        settings=Settings(results_dir=tmp_path / "results", db_path=tmp_path / "h.db"),
        models={"m5_qwen": _model()},
        pricing={},
    )


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "read_files",
                "parameters": {"type": "object", "properties": {"paths": {}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {}, "regex": {}, "file_pattern": {}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "editor",
                "parameters": {"type": "object", "properties": {"path": {}, "content": {}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {}, "timeout": {}},
                },
            },
        },
    ]


def _read_pair(index: int) -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"read_{index}",
                    "type": "function",
                    "function": {
                        "name": "read_files",
                        "arguments": '{"paths":["src/app.py","tests/test_x.py"]}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"read_{index}",
            "content": (
                "FILE tests/test_x.py\ndef test_x():\n    assert True\n\n"
                "FILE src/app.py\n" + ("def work(): pass\n" * 20)
            ),
        },
    ]


def _ready_messages(extra: list[dict] | None = None) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": FIX}]
    messages.extend(_read_pair(0))
    messages.extend(_read_pair(1))
    if extra:
        messages.extend(extra)
    return messages


def _edit_pair(index: int, result: str = "Successfully applied edit to src/app.py\n+ return 1") -> list[dict]:
    return _edit_path_pair(index, "src/app.py", "fixed", result)


def _edit_path_pair(
    index: int,
    path: str,
    content: str,
    result: str = "Successfully applied edit",
) -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"edit_{index}",
                    "type": "function",
                    "function": {
                        "name": "editor",
                        "arguments": json.dumps({"path": path, "content": content}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"edit_{index}", "content": result},
    ]


def _verify_pair(index: int, result: str) -> list[dict]:
    return _command_pair(index, "pytest tests/test_x.py", result)


def _command_pair(index: int, command: str, result: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"ver_{index}",
                    "type": "function",
                    "function": {
                        "name": "execute_command",
                        "arguments": json.dumps({"command": command, "timeout": 60}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"ver_{index}", "content": result},
    ]


def _refresh_pair(index: int, content: str | None = None) -> list[dict]:
    body = content or f"def work():\n    return {index + 1}\n"
    diff = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n+++ b/src/app.py\n"
        f"@@ -1 +1 @@\n-def work(): pass\n+{body.strip()}\n"
    )
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"refresh_read_{index}",
                    "type": "function",
                    "function": {
                        "name": "read_files",
                        "arguments": '{"paths":["src/app.py"]}',
                    },
                },
                {
                    "id": f"refresh_diff_{index}",
                    "type": "function",
                    "function": {
                        "name": "execute_command",
                        "arguments": '{"command":"git diff -- src/app.py","timeout":60}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"refresh_read_{index}",
            "content": f"FILE src/app.py\n{body}",
        },
        {
            "role": "tool",
            "tool_call_id": f"refresh_diff_{index}",
            "content": json.dumps(
                [{"query": "git diff -- src/app.py", "result": diff, "exitCode": 0}]
            ),
        },
    ]


def _expansion_read_pair(index: int, path: str, content: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"expand_read_{index}",
                    "type": "function",
                    "function": {
                        "name": "read_files",
                        "arguments": json.dumps({"paths": [path]}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"expand_read_{index}",
            "content": f"FILE {path}\n{content}",
        },
    ]


def _search_pair(index: int, symbol: str, result: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"expand_search_{index}",
                    "type": "function",
                    "function": {
                        "name": "search_files",
                        "arguments": json.dumps(
                            {"path": ".", "regex": rf"\b{symbol}\b"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"expand_search_{index}",
            "content": result,
        },
    ]


def _report(command: str = "pytest tests/test_x.py", text: str = "replace the return", qa_pass: bool = True) -> DispatchReport:
    packet = Packet(
        id="p1",
        title="fix test",
        prompt="Write the patch.",
        accept=AcceptSpec(commands=(command,) if command else (), invariants=("min_chars 10",)),
    )
    shot = Shot(
        packet=packet,
        worker_id="w",
        model_key="m",
        result=ChatResult(provider="x", model="x", text=text),
        tokens_per_sec=1.0,
        tool_names=[],
        tool_hit=True,
        qa_pass=qa_pass,
        preview=text,
    )
    return DispatchReport(run_id="loop-test", intent=FIX, packets=[packet], shots=[shot])


def _report_commands(commands: list[str]) -> DispatchReport:
    report = _report(command="")
    report.packets[0].accept = AcceptSpec(
        commands=tuple(commands),
        invariants=("min_chars 10",),
    )
    return report


def _edit_call() -> list[dict]:
    return [
        {
            "id": "call_edit",
            "type": "function",
            "function": {
                "name": "editor",
                "arguments": json.dumps({"path": "src/app.py", "content": "fixed"}),
            },
        }
    ]


def _patch_loop(monkeypatch, report: DispatchReport | None = None, captured: dict | None = None):
    captured = captured if captured is not None else {}
    payload = report or _report()

    async def pick(*_a, **_k):
        return "m5_qwen", _model()

    async def dispatch(_cfg, intent, **kwargs):
        captured.setdefault("dispatches", []).append(kwargs.get("packets"))
        captured.setdefault("dispatch_kwargs", []).append(dict(kwargs))
        return payload

    async def actions(*_a, **_k):
        captured["planned"] = True
        queued = captured.get("planned_calls")
        if isinstance(queued, list) and queued:
            return queued.pop(0)
        return _edit_call()

    packet_commands = [
        command
        for packet in payload.packets
        for command in packet.accept.commands
    ]
    contract = SimpleNamespace(
        repo_root="/tmp/test-repo",
        fingerprint="contract-test",
        configs=[".harness.toml"],
        commands=[
            SimpleNamespace(command=command, timeout_s=60)
            for command in packet_commands
        ],
    )

    monkeypatch.setattr("harness.gateway.orch.pick_foreman", pick)
    monkeypatch.setattr("harness.gateway.orch.run_dispatch", dispatch)
    monkeypatch.setattr("harness.gateway.orch.plan_actions", actions)
    monkeypatch.setattr("harness.gateway.orch.plan_orch", dispatch)
    monkeypatch.setattr(
        "harness.gateway.orch.build_repo_contract",
        lambda *_a, **_k: contract if packet_commands else None,
    )

    def capture_checkpoint(_cfg, task_id, state, calls, _workspace, number):
        paths: list[str] = []
        for call in calls:
            fn = call.get("function") or {}
            args = json.loads(fn.get("arguments") or "{}")
            path = args.get("path")
            if path and path not in paths:
                paths.append(path)
        state.checkpoint_task_id = task_id
        state.checkpoint_run_id = state.checkpoint_run_id or "test-run"
        state.checkpoint_pending_paths = paths
        state.checkpoint_pending_number = number
        return True

    def finalize_checkpoint(_cfg, state):
        state.checkpoint_count = state.checkpoint_pending_number
        state.checkpoint_available = True
        state.checkpoint_last_manifest = "/tmp/test-checkpoint.json"
        state.checkpoint_pending_paths = []
        state.checkpoint_pending_number = 0

    monkeypatch.setattr(
        "harness.gateway.orch._capture_apply_baseline",
        capture_checkpoint,
    )
    monkeypatch.setattr(
        "harness.gateway.orch._finalize_pending_checkpoint",
        finalize_checkpoint,
    )
    return captured


def test_allowlist_rejects_shell_and_unknowns():
    assert command_allowed("pytest tests/test_x.py")
    assert command_allowed("python3 -m pytest tests/test_x.py")
    assert command_allowed("ruff check src")
    assert command_allowed("npm run typecheck")
    assert command_allowed("pnpm test:unit")
    assert command_allowed("npx tsc --noEmit")
    assert not command_allowed("pytest tests; rm -rf /")
    assert not command_allowed("npm exec sh")
    assert not command_allowed("bash -c 'pytest'")
    assert parse_argv("pytest tests && rm -rf /") is None
    command, reason = select_verify_command([])
    assert command is None
    assert "accept.commands" in reason
    command, reason = select_verify_command(["pytest tests; rm -rf /tmp/x"])
    assert command is None
    assert "unsafe" in reason
    assert commands_named_in_intent(
        "fix test_add.py. Use pytest test_add.py as the acceptance command."
    ) == ["pytest test_add.py"]
    assert commands_named_in_intent(
        "add a comment to hello.py. Do not invent tests. Do not run pytest."
    ) == []
    assert commands_named_in_intent("fix the failing unit test in tests/test_x.py") == []


def test_parse_command_outcome_reads_client_payloads_and_fails_closed():
    code, timed, out, _err = parse_command_outcome("1 passed\nExit code: 0")
    assert code == 0 and not timed
    code, timed, out, _err = parse_command_outcome(
        json.dumps([{"query": "pytest tests/test_x.py", "result": "1 passed", "exitCode": 0}])
    )
    assert code == 0
    assert "1 passed" in out
    code, timed, out, _err = parse_command_outcome(
        "<terminal_output>ok</terminal_output>\n<exit_code>0</exit_code>"
    )
    assert code == 0
    code, timed, out, _err = parse_command_outcome(
        "Command completed with exit code 1\nFAILED tests/test_x.py"
    )
    assert code == 1
    code, timed, out, _err = parse_command_outcome("===== 1 passed in 0.02s =====")
    assert code is None
    assert not timed
    code, timed, out, _err = parse_command_outcome("pytest timed out after 60s")
    assert timed


def test_diff_hash_preserves_exact_file_state():
    state = LoopState()
    state.working_set.files_changed = ["src/app.py"]
    state.working_set.files_read["src/app.py"] = WorkingFile(
        content="VALUE = 'A'\n",
        content_hash="hash-A",
    )
    first = working_set_diff_hash(state)
    state.working_set.files_read["src/app.py"] = WorkingFile(
        content="value = 'a'\n",
        content_hash="hash-a",
    )
    assert working_set_diff_hash(state) != first


def test_deleted_task_file_changes_hash_and_completes_missing_refresh():
    state = LoopState(last_cmd="pytest tests/test_x.py")
    state.working_set.files_changed = ["src/app.py"]
    state.working_set.files_read["src/app.py"] = WorkingFile(
        content="VALUE = 1\n",
        content_hash="original",
    )
    original = working_set_diff_hash(state)

    remember_write(
        state,
        {"path": "src/app.py", "patch": "*** Delete File: src/app.py"},
        "Successfully deleted src/app.py",
    )
    assert "src/app.py" not in state.working_set.files_read
    assert working_set_diff_hash(state) != original

    state.working_set.refresh_pending = ["src/app.py"]
    messages = [
        *_verify_pair(0, "FAILED\nExit code: 1"),
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "deleted_read",
                    "type": "function",
                    "function": {
                        "name": "read_files",
                        "arguments": '{"paths":["src/app.py"]}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "deleted_read",
            "content": "ERROR: file not found: src/app.py",
        },
    ]
    assert complete_working_set_refresh(state, messages)
    assert state.working_set.refresh_pending == []


def test_resumed_refresh_consumes_fresh_results_without_prior_exchange():
    state = LoopState(last_cmd=".venv/bin/ruff check .")
    state.working_set.refresh_pending = ["src/app.py"]
    state.working_set.refresh_diff_pending = True

    assert complete_working_set_refresh(state, _refresh_pair(0))
    assert state.working_set.refresh_pending == []
    assert state.working_set.refresh_diff_pending is False


def test_headless_read_wrapper_hashes_file_content_not_json_envelope():
    from harness.orch_loop import _read_result_files

    content = "def endpoint():\n    return {'ok': True}\n"
    wrapped = json.dumps(
        {"path": "src/app.py", "result": content, "success": True}
    )
    assert _read_result_files(["src/app.py"], wrapped) == {
        "src/app.py": content
    }


def test_failure_expansion_extracts_only_unseen_workspace_evidence():
    state = LoopState()
    state.working_set.repo_root = "/tmp/workspace"
    state.working_set.files_read["tests/test_service.py"] = WorkingFile(
        "def test_service(): ...\n",
        "known",
    )
    state.stdout_tail = (
        '  File "/tmp/workspace/services/cache/redis.py", line 184, in fetch\n'
        "FAILED tests/test_service.py::test_sparse\n"
        "/tmp/outside/secrets.py:4: should not be read\n"
        "/tmp/workspace/src/absolute.ts:9:2 - error TS2552\n"
        "src/view.tsx:27:11 - error TS2304: Cannot find name 'WidgetFactory'.\n"
        "NameError: name 'WidgetFactory' is not defined\n"
    )

    assert prepare_failure_expansion(state)
    assert state.expansion_paths == [
        "services/cache/redis.py",
        "src/absolute.ts",
        "src/view.tsx",
    ]
    assert state.expansion_symbols == ["WidgetFactory"]


@pytest.mark.asyncio
async def test_happy_path_persists_across_run_orch_calls(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    captured = _patch_loop(monkeypatch)
    tools = _tools()

    first = await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    assert first.tool_calls
    assert first.tool_calls[0]["function"]["name"] == "editor"
    assert first.loop_phase == "apply"
    assert first.loop_iteration == 1
    initial_context = captured["dispatch_kwargs"][0]["compiled_context"]
    assert "<OBJECTIVE>" in initial_context
    assert "<ACCEPTANCE>" in initial_context
    assert "pytest tests/test_x.py" in initial_context
    assert 'path="src/app.py"' in initial_context
    assert captured["dispatch_kwargs"][0]["thread"] == initial_context

    svc = TaskService(Store(cfg.settings.db_path))
    mid = load_loop_state(svc, svc.session_task().task_id)
    assert mid is not None and mid.phase == "apply"

    second = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(_edit_pair(0)),
        tools=tools,
    )
    assert second.tool_calls
    args = json.loads(second.tool_calls[0]["function"]["arguments"])
    assert args["command"] == "pytest tests/test_x.py"
    assert args.get("timeout") == 60
    assert second.loop_phase == "verify"
    assert "rm " not in args["command"]

    third = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(_edit_pair(0) + _verify_pair(0, "1 passed\nExit code: 0")),
        tools=tools,
    )
    assert not third.tool_calls
    assert third.loop_phase == "verified"
    assert "status: verified" in third.text
    assert "exit_code: 0" in third.text
    again = load_loop_state(svc, svc.session_task().task_id)
    assert again is not None and again.phase == "verified"


@pytest.mark.asyncio
async def test_checkpoint_baseline_precedes_apply_and_repair_extends_it(
    tmp_path: Path,
    monkeypatch,
):
    from harness.checkpoints import CheckpointStore
    from harness.gateway.orch import run_orch

    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src/app.py").write_text("def work():\n    return 0\n")
    cfg = _cfg(tmp_path)
    payload = _report()
    actions = [
        _edit_call(),
        [
            {
                "id": "call_generated",
                "type": "function",
                "function": {
                    "name": "editor",
                    "arguments": json.dumps(
                        {
                            "path": "src/generated_helper.py",
                            "content": "VALUE = 2\n",
                        }
                    ),
                },
            }
        ],
    ]

    async def pick(*_a, **_k):
        return "m5_qwen", _model()

    async def dispatch(*_a, **_k):
        return payload

    async def plan(*_a, **_k):
        return actions.pop(0)

    contract = SimpleNamespace(
        repo_root=str(workspace),
        fingerprint="checkpoint-contract",
        configs=[],
        commands=[SimpleNamespace(command="pytest tests/test_x.py", timeout_s=60)],
    )
    monkeypatch.setattr("harness.gateway.orch.pick_foreman", pick)
    monkeypatch.setattr("harness.gateway.orch.run_dispatch", dispatch)
    monkeypatch.setattr("harness.gateway.orch.plan_actions", plan)
    monkeypatch.setattr("harness.gateway.orch.build_repo_contract", lambda *_a, **_k: contract)
    extra = {"workspace_root": str(workspace)}

    first = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(),
        tools=_tools(),
        extra=extra,
    )
    assert first.loop_phase == "apply"
    svc = TaskService(Store(cfg.settings.db_path))
    task_id = svc.session_task().task_id
    state = load_loop_state(svc, task_id)
    assert state is not None
    store = CheckpointStore(cfg.settings.results_dir / "checkpoints")
    baseline = store.load_manifest(task_id, state.checkpoint_run_id, 0)
    assert baseline is not None
    assert set(baseline.files) == {"src/app.py"}
    original_blob = (
        cfg.settings.results_dir
        / "checkpoints"
        / task_id
        / state.checkpoint_run_id
        / "blobs"
        / baseline.files["src/app.py"].content_hash
    )
    assert original_blob.read_text() == "def work():\n    return 0\n"
    assert state.checkpoint_count == 0
    assert state.checkpoint_pending_number == 1

    (workspace / "src/app.py").write_text("fixed")
    edit = _edit_pair(0)
    verify = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(edit),
        tools=_tools(),
        extra=extra,
    )
    assert verify.loop_phase == "verify"
    state = load_loop_state(svc, task_id)
    assert state is not None
    assert state.checkpoint_count == 1
    checkpoint_1 = store.load_manifest(task_id, state.checkpoint_run_id, 1)
    assert checkpoint_1 is not None
    assert checkpoint_1.active_diff_hash == state.active_diff_hash

    failed_pair = _verify_pair(0, "AssertionError: still broken\nExit code: 1")
    failed = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(edit + failed_pair),
        tools=_tools(),
        extra=extra,
    )
    assert failed.loop_phase == "repair"
    history = edit + failed_pair + _refresh_pair(0, content="fixed")
    repair = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(history),
        tools=_tools(),
        extra=extra,
    )
    assert repair.loop_phase == "apply"
    state = load_loop_state(svc, task_id)
    assert state is not None
    baseline = store.load_manifest(task_id, state.checkpoint_run_id, 0)
    assert baseline is not None
    assert set(baseline.files) == {"src/app.py", "src/generated_helper.py"}
    assert baseline.files["src/generated_helper.py"].exists is False
    assert state.checkpoint_count == 1
    assert state.checkpoint_pending_number == 2

    (workspace / "src/generated_helper.py").write_text("VALUE = 2\n")
    generated = _edit_path_pair(
        1,
        "src/generated_helper.py",
        "VALUE = 2\n",
        "Successfully created src/generated_helper.py",
    )
    verify_again = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(history + generated),
        tools=_tools(),
        extra=extra,
    )
    assert verify_again.loop_phase == "verify"
    state = load_loop_state(svc, task_id)
    assert state is not None
    assert state.checkpoint_count == 2
    checkpoint_2 = store.load_manifest(task_id, state.checkpoint_run_id, 2)
    assert checkpoint_2 is not None
    assert checkpoint_2.active_diff_hash == state.active_diff_hash
    assert checkpoint_2.active_diff_hash != checkpoint_1.active_diff_hash


@pytest.mark.asyncio
async def test_all_commands_verify_the_same_diff(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    commands = ["pytest tests/test_x.py", "ruff check src"]
    _patch_loop(monkeypatch, report=_report_commands(commands))
    tools = _tools()
    edit = _edit_pair(0)

    await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    first_verify = await run_orch(cfg, FIX, messages=_ready_messages(edit), tools=tools)
    first_args = json.loads(first_verify.tool_calls[0]["function"]["arguments"])
    assert first_args["command"] == commands[0]

    first_pass = _command_pair(0, commands[0], "1 passed\nExit code: 0")
    second_verify = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(edit + first_pass),
        tools=tools,
    )
    second_args = json.loads(second_verify.tool_calls[0]["function"]["arguments"])
    assert second_args["command"] == commands[1]
    assert second_verify.loop_phase == "verify"

    svc = TaskService(Store(cfg.settings.db_path))
    mid = load_loop_state(svc, svc.session_task().task_id)
    assert mid is not None
    assert mid.verify_index == 1
    assert len(mid.verification_results) == 1
    assert mid.verification_results[0].diff_hash == mid.active_diff_hash

    second_pass = _command_pair(1, commands[1], "All checks passed\nExit code: 0")
    done = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(edit + first_pass + second_pass),
        tools=tools,
    )
    assert done.loop_phase == "verified"
    state = load_loop_state(svc, svc.session_task().task_id)
    assert state is not None
    assert [(row.command, row.exit_code) for row in state.verification_results] == [
        (commands[0], 0),
        (commands[1], 0),
    ]
    assert {row.diff_hash for row in state.verification_results} == {
        state.active_diff_hash
    }


@pytest.mark.asyncio
async def test_mutation_invalidates_prior_command_success(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    commands = ["pytest tests/test_x.py", "ruff check src"]
    captured: dict = {}
    _patch_loop(monkeypatch, report=_report_commands(commands), captured=captured)
    captured["planned_calls"] = [
        _edit_call(),
        [
            {
                "id": "call_generated",
                "type": "function",
                "function": {
                    "name": "editor",
                    "arguments": json.dumps(
                        {
                            "path": "src/generated_helper.py",
                            "content": "VALUE = 'new untracked task file'\n",
                        }
                    ),
                },
            }
        ],
    ]
    tools = _tools()
    edit = _edit_pair(0)
    pytest_pass = _command_pair(0, commands[0], "1 passed\nExit code: 0")
    ruff_fail = _command_pair(1, commands[1], "E501 too long\nExit code: 1")

    await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    await run_orch(cfg, FIX, messages=_ready_messages(edit), tools=tools)
    await run_orch(cfg, FIX, messages=_ready_messages(edit + pytest_pass), tools=tools)
    before_failure = load_loop_state(
        TaskService(Store(cfg.settings.db_path)),
        TaskService(Store(cfg.settings.db_path)).session_task().task_id,
    )
    assert before_failure is not None
    original_diff_hash = before_failure.active_diff_hash
    failed = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(edit + pytest_pass + ruff_fail),
        tools=tools,
    )
    assert failed.loop_phase == "repair"

    history = edit + pytest_pass + ruff_fail + _refresh_pair(0)
    repair = await run_orch(cfg, FIX, messages=_ready_messages(history), tools=tools)
    assert repair.loop_phase == "apply"

    repaired_edit = _edit_path_pair(
        1,
        "src/generated_helper.py",
        "VALUE = 'new untracked task file'\n",
        "Successfully created src/generated_helper.py",
    )
    verify_again = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(history + repaired_edit),
        tools=tools,
    )
    args = json.loads(verify_again.tool_calls[0]["function"]["arguments"])
    assert args["command"] == commands[0]
    state = load_loop_state(
        TaskService(Store(cfg.settings.db_path)),
        TaskService(Store(cfg.settings.db_path)).session_task().task_id,
    )
    assert state is not None
    assert state.verify_index == 0
    assert state.verification_results == []
    assert state.active_diff_hash != original_diff_hash
    assert "src/generated_helper.py" in state.working_set.files_changed
    helper = state.working_set.files_read["src/generated_helper.py"]
    assert helper.content == "VALUE = 'new untracked task file'\n"
    assert len(helper.content_hash) == 64


@pytest.mark.asyncio
async def test_repair_path_feeds_failure_and_reverifies(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    captured = _patch_loop(monkeypatch)
    tools = _tools()

    await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    await run_orch(cfg, FIX, messages=_ready_messages(_edit_pair(0)), tools=tools)
    failed = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(
            _edit_pair(0) + _verify_pair(0, "FAILED tests/test_x.py::test_add\nExit code: 1")
        ),
        tools=tools,
    )
    assert failed.loop_phase == "repair"
    assert failed.loop_iteration == 1
    assert failed.tool_calls
    refresh_names = [call["function"]["name"] for call in failed.tool_calls]
    assert "read_files" in refresh_names
    assert "execute_command" in refresh_names

    repair = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(
            _edit_pair(0)
            + _verify_pair(0, "FAILED tests/test_x.py::test_add\nExit code: 1")
            + _refresh_pair(0)
        ),
        tools=tools,
    )
    assert repair.loop_phase == "apply"
    assert repair.loop_iteration == 2
    repair_packets = captured["dispatches"][-1]
    assert repair_packets
    prompt = repair_packets[0].prompt
    assert 'exit_code="1"' in prompt
    assert "FAILED tests/test_x.py::test_add" in prompt
    assert '<CURRENT_FILE path="src/app.py"' in prompt
    assert "return 1" in prompt
    assert "<CURRENT_DIFF" in prompt
    assert "diff --git a/src/app.py" in prompt
    assert "<OBJECTIVE>" in prompt
    assert "<INSTRUCTION>" in prompt
    assert "Do not restart investigation" in prompt

    svc = TaskService(Store(cfg.settings.db_path))
    persisted = load_loop_state(svc, svc.session_task().task_id)
    assert persisted is not None
    assert persisted.working_set.objective == FIX
    assert persisted.working_set.acceptance_commands == ["pytest tests/test_x.py"]
    assert persisted.working_set.files_changed == ["src/app.py"]
    assert persisted.working_set.files_read["src/app.py"].content.endswith("return 1\n")
    assert len(persisted.working_set.files_read["src/app.py"].content_hash) == 64
    assert "diff --git a/src/app.py" in persisted.working_set.current_diff
    assert persisted.working_set.refresh_pending == []

    after_repair = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(
            _edit_pair(0)
            + _verify_pair(0, "FAILED\nExit code: 1")
            + _refresh_pair(0)
            + _edit_pair(1)
        ),
        tools=tools,
    )
    assert after_repair.loop_phase == "verify"
    done = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(
            _edit_pair(0)
            + _verify_pair(0, "FAILED\nExit code: 1")
            + _refresh_pair(0)
            + _edit_pair(1)
            + _verify_pair(1, "2 passed\nExit code: 0")
        ),
        tools=tools,
    )
    assert done.loop_phase == "verified"
    assert "status: verified" in done.text


@pytest.mark.asyncio
async def test_failure_expands_working_set_without_spending_iteration(
    tmp_path: Path, monkeypatch
):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    captured = _patch_loop(monkeypatch)
    tools = _tools()
    edit = _edit_pair(0)
    failure = (
        'Traceback:\n  File "/tmp/test-repo/services/cache/redis.py", '
        "line 184, in fetch\n"
        "    return WidgetFactory.build()\n"
        "NameError: name 'WidgetFactory' is not defined\n"
        "Exit code: 1"
    )
    failed_verify = _verify_pair(0, failure)

    await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    await run_orch(cfg, FIX, messages=_ready_messages(edit), tools=tools)
    failed = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(edit + failed_verify),
        tools=tools,
    )
    assert failed.loop_phase == "expand"
    assert failed.loop_iteration == 1
    assert {
        call["function"]["name"] for call in failed.tool_calls
    } == {"read_files", "execute_command"}
    svc = TaskService(Store(cfg.settings.db_path))
    expanded_state = load_loop_state(svc, svc.session_task().task_id)
    assert expanded_state is not None
    assert expanded_state.checkpoint_count == 1

    history = edit + failed_verify + _refresh_pair(0)
    exact_read = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(history),
        tools=tools,
    )
    assert exact_read.loop_phase == "expand"
    assert exact_read.loop_iteration == 1
    assert len(exact_read.tool_calls) == 1
    expanded_state = load_loop_state(svc, svc.session_task().task_id)
    assert expanded_state is not None
    assert expanded_state.checkpoint_count == 1
    read_args = json.loads(exact_read.tool_calls[0]["function"]["arguments"])
    assert read_args["paths"] == ["services/cache/redis.py"]

    history += _expansion_read_pair(
        0,
        "services/cache/redis.py",
        "def fetch():\n    return WidgetFactory.build()\n",
    )
    exact_search = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(history),
        tools=tools,
    )
    assert exact_search.loop_phase == "expand"
    assert exact_search.loop_iteration == 1
    assert len(exact_search.tool_calls) == 1
    search_fn = exact_search.tool_calls[0]["function"]
    assert search_fn["name"] == "search_files"
    assert "WidgetFactory" in json.loads(search_fn["arguments"])["regex"]

    history += _search_pair(
        0,
        "WidgetFactory",
        "services/factories.py:10:class WidgetFactory:\n",
    )
    discovered_read = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(history),
        tools=tools,
    )
    assert discovered_read.loop_phase == "expand"
    assert discovered_read.loop_iteration == 1
    assert len(discovered_read.tool_calls) == 1
    discovered_args = json.loads(
        discovered_read.tool_calls[0]["function"]["arguments"]
    )
    assert discovered_args["paths"] == ["services/factories.py"]

    history += _expansion_read_pair(
        1,
        "services/factories.py",
        "class WidgetFactory:\n    @classmethod\n    def build(cls): return cls()\n",
    )
    repair = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(history),
        tools=tools,
    )
    assert repair.loop_phase == "apply"
    assert repair.loop_iteration == 2
    prompt = captured["dispatches"][-1][0].prompt
    assert '<EXPANDED_EVIDENCE path="services/cache/redis.py"' in prompt
    assert '<EXPANDED_EVIDENCE path="services/factories.py"' in prompt
    assert "README.md" not in prompt


@pytest.mark.asyncio
async def test_expansion_uses_workspace_semantic_fallback_after_exact_miss(
    tmp_path: Path, monkeypatch
):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    captured = _patch_loop(monkeypatch)
    semantic_queries: list[tuple[str, list[str]]] = []

    def semantic(_cfg, state):
        semantic_queries.append(
            (state.working_set.repo_root, list(state.expansion_symbols))
        )
        return ["services/semantic_factory.py"]

    monkeypatch.setattr("harness.gateway.orch._semantic_expansion_paths", semantic)
    tools = _tools()
    edit = _edit_pair(0)
    failed_verify = _verify_pair(
        0,
        "NameError: name 'WidgetFactory' is not defined\nExit code: 1",
    )

    await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    await run_orch(cfg, FIX, messages=_ready_messages(edit), tools=tools)
    await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(edit + failed_verify),
        tools=tools,
    )
    history = edit + failed_verify + _refresh_pair(0)
    exact_search = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(history),
        tools=tools,
    )
    assert exact_search.tool_calls[0]["function"]["name"] == "search_files"

    history += _search_pair(0, "WidgetFactory", "No results found")
    semantic_read = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(history),
        tools=tools,
    )
    assert semantic_read.loop_phase == "expand"
    assert semantic_read.loop_iteration == 1
    assert semantic_queries == [("/tmp/test-repo", ["WidgetFactory"])]
    args = json.loads(semantic_read.tool_calls[0]["function"]["arguments"])
    assert args["paths"] == ["services/semantic_factory.py"]

    history += _expansion_read_pair(
        1,
        "services/semantic_factory.py",
        "class WidgetFactory:\n    pass\n",
    )
    repair = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(history),
        tools=tools,
    )
    assert repair.loop_phase == "apply"
    assert repair.loop_iteration == 2
    prompt = captured["dispatches"][-1][0].prompt
    assert 'path="services/semantic_factory.py"' in prompt


@pytest.mark.asyncio
async def test_five_cycles_exhaust_without_sixth_apply(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    _patch_loop(monkeypatch)
    tools = _tools()
    extra: list[dict] = []

    first = await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    assert first.loop_iteration == 1
    extra.extend(_edit_pair(0))
    await run_orch(cfg, FIX, messages=_ready_messages(extra), tools=tools)

    for cycle in range(MAX_CYCLES):
        extra.extend(_verify_pair(cycle, f"FAIL unique-{cycle}\nExit code: 1"))
        result = await run_orch(cfg, FIX, messages=_ready_messages(extra), tools=tools)
        if cycle < MAX_CYCLES - 1:
            assert result.loop_phase == "repair"
            assert result.tool_calls
            extra.extend(_refresh_pair(cycle))
            result = await run_orch(cfg, FIX, messages=_ready_messages(extra), tools=tools)
            assert result.loop_phase == "apply"
            extra.extend(_edit_pair(cycle + 1))
            nxt = await run_orch(cfg, FIX, messages=_ready_messages(extra), tools=tools)
            assert nxt.loop_phase == "verify"
        else:
            assert result.loop_phase == "exhausted"
            assert "status: exhausted" in result.text
            assert result.loop_iteration == MAX_CYCLES
            assert not result.tool_calls or result.tool_calls[0]["function"]["name"] != "editor"


@pytest.mark.asyncio
async def test_gather_rounds_do_not_spend_repair_budget(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    _patch_loop(monkeypatch)

    async def more_gather(*_a, **_k):
        return "gather", [{"name": "read_file", "arguments": {"path": "src/app.py"}}], []

    monkeypatch.setattr("harness.gateway.orch.plan_orch", more_gather)

    first = await run_orch(
        cfg,
        FIX,
        messages=[{"role": "user", "content": FIX}],
        tools=_tools(),
    )
    assert first.tool_calls
    assert first.loop_phase == "gather"
    assert first.loop_iteration == 0

    second = await run_orch(
        cfg,
        FIX,
        messages=[{"role": "user", "content": FIX}, *_read_pair(0)],
        tools=_tools(),
    )
    assert second.loop_phase == "gather"
    assert second.loop_iteration == 0

    svc = TaskService(Store(cfg.settings.db_path))
    state = load_loop_state(svc, svc.session_task().task_id)
    assert state is not None
    assert state.iteration == 0
    assert state.phase == "gather"


@pytest.mark.asyncio
async def test_identical_failure_exhausts_early(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    _patch_loop(monkeypatch)
    tools = _tools()
    same = "AssertionError: expected 4\nExit code: 1"

    await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    await run_orch(cfg, FIX, messages=_ready_messages(_edit_pair(0)), tools=tools)
    await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(_edit_pair(0) + _verify_pair(0, same)),
        tools=tools,
    )
    history = _edit_pair(0) + _verify_pair(0, same) + _refresh_pair(0)
    await run_orch(cfg, FIX, messages=_ready_messages(history), tools=tools)
    await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(history + _edit_pair(1, "no changes made")),
        tools=tools,
    )
    # empty mutation after repair also exhausts; use a real second apply then same fail
    # Reset path: if previous turn exhausted on empty mutation, start a dedicated same-fail sequence.
    # The empty-mutation case is asserted separately below if phase already exhausted.
    svc = TaskService(Store(cfg.settings.db_path))
    state = load_loop_state(svc, svc.session_task().task_id)
    assert state is not None
    assert state.phase == "exhausted"


@pytest.mark.asyncio
async def test_same_failure_hash_after_repair_exhausts(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    _patch_loop(monkeypatch)
    tools = _tools()
    same = "AssertionError: expected 4\nExit code: 1"
    extra = _edit_pair(0)
    await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    await run_orch(cfg, FIX, messages=_ready_messages(extra), tools=tools)
    repaired = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(extra + _verify_pair(0, same)),
        tools=tools,
    )
    assert repaired.loop_phase == "repair"
    extra = extra + _verify_pair(0, same) + _refresh_pair(0)
    repaired = await run_orch(cfg, FIX, messages=_ready_messages(extra), tools=tools)
    assert repaired.loop_phase == "apply"
    extra = extra + _edit_pair(1, "Successfully applied a different hunk")
    await run_orch(cfg, FIX, messages=_ready_messages(extra), tools=tools)
    done = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(extra + _verify_pair(1, same)),
        tools=tools,
    )
    assert done.loop_phase == "exhausted"
    assert "status: exhausted" in done.text


@pytest.mark.asyncio
async def test_repair_delta_can_run_before_semantic_critic_acceptance(
    tmp_path: Path,
    monkeypatch,
):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    report = _report()
    captured = _patch_loop(monkeypatch, report=report)
    tools = _tools()
    failure = "AssertionError: expected 4\nExit code: 1"

    await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    extra = _edit_pair(0)
    await run_orch(cfg, FIX, messages=_ready_messages(extra), tools=tools)
    repair = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(extra + _verify_pair(0, failure)),
        tools=tools,
    )
    assert repair.loop_phase == "repair"

    report.shots[0].qa_pass = False
    report.critic_verdict = "reject"
    repaired = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(extra + _verify_pair(0, failure) + _refresh_pair(0)),
        tools=tools,
    )
    assert repaired.loop_phase == "apply"
    assert repaired.tool_calls
    assert captured["planned"]


@pytest.mark.asyncio
async def test_missing_command_blocks_and_does_not_invent(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    _patch_loop(monkeypatch, report=_report(command=""))
    tools = _tools()
    result = await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    assert result.loop_phase == "blocked"
    assert "status: blocked" in result.text
    assert not result.tool_calls
    assert "pytest" not in result.text or "accept.commands" in result.text


@pytest.mark.asyncio
async def test_written_relay_does_not_enter_verified_mutation_loop(
    tmp_path: Path,
    monkeypatch,
):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    captured = _patch_loop(
        monkeypatch,
        report=_report(command="", text="Here is the requested onboarding message."),
    )
    intent = (
        "Paste this to them. They clone and look only. "
        "If they find a fix, write it locally and show cursor-cr."
    )
    messages: list[dict] = [{"role": "user", "content": intent}]
    for index in range(4):
        messages.extend(_read_pair(index))

    result = await run_orch(cfg, intent, messages=messages, tools=_tools())

    assert result.loop_phase == ""
    assert "status: blocked" not in result.text
    assert "Here is the requested onboarding message." in result.text
    assert not result.tool_calls
    assert captured["dispatch_kwargs"][0]["compiled_context"] == ""
    assert "GLOBAL CODE INTELLIGENCE" not in captured["dispatch_kwargs"][0]["thread"]


@pytest.mark.asyncio
async def test_repository_onboarding_executes_then_reports_command_evidence(
    tmp_path: Path,
    monkeypatch,
):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    captured = _patch_loop(
        monkeypatch,
        report=_report(
            command="",
            text="All three repositories are present on their requested branches.",
        ),
    )
    intent = """
Paste this to them. They clone and look.
git clone git@github.com:beargallbladder/outcited.com.git
git checkout feat/fae-evidence-compiler-knives-v0
git fetch origin staging/w31-private
git fetch origin release/designwins-w46-guided-minima-v5
git clone git@github.com:beargallbladder/octopart-ai.git
git clone git@github.com:beargallbladder/designwins.git
No git commit, no git push, and no deploy.
""".strip()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_commands",
                "parameters": {
                    "type": "object",
                    "properties": {"commands": {"type": "array"}},
                },
            },
        }
    ]

    first = await run_orch(
        cfg,
        intent,
        messages=[{"role": "user", "content": intent}],
        tools=tools,
    )
    assert len(first.tool_calls) == 1
    assert first.tool_calls[0]["function"]["name"] == "run_commands"
    args = json.loads(first.tool_calls[0]["function"]["arguments"])
    commands = "\n".join(args["commands"])
    assert "HARNESS_REPO_BOOTSTRAP" in commands
    assert commands.count("git clone") == 3
    assert "feat/fae-evidence-compiler-knives-v0" in commands
    assert "staging/w31-private" in commands
    assert "release/designwins-w46-guided-minima-v5" in commands
    assert "git commit" not in commands
    assert "git push" not in commands

    messages = [
        {"role": "user", "content": intent},
        {"role": "assistant", "tool_calls": first.tool_calls},
        {
            "role": "tool",
            "tool_call_id": first.tool_calls[0]["id"],
            "content": (
                "outcited.com feat/fae-evidence-compiler-knives-v0 clean\n"
                "octopart-ai main clean\ndesignwins main clean\nexit_code: 0"
            ),
        },
    ]
    second = await run_orch(cfg, intent, messages=messages, tools=tools)

    assert not second.tool_calls
    assert "All three repositories are present" in second.text
    assert "exit_code: 0" in captured["dispatch_kwargs"][0]["thread"]


@pytest.mark.asyncio
async def test_degraded_critic_blocks_code_mutation(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    report = _report()
    report.critic_verdict = "degraded"
    _patch_loop(monkeypatch, report=report)
    result = await run_orch(cfg, FIX, messages=_ready_messages(), tools=_tools())
    assert result.loop_phase == "blocked"
    assert "no critic produced a valid grade" in result.text
    assert not result.tool_calls


@pytest.mark.asyncio
async def test_greenfield_execution_never_queries_global_discovery(
    tmp_path: Path,
    monkeypatch,
):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    _patch_loop(monkeypatch)
    called = 0

    def global_context(*_args, **_kwargs):
        nonlocal called
        called += 1
        return "must not appear"

    monkeypatch.setattr(
        "harness.gci.integration.global_discovery_context",
        global_context,
    )
    result = await run_orch(
        cfg,
        "GREENFIELD RUN gf_test\nImplement the approved API milestone.",
        messages=_ready_messages(),
        tools=_tools(),
    )
    assert result.loop_phase == "apply"
    assert called == 0


def test_verification_gated_repair_allows_only_safe_exact_delta():
    from harness.gateway.orch import _verification_gated_repair_reason

    exact = {
        "function": {
            "name": "replace_in_file",
            "arguments": {
                "path": "tests/test_api.py",
                "old_string": "similarities, differences = result",
                "new_string": "_similarities, differences = result",
            },
        }
    }
    assert _verification_gated_repair_reason([exact]) == ""

    dynamic_execution = {
        "function": {
            "name": "write_to_file",
            "arguments": {
                "path": "tests/test_api.py",
                "content": "result = eval(source)",
            },
        }
    }
    assert "dynamic execution" in _verification_gated_repair_reason(
        [dynamic_execution]
    )

    dependency = {
        "function": {
            "name": "replace_in_file",
            "arguments": {
                "path": "pyproject.toml",
                "old_string": "dependencies = []",
                "new_string": 'dependencies = ["requests"]',
            },
        }
    }
    assert "dependency manifests" in _verification_gated_repair_reason([dependency])


def test_reconstructed_command_result_is_available_without_write_exchange():
    from harness.orch_loop import (
        last_run_for_command,
        last_run_for_command_since_write,
    )

    messages = _verify_pair(0, '{"exit_code":0,"stdout":"1 passed"}')
    assert last_run_for_command_since_write(messages, "pytest tests/test_x.py") is None
    recovered = last_run_for_command(messages, "pytest tests/test_x.py")
    assert recovered is not None
    assert "1 passed" in recovered[2]


@pytest.mark.asyncio
async def test_text_only_answer_blocks_when_no_mutation_can_be_planned(
    tmp_path: Path,
    monkeypatch,
):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    captured = {"planned_calls": [[]]}
    _patch_loop(monkeypatch, captured=captured)
    result = await run_orch(cfg, FIX, messages=_ready_messages(), tools=_tools())
    assert result.loop_phase == "blocked"
    assert "no executable client mutation calls" in result.text
    assert not result.tool_calls


@pytest.mark.asyncio
async def test_intent_named_command_seeds_empty_packet(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    _patch_loop(monkeypatch, report=_report(command=""))
    intent = FIX + " Use pytest tests/test_x.py as the acceptance command."
    result = await run_orch(cfg, intent, messages=_ready_messages(), tools=_tools())
    assert result.loop_phase == "apply"
    assert result.tool_calls
    assert result.tool_calls[0]["function"]["name"] == "editor"


@pytest.mark.asyncio
async def test_unsafe_command_is_blocked_and_never_returned(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    _patch_loop(monkeypatch, report=_report(command="pytest tests; rm -rf /tmp/harness-unsafe"))
    tools = _tools()
    result = await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    assert result.loop_phase == "blocked"
    assert not result.tool_calls
    assert "rm -rf" not in (result.text or "")


@pytest.mark.asyncio
async def test_critic_rejected_output_cannot_drive_apply(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    captured = _patch_loop(monkeypatch, report=_report(qa_pass=False))
    tools = _tools()

    rejected = await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    assert rejected.loop_phase != "apply"
    assert not rejected.tool_calls
    assert "planned" not in captured

    cfg2 = _cfg(tmp_path / "plain")

    async def pick(*_a, **_k):
        return "m5_qwen", _model()

    async def dispatch(*_a, **_k):
        return _report(qa_pass=True)

    async def no_edit(*_a, **_k):
        return []

    monkeypatch.setattr("harness.gateway.orch.pick_foreman", pick)
    monkeypatch.setattr("harness.gateway.orch.run_dispatch", dispatch)
    monkeypatch.setattr("harness.gateway.orch.plan_actions", no_edit)
    text_only = await run_orch(cfg2, FIX, messages=_ready_messages(), tools=tools)
    assert text_only.loop_phase != "verified"
    assert "status: verified" not in (text_only.text or "")


@pytest.mark.asyncio
async def test_verified_frontier_rescue_can_drive_apply(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    report = _report(qa_pass=False)
    report.critic_verdict = "reject"
    report.frontier_verified = True
    report.frontier_text = "Exact independently verified implementation."
    captured = _patch_loop(monkeypatch, report=report)
    result = await run_orch(cfg, FIX, messages=_ready_messages(), tools=_tools())
    assert result.loop_phase == "apply"
    assert result.tool_calls
    assert captured["planned"] is True


@pytest.mark.asyncio
async def test_client_run_commands_json_exit_verifies(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    _patch_loop(monkeypatch)
    tools = _tools()
    payload = json.dumps(
        [{"query": "pytest tests/test_x.py", "result": "1 passed", "exitCode": 0}]
    )
    await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    await run_orch(cfg, FIX, messages=_ready_messages(_edit_pair(0)), tools=tools)
    done = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(_edit_pair(0) + _verify_pair(0, payload)),
        tools=tools,
    )
    assert done.loop_phase == "verified"
    assert "exit_code: 0" in done.text


@pytest.mark.asyncio
async def test_pytest_passed_without_exit_is_not_verified(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    _patch_loop(monkeypatch)
    tools = _tools()
    await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    await run_orch(cfg, FIX, messages=_ready_messages(_edit_pair(0)), tools=tools)
    result = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(
            _edit_pair(0) + _verify_pair(0, "===== 1 passed in 0.02s =====")
        ),
        tools=tools,
    )
    assert result.loop_phase != "verified"
    assert "status: verified" not in (result.text or "")


@pytest.mark.asyncio
async def test_verify_binds_to_client_run_commands_catalog(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    _patch_loop(monkeypatch)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_files",
                "parameters": {"type": "object", "properties": {"paths": {}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "editor",
                "parameters": {"type": "object", "properties": {"path": {}, "content": {}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_commands",
                "parameters": {"type": "object", "properties": {"commands": {}}},
            },
        },
    ]
    await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    second = await run_orch(cfg, FIX, messages=_ready_messages(_edit_pair(0)), tools=tools)
    assert second.tool_calls
    assert second.tool_calls[0]["function"]["name"] == "run_commands"
    args = json.loads(second.tool_calls[0]["function"]["arguments"])
    assert args["commands"] == ["pytest tests/test_x.py"]
    assert "rm " not in json.dumps(args)
