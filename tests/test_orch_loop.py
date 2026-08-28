from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.config import AppConfig, ModelConfig, Settings
from harness.dispatch import AcceptSpec, DispatchReport, Packet, Shot
from harness.orch_loop import (
    MAX_CYCLES,
    command_allowed,
    load_loop_state,
    parse_argv,
    select_verify_command,
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
                        "arguments": '{"paths":["src/app.py"]}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"read_{index}",
            "content": "src/app.py\n" + ("def work(): pass\n" * 20),
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
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"edit_{index}",
                    "type": "function",
                    "function": {
                        "name": "editor",
                        "arguments": '{"path":"src/app.py","content":"fixed"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"edit_{index}", "content": result},
    ]


def _verify_pair(index: int, result: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"ver_{index}",
                    "type": "function",
                    "function": {
                        "name": "execute_command",
                        "arguments": '{"command":"pytest tests/test_x.py","timeout":60}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"ver_{index}", "content": result},
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
        return payload

    async def actions(*_a, **_k):
        captured["planned"] = True
        return _edit_call()

    monkeypatch.setattr("harness.gateway.orch.pick_foreman", pick)
    monkeypatch.setattr("harness.gateway.orch.run_dispatch", dispatch)
    monkeypatch.setattr("harness.gateway.orch.plan_actions", actions)
    monkeypatch.setattr("harness.gateway.orch.plan_orch", dispatch)
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


@pytest.mark.asyncio
async def test_happy_path_persists_across_run_orch_calls(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    _patch_loop(monkeypatch)
    tools = _tools()

    first = await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    assert first.tool_calls
    assert first.tool_calls[0]["function"]["name"] == "editor"
    assert first.loop_phase == "apply"
    assert first.loop_iteration == 1

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
    assert failed.loop_phase == "apply"
    assert failed.loop_iteration == 2
    repair_packets = captured["dispatches"][-1]
    assert repair_packets
    prompt = repair_packets[0].prompt
    assert "EXIT CODE: 1" in prompt
    assert "FAILED tests/test_x.py::test_add" in prompt

    after_repair = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(_edit_pair(0) + _verify_pair(0, "FAILED\nExit code: 1") + _edit_pair(1)),
        tools=tools,
    )
    assert after_repair.loop_phase == "verify"
    done = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(
            _edit_pair(0)
            + _verify_pair(0, "FAILED\nExit code: 1")
            + _edit_pair(1)
            + _verify_pair(1, "2 passed\nExit code: 0")
        ),
        tools=tools,
    )
    assert done.loop_phase == "verified"
    assert "status: verified" in done.text


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
    await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(_edit_pair(0) + _verify_pair(0, same) + _edit_pair(1, "no changes made")),
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
    assert repaired.loop_phase == "apply"
    extra = extra + _verify_pair(0, same) + _edit_pair(1, "Successfully applied a different hunk")
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
async def test_missing_command_blocks_and_does_not_invent(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    _patch_loop(monkeypatch, report=_report(command=""))
    tools = _tools()
    await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    result = await run_orch(cfg, FIX, messages=_ready_messages(_edit_pair(0)), tools=tools)
    assert result.loop_phase == "blocked"
    assert "status: blocked" in result.text
    assert not result.tool_calls
    assert "pytest" not in result.text or "accept.commands" in result.text


@pytest.mark.asyncio
async def test_unsafe_command_is_blocked_and_never_returned(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    _patch_loop(monkeypatch, report=_report(command="pytest tests; rm -rf /tmp/harness-unsafe"))
    tools = _tools()
    await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    result = await run_orch(cfg, FIX, messages=_ready_messages(_edit_pair(0)), tools=tools)
    assert result.loop_phase == "blocked"
    assert not result.tool_calls
    assert "rm -rf" not in (result.text or "")


@pytest.mark.asyncio
async def test_qa_pass_does_not_establish_or_revoke_verified(tmp_path: Path, monkeypatch):
    from harness.gateway.orch import run_orch

    cfg = _cfg(tmp_path)
    captured = _patch_loop(monkeypatch, report=_report(qa_pass=False))
    tools = _tools()

    first = await run_orch(cfg, FIX, messages=_ready_messages(), tools=tools)
    assert first.loop_phase == "apply"
    assert captured["planned"] is True

    await run_orch(cfg, FIX, messages=_ready_messages(_edit_pair(0)), tools=tools)
    verified = await run_orch(
        cfg,
        FIX,
        messages=_ready_messages(_edit_pair(0) + _verify_pair(0, "not yet mentioned\nExit code: 0")),
        tools=tools,
    )
    assert verified.loop_phase == "verified"
    assert "not yet" not in verified.text.split("status:")[0] or "status: verified" in verified.text

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
