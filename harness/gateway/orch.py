"""Cline Model ID harness-orch: Cline is the hands, the fleet is the workers."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.task.code_index import normalize_repo_root

from harness.dispatch import (
    DispatchReport,
    bind_gather_calls,
    default_gather_calls,
    evidence_covers_intent,
    is_change_job,
    is_orch_echo,
    merge_tool_catalog,
    packets_claim_unread,
    pick_foreman,
    plan_actions,
    plan_orch,
    run_dispatch,
    sanitize_packets,
)
from harness.orch_loop import (
    MAX_CYCLES,
    TERMINAL,
    LoopState,
    commands_from_packets,
    failure_hash,
    last_run_since_write,
    last_tool_exchanges,
    last_write_result,
    load_loop_state,
    parse_command_outcome,
    parse_mutation,
    remember_failure,
    remember_write,
    repair_packets,
    save_loop_state,
    select_verify_command,
    shot_texts,
    terminal_text,
    verify_tool_calls,
    working_set_text,
)


_ENV_BLOCK = re.compile(r"<env>(.*?)</env>", re.IGNORECASE | re.DOTALL)
_ENV_WORKDIR = re.compile(r"Working Directory:\s*([^\n<]+)", re.IGNORECASE)
_BODY_WORKSPACE_KEYS = ("workspace_root", "workspace_path", "workspace", "cwd")


def _add_workspace_candidate(found: list[str], raw: str) -> None:
    text = raw.strip().strip("\"'")
    if not text:
        return
    path = Path(text).expanduser()
    if not path.is_absolute():
        return
    found.append(str(normalize_repo_root(path)))


def cline_workspace_root(
    messages: list[Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    """Active Cline workspace. Fail closed if missing or if two roots disagree."""
    found: list[str] = []
    if extra:
        for key in _BODY_WORKSPACE_KEYS:
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                _add_workspace_candidate(found, value)
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        text = _content_text(item.get("content"))
        if not text:
            continue
        for block in _ENV_BLOCK.findall(text):
            match = _ENV_WORKDIR.search(block)
            if match:
                _add_workspace_candidate(found, match.group(1))
    unique = list(dict.fromkeys(found))
    if len(unique) != 1:
        return None
    return Path(unique[0])


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"text", "output_text", "tool_result"}:
                parts.append(str(block.get("text") or block.get("content") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return ""


def _is_toolish(item: dict[str, Any]) -> bool:
    if item.get("role") == "tool":
        return True
    content = item.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"tool_result", "tool_use"}:
                return True
    text = _content_text(content)
    if text.startswith("[") and "Result" in text[:120]:
        return True
    return False


def last_user_text(messages: list[Any]) -> str:
    for item in reversed(messages or []):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        if _is_toolish(item):
            continue
        text = _content_text(item.get("content"))
        if text:
            return text
    for item in reversed(messages or []):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        text = _content_text(item.get("content"))
        if text:
            return text
    return ""


def gather_rounds(messages: list[Any]) -> int:
    n = 0
    for item in messages or []:
        if isinstance(item, dict) and item.get("role") == "assistant" and item.get("tool_calls"):
            n += 1
    return n


def has_action_round(messages: list[Any]) -> bool:
    from harness.dispatch import WRITE_TOOLS

    for item in messages or []:
        if not isinstance(item, dict) or item.get("role") != "assistant":
            continue
        for call in item.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else {}
            name = str((fn or {}).get("name") or "").lower()
            if name in WRITE_TOOLS:
                return True
    return False


def _is_change_job(intent: str) -> bool:
    return is_change_job(intent)


def cline_tool_catalog(tools: list[Any] | None) -> dict[str, tuple[str, ...]]:
    catalog: dict[str, tuple[str, ...]] = {}
    for item in tools or []:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = str(fn.get("name") or item.get("name") or "").strip()
        if not name:
            continue
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        catalog[name] = tuple(str(k) for k in props.keys())
    return catalog


def _is_orch_noise(text: str) -> bool:
    t = text.lstrip()
    return t.startswith("Harness orch") or t.startswith("QA FAIL closed") or t.startswith("Running coder pool")


def has_repo_evidence(messages: list[Any]) -> bool:
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "tool":
            text = _content_text(item.get("content"))
            if len(text) > 20 and not _is_orch_noise(text):
                return True
        if _is_toolish(item):
            text = _content_text(item.get("content"))
            if len(text) > 80 and not _is_orch_noise(text) and "tools=execute_command" not in text:
                return True
    return False


def compact_thread(messages: list[Any], limit: int = 60000) -> str:
    parts: list[str] = []
    pending: dict[str, str] = {}
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        if role == "assistant" and item.get("tool_calls"):
            for call in item.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = str(fn.get("name") or "")
                args = fn.get("arguments") or ""
                cid = str(call.get("id") or "")
                parts.append(f"assistant-tool {name} {args}"[:400])
                if cid:
                    pending[cid] = name
            continue
        if role == "assistant":
            text = _content_text(item.get("content"))
            if _is_orch_noise(text):
                continue
        if role == "tool":
            name = pending.get(str(item.get("tool_call_id") or ""), "tool")
            text = _content_text(item.get("content"))
            if text and not _is_orch_noise(text):
                snippet = text[:14000]
                if len(text) > len(snippet):
                    snippet += (
                        "\n[TOOL RESULT TRUNCATED BY HARNESS; the cutoff is not "
                        "evidence of a source-code defect]"
                    )
                parts.append(f"tool({name}): {snippet}")
            continue
        text = _content_text(item.get("content"))
        if not text or _is_orch_noise(text) or is_orch_echo(text):
            continue
        cap = 14000 if _is_toolish(item) else 1500
        snippet = text[:cap]
        if len(text) > len(snippet) and _is_toolish(item):
            snippet += (
                "\n[TOOL RESULT TRUNCATED BY HARNESS; the cutoff is not "
                "evidence of a source-code defect]"
            )
        parts.append(f"{role}: {snippet}")
    blob = "\n".join(parts[-40:])
    return blob[-limit:] if len(blob) > limit else blob


def stitch_report(report: DispatchReport) -> str:
    if report.slice_error:
        return f"Harness orch {report.run_id}\nQA FAIL closed: {report.slice_error}\n"
    passed = [s for s in report.shots if s.qa_pass and (s.result.text or "").strip()]
    answers: list[str] = []
    for shot in passed:
        body = (shot.result.text or "").strip()
        if len(passed) > 1:
            answers.append(f"### {shot.packet.id} {shot.packet.title}\n{body}")
        else:
            answers.append(body)
    if report.frontier_verified and report.frontier_text.strip():
        rescued = report.frontier_text.strip()
        if answers:
            answers.append(f"### Harness completion\n{rescued}")
        else:
            answers.append(rescued)
    footer = (
        f"harness-orch {report.run_id}  qa {len(passed)}/{len(report.shots)}  "
        f"verdict: {report.critic_verdict or 'python'}"
    )
    if report.frontier_run_id:
        footer += f"  rescue: {'verified' if report.frontier_verified else 'unresolved'}"
    if report.critic_verdict == "degraded" and answers:
        answers.insert(
            0,
            "QA DEGRADED: adversarial critic unavailable; the answers below passed "
            "machine acceptance only and are not critic-verified.",
        )
    if not answers:
        lines = [
            "QA FAIL closed. No shot returned a written answer that passed accept.",
            footer,
        ]
        for shot in report.shots:
            why = shot.qa_why or "miss"
            lines.append(f"- {shot.packet.id}: {why}")
        if report.frontier_run_id or report.frontier_why:
            lines.append("- rescue: the additional answer did not pass harness verification")
        return "\n".join(lines).strip() + "\n"
    # Do not append the results-json path here: Cline treats it as a file to read
    # and drags harness internals into the next gather round.
    lines = answers + ["", "---", footer]
    return "\n".join(lines).strip() + "\n"


def completion_body(text: str, model_id: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-orch-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": max(1, len(text.split()))},
    }


def completion_body_tools(calls: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-orch-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "", "tool_calls": calls},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": max(1, len(calls))},
    }


def sse_chunk(model_id: str, delta: dict[str, Any], finish: str | None = None) -> bytes:
    payload = {
        "id": "chatcmpl-orch",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


@dataclass
class OrchResult:
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    loop_phase: str = ""
    loop_iteration: int = 0


def _task_service(cfg) -> TaskService | None:
    settings = getattr(cfg, "settings", None)
    db_path = getattr(settings, "db_path", None)
    if not db_path:
        return None
    from harness.storage.db import Store
    from harness.task.service import TaskService

    return TaskService(Store(db_path))


def _with_loop(result: OrchResult, state: LoopState | None) -> OrchResult:
    if state is not None:
        result.loop_phase = state.phase
        result.loop_iteration = state.iteration
    return result


def _waiting(state: LoopState, what: str) -> OrchResult:
    return _with_loop(
        OrchResult(text=f"Harness orch loop\nstatus: {state.phase}\nwaiting: {what}\n"),
        state,
    )


def _terminal(svc: TaskService | None, task_id: str | None, state: LoopState) -> OrchResult:
    if svc and task_id:
        save_loop_state(svc, task_id, state)
    return _with_loop(OrchResult(text=terminal_text(state)), state)


async def _emit_apply(
    cfg,
    intent: str,
    thread: str,
    catalog: dict[str, tuple[str, ...]],
    tools: list[Any],
    report,
    svc: TaskService,
    task_id: str,
    state: LoopState,
    *,
    picked,
) -> OrchResult | None:
    if state.iteration >= MAX_CYCLES:
        state.phase = "exhausted"
        return _terminal(svc, task_id, state)
    texts = shot_texts(report, require_qa=False)
    if not texts or picked is None:
        return None
    _key, foreman = picked
    calls = await plan_actions(
        foreman,
        intent,
        thread,
        "\n\n".join(texts),
        catalog,
        list(tools or []),
        working_set=working_set_text(state),
    )
    if not calls:
        return None
    if report is not None:
        state.commands = commands_from_packets(report.packets) or state.commands
    state.iteration += 1
    if state.iteration > MAX_CYCLES:
        state.phase = "exhausted"
        state.iteration = MAX_CYCLES
        return _terminal(svc, task_id, state)
    state.phase = "apply"
    save_loop_state(svc, task_id, state)
    return _with_loop(OrchResult(tool_calls=calls), state)


async def _advance_after_apply(
    messages: list[Any],
    catalog: dict[str, tuple[str, ...]],
    svc: TaskService,
    task_id: str,
    state: LoopState,
) -> OrchResult:
    write = last_write_result(messages)
    if write is None:
        return _waiting(state, "Cline mutation result")
    from harness.dispatch import WRITE_TOOLS

    for name, args, text in last_tool_exchanges(messages):
        if name.lower() in WRITE_TOOLS:
            remember_write(state, args, text)
    changed, diff_hash = parse_mutation(write[2])
    state.prev_diff_hash = state.last_diff_hash
    state.last_diff_hash = diff_hash
    if not changed and state.last_failure_hash:
        state.phase = "exhausted"
        state.blocked_reason = state.blocked_reason or "empty mutation after repair"
        return _terminal(svc, task_id, state)
    command, reason = select_verify_command(state.commands)
    if not command:
        state.phase = "blocked"
        state.blocked_reason = reason
        return _terminal(svc, task_id, state)
    calls = verify_tool_calls(command, catalog)
    if not calls:
        state.phase = "blocked"
        state.blocked_reason = "no Cline run tool available"
        return _terminal(svc, task_id, state)
    state.phase = "verify"
    state.last_cmd = command
    save_loop_state(svc, task_id, state)
    return _with_loop(OrchResult(tool_calls=calls), state)


async def _advance_after_verify(
    cfg,
    intent: str,
    thread: str,
    catalog: dict[str, tuple[str, ...]],
    tools: list[Any],
    messages: list[Any],
    svc: TaskService,
    task_id: str,
    state: LoopState,
    picked,
) -> OrchResult:
    run = last_run_since_write(messages)
    if run is None:
        return _waiting(state, "Cline verification result")
    exit_code, timed_out, stdout, stderr = parse_command_outcome(run[2])
    state.last_exit = exit_code
    state.timed_out = timed_out
    state.stdout_tail = stdout
    state.stderr_tail = stderr
    if (not timed_out) and exit_code == 0:
        state.phase = "verified"
        return _terminal(svc, task_id, state)
    remember_failure(state)
    fail_fp = failure_hash(state)
    same_fail = bool(state.last_failure_hash and fail_fp == state.last_failure_hash)
    same_diff = bool(
        state.prev_diff_hash
        and state.last_diff_hash
        and state.last_diff_hash == state.prev_diff_hash
    )
    state.prev_failure_hash = state.last_failure_hash
    state.last_failure_hash = fail_fp
    if same_fail or (same_fail and same_diff) or state.iteration >= MAX_CYCLES:
        state.phase = "exhausted"
        return _terminal(svc, task_id, state)
    state.phase = "repair"
    save_loop_state(svc, task_id, state)
    return await _run_repair_apply(
        cfg, intent, thread, catalog, tools, svc, task_id, state, picked
    )


async def _run_repair_apply(
    cfg,
    intent: str,
    thread: str,
    catalog: dict[str, tuple[str, ...]],
    tools: list[Any],
    svc: TaskService,
    task_id: str,
    state: LoopState,
    picked,
) -> OrchResult:
    if state.iteration >= MAX_CYCLES:
        state.phase = "exhausted"
        return _terminal(svc, task_id, state)
    if picked is None:
        picked = await pick_foreman(cfg)
    if picked is None:
        state.phase = "blocked"
        state.blocked_reason = "no foreman for repair"
        return _terminal(svc, task_id, state)
    packets = repair_packets(intent, thread, state)
    report = await run_dispatch(cfg, intent, thread=thread, packets=packets)
    applied = await _emit_apply(
        cfg, intent, thread, catalog, tools, report, svc, task_id, state, picked=picked
    )
    if applied is not None:
        return applied
    state.phase = "exhausted"
    state.blocked_reason = state.blocked_reason or "repair produced no mutation"
    return _terminal(svc, task_id, state)


async def run_orch(
    cfg,
    intent: str,
    *,
    thread: str = "",
    messages: list[Any] | None = None,
    tools: list[Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> OrchResult:
    intent = (intent or "").strip()
    if not intent:
        return OrchResult(text="Harness orch needs a user message.", error="no intent")
    if not thread and messages:
        thread = compact_thread(messages)
    workspace = cline_workspace_root(messages, extra)
    from harness.task.search import embedder_thread_block

    embedder_hits = embedder_thread_block(intent)
    if embedder_hits:
        thread = f"{thread}\n\n{embedder_hits}".strip()
    if is_orch_echo(intent):
        return OrchResult(
            text=(
                "QA FAIL closed: that message is a previous harness-orch dump, not a new job. "
                "Ask the actual question.\n"
            )
        )
    catalog = merge_tool_catalog(cline_tool_catalog(tools))
    rounds = gather_rounds(messages or [])
    evidence = has_repo_evidence(messages or [])
    acted = has_action_round(messages or [])
    covered = evidence_covers_intent(intent, thread)
    svc = _task_service(cfg) if tools and _is_change_job(intent) else None
    task_id = ""
    state: LoopState | None = None
    if svc is not None:
        task = svc.session_task()
        task_id = task.task_id
        state = load_loop_state(svc, task_id)
        if state and state.intent and state.intent != intent:
            state = None
        if state is None and last_write_result(messages or []):
            state = LoopState(phase="apply", intent=intent, iteration=1)
        if state is None:
            state = LoopState(phase="gather", intent=intent)
            save_loop_state(svc, task_id, state)
        if state.phase in TERMINAL:
            return _with_loop(OrchResult(text=terminal_text(state)), state)
        if state.phase == "apply":
            return await _advance_after_apply(messages or [], catalog, svc, task_id, state)
        if state.phase == "verify":
            picked = await pick_foreman(cfg)
            return await _advance_after_verify(
                cfg,
                intent,
                thread,
                catalog,
                list(tools or []),
                messages or [],
                svc,
                task_id,
                state,
                picked,
            )
        if state.phase == "repair":
            picked = await pick_foreman(cfg)
            return await _run_repair_apply(
                cfg,
                intent,
                thread,
                catalog,
                list(tools or []),
                svc,
                task_id,
                state,
                picked,
            )

    # A client that sent no tools (plain curl, SDK chat) has no hands to run
    # gather calls; a tool_calls turn would render as an empty reply there.
    # Go straight to dispatch so orch always answers in text. With Cline,
    # permit one targeted follow-up after the initial broad gather, then use
    # the evidence instead of spending four rounds asking for more context.
    # README/package.json is not enough for a frontend/UI assessment: keep
    # gathering the source rather than dispatching a 'no frontend' packet.
    force_dispatch = (
        not tools or rounds >= 4 or (evidence and rounds >= 2 and not acted and covered)
    )
    if not force_dispatch and (not evidence or not covered):
        calls = default_gather_calls(catalog, intent, workspace=workspace)
        if calls:
            if svc and task_id and state is not None:
                state.phase = "gather"
                save_loop_state(svc, task_id, state)
            return _with_loop(OrchResult(tool_calls=calls), state)
    packets = None
    picked = await pick_foreman(cfg)
    if picked is None:
        return OrchResult(
            text=(
                "QA FAIL closed: the harness planning layer is temporarily unavailable.\n"
            ),
            error="no foreman",
        )
    if not force_dispatch and evidence and picked is not None:
        _foreman_key, foreman = picked
        mode, raw_calls, planned = await plan_orch(
            foreman,
            intent,
            thread,
            catalog,
            limit=8,
            gather_round=rounds,
        )
        if mode == "gather":
            calls = bind_gather_calls(raw_calls, catalog) or default_gather_calls(
                catalog, intent, workspace=workspace
            )
            if calls:
                if svc and task_id and state is not None:
                    state.phase = "gather"
                    save_loop_state(svc, task_id, state)
                return _with_loop(OrchResult(tool_calls=calls), state)
        if planned:
            planned = sanitize_packets(planned, thread)
            if packets_claim_unread(planned) and not covered:
                calls = default_gather_calls(catalog, intent, workspace=workspace)
                if calls:
                    return _with_loop(OrchResult(tool_calls=calls), state)
            packets = planned
    report = await run_dispatch(cfg, intent, thread=thread, packets=packets)
    if report.slice_error and not force_dispatch:
        calls = default_gather_calls(catalog, intent, workspace=workspace)
        if calls:
            return _with_loop(OrchResult(tool_calls=calls), state)
    if svc and task_id and state is not None and picked is not None:
        applied = await _emit_apply(
            cfg,
            intent,
            thread,
            catalog,
            list(tools or []),
            report,
            svc,
            task_id,
            state,
            picked=picked,
        )
        if applied is not None:
            return applied
    text = stitch_report(report)
    if tools and _is_change_job(intent) and picked is not None and svc is None:
        accepted = [
            (shot.result.text or "").strip()
            for shot in report.shots
            if shot.qa_pass and (shot.result.text or "").strip()
        ]
        if report.frontier_verified and report.frontier_text.strip():
            accepted.append(report.frontier_text.strip())
        if accepted:
            _foreman_key, foreman = picked
            calls = await plan_actions(
                foreman,
                intent,
                thread,
                "\n\n".join(accepted),
                catalog,
                list(tools or []),
            )
            if calls:
                return OrchResult(tool_calls=calls)
    return _with_loop(OrchResult(text=text), state)


async def run_orch_text(cfg, intent: str, thread: str = "") -> str:
    result = await run_orch(cfg, intent, thread=thread)
    return result.text
