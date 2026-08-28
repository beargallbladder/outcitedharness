"""Cline Model ID harness-orch: Cline is the hands, the fleet is the workers."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.checkpoints import CheckpointError, CheckpointStore
from harness.task.code_index import normalize_repo_root
from harness.context_compiler import compile_context
from harness.repo_contract import build_repo_contract

from harness.dispatch import (
    DispatchReport,
    bind_gather_calls,
    coder_models,
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
    VerificationRecord,
    add_semantic_expansion_paths,
    commands_named_in_intent,
    complete_working_set_refresh,
    consume_expansion_results,
    expansion_complete,
    expansion_needs_semantic,
    expansion_tool_calls,
    failure_hash,
    last_run_for_command_since_write,
    last_tool_exchanges,
    last_write_result,
    load_loop_state,
    parse_command_outcome,
    parse_mutation,
    prepare_failure_expansion,
    remember_failure,
    remember_reads,
    remember_write,
    refresh_working_set_calls,
    repair_packets,
    save_loop_state,
    select_verify_commands,
    shot_texts,
    terminal_text,
    verify_tool_calls,
    working_set_diff_hash,
)

log = logging.getLogger("harness.orch")


_ENV_BLOCK = re.compile(r"<env>(.*?)</env>", re.IGNORECASE | re.DOTALL)
_ENV_WORKDIR = re.compile(r"Working Directory:\s*([^\n<]+)", re.IGNORECASE)
_BODY_WORKSPACE_KEYS = ("workspace_root", "workspace_path", "workspace", "cwd")
_PATCH_PATH_RE = re.compile(
    r"(?m)^(?:\*\*\* (?:Add|Update) File: |(?:\+\+\+|---) [ab]/)([^\n]+)$"
)


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
        if report.critic_verdict in {"revise", "reject", "insufficient", "degraded"}:
            # A verified rescue resolves the failed local set. Mixing it with
            # partially accepted local prose can surface contradictory fixes.
            answers = [rescued]
        elif answers:
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


def _coder_context_tokens(cfg) -> int:
    default = int(getattr(getattr(cfg, "settings", None), "coder_context_tokens", 6_000))
    try:
        budgets = [
            int(model.coder_context_tokens or default)
            for _worker, model in coder_models(cfg)
        ]
    except Exception:
        budgets = []
    return min(budgets) if budgets else default


def _checkpoint_store(cfg) -> CheckpointStore:
    settings = getattr(cfg, "settings", None)
    results_dir = Path(getattr(settings, "results_dir"))
    max_bytes = int(getattr(settings, "checkpoint_max_file_bytes", 1_000_000))
    return CheckpointStore(results_dir / "checkpoints", max_file_bytes=max_bytes)


def _normalized_mutation_path(raw: Any, workspace: Path | None = None) -> str:
    value = str(raw or "").strip().replace("\\", "/")
    if not value or "\x00" in value:
        return ""
    path = Path(value)
    if path.is_absolute():
        if workspace is None:
            return ""
        try:
            value = path.resolve(strict=False).relative_to(
                workspace.resolve()
            ).as_posix()
        except (OSError, ValueError):
            return ""
    parts = [part for part in value.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        return ""
    return "/".join(parts)


def _function_call(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    name = str((fn or {}).get("name") or "").lower().replace("-", "_")
    raw = (fn or {}).get("arguments")
    if isinstance(raw, dict):
        return name, raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        return name, parsed if isinstance(parsed, dict) else {}
    return name, {}


def _paths_from_write(
    name: str,
    args: dict[str, Any],
    workspace: Path | None = None,
) -> list[str]:
    out: list[str] = []

    def add(raw: Any) -> None:
        path = _normalized_mutation_path(raw, workspace)
        if path and path not in out:
            out.append(path)

    add(args.get("path") or args.get("file_path") or args.get("rel_path"))
    if isinstance(args.get("paths"), list):
        for raw in args["paths"]:
            add(raw)
    for key in ("edits", "changes", "files"):
        rows = args.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                add(row.get("path") or row.get("file_path") or row.get("rel_path"))
    if not out and name == "apply_diff":
        patch = str(args.get("patch") or args.get("diff") or "")
        for match in _PATCH_PATH_RE.finditer(patch):
            add(match.group(1).strip())
    return out


def _mutation_paths_from_calls(
    calls: list[dict[str, Any]],
    workspace: Path | None = None,
) -> tuple[list[str], str]:
    from harness.dispatch import WRITE_TOOLS

    paths: list[str] = []
    saw_write = False
    for call in calls:
        if not isinstance(call, dict):
            continue
        name, args = _function_call(call)
        if name not in WRITE_TOOLS or name == "attempt_completion":
            continue
        saw_write = True
        current = _paths_from_write(name, args, workspace)
        if not current:
            return [], f"write call {name} has no safe attributable path"
        for path in current:
            if path not in paths:
                paths.append(path)
    if not saw_write:
        return [], "planned action contains no mutation"
    return paths, ""


def _latest_write_exchanges(
    messages: list[Any],
) -> list[tuple[str, dict[str, Any], str]]:
    from harness.dispatch import WRITE_TOOLS

    start = -1
    for index, item in enumerate(messages or []):
        if not isinstance(item, dict) or item.get("role") != "assistant":
            continue
        calls = item.get("tool_calls") if isinstance(item.get("tool_calls"), list) else []
        if any(_function_call(call)[0] in WRITE_TOOLS for call in calls if isinstance(call, dict)):
            start = index
    if start < 0:
        return []
    return [
        exchange
        for exchange in last_tool_exchanges(messages[start:])
        if exchange[0].lower().replace("-", "_") in WRITE_TOOLS
        and exchange[0].lower().replace("-", "_") != "attempt_completion"
    ]


def _capture_apply_baseline(
    cfg,
    task_id: str,
    state: LoopState,
    calls: list[dict[str, Any]],
    workspace: Path | None,
    next_iteration: int,
) -> bool:
    root = workspace
    if root is None and state.working_set.repo_root:
        root = Path(state.working_set.repo_root)
    if root is None:
        state.checkpoint_error = "active workspace root is unavailable"
        state.blocked_reason = "checkpoint refused mutation: active workspace root is unavailable"
        return False
    paths, reason = _mutation_paths_from_calls(calls, root)
    if not paths:
        state.checkpoint_error = reason
        state.blocked_reason = f"checkpoint refused mutation: {reason}"
        return False
    if not state.checkpoint_run_id:
        state.checkpoint_run_id = uuid.uuid4().hex
    try:
        _checkpoint_store(cfg).capture_baseline(
            task_id=task_id,
            run_id=state.checkpoint_run_id,
            repo_root=root,
            intent=state.working_set.objective or state.intent,
            paths=paths,
            before_iteration=next_iteration,
        )
    except CheckpointError as exc:
        state.checkpoint_error = str(exc)
        state.blocked_reason = f"checkpoint refused mutation: {exc}"
        return False
    state.checkpoint_task_id = task_id
    state.checkpoint_pending_paths = paths
    state.checkpoint_pending_number = next_iteration
    state.checkpoint_error = ""
    return True


def _finalize_pending_checkpoint(cfg, state: LoopState) -> None:
    number = state.checkpoint_pending_number
    if not number or number <= state.checkpoint_count:
        return
    store = _checkpoint_store(cfg)
    manifest = store.record_checkpoint(
        task_id=state.checkpoint_task_id,
        run_id=state.checkpoint_run_id,
        number=number,
    )
    state.checkpoint_available = True
    state.checkpoint_count = number
    state.checkpoint_last_manifest = str(
        store.manifest_path(
            state.checkpoint_task_id, state.checkpoint_run_id, number
        )
    )
    state.active_diff_hash = manifest.active_diff_hash
    mismatched: list[str] = []
    for path in state.checkpoint_pending_paths:
        actual = manifest.files.get(path)
        current = state.working_set.files_read.get(path)
        if actual is None:
            mismatched.append(path)
        elif actual.exists and (
            current is None or current.content_hash != actual.content_hash
        ):
            mismatched.append(path)
        elif not actual.exists and current is not None:
            mismatched.append(path)
    if mismatched:
        state.checkpoint_error = (
            "checkpoint filesystem state does not match refreshed working set: "
            + ", ".join(sorted(mismatched))
        )
        raise CheckpointError(state.checkpoint_error)
    state.checkpoint_pending_paths = []
    state.checkpoint_pending_number = 0
    state.checkpoint_error = ""


def _prepare_verification_contract(
    state: LoopState,
    intent: str,
    workspace: Path | None,
) -> None:
    if state.iteration != 0:
        return
    explicit = commands_named_in_intent(intent)
    contract = build_repo_contract(workspace)
    if explicit:
        state.commands = explicit
        state.command_timeouts = {command: 60 for command in explicit}
    elif contract and contract.commands:
        state.commands = [spec.command for spec in contract.commands]
        state.command_timeouts = {
            spec.command: spec.timeout_s for spec in contract.commands
        }
    else:
        state.commands = []
        state.command_timeouts = {}
    if contract:
        state.working_set.repo_root = contract.repo_root
        state.working_set.contract_fingerprint = contract.fingerprint
        state.working_set.contract_configs = list(contract.configs)
    elif workspace:
        state.working_set.repo_root = str(workspace)
    state.working_set.objective = state.working_set.objective or intent
    state.working_set.acceptance_commands = list(state.commands)


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
    workspace: Path | None = None,
) -> OrchResult | None:
    if state.iteration >= MAX_CYCLES:
        state.phase = "exhausted"
        return _terminal(svc, task_id, state)
    _prepare_verification_contract(state, intent, workspace)
    state.working_set.objective = state.working_set.objective or intent
    state.working_set.acceptance_commands = list(state.commands)
    commands, reason = select_verify_commands(state.commands)
    if not commands:
        state.phase = "blocked"
        state.blocked_reason = reason
        return _terminal(svc, task_id, state)
    texts = shot_texts(report, require_qa=True)
    if not texts or picked is None:
        return None
    _key, foreman = picked
    compiled = compile_context(
        state,
        budget_tokens=_coder_context_tokens(cfg),
    )
    calls = await plan_actions(
        foreman,
        intent,
        "",
        "\n\n".join(texts),
        catalog,
        list(tools or []),
        working_set=compiled.text,
    )
    if not calls:
        return None
    next_iteration = state.iteration + 1
    if not _capture_apply_baseline(
        cfg,
        task_id,
        state,
        calls,
        workspace,
        next_iteration,
    ):
        state.phase = "blocked"
        return _terminal(svc, task_id, state)
    state.last_cmd = commands[0]
    state.iteration = next_iteration
    if state.iteration > MAX_CYCLES:
        state.phase = "exhausted"
        state.iteration = MAX_CYCLES
        return _terminal(svc, task_id, state)
    state.phase = "apply"
    save_loop_state(svc, task_id, state)
    return _with_loop(OrchResult(tool_calls=calls), state)


async def _advance_after_apply(
    cfg,
    messages: list[Any],
    catalog: dict[str, tuple[str, ...]],
    svc: TaskService,
    task_id: str,
    state: LoopState,
) -> OrchResult:
    writes = _latest_write_exchanges(messages)
    if not writes:
        return _waiting(state, "Cline mutation result")

    observed: list[str] = []
    workspace = (
        Path(state.working_set.repo_root)
        if state.working_set.repo_root
        else None
    )
    for name, args, _text in writes:
        current = _paths_from_write(
            name.lower().replace("-", "_"),
            args,
            workspace,
        )
        if not current:
            state.phase = "blocked"
            state.blocked_reason = f"completed write {name} has no attributable path"
            return _terminal(svc, task_id, state)
        for path in current:
            if path not in observed:
                observed.append(path)
    expected = set(state.checkpoint_pending_paths)
    unexpected = sorted(set(observed) - expected)
    missing = sorted(expected - set(observed))
    if unexpected:
        state.phase = "blocked"
        state.blocked_reason = (
            "completed write paths differ from checkpoint attribution: "
            + ", ".join(unexpected)
        )
        return _terminal(svc, task_id, state)
    if missing:
        return _waiting(state, "remaining Cline mutation results")

    for _name, args, text in writes:
        remember_write(state, args, text)
    changed, diff_hash = parse_mutation("\n".join(text for _name, _args, text in writes))
    state.prev_diff_hash = state.last_diff_hash
    state.last_diff_hash = diff_hash
    if not changed and state.last_failure_hash:
        state.phase = "exhausted"
        state.blocked_reason = state.blocked_reason or "empty mutation after repair"
        return _terminal(svc, task_id, state)
    commands, reason = select_verify_commands(state.commands)
    if not commands:
        state.phase = "blocked"
        state.blocked_reason = reason
        return _terminal(svc, task_id, state)
    state.verify_index = 0
    state.active_diff_hash = None
    state.verification_results = []
    state.phase = "verify"
    state.last_cmd = commands[0]
    calls = refresh_working_set_calls(
        state,
        catalog,
        paths=list(state.working_set.stale_files),
    )
    if not calls:
        state.active_diff_hash = working_set_diff_hash(state)
        try:
            _finalize_pending_checkpoint(cfg, state)
        except CheckpointError as exc:
            state.phase = "blocked"
            state.blocked_reason = f"checkpoint finalization failed: {exc}"
            return _terminal(svc, task_id, state)
        calls = verify_tool_calls(
            state.last_cmd,
            catalog,
            state.command_timeouts.get(state.last_cmd, 60),
        )
        if not calls:
            state.phase = "blocked"
            state.blocked_reason = "no Cline run tool available"
            return _terminal(svc, task_id, state)
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
    if not complete_working_set_refresh(state, messages):
        return _waiting(state, "post-mutation working-set snapshot")
    commands, reason = select_verify_commands(state.commands)
    if not commands:
        state.phase = "blocked"
        state.blocked_reason = reason
        return _terminal(svc, task_id, state)
    if state.active_diff_hash is None:
        state.active_diff_hash = working_set_diff_hash(state)
    if state.checkpoint_pending_number:
        try:
            _finalize_pending_checkpoint(cfg, state)
        except CheckpointError as exc:
            state.phase = "blocked"
            state.blocked_reason = f"checkpoint finalization failed: {exc}"
            return _terminal(svc, task_id, state)
    if state.verify_index >= len(commands):
        state.phase = "blocked"
        state.blocked_reason = "invalid verification cursor"
        return _terminal(svc, task_id, state)
    command = commands[state.verify_index]
    state.last_cmd = command
    run = last_run_for_command_since_write(messages, command)
    if run is None:
        calls = verify_tool_calls(
            command,
            catalog,
            state.command_timeouts.get(command, 60),
        )
        if not calls:
            state.phase = "blocked"
            state.blocked_reason = "no Cline run tool available"
            return _terminal(svc, task_id, state)
        save_loop_state(svc, task_id, state)
        return _with_loop(OrchResult(tool_calls=calls), state)
    exit_code, timed_out, stdout, stderr = parse_command_outcome(run[2])
    state.last_exit = exit_code
    state.timed_out = timed_out
    state.stdout_tail = stdout
    state.stderr_tail = stderr
    if (not timed_out) and exit_code == 0:
        state.verification_results.append(
            VerificationRecord(
                diff_hash=state.active_diff_hash,
                command=command,
                exit_code=0,
            )
        )
        state.verify_index += 1
        if state.verify_index == len(commands):
            expected = {(state.active_diff_hash, item) for item in commands}
            observed = {
                (record.diff_hash, record.command)
                for record in state.verification_results
                if record.exit_code == 0
            }
            if observed == expected:
                state.phase = "verified"
                return _terminal(svc, task_id, state)
            state.phase = "blocked"
            state.blocked_reason = "verification evidence does not match current diff"
            return _terminal(svc, task_id, state)
        state.last_cmd = commands[state.verify_index]
        calls = verify_tool_calls(
            state.last_cmd,
            catalog,
            state.command_timeouts.get(state.last_cmd, 60),
        )
        if not calls:
            state.phase = "blocked"
            state.blocked_reason = "no Cline run tool available"
            return _terminal(svc, task_id, state)
        save_loop_state(svc, task_id, state)
        return _with_loop(OrchResult(tool_calls=calls), state)
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
    needs_expansion = prepare_failure_expansion(state)
    state.phase = "expand" if needs_expansion else "repair"
    refresh_calls = refresh_working_set_calls(state, catalog)
    save_loop_state(svc, task_id, state)
    if refresh_calls:
        return _with_loop(OrchResult(tool_calls=refresh_calls), state)
    if needs_expansion:
        return await _advance_expansion(
            cfg,
            intent,
            thread,
            catalog,
            tools,
            messages,
            svc,
            task_id,
            state,
            picked,
        )
    return await _run_repair_apply(
        cfg, intent, thread, catalog, tools, svc, task_id, state, picked
    )


def _semantic_expansion_paths(cfg, state: LoopState) -> list[str]:
    root = state.working_set.repo_root
    if not root:
        return []
    query = " ".join(
        [
            *state.expansion_symbols,
            (state.stderr_tail or state.stdout_tail)[-1200:],
        ]
    ).strip()
    if not query:
        return []
    try:
        from harness.task.code_index import (
            default_index_path,
            gather_paths_for_intent,
        )

        settings = getattr(cfg, "settings", None)
        db_path = getattr(settings, "code_index_path", None)
        return gather_paths_for_intent(
            query,
            db_path=db_path or default_index_path(getattr(cfg, "root", None)),
            workspace=Path(root),
            limit=4,
        )
    except Exception:
        log.exception("workspace-scoped semantic expansion failed")
        return []


async def _advance_expansion(
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
    if not complete_working_set_refresh(state, messages):
        return _waiting(state, "changed-file refresh results")
    if not consume_expansion_results(state, messages):
        return _waiting(state, "causal expansion results")
    calls = expansion_tool_calls(state, catalog)
    if calls:
        save_loop_state(svc, task_id, state)
        return _with_loop(OrchResult(tool_calls=calls), state)
    if expansion_needs_semantic(state):
        add_semantic_expansion_paths(state, _semantic_expansion_paths(cfg, state))
        calls = expansion_tool_calls(state, catalog)
        if calls:
            save_loop_state(svc, task_id, state)
            return _with_loop(OrchResult(tool_calls=calls), state)
    if not expansion_complete(state):
        save_loop_state(svc, task_id, state)
        return _waiting(state, "causal expansion results")
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
    if not complete_working_set_refresh(state, messages=[]):
        return _waiting(state, "changed-file refresh results")
    if picked is None:
        picked = await pick_foreman(cfg)
    if picked is None:
        state.phase = "blocked"
        state.blocked_reason = "no foreman for repair"
        return _terminal(svc, task_id, state)
    compiled = compile_context(
        state,
        budget_tokens=_coder_context_tokens(cfg),
        instruction=(
            "Repair the observed current failure from this state. Preserve correct "
            "existing behavior. Do not restart investigation or broad gather."
        ),
    )
    packets = repair_packets(
        intent,
        thread,
        state,
        compiled_context=compiled.text,
    )
    report = await run_dispatch(cfg, intent, thread="", packets=packets)
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
        state.working_set.objective = state.working_set.objective or intent
        remember_reads(state, messages or [])
        if state.phase in {"expand", "repair"}:
            complete_working_set_refresh(state, messages or [])
        if state.phase in TERMINAL:
            return _with_loop(OrchResult(text=terminal_text(state)), state)
        if state.phase == "apply":
            return await _advance_after_apply(
                cfg,
                messages or [],
                catalog,
                svc,
                task_id,
                state,
            )
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
        if state.phase == "expand":
            picked = await pick_foreman(cfg)
            return await _advance_expansion(
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
    dispatch_thread = thread
    compiled_context = ""
    if svc and state is not None:
        _prepare_verification_contract(state, intent, workspace)
        compiled_context = compile_context(
            state,
            budget_tokens=_coder_context_tokens(cfg),
            instruction=(
                "Produce an exact implementation for the current objective from this "
                "state. Return concrete code changes, not a prose-only answer."
            ),
        ).text
        dispatch_thread = compiled_context
        save_loop_state(svc, task_id, state)
    report = await run_dispatch(
        cfg,
        intent,
        thread=dispatch_thread,
        packets=packets,
        compiled_context=compiled_context,
    )
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
            workspace=workspace,
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
