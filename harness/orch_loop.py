"""Persistent apply → verify → repair policy. Cline executes; run_orch ticks once."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from harness.dispatch import (
    AcceptSpec,
    Packet,
    WRITE_TOOLS,
    bind_gather_calls,
    _clip_context,
)
from harness.task.models import Evidence
from harness.task.service import TaskService

log = logging.getLogger("harness.orch")

LOOP_KIND = "orch_loop"
MAX_CYCLES = 5
VERIFY_TIMEOUT_S = 60
TAIL_CHARS = 8000

Phase = Literal[
    "gather",
    "apply",
    "verify",
    "repair",
    "verified",
    "blocked",
    "exhausted",
]

TERMINAL = frozenset({"verified", "blocked", "exhausted"})

_UNSAFE = re.compile(r"""[;&|`$<>]|&&|\|\||\$\(""")
_NPM_SCRIPT = re.compile(r"^(typecheck|lint|test|test:[A-Za-z0-9_-]+|check)$")
_EXIT_RE = re.compile(r"(?i)exit(?:[\s_-]*code)?\s*[:=]\s*(-?\d+)")
_TIMEOUT_RE = re.compile(r"(?i)\b(timed?\s*out|timeout(?:error)?)\b")
_EMPTY_MUTATION = re.compile(
    r"(?i)(no changes?|did not change|no modifications?|empty diff|nothing to (?:apply|change))"
)
_RUN_TOOLS = frozenset(
    {
        "execute_command",
        "run_commands",
        "run_command",
        "bash",
        "shell",
    }
)


@dataclass
class LoopState:
    phase: Phase = "gather"
    intent: str = ""
    iteration: int = 0
    commands: list[str] = field(default_factory=list)
    last_cmd: str | None = None
    last_exit: int | None = None
    timed_out: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""
    last_diff_hash: str | None = None
    last_failure_hash: str | None = None
    prev_diff_hash: str | None = None
    prev_failure_hash: str | None = None
    blocked_reason: str = ""
    files: list[str] = field(default_factory=list)
    diff: str = ""
    failed_tests: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> LoopState:
        data = dict(raw or {})
        commands = data.get("commands") or []
        if not isinstance(commands, list):
            commands = []
        phase = str(data.get("phase") or "gather")
        if phase not in {
            "gather",
            "apply",
            "verify",
            "repair",
            "verified",
            "blocked",
            "exhausted",
        }:
            phase = "gather"
        return cls(
            phase=phase,  # type: ignore[arg-type]
            intent=str(data.get("intent") or ""),
            iteration=int(data.get("iteration") or 0),
            commands=[str(c) for c in commands if str(c).strip()],
            last_cmd=data.get("last_cmd") or None,
            last_exit=data.get("last_exit") if data.get("last_exit") is not None else None,
            timed_out=bool(data.get("timed_out")),
            stdout_tail=str(data.get("stdout_tail") or ""),
            stderr_tail=str(data.get("stderr_tail") or ""),
            last_diff_hash=data.get("last_diff_hash") or None,
            last_failure_hash=data.get("last_failure_hash") or None,
            prev_diff_hash=data.get("prev_diff_hash") or None,
            prev_failure_hash=data.get("prev_failure_hash") or None,
            blocked_reason=str(data.get("blocked_reason") or ""),
            files=[str(p) for p in (data.get("files") or []) if str(p).strip()],
            diff=str(data.get("diff") or ""),
            failed_tests=str(data.get("failed_tests") or ""),
        )


def load_loop_state(svc: TaskService, task_id: str) -> LoopState | None:
    rows = svc.evidence(task_id, kind=LOOP_KIND)
    if not rows:
        return None
    payload = rows[-1].payload
    if not isinstance(payload, dict):
        return None
    return LoopState.from_dict(payload)


def save_loop_state(svc: TaskService, task_id: str, state: LoopState) -> None:
    svc.add_evidence(Evidence(task_id=task_id, kind=LOOP_KIND, payload=state.to_dict()))
    outcome = state.phase if state.phase in TERMINAL else ""
    svc.set_stage(task_id, f"loop_{state.phase}", outcome)
    extra = ""
    if state.phase == "verify" and state.last_cmd:
        extra = f' cmd="{state.last_cmd}"'
    elif state.phase == "blocked" and state.blocked_reason:
        extra = f" reason={state.blocked_reason}"
    elif state.last_exit is not None and state.phase in {"repair", "verified", "exhausted"}:
        extra = f" verify_exit={state.last_exit}"
    log.info(
        "orch_loop task=%s phase=%s iter=%s%s",
        task_id,
        state.phase,
        state.iteration,
        extra,
    )


def tail_text(text: str, limit: int = TAIL_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def fingerprint(parts: list[str]) -> str:
    blob = "\n".join(re.sub(r"\s+", " ", (p or "").strip().lower()) for p in parts)
    return hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()[:16]


def parse_argv(command: str) -> list[str] | None:
    raw = (command or "").strip()
    if not raw or _UNSAFE.search(raw):
        return None
    try:
        argv = shlex.split(raw, posix=True)
    except ValueError:
        return None
    if not argv:
        return None
    return argv


def command_allowed(command: str) -> bool:
    argv = parse_argv(command)
    if not argv:
        return False
    head = Path(argv[0]).name.lower()
    if head in {"pytest", "ruff", "eslint", "tsc"}:
        return True
    if head in {"python", "python3"}:
        return len(argv) >= 3 and argv[1] == "-m" and argv[2] in {"pytest", "ruff"}
    if head == "npx":
        return len(argv) >= 2 and Path(argv[1]).name.lower() == "tsc"
    if head == "npm":
        return (
            len(argv) >= 3
            and argv[1] in {"run", "run-script"}
            and bool(_NPM_SCRIPT.match(argv[2]))
        )
    if head == "pnpm":
        if len(argv) >= 3 and argv[1] == "run" and _NPM_SCRIPT.match(argv[2]):
            return True
        return len(argv) >= 2 and bool(_NPM_SCRIPT.match(argv[1]))
    return False


def select_verify_command(commands: list[str]) -> tuple[str | None, str]:
    usable = [str(c).strip() for c in commands if str(c).strip()]
    if not usable:
        return None, "no accept.commands"
    for raw in usable:
        if parse_argv(raw) is None:
            return None, "unsafe or unparseable command"
        if command_allowed(raw):
            return raw, ""
        return None, "command not on allowlist"
    return None, "no allowed accept.commands"


def commands_from_packets(packets: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for packet in packets or []:
        accept = getattr(packet, "accept", None)
        rows = getattr(accept, "commands", ()) if accept is not None else ()
        for raw in rows:
            cmd = str(raw).strip()
            if cmd and cmd not in seen:
                seen.add(cmd)
                out.append(cmd)
    return out


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content or "")


def _args_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(data, dict):
            return data
    return {}


def last_tool_exchanges(messages: list[Any]) -> list[tuple[str, dict[str, Any], str]]:
    """Most recent first-class tool exchanges: (name, args, result)."""
    pending: dict[str, tuple[str, dict[str, Any]]] = {}
    out: list[tuple[str, dict[str, Any], str]] = []
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        if role == "assistant":
            for call in item.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = str((fn or {}).get("name") or "")
                args = _args_dict((fn or {}).get("arguments"))
                cid = str(call.get("id") or "")
                if cid:
                    pending[cid] = (name, args)
                elif name:
                    out.append((name, args, ""))
        if role == "tool":
            cid = str(item.get("tool_call_id") or "")
            name, args = pending.pop(cid, ("tool", {}))
            out.append((name, args, _content_text(item.get("content"))))
            continue
        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") not in {"tool_result", "tool_use"}:
                    continue
                name = str(block.get("name") or block.get("tool_name") or "tool")
                out.append((name, _args_dict(block.get("input")), _content_text(block.get("content") or block.get("text"))))
    return out


def last_write_result(messages: list[Any]) -> tuple[str, dict[str, Any], str] | None:
    for name, args, text in reversed(last_tool_exchanges(messages)):
        if name.lower() in WRITE_TOOLS:
            return name, args, text
    return None


def last_run_result(messages: list[Any]) -> tuple[str, dict[str, Any], str] | None:
    for name, args, text in reversed(last_tool_exchanges(messages)):
        if name.lower() in _RUN_TOOLS:
            return name, args, text
    return None


def last_run_since_write(messages: list[Any]) -> tuple[str, dict[str, Any], str] | None:
    """Only a command result that follows the latest mutation counts as verify."""
    exchanges = last_tool_exchanges(messages)
    write_at = -1
    for index, (name, _args, _text) in enumerate(exchanges):
        if name.lower() in WRITE_TOOLS:
            write_at = index
    if write_at < 0:
        return None
    latest: tuple[str, dict[str, Any], str] | None = None
    for name, args, text in exchanges[write_at + 1 :]:
        if name.lower() in _RUN_TOOLS:
            latest = (name, args, text)
    return latest


def parse_command_outcome(text: str) -> tuple[int | None, bool, str, str]:
    """Return (exit_code, timed_out, stdout_tail, stderr_tail)."""
    blob = text or ""
    timed_out = bool(_TIMEOUT_RE.search(blob))
    exit_code: int | None = None
    match = _EXIT_RE.search(blob)
    if match:
        exit_code = int(match.group(1))
    else:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            for key in ("exit_code", "exitCode", "code"):
                if key in data and data[key] is not None:
                    try:
                        exit_code = int(data[key])
                    except (TypeError, ValueError):
                        pass
                    break
            if data.get("timeout") or data.get("timed_out"):
                timed_out = True
            stdout = str(data.get("stdout") or data.get("output") or blob)
            stderr = str(data.get("stderr") or "")
            return exit_code, timed_out, tail_text(stdout), tail_text(stderr)
    if timed_out and exit_code is None:
        exit_code = None
    return exit_code, timed_out, tail_text(blob), ""


def parse_mutation(text: str) -> tuple[bool, str]:
    blob = text or ""
    digest = fingerprint([blob[-4000:]])
    if _EMPTY_MUTATION.search(blob):
        return False, digest
    return True, digest


def failure_hash(state: LoopState) -> str:
    return fingerprint(
        [
            state.last_cmd or "",
            str(state.last_exit if state.last_exit is not None else "timeout"),
            "timeout" if state.timed_out else "",
            state.stderr_tail[-2000:],
            state.stdout_tail[-2000:],
        ]
    )


def verify_tool_calls(command: str, catalog: dict[str, tuple[str, ...]]) -> list[dict[str, Any]]:
    raw = [
        {
            "name": "execute_command",
            "arguments": {
                "command": command,
                "timeout": VERIFY_TIMEOUT_S,
                "timeout_s": VERIFY_TIMEOUT_S,
            },
        }
    ]
    return bind_gather_calls(raw, catalog)


def remember_write(state: LoopState, args: dict[str, Any], text: str) -> None:
    path = str(args.get("path") or args.get("file_path") or args.get("rel_path") or "").strip()
    if path and path not in state.files:
        state.files.append(path)
    state.diff = tail_text(text, 4000)


def remember_failure(state: LoopState) -> None:
    state.failed_tests = tail_text(
        f"cmd={state.last_cmd} exit={state.last_exit} timeout={state.timed_out}\n"
        f"{state.stderr_tail or state.stdout_tail}",
        4000,
    )


def working_set_text(state: LoopState) -> str:
    files = "\n".join(f"- {p}" for p in state.files) or "- (none)"
    return (
        f"FILES:\n{files}\n\n"
        f"ACTIVE DIFF:\n{state.diff or '(none)'}\n\n"
        f"FAILED TESTS:\n{state.failed_tests or '(none)'}\n"
    )


def repair_packets(intent: str, thread: str, state: LoopState) -> list[Packet]:
    remaining = max(0, MAX_CYCLES - state.iteration)
    prompt = (
        f"USER REQUEST:\n{intent[:1200]}\n\n"
        f"REPAIR the actual verification failure. Do not re-investigate from scratch.\n"
        f"ITERATION: {state.iteration} of {MAX_CYCLES} ({remaining} remaining after this repair)\n"
        f"ACCEPTANCE COMMAND:\n{state.last_cmd or '(none)'}\n"
        f"EXIT CODE: {state.last_exit if state.last_exit is not None else 'timeout/unknown'}\n"
        f"TIMED OUT: {state.timed_out}\n"
        f"STDOUT TAIL:\n{state.stdout_tail or '(empty)'}\n"
        f"STDERR TAIL:\n{state.stderr_tail or '(empty)'}\n\n"
        f"{working_set_text(state)}\n"
        f"WORKSPACE EVIDENCE:\n{_clip_context(thread, 8000)}\n\n"
        "Produce the exact patch that fixes this failure."
    )
    return [
        Packet(
            id="repair-1",
            title="Repair verification failure",
            prompt=prompt[:16000],
            accept=AcceptSpec(
                commands=tuple(state.commands),
                invariants=("min_chars 40",),
            ),
        )
    ]


def terminal_text(state: LoopState) -> str:
    lines = [
        "Harness orch loop",
        f"status: {state.phase}",
        f"command: {state.last_cmd or '(none)'}",
        f"exit_code: {state.last_exit if state.last_exit is not None else 'n/a'}",
        f"iterations: {state.iteration}",
    ]
    if state.timed_out:
        lines.append("timeout: true")
    if state.blocked_reason:
        lines.append(f"reason: {state.blocked_reason}")
    if state.phase == "exhausted" and (state.stdout_tail or state.stderr_tail):
        lines.append("last_output:")
        lines.append(tail_text(state.stderr_tail or state.stdout_tail, 1200))
    return "\n".join(lines) + "\n"


def shot_texts(report: Any, *, require_qa: bool = False) -> list[str]:
    out: list[str] = []
    for shot in getattr(report, "shots", []) or []:
        text = ((getattr(shot, "result", None) and shot.result.text) or "").strip()
        if not text:
            continue
        if require_qa and not getattr(shot, "qa_pass", False):
            continue
        out.append(text)
    frontier = (getattr(report, "frontier_text", "") or "").strip()
    if getattr(report, "frontier_verified", False) and frontier:
        out.append(frontier)
    return out
