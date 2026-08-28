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
WORKING_FILE_MAX_CHARS = 16000

Phase = Literal[
    "gather",
    "apply",
    "verify",
    "expand",
    "repair",
    "verified",
    "blocked",
    "exhausted",
]

TERMINAL = frozenset({"verified", "blocked", "exhausted"})

_UNSAFE = re.compile(r"""[;&|`$<>]|&&|\|\||\$\(""")
_NPM_SCRIPT = re.compile(r"^(typecheck|lint|test|test:[A-Za-z0-9_-]+|check)$")
_EXIT_RES = (
    re.compile(r"<exit_code>\s*(-?\d+)\s*</exit_code>", re.I),
    re.compile(r"(?i)exit(?:[\s_-]*code)?\s*[:=]\s*(-?\d+)"),
    re.compile(
        r"(?i)(?:command|process)\s+(?:completed|finished|ended|exited|failed)"
        r"[^\n]{0,80}(?:exit(?:[\s_-]*code)?|code)\s*[:=]?\s*(-?\d+)"
    ),
    re.compile(r"(?i)exited with (?:status |code )?(-?\d+)"),
)
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
_READ_TOOLS = frozenset({"read_file", "read_files", "readfile", "read"})
_SEARCH_TOOLS = frozenset(
    {"search_files", "search_codebase", "codebase_search", "grep", "glob"}
)
_SOURCE_SUFFIXES = (
    "py",
    "pyi",
    "ts",
    "tsx",
    "js",
    "jsx",
    "mjs",
    "cjs",
    "go",
    "rs",
    "java",
    "kt",
    "kts",
    "rb",
    "php",
    "cs",
    "c",
    "cc",
    "cpp",
    "h",
    "hpp",
)
_SOURCE_PATH_RE = re.compile(
    rf"(?<![A-Za-z0-9_./-])((?:[A-Za-z0-9_.-]+/)*"
    rf"[A-Za-z0-9_.-]+\.(?:{'|'.join(_SOURCE_SUFFIXES)}))"
    rf"(?=:\d|::|\s|$)"
)
_ABS_SOURCE_PATH_RE = re.compile(
    rf"(?<![A-Za-z0-9_.-])((?:/[A-Za-z0-9_.-]+)+"
    rf"\.(?:{'|'.join(_SOURCE_SUFFIXES)}))(?=:\d|::|\s|$)"
)
_TRACEBACK_PATH_RE = re.compile(r"""(?i)\bFile\s+["']([^"']+)["']\s*,\s*line\s+\d+""")
_MISSING_FILE_RE = re.compile(
    r"(?i)\b(?:no such file|not found|does not exist|cannot find the file)\b"
)
_SYMBOL_RES = (
    re.compile(
        r"""(?i)\bNameError:\s*(?:name\s+)?["']?([A-Za-z_][A-Za-z0-9_]*)["']?"""
    ),
    re.compile(
        r"""(?i)\bcannot\s+find\s+(?:name|symbol)\s+["']?([A-Za-z_][A-Za-z0-9_]*)["']?"""
    ),
    re.compile(
        r"""(?i)\bcannot\s+import\s+name\s+["']?([A-Za-z_][A-Za-z0-9_]*)["']?"""
    ),
)


@dataclass
class WorkingFile:
    content: str = ""
    content_hash: str = ""

    @classmethod
    def from_dict(cls, raw: Any) -> WorkingFile:
        data = raw if isinstance(raw, dict) else {}
        return cls(
            content=str(data.get("content") or ""),
            content_hash=str(data.get("content_hash") or ""),
        )


@dataclass
class WorkingSet:
    objective: str = ""
    repo_root: str = ""
    contract_fingerprint: str = ""
    contract_configs: list[str] = field(default_factory=list)
    acceptance_commands: list[str] = field(default_factory=list)
    files_read: dict[str, WorkingFile] = field(default_factory=dict)
    files_changed: list[str] = field(default_factory=list)
    stale_files: list[str] = field(default_factory=list)
    current_diff: str = ""
    refresh_pending: list[str] = field(default_factory=list)
    refresh_diff_pending: bool = False

    @classmethod
    def from_dict(cls, raw: Any) -> WorkingSet:
        data = raw if isinstance(raw, dict) else {}
        files = data.get("files_read") if isinstance(data.get("files_read"), dict) else {}
        return cls(
            objective=str(data.get("objective") or ""),
            repo_root=str(data.get("repo_root") or ""),
            contract_fingerprint=str(data.get("contract_fingerprint") or ""),
            contract_configs=[
                str(path) for path in (data.get("contract_configs") or []) if str(path).strip()
            ],
            acceptance_commands=[
                str(c) for c in (data.get("acceptance_commands") or []) if str(c).strip()
            ],
            files_read={
                str(path): WorkingFile.from_dict(value)
                for path, value in files.items()
                if str(path).strip()
            },
            files_changed=[
                str(path) for path in (data.get("files_changed") or []) if str(path).strip()
            ],
            stale_files=[
                str(path) for path in (data.get("stale_files") or []) if str(path).strip()
            ],
            current_diff=str(data.get("current_diff") or ""),
            refresh_pending=[
                str(path) for path in (data.get("refresh_pending") or []) if str(path).strip()
            ],
            refresh_diff_pending=bool(data.get("refresh_diff_pending")),
        )


@dataclass
class VerificationRecord:
    diff_hash: str = ""
    command: str = ""
    exit_code: int | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> VerificationRecord:
        data = raw if isinstance(raw, dict) else {}
        return cls(
            diff_hash=str(data.get("diff_hash") or ""),
            command=str(data.get("command") or ""),
            exit_code=data.get("exit_code") if data.get("exit_code") is not None else None,
        )


@dataclass
class LoopState:
    phase: Phase = "gather"
    intent: str = ""
    iteration: int = 0
    commands: list[str] = field(default_factory=list)
    command_timeouts: dict[str, int] = field(default_factory=dict)
    verify_index: int = 0
    active_diff_hash: str | None = None
    verification_results: list[VerificationRecord] = field(default_factory=list)
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
    failed_tests: str = ""
    attempt_summaries: list[dict[str, Any]] = field(default_factory=list)
    expansion_paths: list[str] = field(default_factory=list)
    expansion_symbols: list[str] = field(default_factory=list)
    expansion_attempted_paths: list[str] = field(default_factory=list)
    expansion_attempted_symbols: list[str] = field(default_factory=list)
    expansion_pending_paths: list[str] = field(default_factory=list)
    expansion_pending_symbols: list[str] = field(default_factory=list)
    semantic_expansion_paths: list[str] = field(default_factory=list)
    expansion_semantic_attempted: bool = False
    working_set: WorkingSet = field(default_factory=WorkingSet)

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
            "expand",
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
            command_timeouts={
                str(command): int(timeout)
                for command, timeout in (
                    data.get("command_timeouts")
                    if isinstance(data.get("command_timeouts"), dict)
                    else {}
                ).items()
                if str(command).strip()
            },
            verify_index=int(data.get("verify_index") or 0),
            active_diff_hash=data.get("active_diff_hash") or None,
            verification_results=[
                VerificationRecord.from_dict(row)
                for row in (data.get("verification_results") or [])
                if isinstance(row, dict)
            ],
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
            failed_tests=str(data.get("failed_tests") or ""),
            attempt_summaries=[
                {
                    "iteration": int(row.get("iteration") or 0),
                    "command": str(row.get("command") or ""),
                    "exit_code": (
                        int(row["exit_code"])
                        if row.get("exit_code") is not None
                        else None
                    ),
                    "failure": str(row.get("failure") or ""),
                    "changed_files": [
                        str(path)
                        for path in (row.get("changed_files") or [])
                        if str(path).strip()
                    ],
                    "diff_hash": str(row.get("diff_hash") or ""),
                }
                for row in (data.get("attempt_summaries") or [])
                if isinstance(row, dict)
            ][-5:],
            expansion_paths=[
                str(path)
                for path in (data.get("expansion_paths") or [])
                if str(path).strip()
            ],
            expansion_symbols=[
                str(symbol)
                for symbol in (data.get("expansion_symbols") or [])
                if str(symbol).strip()
            ],
            expansion_attempted_paths=[
                str(path)
                for path in (data.get("expansion_attempted_paths") or [])
                if str(path).strip()
            ],
            expansion_attempted_symbols=[
                str(symbol)
                for symbol in (data.get("expansion_attempted_symbols") or [])
                if str(symbol).strip()
            ],
            expansion_pending_paths=[
                str(path)
                for path in (data.get("expansion_pending_paths") or [])
                if str(path).strip()
            ],
            expansion_pending_symbols=[
                str(symbol)
                for symbol in (data.get("expansion_pending_symbols") or [])
                if str(symbol).strip()
            ],
            semantic_expansion_paths=[
                str(path)
                for path in (data.get("semantic_expansion_paths") or [])
                if str(path).strip()
            ],
            expansion_semantic_attempted=bool(
                data.get("expansion_semantic_attempted")
            ),
            working_set=WorkingSet.from_dict(data.get("working_set")),
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
    selected, reason = select_verify_commands(commands)
    return (selected[0], "") if selected else (None, reason)


def select_verify_commands(commands: list[str]) -> tuple[list[str], str]:
    usable = [str(c).strip() for c in commands if str(c).strip()]
    if not usable:
        return [], "no accept.commands"
    selected: list[str] = []
    for raw in usable:
        if parse_argv(raw) is None:
            return [], "unsafe or unparseable command"
        if not command_allowed(raw):
            return [], "command not on allowlist"
        if raw not in selected:
            selected.append(raw)
    return selected, ""


_NEGATED_VERIFY_RE = re.compile(
    r"(?i)\b(?:do\s+not|don't|dont|never|without)\b[\s\w,-]{0,48}"
    r"\b(?:pytest|ruff|eslint|tsc|npx|npm|pnpm)\b"
)
_VERIFY_HEAD_RE = re.compile(
    r"(?i)\b(?:python3?\s+-m\s+pytest|python3?\s+-m\s+ruff|npx\s+tsc|"
    r"npm\s+run\s+[A-Za-z0-9_:-]+|pnpm\s+(?:run\s+)?[A-Za-z0-9_:-]+|"
    r"pytest|ruff|eslint|tsc)\b"
)
_CMD_STOP = frozenset(
    {
        "as",
        "the",
        "a",
        "an",
        "to",
        "for",
        "and",
        "with",
        "after",
        "before",
        "then",
        "use",
        "using",
        "acceptance",
        "command",
        "must",
        "please",
    }
)


def commands_named_in_intent(intent: str) -> list[str]:
    """Copy an allowlisted command the user already named. Never invent one."""
    text = intent or ""
    if _NEGATED_VERIFY_RE.search(text):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for match in _VERIFY_HEAD_RE.finditer(text):
        tokens = [re.sub(r"\s+", " ", match.group(0)).strip()]
        for tok in re.findall(r"[A-Za-z0-9_./:-]+", text[match.end() :]):
            if tok.lower() in _CMD_STOP:
                break
            tokens.append(tok)
            if len(tokens) >= 6:
                break
        cmd = " ".join(tokens).strip()
        if not command_allowed(cmd) or cmd in seen:
            continue
        seen.add(cmd)
        out.append(cmd)
    return out


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


def last_run_for_command_since_write(
    messages: list[Any], command: str
) -> tuple[str, dict[str, Any], str] | None:
    """Latest result for COMMAND after the latest mutation."""
    exchanges = last_tool_exchanges(messages)
    write_at = -1
    for index, (name, _args, _text) in enumerate(exchanges):
        if name.lower() in WRITE_TOOLS:
            write_at = index
    if write_at < 0:
        return None
    latest: tuple[str, dict[str, Any], str] | None = None
    for name, args, text in exchanges[write_at + 1 :]:
        if name.lower() in _RUN_TOOLS and command in _command_args(args):
            latest = (name, args, text)
    return latest


def _coerce_exit(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _exit_from_mapping(data: dict[str, Any]) -> int | None:
    for key in ("exit_code", "exitCode", "exitcode"):
        if key in data and data[key] is not None:
            code = _coerce_exit(data[key])
            if code is not None:
                return code
    return None


def _flatten_command_blob(blob: str) -> tuple[str, int | None, bool]:
    """Unwrap Cline JSON wrappers. Return (text, exit_from_json, timed_out)."""
    raw = blob or ""
    timed_out = False
    exit_code: int | None = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw, None, False
    chunks: list[str] = []

    def walk(node: Any) -> None:
        nonlocal exit_code, timed_out
        if isinstance(node, dict):
            found = _exit_from_mapping(node)
            if found is not None:
                exit_code = found
            if node.get("timeout") or node.get("timed_out"):
                timed_out = True
            for key in ("stdout", "stderr", "output", "result", "text", "content"):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    chunks.append(value)
            for value in node.values():
                if isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str) and node.strip():
            chunks.append(node)

    walk(data)
    text = "\n".join(chunks) if chunks else raw
    return text, exit_code, timed_out


def parse_command_outcome(text: str) -> tuple[int | None, bool, str, str]:
    """Return (exit_code, timed_out, stdout_tail, stderr_tail).

    Fail closed: a missing exit code is not 0. 'passed' in pytest output is not enough.
    """
    blob = text or ""
    flat, json_exit, json_timeout = _flatten_command_blob(blob)
    timed_out = json_timeout or bool(_TIMEOUT_RE.search(blob)) or bool(_TIMEOUT_RE.search(flat))
    exit_code = json_exit
    if exit_code is None:
        last: int | None = None
        for pattern in _EXIT_RES:
            for match in pattern.finditer(flat):
                last = int(match.group(1))
        if last is None:
            for pattern in _EXIT_RES:
                for match in pattern.finditer(blob):
                    last = int(match.group(1))
        exit_code = last
    if timed_out and exit_code is None:
        exit_code = None
    stdout = tail_text(flat or blob)
    return exit_code, timed_out, stdout, ""


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


def verify_tool_calls(
    command: str,
    catalog: dict[str, tuple[str, ...]],
    timeout_s: int = VERIFY_TIMEOUT_S,
) -> list[dict[str, Any]]:
    timeout = max(1, min(int(timeout_s), 600))
    raw = [
        {
            "name": "execute_command",
            "arguments": {
                "command": command,
                "timeout": timeout,
                "timeout_s": timeout,
            },
        }
    ]
    return bind_gather_calls(raw, catalog)


def _normalized_workspace_path(raw: Any) -> str:
    path = str(raw or "").strip().replace("\\", "/")
    if not path or path.startswith("/") or "\x00" in path:
        return ""
    parts = [part for part in path.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        return ""
    return "/".join(parts)


def _read_paths(args: dict[str, Any]) -> list[str]:
    raw = args.get("paths")
    values = raw if isinstance(raw, list) else [
        args.get("path") or args.get("file_path") or args.get("rel_path")
    ]
    out: list[str] = []
    for value in values:
        path = _normalized_workspace_path(value)
        if path and path not in out:
            out.append(path)
    return out


def _read_result_files(paths: list[str], text: str) -> dict[str, str]:
    """Map a successful Cline read result back to its requested paths."""
    blob = text or ""
    if not paths or not blob.strip() or re.search(r"(?im)^\s*ERROR:", blob):
        return {}
    markers = list(re.finditer(r"(?im)^FILE\s+(.+?)\s*\r?\n", blob))
    if markers:
        out: dict[str, str] = {}
        requested = {path.lower(): path for path in paths}
        by_name = {Path(path).name.lower(): path for path in paths}
        for index, marker in enumerate(markers):
            raw_path = _normalized_workspace_path(marker.group(1))
            path = requested.get(raw_path.lower()) or by_name.get(Path(raw_path).name.lower())
            if not path:
                continue
            end = markers[index + 1].start() if index + 1 < len(markers) else len(blob)
            out[path] = blob[marker.end() : end]
        return out
    if len(paths) == 1:
        return {paths[0]: blob}
    return {}


def remember_reads(state: LoopState, messages: list[Any]) -> None:
    """Persist the latest successful content for every task-scoped file read."""
    state.working_set.objective = state.working_set.objective or state.intent
    for name, args, text in last_tool_exchanges(messages):
        if name.lower().replace("-", "_") not in _READ_TOOLS:
            continue
        for path, content in _read_result_files(_read_paths(args), text).items():
            full_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
            stored = content
            if len(stored) > WORKING_FILE_MAX_CHARS:
                stored = (
                    stored[:WORKING_FILE_MAX_CHARS]
                    + "\n[WORKING FILE TRUNCATED BY HARNESS]"
                )
            state.working_set.files_read[path] = WorkingFile(
                content=stored,
                content_hash=full_hash,
            )
            state.working_set.stale_files = [
                stale for stale in state.working_set.stale_files if stale != path
            ]


def _command_args(args: dict[str, Any]) -> list[str]:
    commands = args.get("commands")
    if isinstance(commands, list):
        return [str(command).strip() for command in commands if str(command).strip()]
    command = args.get("command") or args.get("query")
    return [str(command).strip()] if str(command or "").strip() else []


def _after_latest_verification(
    messages: list[Any], command: str | None
) -> list[tuple[str, dict[str, Any], str]]:
    exchanges = last_tool_exchanges(messages)
    verify_at = -1
    for index, (name, args, _text) in enumerate(exchanges):
        if name.lower() in _RUN_TOOLS and command in _command_args(args):
            verify_at = index
    return exchanges[verify_at + 1 :] if verify_at >= 0 else []


def complete_working_set_refresh(state: LoopState, messages: list[Any]) -> bool:
    """Consume only reads/diff results that arrived after the failed verify."""
    pending = list(state.working_set.refresh_pending)
    if not pending and not state.working_set.refresh_diff_pending:
        return True
    seen: set[str] = set()
    diff_seen = False
    for name, args, text in _after_latest_verification(messages, state.last_cmd):
        lower = name.lower().replace("-", "_")
        if lower in _READ_TOOLS:
            requested = _read_paths(args)
            results = _read_result_files(requested, text)
            for path, content in results.items():
                if path not in pending:
                    continue
                full_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
                stored = content
                if len(stored) > WORKING_FILE_MAX_CHARS:
                    stored = (
                        stored[:WORKING_FILE_MAX_CHARS]
                        + "\n[WORKING FILE TRUNCATED BY HARNESS]"
                    )
                state.working_set.files_read[path] = WorkingFile(stored, full_hash)
                state.working_set.stale_files = [
                    stale for stale in state.working_set.stale_files if stale != path
                ]
                seen.add(path)
            if not results and _MISSING_FILE_RE.search(text or ""):
                for path in requested:
                    if path not in pending:
                        continue
                    state.working_set.files_read.pop(path, None)
                    state.working_set.stale_files = [
                        stale
                        for stale in state.working_set.stale_files
                        if stale != path
                    ]
                    seen.add(path)
        if lower in _RUN_TOOLS and any(
            command.startswith("git diff --") for command in _command_args(args)
        ):
            flat, _exit_code, _timed_out = _flatten_command_blob(text)
            state.working_set.current_diff = tail_text(flat or text, 8000)
            diff_seen = True
    state.working_set.refresh_pending = [path for path in pending if path not in seen]
    if diff_seen:
        state.working_set.refresh_diff_pending = False
    return not state.working_set.refresh_pending and not state.working_set.refresh_diff_pending


def working_set_diff_hash(state: LoopState) -> str:
    """Content-state hash shared by every verifier in one logical cycle."""
    digest = hashlib.sha256()
    has_snapshot = False
    for path in sorted(state.working_set.files_changed):
        snapshot = state.working_set.files_read.get(path)
        digest.update(path.encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(
            (snapshot.content_hash if snapshot else "(unread)").encode(
                "utf-8", errors="replace"
            )
        )
        digest.update(b"\0")
        has_snapshot = has_snapshot or snapshot is not None
    if not has_snapshot:
        digest.update(
            state.working_set.current_diff.encode("utf-8", errors="replace")
        )
    return digest.hexdigest()[:16]


def refresh_working_set_calls(
    state: LoopState,
    catalog: dict[str, tuple[str, ...]],
    paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected = paths if paths is not None else state.working_set.files_changed
    safe_paths = [
        path
        for path in selected
        if _normalized_workspace_path(path) == path
    ][:5]
    if not safe_paths:
        return []
    raw: list[dict[str, Any]] = [
        {"name": "read_file", "arguments": {"path": path}} for path in safe_paths
    ]
    diff_command = "git diff -- " + " ".join(shlex.quote(path) for path in safe_paths)
    raw.append(
        {
            "name": "execute_command",
            "arguments": {
                "command": diff_command,
                "timeout": VERIFY_TIMEOUT_S,
                "timeout_s": VERIFY_TIMEOUT_S,
            },
        }
    )
    calls = bind_gather_calls(raw, catalog, limit=8)
    pending: list[str] = []
    diff_pending = False
    for call in calls:
        fn = call.get("function") if isinstance(call, dict) else {}
        name = str((fn or {}).get("name") or "").lower().replace("-", "_")
        args = _args_dict((fn or {}).get("arguments"))
        if name in _READ_TOOLS:
            pending.extend(path for path in _read_paths(args) if path not in pending)
        if name in _RUN_TOOLS and any(
            command.startswith("git diff --") for command in _command_args(args)
        ):
            diff_pending = True
    state.working_set.refresh_pending = pending
    state.working_set.refresh_diff_pending = diff_pending
    return calls


def remember_write(state: LoopState, args: dict[str, Any], text: str) -> None:
    path = _normalized_workspace_path(
        args.get("path") or args.get("file_path") or args.get("rel_path")
    )
    if path and path not in state.working_set.files_changed:
        state.working_set.files_changed.append(path)
    content = args.get("content")
    if content is None:
        content = args.get("new_content")
    if path and isinstance(content, str):
        full_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
        stored = content
        if len(stored) > WORKING_FILE_MAX_CHARS:
            stored = stored[:WORKING_FILE_MAX_CHARS] + "\n[WORKING FILE TRUNCATED BY HARNESS]"
        state.working_set.files_read[path] = WorkingFile(stored, full_hash)
        state.working_set.stale_files = [
            stale for stale in state.working_set.stale_files if stale != path
        ]
    elif path:
        state.working_set.files_read.pop(path, None)
        if path not in state.working_set.stale_files:
            state.working_set.stale_files.append(path)
    # Until git diff returns, retain the edit result as explicit provisional state.
    state.working_set.current_diff = tail_text(text, 4000)


def remember_failure(state: LoopState) -> None:
    failure = state.stderr_tail or state.stdout_tail
    state.failed_tests = tail_text(
        f"cmd={state.last_cmd} exit={state.last_exit} timeout={state.timed_out}\n"
        f"{failure}",
        4000,
    )
    compact = " | ".join(
        line.strip()
        for line in failure.splitlines()
        if line.strip() and not _EXIT_RES[1].search(line)
    )
    if len(compact) > 360:
        compact = compact[:357] + "..."
    state.attempt_summaries.append(
        {
            "iteration": state.iteration,
            "command": state.last_cmd or "",
            "exit_code": state.last_exit,
            "failure": compact or "(empty failure output)",
            "changed_files": sorted(state.working_set.files_changed),
            "diff_hash": state.active_diff_hash or "",
        }
    )
    state.attempt_summaries = state.attempt_summaries[-5:]


def _workspace_evidence_path(raw: str, repo_root: str) -> str:
    value = (raw or "").strip().replace("\\", "/")
    if not value or "\x00" in value:
        return ""
    candidate = Path(value)
    if candidate.is_absolute():
        if not repo_root:
            return ""
        try:
            value = candidate.resolve().relative_to(Path(repo_root).resolve()).as_posix()
        except (OSError, ValueError):
            return ""
    path = _normalized_workspace_path(value)
    if not path or Path(path).suffix.lower().lstrip(".") not in _SOURCE_SUFFIXES:
        return ""
    return path


def _paths_in_text(text: str, repo_root: str) -> list[str]:
    out: list[str] = []
    raw_paths = [match.group(1) for match in _TRACEBACK_PATH_RE.finditer(text or "")]
    raw_paths.extend(match.group(1) for match in _ABS_SOURCE_PATH_RE.finditer(text or ""))
    raw_paths.extend(match.group(1) for match in _SOURCE_PATH_RE.finditer(text or ""))
    for raw in raw_paths:
        path = _workspace_evidence_path(raw, repo_root)
        if path and path not in out:
            out.append(path)
    return out


def prepare_failure_expansion(state: LoopState) -> bool:
    """Extract concrete unseen paths/symbols without broad rediscovery."""
    failure = state.stderr_tail or state.stdout_tail or ""
    known = set(state.working_set.files_read)
    state.expansion_paths = [
        path
        for path in _paths_in_text(failure, state.working_set.repo_root)
        if path not in known
    ][:5]
    symbols: list[str] = []
    for pattern in _SYMBOL_RES:
        for match in pattern.finditer(failure):
            symbol = match.group(1)
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    state.expansion_symbols = symbols[:3]
    state.expansion_attempted_paths = []
    state.expansion_attempted_symbols = []
    state.expansion_pending_paths = []
    state.expansion_pending_symbols = []
    state.semantic_expansion_paths = []
    state.expansion_semantic_attempted = False
    return bool(state.expansion_paths or state.expansion_symbols)


def _symbol_present(state: LoopState, symbol: str) -> bool:
    escaped = re.escape(symbol)
    declaration = re.compile(
        rf"(?m)^\s*(?:(?:export|default|public|private|protected|abstract)\s+)*"
        rf"(?:class|def|function|interface|type|enum|const|let|var)\s+{escaped}\b"
    )
    imported = re.compile(
        rf"(?m)^\s*(?:from\s+\S+\s+import|import)\s+[^\n]*\b{escaped}\b"
    )
    return any(
        declaration.search(snapshot.content) or imported.search(snapshot.content)
        for snapshot in state.working_set.files_read.values()
    )


def consume_expansion_results(state: LoopState, messages: list[Any]) -> bool:
    """Consume only deterministic expansion reads/searches after verification."""
    if not state.expansion_pending_paths and not state.expansion_pending_symbols:
        return True
    seen_paths: set[str] = set()
    seen_symbols: set[str] = set()
    for name, args, text in _after_latest_verification(messages, state.last_cmd):
        lower = name.lower().replace("-", "_")
        if lower in _READ_TOOLS:
            requested = _read_paths(args)
            seen_paths.update(
                path for path in requested if path in state.expansion_pending_paths
            )
        if lower in _SEARCH_TOOLS:
            query = str(args.get("regex") or args.get("query") or "")
            matched = [
                symbol
                for symbol in state.expansion_pending_symbols
                if symbol in query
            ]
            if not matched:
                continue
            seen_symbols.update(matched)
            for path in _paths_in_text(text, state.working_set.repo_root):
                if (
                    path not in state.working_set.files_read
                    and path not in state.expansion_paths
                ):
                    state.expansion_paths.append(path)
    state.expansion_pending_paths = [
        path for path in state.expansion_pending_paths if path not in seen_paths
    ]
    state.expansion_pending_symbols = [
        symbol
        for symbol in state.expansion_pending_symbols
        if symbol not in seen_symbols
    ]
    return not state.expansion_pending_paths and not state.expansion_pending_symbols


def expansion_tool_calls(
    state: LoopState,
    catalog: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    """Read causal paths first, then search only unresolved exact symbols."""
    if state.expansion_pending_paths or state.expansion_pending_symbols:
        return []
    unread = [
        path
        for path in state.expansion_paths
        if path not in state.working_set.files_read
        and path not in state.expansion_attempted_paths
    ][:5]
    if unread:
        calls = bind_gather_calls(
            [{"name": "read_file", "arguments": {"path": path}} for path in unread],
            catalog,
            limit=5,
        )
        requested: list[str] = []
        for call in calls:
            fn = call.get("function") if isinstance(call, dict) else {}
            requested.extend(_read_paths(_args_dict((fn or {}).get("arguments"))))
        state.expansion_attempted_paths.extend(
            path for path in unread if path not in state.expansion_attempted_paths
        )
        state.expansion_pending_paths = list(dict.fromkeys(requested))
        if calls:
            return calls
    unresolved = [
        symbol
        for symbol in state.expansion_symbols
        if not _symbol_present(state, symbol)
        and symbol not in state.expansion_attempted_symbols
    ][:3]
    if unresolved:
        calls = bind_gather_calls(
            [
                {
                    "name": "search_files",
                    "arguments": {
                        "path": ".",
                        "regex": rf"\b{re.escape(symbol)}\b",
                        "query": symbol,
                    },
                }
                for symbol in unresolved
            ],
            catalog,
            limit=3,
        )
        searched: list[str] = []
        for call in calls:
            fn = call.get("function") if isinstance(call, dict) else {}
            args = _args_dict((fn or {}).get("arguments"))
            query = str(args.get("regex") or args.get("query") or "")
            searched.extend(symbol for symbol in unresolved if symbol in query)
        state.expansion_attempted_symbols.extend(
            symbol
            for symbol in unresolved
            if symbol not in state.expansion_attempted_symbols
        )
        state.expansion_pending_symbols = list(dict.fromkeys(searched))
        if calls:
            return calls
    return []


def expansion_needs_semantic(state: LoopState) -> bool:
    if (
        state.expansion_pending_paths
        or state.expansion_pending_symbols
        or state.expansion_semantic_attempted
    ):
        return False
    unresolved = [
        symbol
        for symbol in state.expansion_symbols
        if not _symbol_present(state, symbol)
    ]
    unread = [
        path
        for path in state.expansion_paths
        if path not in state.working_set.files_read
        and path not in state.expansion_attempted_paths
    ]
    return bool(unresolved and not unread) and all(
        symbol in state.expansion_attempted_symbols for symbol in unresolved
    )


def add_semantic_expansion_paths(state: LoopState, paths: list[str]) -> None:
    state.expansion_semantic_attempted = True
    for raw in paths:
        path = _normalized_workspace_path(raw)
        if (
            path
            and Path(path).suffix.lower().lstrip(".") in _SOURCE_SUFFIXES
            and path not in state.working_set.files_read
            and path not in state.expansion_paths
        ):
            state.expansion_paths.append(path)
            state.semantic_expansion_paths.append(path)


def expansion_complete(state: LoopState) -> bool:
    if state.expansion_pending_paths or state.expansion_pending_symbols:
        return False
    unread = [
        path
        for path in state.expansion_paths
        if path not in state.working_set.files_read
        and path not in state.expansion_attempted_paths
    ]
    unsearched = [
        symbol
        for symbol in state.expansion_symbols
        if not _symbol_present(state, symbol)
        and symbol not in state.expansion_attempted_symbols
    ]
    return not unread and not unsearched and not expansion_needs_semantic(state)


def working_files_text(state: LoopState) -> str:
    files: list[str] = []
    for path, snapshot in state.working_set.files_read.items():
        files.append(
            f"FILE: {path}\nCONTENT_HASH: {snapshot.content_hash}\n{snapshot.content}"
        )
    return "\n\n".join(files) or "(none)"


def working_set_text(state: LoopState) -> str:
    return (
        f"OBJECTIVE:\n{state.working_set.objective or state.intent}\n\n"
        f"ACCEPTANCE COMMANDS:\n"
        f"{chr(10).join(state.working_set.acceptance_commands) or '(none)'}\n\n"
        f"WORKING FILES:\n{_clip_context(working_files_text(state), 8000)}\n\n"
        f"CURRENT DIFF:\n"
        f"{_clip_context(state.working_set.current_diff or '(none)', 3000)}\n"
    )


def repair_packets(
    intent: str,
    thread: str,
    state: LoopState,
    *,
    compiled_context: str = "",
) -> list[Packet]:
    remaining = max(0, MAX_CYCLES - state.iteration)
    objective = _clip_context(state.working_set.objective or intent, 1200)
    files = _clip_context(working_files_text(state), 6500)
    diff = _clip_context(state.working_set.current_diff or "(none)", 3500)
    failure = _clip_context(
        state.stderr_tail or state.stdout_tail or "(empty)",
        3000,
    )
    prompt = compiled_context or (
        f"ORIGINAL OBJECTIVE\n{objective}\n\n"
        f"WORKING FILES\n{files}\n\n"
        f"CURRENT DIFF\n{diff}\n\n"
        f"FAILED VERIFICATION\n"
        f"command: {state.last_cmd or '(none)'}\n"
        f"exit: {state.last_exit if state.last_exit is not None else 'timeout/unknown'}\n"
        f"timed_out: {state.timed_out}\n"
        f"stdout/stderr tail:\n{failure}\n\n"
        f"ITERATION\n{state.iteration} of {MAX_CYCLES} ({remaining} remaining)\n\n"
        "INSTRUCTION\n"
        "Repair the observed failure. Do not restart investigation unless the "
        "failure points to code outside the working set."
    )
    return [
        Packet(
            id="repair-1",
            title="Repair verification failure",
            prompt=prompt if compiled_context else prompt[:16000],
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
