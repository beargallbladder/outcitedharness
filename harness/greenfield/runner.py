from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.config import AppConfig
from harness.gateway.orch import OrchResult, run_orch
from harness.orch_loop import (
    MAX_CYCLES,
    command_allowed,
    load_loop_state,
    parse_argv,
    save_loop_state,
)


class GreenfieldRunnerError(RuntimeError):
    pass


HEADLESS_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one workspace-relative text file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List workspace-relative files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search text files inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "regex": {"type": "string"},
                    "file_pattern": {"type": "string"},
                },
                "required": ["path", "regex"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_to_file",
            "description": "Write complete text content to one workspace-relative file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Replace one exact unique text block in a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Run one allowlisted verification command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
]


@dataclass(frozen=True)
class HeadlessRunResult:
    run_id: str
    turns: int
    status: str
    text: str


def _arguments(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(fn.get("name") or "")
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        return name, raw
    try:
        data = json.loads(str(raw or "{}"))
    except json.JSONDecodeError as exc:
        raise GreenfieldRunnerError(f"invalid tool arguments for {name}") from exc
    if not isinstance(data, dict):
        raise GreenfieldRunnerError(f"non-object tool arguments for {name}")
    return name, data


def _workspace_path(root: Path, raw: Any, *, must_exist: bool = False) -> Path:
    value = str(raw or "").strip()
    if not value or "\x00" in value:
        raise GreenfieldRunnerError("empty workspace path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise GreenfieldRunnerError(f"path escapes workspace: {value}")
    candidate = root.joinpath(relative)
    resolved_root = root.resolve()
    resolved = candidate.resolve(strict=False)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise GreenfieldRunnerError(f"path escapes workspace: {value}")
    if candidate.is_symlink():
        raise GreenfieldRunnerError(f"symlink path rejected: {value}")
    if must_exist and not candidate.exists():
        raise FileNotFoundError(value)
    return candidate


def _ignored(path: Path, root: Path) -> bool:
    ignored = {".git", ".venv", "node_modules", ".pytest_cache", ".ruff_cache"}
    return any(part in ignored for part in path.relative_to(root).parts)


def _command_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "npm_config_audit": "false",
            "npm_config_fund": "false",
        }
    )
    return env


def _workspace_file_state(root: Path) -> dict[str, str]:
    state: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or _ignored(path, root):
            continue
        state[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return state


def _try_deterministic_lint_repair(root: Path, state: Any) -> bool:
    """Apply a contract-owned Ruff safe fix without model interpretation."""
    if state is None or state.phase != "repair" or not state.last_cmd:
        return False
    if getattr(state, "result_cmd", None) != state.last_cmd:
        return False
    failure = f"{state.stdout_tail}\n{state.stderr_tail}".lower()
    if "fixable" not in failure:
        return False
    argv = parse_argv(state.last_cmd)
    if (
        len(argv) < 2
        or Path(argv[0]).name.lower() != "ruff"
        or argv[1] != "check"
        or any(arg in {"--fix", "--unsafe-fixes"} for arg in argv)
    ):
        return False
    before = _workspace_file_state(root)
    proc = subprocess.run(
        [*argv, "--fix"],
        cwd=root,
        env=_command_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    after = _workspace_file_state(root)
    changed = sorted(
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )
    if not changed:
        return False
    state.iteration = min(state.iteration + 1, MAX_CYCLES)
    state.phase = "verify"
    state.verify_index = 0
    state.active_diff_hash = None
    state.verification_results = []
    state.result_cmd = None
    state.last_exit = proc.returncode
    state.stdout_tail = (proc.stdout or "")[-4000:]
    state.stderr_tail = (proc.stderr or "")[-4000:]
    for path in changed:
        if path not in state.working_set.files_changed:
            state.working_set.files_changed.append(path)
    state.working_set.refresh_pending = changed
    state.working_set.refresh_diff_pending = True
    return True


def _read(root: Path, args: dict[str, Any]) -> dict[str, Any]:
    path = _workspace_path(root, args.get("path"), must_exist=True)
    if not path.is_file():
        raise GreenfieldRunnerError(f"not a file: {path.relative_to(root)}")
    text = path.read_text()
    return {"path": str(path.relative_to(root)), "result": text[:80_000], "success": True}


def _list(root: Path, args: dict[str, Any]) -> dict[str, Any]:
    base = _workspace_path(root, args.get("path") or ".", must_exist=True)
    recursive = bool(args.get("recursive"))
    rows: list[str] = []
    iterator = base.rglob("*") if recursive else base.iterdir()
    for path in iterator:
        if _ignored(path, root):
            continue
        suffix = "/" if path.is_dir() else ""
        rows.append(f"{path.relative_to(root)}{suffix}")
        if len(rows) >= 500:
            break
    return {"path": str(base.relative_to(root)), "result": "\n".join(sorted(rows)), "success": True}


def _search(root: Path, args: dict[str, Any]) -> dict[str, Any]:
    base = _workspace_path(root, args.get("path") or ".", must_exist=True)
    try:
        pattern = re.compile(str(args.get("regex") or "."))
    except re.error as exc:
        raise GreenfieldRunnerError(f"invalid search regex: {exc}") from exc
    glob = str(args.get("file_pattern") or "*")
    rows: list[str] = []
    iterator = base.rglob(glob) if base.is_dir() else [base]
    for path in iterator:
        if not path.is_file() or _ignored(path, root):
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                rows.append(f"{path.relative_to(root)}:{number}:{line[:500]}")
                if len(rows) >= 300:
                    break
        if len(rows) >= 300:
            break
    return {"result": "\n".join(rows), "success": True}


def _write(root: Path, args: dict[str, Any]) -> dict[str, Any]:
    path = _workspace_path(root, args.get("path"))
    content = args.get("content")
    if not isinstance(content, str):
        raise GreenfieldRunnerError("write content must be text")
    if len(content) > 500_000:
        raise GreenfieldRunnerError("write content exceeds 500000 characters")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if any(part.is_symlink() for part in [parent, *parent.parents] if part != root.parent):
        raise GreenfieldRunnerError(f"symlink parent rejected: {path.relative_to(root)}")
    path.write_text(content)
    return {
        "path": str(path.relative_to(root)),
        "result": f"Wrote {len(content)} characters to {path.relative_to(root)}",
        "success": True,
    }


def _replace(root: Path, args: dict[str, Any]) -> dict[str, Any]:
    path = _workspace_path(root, args.get("path"), must_exist=True)
    if not path.is_file():
        raise GreenfieldRunnerError(f"not a file: {path.relative_to(root)}")
    old = args.get("old_string")
    new = args.get("new_string")
    if not isinstance(old, str) or not old:
        raise GreenfieldRunnerError("old_string must be non-empty text")
    if not isinstance(new, str):
        raise GreenfieldRunnerError("new_string must be text")
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise GreenfieldRunnerError(
            f"old_string matched {count} times in {path.relative_to(root)}"
        )
    path.write_text(text.replace(old, new, 1))
    return {
        "path": str(path.relative_to(root)),
        "result": f"Replaced exact text in {path.relative_to(root)}",
        "success": True,
    }


def _execute(root: Path, args: dict[str, Any]) -> dict[str, Any]:
    command = str(args.get("command") or "")
    argv = parse_argv(command)
    if not argv:
        raise GreenfieldRunnerError(f"command could not be parsed: {command}")
    read_only_diff = len(argv) >= 4 and argv[:3] == ["git", "diff", "--"]
    if read_only_diff:
        for path in argv[3:]:
            _workspace_path(root, path)
    elif not command_allowed(command):
        raise GreenfieldRunnerError(f"command rejected by allowlist: {command}")
    timeout = max(1, min(int(args.get("timeout") or 60), 600))
    try:
        result = subprocess.run(
            argv,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_command_env(),
            check=False,
        )
        return {
            "command": command,
            "stdout": result.stdout[-40_000:],
            "stderr": result.stderr[-40_000:],
            "exit_code": result.returncode,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "stdout": str(exc.stdout or "")[-40_000:],
            "stderr": str(exc.stderr or "")[-40_000:],
            "exit_code": None,
            "timed_out": True,
            "success": False,
        }


def execute_headless_call(root: Path, call: dict[str, Any]) -> str:
    name, args = _arguments(call)
    handlers = {
        "read_file": _read,
        "list_files": _list,
        "search_files": _search,
        "write_to_file": _write,
        "replace_in_file": _replace,
        "execute_command": _execute,
    }
    handler = handlers.get(name)
    if handler is None:
        raise GreenfieldRunnerError(f"unsupported headless tool: {name}")
    try:
        payload = handler(root, args)
    except Exception as exc:
        payload = {
            "error": f"{type(exc).__name__}: {exc}",
            "success": False,
        }
    return json.dumps(payload, sort_keys=True)


def _initial_messages(controller: Any, run_id: str, prompt: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    run = controller.service.get(run_id)
    active = next(
        (
            row
            for row in run.milestones
            if row.ordinal > 0 and row.state == "active" and row.task_id
        ),
        None,
    )
    if active is None:
        return messages
    state = load_loop_state(controller.tasks, active.task_id)
    if (
        state is not None
        and state.phase in {"expand", "repair", "verify"}
        and state.last_cmd
        and (
            state.phase in {"expand", "repair"}
            or getattr(state, "result_cmd", None) == state.last_cmd
        )
    ):
        call_id = "recovered_verification"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "execute_command",
                                "arguments": json.dumps(
                                    {"command": state.last_cmd}
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        {
                            "command": state.last_cmd,
                            "stdout": state.stdout_tail,
                            "stderr": state.stderr_tail,
                            "exit_code": state.last_exit,
                            "timed_out": state.timed_out,
                            "success": state.last_exit == 0
                            and not state.timed_out,
                        },
                        sort_keys=True,
                    ),
                },
            ]
        )
    return messages


async def run_greenfield_headless(
    cfg: AppConfig,
    run_id: str,
    *,
    max_turns: int = 80,
) -> HeadlessRunResult:
    from harness.greenfield.controller import GreenfieldController

    controller = GreenfieldController(cfg)
    run = controller.resume(run_id)
    if run.status != "running" or not run.workspace_root:
        raise GreenfieldRunnerError(
            f"greenfield run must be running, got {run.status}"
        )
    root = Path(run.workspace_root)
    prompt = f"Continue greenfield {run_id}"
    active = next(
        (
            row
            for row in run.milestones
            if row.ordinal > 0 and row.state == "active" and row.task_id
        ),
        None,
    )
    if active is not None:
        state = load_loop_state(controller.tasks, active.task_id)
        if _try_deterministic_lint_repair(root, state):
            save_loop_state(controller.tasks, active.task_id, state)
    messages = _initial_messages(controller, run_id, prompt)
    active_task = ""
    last_text = ""
    for turn in range(1, max_turns + 1):
        current = controller.service.get(run_id)
        active = next(
            (
                row
                for row in current.milestones
                if row.ordinal > 0 and row.state == "active"
            ),
            None,
        )
        current_task = active.task_id if active and active.task_id else ""
        if active_task and current_task and current_task != active_task:
            messages = _initial_messages(controller, run_id, prompt)
        active_task = current_task or active_task
        result: OrchResult = await run_orch(
            cfg,
            prompt,
            messages=messages,
            tools=HEADLESS_TOOLS,
            extra={"workspace_root": str(root)},
        )
        if result.text:
            last_text = result.text
        if not result.tool_calls:
            latest = controller.service.get(run_id)
            return HeadlessRunResult(run_id, turn, latest.status, last_text)
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": result.tool_calls,
            }
        )
        for call in result.tool_calls:
            tool_id = str(call.get("id") or "")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": execute_headless_call(root, call),
                }
            )
        await asyncio.sleep(0)
    raise GreenfieldRunnerError(f"headless execution exceeded {max_turns} turns")
