#!/usr/bin/env python3
"""Long-context, tool-using repository repair bakeoff."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "long_context_agent_workspaces"
OUTPUT = ROOT / "results" / "long_context_agent_bakeoff_20260828.json"


@dataclass(frozen=True)
class Model:
    name: str
    base_url: str
    model: str
    deepseek: bool = False


MODELS = (
    Model(
        "deepseek_tp2_max",
        "http://100.68.133.1:9000/v1",
        "deepseek-v4-flash-0731",
        True,
    ),
    Model(
        "qwen_coder_single",
        "http://100.116.221.82:8900/v1",
        "qwen3-coder-next",
    ),
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List repository files.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 repository file.",
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
            "name": "search_files",
            "description": "Search repository text and return matching lines.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Replace a UTF-8 repository file with complete content.",
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
            "description": "Replace one exact string occurrence in a repository file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_public_tests",
            "description": "Run the repository's visible test suite.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish after implementing and testing the repair.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]


def _write_fixture(root: Path) -> None:
    (root / "src" / "orbit_jobs").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "docs").mkdir()
    (root / "pyproject.toml").write_text(
        """\
[project]
name = "orbit-jobs"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
pythonpath = ["src"]
"""
    )
    (root / "README.md").write_text(
        """\
# Orbit Jobs

Orbit Jobs reduces replicated job events and creates deterministic execution
batches. Operational protocol details live in docs/operations_reference.md.
"""
    )
    (root / "src" / "orbit_jobs" / "__init__.py").write_text(
        """\
from .reducer import reduce_events
from .scheduler import DependencyCycleError, ready_batches

__all__ = ["DependencyCycleError", "ready_batches", "reduce_events"]
"""
    )
    (root / "src" / "orbit_jobs" / "reducer.py").write_text(
        """\
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def reduce_events(events: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    \"\"\"Return the visible job record for each id after reducing events.\"\"\"
    visible: dict[str, dict[str, Any]] = {}
    highest_revision: dict[str, int] = {}
    for event in events:
        job_id = str(event["id"])
        revision = int(event["revision"])
        if revision >= highest_revision.get(job_id, -1):
            highest_revision[job_id] = revision
            if event["state"] == "deleted":
                visible.pop(job_id, None)
                highest_revision.pop(job_id, None)
            else:
                visible[job_id] = dict(event)
    return visible
"""
    )
    (root / "src" / "orbit_jobs" / "scheduler.py").write_text(
        """\
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class DependencyCycleError(ValueError):
    pass


def ready_batches(jobs: Mapping[str, Mapping[str, Any]]) -> list[list[str]]:
    \"\"\"Build deterministic execution waves for queued jobs.\"\"\"
    pending = {
        job_id: set(job.get("dependencies", ()))
        for job_id, job in jobs.items()
        if job.get("state") == "queued"
    }
    completed = {
        job_id for job_id, job in jobs.items() if job.get("state") == "done"
    }
    batches: list[list[str]] = []
    while pending:
        ready = sorted(
            job_id
            for job_id, dependencies in pending.items()
            if dependencies <= completed
            or any(dependency not in jobs for dependency in dependencies)
        )
        if not ready:
            break
        job_id = ready[0]
        batches.append([job_id])
        completed.add(job_id)
        pending.pop(job_id)
    return batches
"""
    )
    (root / "src" / "orbit_jobs" / "service.py").write_text(
        """\
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .reducer import reduce_events
from .scheduler import ready_batches


def plan(events: Iterable[Mapping[str, Any]]) -> list[list[str]]:
    return ready_batches(reduce_events(events))
"""
    )
    (root / "tests" / "test_public.py").write_text(
        """\
from orbit_jobs import ready_batches, reduce_events


def test_reducer_keeps_latest_revision():
    events = [
        {"id": "a", "revision": 1, "state": "queued", "dependencies": []},
        {"id": "a", "revision": 2, "state": "done", "dependencies": []},
    ]
    assert reduce_events(events)["a"]["state"] == "done"


def test_scheduler_orders_simple_chain():
    jobs = {
        "build": {"state": "queued", "dependencies": ["test"]},
        "test": {"state": "queued", "dependencies": ["lint"]},
        "lint": {"state": "queued", "dependencies": []},
    }
    assert ready_batches(jobs) == [["lint"], ["test"], ["build"]]


def test_done_dependency_is_satisfied():
    jobs = {
        "base": {"state": "done", "dependencies": []},
        "ship": {"state": "queued", "dependencies": ["base"]},
    }
    assert ready_batches(jobs) == [["ship"]]
"""
    )
    filler = []
    for index in range(1100):
        filler.append(
            f"record={index:04d} subsystem=relay-{index % 113:03d} "
            f"owner=crew-{index % 47:02d} port={7200 + index % 1700} "
            f"retention={7 + index % 29} checksum={index * 7919 + 104729} "
            "policy=observe-before-mutate status=archived "
            "note=This operational entry describes an unrelated telemetry relay, "
            "its rotation window, escalation owner, and disaster-recovery marker."
        )
        if index == 1033:
            filler.extend(
                [
                    "",
                    "[PROTOCOL ORBIT-REVISION-GATE — NORMATIVE]",
                    "Process events in their supplied order. An event is accepted only",
                    "when its integer revision is strictly greater than the highest",
                    "revision ever observed for that job id. Equal revisions are",
                    "duplicates: the first accepted event wins. A deleted event removes",
                    "the visible job but its revision watermark remains, so stale events",
                    "cannot resurrect it. Returned records must be copies; never mutate",
                    "the input events or nested dependency lists.",
                    "",
                    "[PROTOCOL ORBIT-READY-BATCH — NORMATIVE]",
                    "Only queued jobs are scheduling candidates. A dependency is",
                    "satisfied only when its current visible state is done or when it",
                    "completed in an earlier generated wave. Unknown dependencies and",
                    "dependencies currently failed, cancelled, deleted, or blocked leave",
                    "the job blocked and omitted. Every wave contains all jobs ready at",
                    "that moment, sorted lexicographically. If no wave is ready and every",
                    "remaining blocker refers to another queued candidate, raise",
                    "DependencyCycleError. Mixed sets containing an external or terminal",
                    "blocker are blocked, not cycles, and are omitted.",
                    "",
                ]
            )
    (root / "docs" / "operations_reference.md").write_text(
        "# Operations reference\n\n" + "\n".join(filler) + "\n"
    )


def _bundle(root: Path) -> str:
    sections = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        sections.append(f"\n===== {rel} =====\n{path.read_text()}")
    return "".join(sections)


def _safe_path(root: Path, raw: str) -> Path:
    path = (root / raw).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("path escapes repository")
    return path


def _tool_result(root: Path, name: str, arguments: dict[str, Any]) -> str:
    try:
        if name == "list_files":
            value: Any = [
                str(path.relative_to(root))
                for path in sorted(p for p in root.rglob("*") if p.is_file())
            ]
        elif name == "read_file":
            value = _safe_path(root, str(arguments["path"])).read_text()[:200_000]
        elif name == "search_files":
            query = str(arguments["query"])
            matches = []
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                for number, line in enumerate(path.read_text().splitlines(), 1):
                    if query.lower() in line.lower():
                        matches.append(
                            f"{path.relative_to(root)}:{number}:{line[:500]}"
                        )
                        if len(matches) >= 100:
                            break
                if len(matches) >= 100:
                    break
            value = matches
        elif name == "write_file":
            path = _safe_path(root, str(arguments["path"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(arguments["content"]))
            value = {"success": True, "bytes": path.stat().st_size}
        elif name == "replace_in_file":
            path = _safe_path(root, str(arguments["path"]))
            old = str(arguments["old"])
            text = path.read_text()
            if text.count(old) != 1:
                raise ValueError(f"old string occurs {text.count(old)} times")
            path.write_text(text.replace(old, str(arguments["new"]), 1))
            value = {"success": True}
        elif name == "run_public_tests":
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "tests/test_public.py"],
                cwd=root,
                text=True,
                capture_output=True,
                timeout=60,
            )
            value = {
                "success": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": proc.stdout[-8000:],
                "stderr": proc.stderr[-8000:],
            }
        elif name == "finish":
            value = {"success": True, "summary": str(arguments["summary"])}
        else:
            raise ValueError(f"unsupported tool {name}")
        return json.dumps({"ok": True, "result": value}, sort_keys=True)
    except Exception as exc:
        return json.dumps(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            sort_keys=True,
        )


HIDDEN_TESTS = r"""
import copy
from orbit_jobs import DependencyCycleError, ready_batches, reduce_events

tests = []

def check(name):
    def register(fn):
        tests.append((name, fn))
        return fn
    return register

@check("equal revision first wins")
def _():
    events = [
        {"id": "a", "revision": 4, "state": "queued", "dependencies": []},
        {"id": "a", "revision": 4, "state": "failed", "dependencies": []},
    ]
    assert reduce_events(events)["a"]["state"] == "queued"

@check("tombstone watermark prevents resurrection")
def _():
    events = [
        {"id": "a", "revision": 8, "state": "deleted", "dependencies": []},
        {"id": "a", "revision": 7, "state": "queued", "dependencies": []},
    ]
    assert reduce_events(events) == {}

@check("reducer does not mutate inputs")
def _():
    events = [
        {"id": "a", "revision": 1, "state": "queued", "dependencies": ["x"]}
    ]
    original = copy.deepcopy(events)
    reduced = reduce_events(events)
    reduced["a"]["dependencies"].append("later")
    assert events == original

@check("all ready jobs share one sorted wave")
def _():
    jobs = {
        "z": {"state": "queued", "dependencies": []},
        "a": {"state": "queued", "dependencies": []},
        "m": {"state": "queued", "dependencies": []},
    }
    assert ready_batches(jobs) == [["a", "m", "z"]]

@check("unknown dependency blocks")
def _():
    jobs = {"ship": {"state": "queued", "dependencies": ["ghost"]}}
    assert ready_batches(jobs) == []

@check("failed dependency blocks")
def _():
    jobs = {
        "base": {"state": "failed", "dependencies": []},
        "ship": {"state": "queued", "dependencies": ["base"]},
    }
    assert ready_batches(jobs) == []

@check("pure queued cycle raises")
def _():
    jobs = {
        "a": {"state": "queued", "dependencies": ["b"]},
        "b": {"state": "queued", "dependencies": ["a"]},
    }
    try:
        ready_batches(jobs)
    except DependencyCycleError:
        return
    raise AssertionError("cycle did not raise")

@check("mixed external blocker is not reported as cycle")
def _():
    jobs = {
        "a": {"state": "queued", "dependencies": ["b", "external"]},
        "b": {"state": "queued", "dependencies": ["a"]},
    }
    assert ready_batches(jobs) == []

passed = 0
details = []
for name, test in tests:
    try:
        test()
        passed += 1
        details.append({"name": name, "passed": True})
    except Exception as exc:
        details.append({"name": name, "passed": False, "error": repr(exc)})
print(__import__("json").dumps({"passed": passed, "total": len(tests), "details": details}))
"""


def _score(root: Path) -> dict[str, Any]:
    public = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_public.py"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=60,
    )
    hidden = subprocess.run(
        [sys.executable, "-c", HIDDEN_TESTS],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        text=True,
        capture_output=True,
        timeout=60,
    )
    try:
        hidden_result = json.loads(hidden.stdout.strip().splitlines()[-1])
    except Exception:
        hidden_result = {
            "passed": 0,
            "total": 8,
            "details": [],
            "parse_error": hidden.stdout[-2000:],
        }
    return {
        "public_passed": public.returncode == 0,
        "public_output": (public.stdout + public.stderr)[-4000:],
        "hidden": hidden_result,
        "hidden_stderr": hidden.stderr[-2000:],
    }


def _parse_arguments(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(function.get("name") or call.get("name") or "")
    raw = function.get("arguments", call.get("arguments", {}))
    if isinstance(raw, dict):
        return name, raw
    try:
        value = json.loads(str(raw or "{}"))
        return name, value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return name, {}


async def _run_agent(model: Model, template: Path) -> dict[str, Any]:
    workspace = RUN_ROOT / model.name
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(template, workspace)
    prompt = f"""\
Repair this repository. Two production bugs remain in replicated event reduction
and deterministic scheduling. The visible tests are incomplete. Treat the
operations reference as normative, inspect the implementation, edit the files,
and run the public tests. Do not merely describe a patch: use the supplied tools
to implement it. Finish only after tests pass.

The complete starting repository is bundled below for long-context orientation.
Tool reads reflect the current editable workspace after your changes.

<repository>
{_bundle(workspace)}
</repository>
"""
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a repository repair agent. Use tools autonomously. "
                "Follow repository evidence exactly and verify your edits."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    started = time.perf_counter()
    turns: list[dict[str, Any]] = []
    total_input = 0
    total_output = 0
    first_prompt_tokens: int | None = None
    finished = False
    async with httpx.AsyncClient(timeout=900) as client:
        for turn in range(1, 13):
            payload: dict[str, Any] = {
                "model": model.model,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 32768 if model.deepseek else 8192,
                "seed": 20260828,
            }
            if model.deepseek:
                payload["reasoning_effort"] = "max"
            request_started = time.perf_counter()
            response = await client.post(
                f"{model.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            latency_ms = (time.perf_counter() - request_started) * 1000
            try:
                body = response.json()
            except Exception:
                body = {"raw": response.text}
            if response.status_code >= 400:
                turns.append(
                    {
                        "turn": turn,
                        "latency_ms": round(latency_ms, 2),
                        "error": f"HTTP {response.status_code}: {str(body)[:1000]}",
                    }
                )
                break
            usage = body.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            if first_prompt_tokens is None:
                first_prompt_tokens = prompt_tokens
            total_input += prompt_tokens
            total_output += completion_tokens
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            calls = [
                call
                for call in message.get("tool_calls") or []
                if isinstance(call, dict)
            ]
            turn_row: dict[str, Any] = {
                "turn": turn,
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "finish_reason": choice.get("finish_reason"),
                "content_head": str(message.get("content") or "")[:1000],
                "tools": [],
            }
            assistant_message = {
                key: message[key]
                for key in (
                    "role",
                    "content",
                    "reasoning_content",
                    "reasoning",
                    "tool_calls",
                )
                if key in message
            }
            assistant_message.setdefault("role", "assistant")
            assistant_message.setdefault("content", "")
            messages.append(assistant_message)
            for call in calls:
                name, arguments = _parse_arguments(call)
                result = _tool_result(workspace, name, arguments)
                turn_row["tools"].append(
                    {
                        "name": name,
                        "arguments": arguments,
                        "result_head": result[:1000],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or f"call_{turn}"),
                        "content": result,
                    }
                )
                if name == "finish":
                    finished = True
            turns.append(turn_row)
            if finished:
                break
            if not calls:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue the repair using tools. Implement and test the "
                            "changes before finishing."
                        ),
                    }
                )
    elapsed = time.perf_counter() - started
    result = {
        "model": model.name,
        "workspace": str(workspace),
        "first_prompt_tokens": first_prompt_tokens,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "elapsed_s": round(elapsed, 3),
        "turn_count": len(turns),
        "finished": finished,
        "score": _score(workspace),
        "turns": turns,
    }
    partial = OUTPUT.with_name(f"{OUTPUT.stem}_{model.name}.json")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    return result


async def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    template = RUN_ROOT / "_template"
    if template.exists():
        shutil.rmtree(template)
    template.mkdir()
    _write_fixture(template)
    rows = await asyncio.gather(*(_run_agent(model, template) for model in MODELS))
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": (
            "Identical long repository bundle, editable isolated workspaces, native "
            "tool calls, visible tests, and eight held-out deterministic checks."
        ),
        "models": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    summary = [
        {
            "model": row["model"],
            "first_prompt_tokens": row["first_prompt_tokens"],
            "elapsed_s": row["elapsed_s"],
            "turn_count": row["turn_count"],
            "finished": row["finished"],
            "public_passed": row["score"]["public_passed"],
            "hidden_passed": row["score"]["hidden"]["passed"],
            "hidden_total": row["score"]["hidden"]["total"],
        }
        for row in rows
    ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
