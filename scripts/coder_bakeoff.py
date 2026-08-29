#!/usr/bin/env python3
"""Compare identical coding prompts on Qwen coders and DeepSeek TP=2."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.config import ModelConfig, load_config
from harness.providers.base import ChatMessage, ChatRequest, ChatResult
from harness.providers.openai_compatible import OpenAICompatibleProvider


CODE_MODE = os.environ.get("DEEPSEEK_CODE_MODE") == "1"
OUTPUT = Path(
    os.environ.get(
        "CODER_BAKEOFF_OUTPUT",
        (
            "results/coder_bakeoff_code_mode_20260828.json"
            if CODE_MODE
            else "results/coder_bakeoff_20260828.json"
        ),
    )
)
MODEL_KEYS = tuple(
    key.strip()
    for key in os.environ.get(
        "CODER_BAKEOFF_MODEL_KEYS",
        "dgx2_qwen,asus_qwen,dgx3_qwen,deepseek_flash_tp2_shadow",
    ).split(",")
    if key.strip()
)
OVERRIDE_KEY = os.environ.get("CODER_BAKEOFF_OVERRIDE_KEY")
OVERRIDE_BASE_URL = os.environ.get("CODER_BAKEOFF_BASE_URL")
OVERRIDE_MODEL = os.environ.get("CODER_BAKEOFF_MODEL")
OVERRIDE_DISPLAY_NAME = os.environ.get("CODER_BAKEOFF_DISPLAY_NAME")


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str
    tests: str


TASKS = (
    Task(
        "merge_ranges",
        """\
Implement this Python function using only built-ins:

def merge_ranges(ranges):
    \"\"\"Normalize integer endpoint pairs and merge overlapping or touching ranges.

    Each input pair may be reversed. Return a sorted list of (start, end) tuples.
    Two ranges touch when the next start is at most current_end + 1.
    Do not mutate the input.
    \"\"\"

Return only the complete function definition, without markdown or explanation.
""",
        """\
assert merge_ranges([]) == []
assert merge_ranges([(5, 1)]) == [(1, 5)]
assert merge_ranges([(1, 2), (3, 4), (9, 7), (8, 8)]) == [(1, 4), (7, 9)]
source = [[10, 12], [2, 3], [4, 4], [20, 18]]
copy = [row[:] for row in source]
assert merge_ranges(source) == [(2, 4), (10, 12), (18, 20)]
assert source == copy
assert merge_ranges([(-3, -1), (0, 0), (2, 2)]) == [(-3, 0), (2, 2)]
""",
    ),
    Task(
        "topological_batches",
        """\
Implement this Python function using only built-ins:

def topological_batches(graph):
    \"\"\"Return deterministic dependency batches.

    graph maps a node to an iterable of nodes it depends on. Dependencies absent
    as graph keys are still nodes. Each returned batch is lexicographically
    sorted, and all currently-ready nodes belong in the same batch. Raise
    ValueError if a cycle exists. Do not mutate the input.
    \"\"\"

Return only the complete function definition, without markdown or explanation.
""",
        """\
assert topological_batches({}) == []
assert topological_batches({"build": {"lint", "test"}, "lint": {"parse"}, "test": {"parse"}}) == [
    ["parse"], ["lint", "test"], ["build"]
]
assert topological_batches({"b": ["a"], "c": [], "d": ["b", "c"]}) == [
    ["a", "c"], ["b"], ["d"]
]
graph = {"a": ["b"], "b": ["a"]}
try:
    topological_batches(graph)
except ValueError:
    pass
else:
    raise AssertionError("cycle was not rejected")
assert graph == {"a": ["b"], "b": ["a"]}
""",
    ),
    Task(
        "reduce_job_events",
        """\
Implement this Python function using only built-ins:

def reduce_job_events(events):
    \"\"\"Reduce ordered job events to current states.

    Each event has string id, integer revision, and state. For each id, accept an
    event only when its revision is strictly greater than the highest revision
    already seen for that id. Return a dict mapping id to the accepted state.
    Do not mutate events.
    \"\"\"

Return only the complete function definition, without markdown or explanation.
""",
        """\
assert reduce_job_events([]) == {}
events = [
    {"id": "a", "revision": 2, "state": "running"},
    {"id": "b", "revision": 1, "state": "queued"},
    {"id": "a", "revision": 1, "state": "stale"},
    {"id": "a", "revision": 2, "state": "duplicate"},
    {"id": "b", "revision": 4, "state": "done"},
    {"id": "a", "revision": 7, "state": "blocked"},
]
original = [dict(row) for row in events]
assert reduce_job_events(events) == {"a": "blocked", "b": "done"}
assert events == original
assert reduce_job_events([
    {"id": "x", "revision": -2, "state": "old"},
    {"id": "x", "revision": -1, "state": "new"},
]) == {"x": "new"}
""",
    ),
    Task(
        "json_pointer_get",
        """\
Implement this Python function using only built-ins:

def json_pointer_get(document, pointer):
    \"\"\"Resolve an RFC 6901 JSON Pointer for reading.

    Empty pointer returns document. Non-empty pointers must begin with '/'.
    Decode ~1 as '/' and ~0 as '~'; reject every other '~' escape with
    ValueError. Dict segments are keys. List segments must be canonical
    non-negative decimal indexes: '0' is valid, leading zeroes and '-' are not.
    Let missing keys and out-of-range indexes raise their normal exceptions.
    \"\"\"

Return only the complete function definition, without markdown or explanation.
""",
        """\
doc = {"a/b": {"~key": [10, {"": "empty"}]}, "0": "dict-zero"}
assert json_pointer_get(doc, "") is doc
assert json_pointer_get(doc, "/a~1b/~0key/1/") == "empty"
assert json_pointer_get(doc, "/0") == "dict-zero"
for pointer in ("a", "/a~2b", "/a~", "/a~01b"):
    try:
        json_pointer_get(doc, pointer)
    except (ValueError, KeyError):
        pass
    else:
        raise AssertionError(f"invalid pointer accepted: {pointer}")
for pointer in ("/items/01", "/items/-", "/items/-1"):
    try:
        json_pointer_get({"items": [1, 2]}, pointer)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError(f"invalid list index accepted: {pointer}")
try:
    json_pointer_get({"items": [1]}, "/items/2")
except IndexError:
    pass
else:
    raise AssertionError("out-of-range index was not rejected")
""",
    ),
)

BANNED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}


def _code(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return value


def _validate_ast(code: str, function_name: str) -> None:
    tree = ast.parse(code)
    top_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(top_functions) != 1 or top_functions[0].name != function_name:
        raise ValueError(f"expected exactly one {function_name} function")
    if len(tree.body) != 1:
        raise ValueError("only one top-level function is allowed")
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Global,
                ast.Import,
                ast.ImportFrom,
                ast.Nonlocal,
            ),
        ):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_CALLS:
                raise ValueError(f"disallowed call: {node.func.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access is disallowed")


def _limits() -> None:
    for kind, value in (
        (resource.RLIMIT_CPU, (2, 2)),
        (resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024)),
    ):
        try:
            resource.setrlimit(kind, value)
        except (OSError, ValueError):
            # RLIMIT_AS is not consistently settable from a macOS preexec hook.
            pass


def _run_hidden_tests(code: str, task: Task) -> tuple[bool, str]:
    try:
        _validate_ast(code, task.name)
    except Exception as exc:
        return False, f"AST validation: {exc}"
    source = f"{code}\n\n{task.tests}\n"
    with tempfile.TemporaryDirectory(prefix="harness-coder-bakeoff-") as root:
        path = Path(root) / "candidate.py"
        path.write_text(source)
        try:
            run = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=4,
                preexec_fn=_limits,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "hidden tests timed out"
    if run.returncode:
        detail = (run.stderr or run.stdout).strip().splitlines()
        return False, detail[-1][:300] if detail else f"exit {run.returncode}"
    return True, "passed"


def _reasoning_chars(result: ChatResult) -> int:
    try:
        message = result.raw_response["choices"][0]["message"]
        return len(message.get("reasoning") or "")
    except Exception:
        return 0


async def _call(model: ModelConfig, task: Task) -> ChatResult:
    return await OpenAICompatibleProvider(model).chat(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "You are a coding worker. Produce only the requested "
                        "function and obey all constraints."
                    ),
                ),
                ChatMessage(role="user", content=task.prompt),
            ],
            temperature=0,
            max_tokens=2048,
            extra_body=model.extra_body,
            timeout_s=max(model.timeout_s, 300),
            seed=0,
        )
    )


async def _run_model(key: str) -> dict[str, Any]:
    model = load_config().models[key]
    result_key = key
    if key == OVERRIDE_KEY:
        updates: dict[str, Any] = {
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "preserve_thinking": False,
                }
            }
        }
        if OVERRIDE_BASE_URL:
            updates["base_url"] = OVERRIDE_BASE_URL
        if OVERRIDE_MODEL:
            updates["model"] = OVERRIDE_MODEL
        if OVERRIDE_DISPLAY_NAME:
            updates["display_name"] = OVERRIDE_DISPLAY_NAME
        model = model.model_copy(update=updates)
    if CODE_MODE and key == "deepseek_flash_tp2_shadow":
        model = model.model_copy(
            update={
                "extra_body": {
                    "chat_template_kwargs": {"thinking": False},
                }
            }
        )
        result_key = "deepseek_flash_tp2_code_mode"
    rows = []
    for task in TASKS:
        result = await _call(model, task)
        code = _code(result.text)
        passed, detail = (
            _run_hidden_tests(code, task)
            if not result.error
            else (False, result.error)
        )
        rows.append(
            {
                "task": task.name,
                "passed": passed,
                "detail": detail,
                "latency_ms": round(result.latency_ms, 2),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "reasoning_chars": _reasoning_chars(result),
                "code": code,
            }
        )
    passed = sum(row["passed"] for row in rows)
    latency = sum(row["latency_ms"] for row in rows)
    tokens = sum(row["output_tokens"] or 0 for row in rows)
    return {
        "model_key": result_key,
        "display_name": model.display_name,
        "passed": passed,
        "total": len(TASKS),
        "all_verified": passed == len(TASKS),
        "total_latency_ms": round(latency, 2),
        "median_task_latency_ms": round(
            statistics.median(row["latency_ms"] for row in rows), 2
        ),
        "output_tokens": tokens,
        "output_tokens_per_second": (
            round(tokens / (latency / 1000), 3) if tokens and latency else None
        ),
        "reasoning_chars": sum(row["reasoning_chars"] for row in rows),
        "tasks": rows,
    }


async def main() -> None:
    rows = await asyncio.gather(*(_run_model(key) for key in MODEL_KEYS))
    rows.sort(key=lambda row: (-row["passed"], row["total_latency_ms"]))
    qwen = [row for row in rows if not row["model_key"].startswith("deepseek_")]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": (
            "Identical prompts and hidden tests; four sequential tasks per model; "
            "models run concurrently."
        ),
        "qwen_median_latency_ms": round(
            statistics.median(row["total_latency_ms"] for row in qwen), 2
        ),
        "models": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    key: row[key]
                    for key in (
                        "model_key",
                        "passed",
                        "total",
                        "all_verified",
                        "total_latency_ms",
                        "median_task_latency_ms",
                        "output_tokens_per_second",
                        "reasoning_chars",
                    )
                }
                for row in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
