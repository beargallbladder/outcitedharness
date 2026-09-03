from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from harness.repo_contract import build_repo_contract
from harness.shadow.models import (
    ModelRuntime,
    ShadowAttempt,
    ShadowTask,
    canonical_json,
)
from harness.shadow.objects import ShadowObjectStore
from harness.shadow.spool import ShadowLease, ShadowSpool
from harness.shadow.workspace import ShadowWorkspace, materialize_snapshot
from harness.training.security import assert_no_secrets, redact_text


SYSTEM_PROMPT = """You are an independent local shadow coding agent.
You are evaluating the user's request without access to any frontier-model answer.
You may inspect and modify only the isolated repository through the supplied tools.
Never request credentials, network access, deployment access, or destructive Git actions.
Use one tool call at a time. Inspect relevant code before changing it.
The apply_patch tool is writable and available. It applies a unified Git diff to
the isolated workspace. Never claim that all supplied tools are read-only.
For a repair task, you must call apply_patch successfully before finishing.
When your best independent attempt is complete, call finish exactly once.
Do not claim tests passed: this phase does not expose a command-execution tool.
"""

_REPAIR_SENTINEL = "<TASK_KIND>repair</TASK_KIND>"
_SHADOW_SENTINEL = "<TASK_KIND>shadow</TASK_KIND>"
_CHANGE_REQUEST = re.compile(
    r"\b(?:add|build|change|create|fix|implement|integrate|migrate|modify|patch|"
    r"refactor|remove|rename|repair|replace|update|wire)\b",
    re.IGNORECASE,
)
_QUESTION_PREFIX = re.compile(
    r"^\s*(?:are|can|could|did|do|does|how|is|should|what|when|where|who|why|would)\b",
    re.IGNORECASE,
)


def repair_expected(prompt: str) -> bool:
    """Conservatively distinguish code mutations from ordinary questions."""

    if _REPAIR_SENTINEL.casefold() in prompt.casefold():
        return True
    if not _CHANGE_REQUEST.search(prompt):
        return False
    first_line = prompt.strip().splitlines()[0] if prompt.strip() else ""
    if _QUESTION_PREFIX.search(first_line) and "?" in first_line:
        return False
    return True


def shadow_actionable(prompt: str) -> bool:
    """Return whether a prompt belongs in the repository code shadow lane."""

    return _SHADOW_SENTINEL.casefold() in prompt.casefold() or repair_expected(prompt)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List tracked and allowed untracked repository files.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a bounded line range from one repository file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "max_lines": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Regex-search bounded text files in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply one text-only unified Git patch in the isolated workspace.",
            "parameters": {
                "type": "object",
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish the independent attempt with a concise answer.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    },
]


class RetryableShadowError(RuntimeError):
    pass


def _headers(runtime: ModelRuntime) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(runtime.api_key_env) if runtime.api_key_env else None
    if not key and runtime.api_key_file is not None:
        path = runtime.api_key_file.expanduser()
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise RetryableShadowError("model API key file is unavailable or unsafe")
        if path.stat().st_mode & 0o077:
            raise RetryableShadowError("model API key file permissions are too broad")
        key = path.read_text().strip()
    if runtime.api_key_env is None and runtime.api_key_file is None:
        return headers
    if not key:
        raise RetryableShadowError(
            "required model API key is unavailable from environment or file"
        )
    headers["Authorization"] = f"Bearer {key}"
    return headers


def _model_identity(runtime: ModelRuntime) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "base_url": runtime.base_url,
                "model": runtime.model,
            }
        )
    ).hexdigest()


def _verify_model_identity(
    client: httpx.Client,
    runtime: ModelRuntime,
    headers: dict[str, str],
) -> None:
    try:
        response = client.get(runtime.base_url + "models", headers=headers)
    except httpx.HTTPError as exc:
        raise RetryableShadowError(
            f"local model identity probe failed: {type(exc).__name__}"
        ) from exc
    if response.status_code >= 500:
        raise RetryableShadowError(
            f"local model identity probe returned HTTP {response.status_code}"
        )
    if response.status_code >= 400:
        raise RuntimeError(
            f"local model identity probe returned HTTP {response.status_code}"
        )
    body = response.json()
    rows = body.get("data") if isinstance(body, dict) else None
    names = {
        str(row.get("id") or row.get("name"))
        for row in rows or []
        if isinstance(row, dict) and (row.get("id") or row.get("name"))
    }
    if runtime.model not in names:
        raise RuntimeError("configured local shadow model is not listed by endpoint")


def _contract_text(task: ShadowTask) -> str:
    contract = build_repo_contract(task.snapshot.repository_root)
    if contract is None:
        return "No repository verification contract was detected."
    commands = "\n".join(
        f"- {command.name}: {command.command}" for command in contract.commands
    )
    return (
        f"Repository contract fingerprint: {contract.fingerprint}\n"
        f"Languages: {', '.join(contract.languages) or '(unknown)'}\n"
        f"Verification commands (not executable in this shadow phase):\n"
        f"{commands or '(none)'}"
    )


def _tool_result(
    workspace: ShadowWorkspace,
    name: str,
    arguments: dict[str, Any],
) -> tuple[str, str | None]:
    if name == "list_files":
        return workspace.list_files(str(arguments.get("pattern") or "")), None
    if name == "read_file":
        return (
            workspace.read_file(
                str(arguments["path"]),
                start_line=int(arguments.get("start_line") or 1),
                max_lines=int(arguments.get("max_lines") or 400),
            ),
            None,
        )
    if name == "search_files":
        return (
            workspace.search_files(
                str(arguments["pattern"]),
                str(arguments.get("path") or "."),
            ),
            None,
        )
    if name == "apply_patch":
        return workspace.apply_patch(str(arguments["patch"])), None
    if name == "finish":
        summary = redact_text(str(arguments["summary"]).strip())
        assert_no_secrets(summary, field="shadow finish summary")
        if not summary:
            raise ValueError("finish summary cannot be empty")
        return "FINISHED", summary
    raise ValueError(f"unknown shadow tool: {name}")


def run_shadow_attempt(
    lease: ShadowLease,
    runtime: ModelRuntime,
    *,
    spool_root: Path,
) -> ShadowAttempt:
    started = time.perf_counter()
    task = lease.task
    workspace_path = materialize_snapshot(
        task.snapshot,
        task.policy,
        work_root=runtime.work_root,
        object_store=ShadowObjectStore(spool_root),
        workspace_id=f"{task.task_id}-attempt-{lease.attempt}",
    )
    workspace = ShadowWorkspace(workspace_path, task.policy)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"<TASK>\n{task.prompt}\n</TASK>\n"
                f"<REPOSITORY_STATE>{task.snapshot.state_sha256}</REPOSITORY_STATE>\n"
                f"<CONTRACT>\n{_contract_text(task)}\n</CONTRACT>"
            ),
        },
    ]
    transcript: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    final_answer = ""
    last_content = ""
    seen_tool_calls: dict[str, int] = {}
    requires_patch = repair_expected(task.prompt)
    patch_applied = False
    context_chars = sum(len(str(message.get("content") or "")) for message in messages)
    messages[1]["content"] += (
        "\n<REPAIR_EXPECTED>"
        + ("true" if requires_patch else "false")
        + "</REPAIR_EXPECTED>"
    )
    headers = _headers(runtime)
    endpoint = runtime.base_url + "chat/completions"

    with httpx.Client(timeout=runtime.timeout_seconds) as client:
        _verify_model_identity(client, runtime, headers)
        for turn in range(1, task.policy.max_agent_turns + 1):
            payload = {
                "model": runtime.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 4096,
                "tools": TOOLS,
                "tool_choice": (
                    {
                        "type": "function",
                        "function": {
                            "name": (
                                "apply_patch"
                                if requires_patch and not patch_applied
                                else "finish"
                            )
                        },
                    }
                    if turn == task.policy.max_agent_turns
                    else "required"
                ),
                "parallel_tool_calls": False,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "preserve_thinking": False,
                },
            }
            try:
                response = client.post(endpoint, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                raise RetryableShadowError(
                    f"local model request failed: {type(exc).__name__}"
                ) from exc
            if response.status_code >= 500:
                raise RetryableShadowError(
                    f"local model returned HTTP {response.status_code}"
                )
            if response.status_code >= 400:
                raise RuntimeError(f"local model rejected request: HTTP {response.status_code}")
            body = response.json()
            choices = body.get("choices") if isinstance(body, dict) else None
            if not isinstance(choices, list) or len(choices) != 1:
                raise RuntimeError("local model response has invalid choices")
            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise RuntimeError("local model response has no assistant message")
            usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
            input_tokens += int(usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("completion_tokens") or 0)
            content = redact_text(str(message.get("content") or ""))
            assert_no_secrets(content, field="shadow model response")
            if content.strip():
                last_content = content.strip()
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise RuntimeError("local model tool_calls must be a list")
            safe_message: dict[str, Any] = {
                "role": "assistant",
                "content": content or None,
            }
            if calls:
                safe_message["tool_calls"] = calls
            messages.append(safe_message)
            transcript.append(
                {
                    "turn": turn,
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": str(call.get("id") or ""),
                            "name": str((call.get("function") or {}).get("name") or ""),
                            "arguments_sha256": hashlib.sha256(
                                str((call.get("function") or {}).get("arguments") or "{}").encode()
                            ).hexdigest(),
                        }
                        for call in calls
                        if isinstance(call, dict)
                        and isinstance(call.get("function"), dict)
                    ],
                }
            )
            if not calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "Call exactly one available tool; use finish when done.",
                    }
                )
                continue
            if not isinstance(calls, list) or len(calls) > 4:
                raise RuntimeError("shadow agent emitted an invalid tool-call batch")
            for call_index, call in enumerate(calls, 1):
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    raise RuntimeError("shadow tool call is malformed")
                name = str(function.get("name") or "")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    raise RuntimeError("shadow tool arguments are invalid JSON") from exc
                if not isinstance(arguments, dict):
                    raise RuntimeError("shadow tool arguments must be an object")
                call_key = hashlib.sha256(
                    canonical_json({"name": name, "arguments": arguments})
                ).hexdigest()
                seen_tool_calls[call_key] = seen_tool_calls.get(call_key, 0) + 1
                if seen_tool_calls[call_key] > 1:
                    observation = (
                        "DUPLICATE_TOOL_CALL: this exact observation is already "
                        "in context; use it or call finish."
                    )
                    finished = None
                elif name == "finish" and requires_patch and not patch_applied:
                    observation = (
                        "REPAIR_REQUIRED: apply_patch is writable and must be used "
                        "successfully before finish for this task."
                    )
                    finished = None
                else:
                    try:
                        observation, finished = _tool_result(
                            workspace,
                            name,
                            arguments,
                        )
                        if name == "apply_patch" and observation == "PATCH_APPLIED":
                            patch_applied = True
                    except Exception as exc:
                        observation = f"TOOL_ERROR: {type(exc).__name__}: {exc}"
                        finished = None
                observation = redact_text(observation)[:20_000]
                remaining = max(512, task.policy.max_context_chars - context_chars)
                if len(observation) > remaining:
                    observation = (
                        observation[:remaining]
                        + "\n[CONTEXT_BUDGET_REACHED: call finish or use a narrower tool]"
                    )
                context_chars += len(observation)
                assert_no_secrets(observation, field="shadow tool observation")
                transcript.append(
                    {
                        "turn": turn,
                        "role": "tool",
                        "name": name,
                        "result": observation,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(
                            call.get("id") or f"call-{turn}-{call_index}"
                        ),
                        "content": observation,
                    }
                )
                if finished is not None:
                    final_answer = finished
                    break
            if final_answer:
                break

    patch = workspace.diff()
    if requires_patch and patch_applied and patch and not final_answer:
        final_answer = last_content or "Applied an independent repair patch."
    if not final_answer:
        error = (
            "repair-class task exhausted its turn budget without a successful "
            "apply_patch"
            if requires_patch and not patch
            else "shadow agent exhausted its turn budget without finishing"
        )
        return ShadowAttempt(
            attempt_id=f"attempt-{task.task_id}-{lease.attempt}",
            task_id=task.task_id,
            status="quarantined",
            model=runtime.model,
            model_endpoint_sha256=_model_identity(runtime),
            answer=last_content,
            patch=patch,
            transcript=tuple(transcript),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=error,
            workspace_state_sha256=workspace.state_sha256(),
            created_at=datetime.now(timezone.utc),
        )
    return ShadowAttempt(
        attempt_id=f"attempt-{task.task_id}-{lease.attempt}",
        task_id=task.task_id,
        status="completed",
        model=runtime.model,
        model_endpoint_sha256=_model_identity(runtime),
        answer=final_answer,
        patch=patch,
        transcript=tuple(transcript),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=(time.perf_counter() - started) * 1000,
        workspace_state_sha256=workspace.state_sha256(),
        created_at=datetime.now(timezone.utc),
    )


def run_one(
    spool: ShadowSpool,
    runtime: ModelRuntime,
    *,
    lease_seconds: int = 1800,
) -> str | None:
    lease = spool.claim(lease_seconds=lease_seconds)
    if lease is None:
        return None
    try:
        attempt = run_shadow_attempt(
            lease,
            runtime,
            spool_root=spool.root,
        )
    except RetryableShadowError as exc:
        spool.fail(lease, str(exc), retryable=True)
        return lease.task.task_id
    except Exception as exc:
        error = redact_text(f"{type(exc).__name__}: {exc}")[:2000]
        attempt = ShadowAttempt(
            attempt_id=f"attempt-{lease.task.task_id}-{lease.attempt}",
            task_id=lease.task.task_id,
            status="failed",
            model=runtime.model,
            model_endpoint_sha256=_model_identity(runtime),
            error=error,
            created_at=datetime.now(timezone.utc),
        )
        spool.complete(lease, attempt)
        return lease.task.task_id
    spool.complete(lease, attempt)
    return lease.task.task_id
