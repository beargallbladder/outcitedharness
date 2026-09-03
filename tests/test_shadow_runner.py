from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from harness.shadow.hook import capture_hook_event
from harness.shadow.models import ModelRuntime
from harness.shadow.runner import _headers, repair_expected, run_shadow_attempt
from harness.shadow.spool import ShadowSpool


def _run(root: Path, *argv: str) -> None:
    result = subprocess.run(
        list(argv),
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _lease(
    tmp_path: Path,
    prompt: str = "<TASK_KIND>shadow</TASK_KIND>\nExplain the application.",
):
    root = tmp_path / "repo"
    root.mkdir()
    _run(root, "git", "init", "-q")
    (root / "app.py").write_text("value = 1\n")
    _run(root, "git", "add", "app.py")
    _run(
        root,
        "git",
        "-c",
        "user.name=Harness Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "base",
    )
    (root / ".harness-shadow.json").write_text(
        json.dumps(
            {
                "version": 1,
                "enabled": True,
                "repository_id": "owner/example",
                "allowed_paths": ["."],
                "excluded_paths": [
                    ".git",
                    ".harness-shadow.json",
                    "**/.git/**",
                ],
                "max_agent_turns": 3,
            }
        )
    )
    spool = ShadowSpool(tmp_path / "spool")
    task_id = capture_hook_event(
        "beforeSubmitPrompt",
        {
            "session_id": "session-one",
            "generation_id": "generation-one",
            "prompt": prompt,
        },
        repository_root=root,
        spool_root=spool.root,
    )
    assert task_id
    lease = spool.claim()
    assert lease
    return spool, lease


def test_runner_uses_verified_model_and_bounded_native_tools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spool, lease = _lease(tmp_path)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"data": [{"id": "qwen-local"}]},
            )
        calls += 1
        body = json.loads(request.content)
        assert body["model"] == "qwen-local"
        assert body["parallel_tool_calls"] is False
        if calls == 1:
            tools = [
                {
                    "id": "call-one",
                    "type": "function",
                    "function": {
                        "name": "list_files",
                        "arguments": "{}",
                    },
                },
                {
                    "id": "call-one-b",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "app.py"}),
                    },
                },
            ]
        else:
            tools = [
                {
                    "id": "call-two",
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": json.dumps(
                            {"summary": "The repository contains app.py."}
                        ),
                    },
                }
            ]
        return httpx.Response(
            200,
            json={
                "model": "qwen-local",
                "choices": [{"message": {"role": "assistant", "tool_calls": tools}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    transport = httpx.MockTransport(handler)
    original = httpx.Client
    monkeypatch.setattr(
        "harness.shadow.runner.httpx.Client",
        lambda **kwargs: original(transport=transport, **kwargs),
    )
    runtime = ModelRuntime(
        base_url="http://qwen.test/v1",
        model="qwen-local",
        spool_root=spool.root,
        work_root=tmp_path / "work",
    )

    attempt = run_shadow_attempt(
        lease,
        runtime,
        spool_root=spool.root,
    )

    assert attempt.status == "completed"
    assert attempt.answer == "The repository contains app.py."
    assert attempt.patch == ""
    assert attempt.input_tokens == 20
    assert attempt.output_tokens == 8
    assert [row["name"] for row in attempt.transcript if row["role"] == "tool"] == [
        "list_files",
        "read_file",
        "finish",
    ]


def test_repair_task_cannot_finish_before_apply_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _spool, lease = _lease(tmp_path, "Repair the incorrect application value.")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "qwen-local"}]})
        calls += 1
        body = json.loads(request.content)
        if calls == 1:
            tools = [
                {
                    "id": "finish-too-early",
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": json.dumps({"summary": "No writable tools exist."}),
                    },
                }
            ]
        elif calls == 2:
            assert "REPAIR_REQUIRED" in body["messages"][-1]["content"]
            tools = [
                {
                    "id": "apply-repair",
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "arguments": json.dumps(
                            {
                                "patch": (
                                    "diff --git a/app.py b/app.py\n"
                                    "--- a/app.py\n"
                                    "+++ b/app.py\n"
                                    "@@ -1 +1 @@\n"
                                    "-value = 1\n"
                                    "+value = 2\n"
                                )
                            }
                        ),
                    },
                }
            ]
        else:
            assert body["tool_choice"]["function"]["name"] == "finish"
            tools = [
                {
                    "id": "finish-after-repair",
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": json.dumps({"summary": "Repaired app.py."}),
                    },
                }
            ]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "tool_calls": tools}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    transport = httpx.MockTransport(handler)
    original = httpx.Client
    monkeypatch.setattr(
        "harness.shadow.runner.httpx.Client",
        lambda **kwargs: original(transport=transport, **kwargs),
    )
    attempt = run_shadow_attempt(
        lease,
        ModelRuntime(
            base_url="http://qwen.test/v1",
            model="qwen-local",
            spool_root=tmp_path / "spool",
            work_root=tmp_path / "work",
        ),
        spool_root=tmp_path / "spool",
    )

    assert attempt.status == "completed"
    assert "+value = 2" in attempt.patch
    assert attempt.answer == "Repaired app.py."
    assert [row["name"] for row in attempt.transcript if row["role"] == "tool"] == [
        "finish",
        "apply_patch",
        "finish",
    ]


def test_repair_classification_ignores_questions() -> None:
    assert repair_expected("<TASK_KIND>repair</TASK_KIND>\nFix it")
    assert repair_expected("Implement the requested endpoint.")
    assert not repair_expected("Why can't you implement this differently?")
    assert not repair_expected("Explain the application.")


def test_private_api_key_file_authenticates_background_worker(tmp_path: Path) -> None:
    key_file = tmp_path / "model-api-key"
    key_file.write_text("private-test-credential\n")
    key_file.chmod(0o600)
    runtime = ModelRuntime(
        base_url="http://qwen.test/v1",
        model="qwen-local",
        api_key_env="UNAVAILABLE_QWEN_KEY",
        api_key_file=key_file,
    )

    assert _headers(runtime)["Authorization"] == "Bearer private-test-credential"


def test_runner_fails_closed_on_model_identity_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spool, lease = _lease(tmp_path)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"data": [{"id": "different-model"}]},
        )
    )
    original = httpx.Client
    monkeypatch.setattr(
        "harness.shadow.runner.httpx.Client",
        lambda **kwargs: original(transport=transport, **kwargs),
    )
    runtime = ModelRuntime(
        base_url="http://qwen.test/v1",
        model="qwen-local",
        spool_root=spool.root,
        work_root=tmp_path / "work",
    )

    with pytest.raises(RuntimeError, match="not listed"):
        run_shadow_attempt(lease, runtime, spool_root=spool.root)
