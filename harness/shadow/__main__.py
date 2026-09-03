from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from harness.shadow.hook import (
    capture_hook_event,
    default_spool_root,
    read_hook_input,
)
from harness.shadow.policy import load_policy, load_runtime
from harness.shadow.processor import process_task
from harness.shadow.repository import discover_repository
from harness.shadow.runner import run_one
from harness.shadow.spool import ShadowSpool
from harness.storage.db import Store
from harness.training.ledger import LearningLedger
from harness.training.security import redact_text


def _record_hook_error(root: Path, event: str, error: BaseException) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    path = root / "hook-errors.jsonl"
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": socket.gethostname(),
        "event": event,
        "error": redact_text(f"{type(error).__name__}: {error}")[:2000],
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def hook_command(arguments: argparse.Namespace) -> int:
    spool_root = Path(arguments.spool).expanduser()
    if arguments.event == "beforeSubmitPrompt":
        output = {"continue": True}
    elif arguments.event == "beforeReadFile":
        output = {"permission": "allow"}
    else:
        output = {}
    try:
        payload = read_hook_input()
        capture_hook_event(
            arguments.event,
            payload,
            repository_root=Path(arguments.repository_root)
            if arguments.repository_root
            else None,
            spool_root=spool_root,
        )
        print(json.dumps(output))
    except BaseException as exc:
        _record_hook_error(spool_root, arguments.event, exc)
        print(json.dumps(output))
    return 0


def worker_command(arguments: argparse.Namespace) -> int:
    runtime = load_runtime(
        Path(arguments.runtime).expanduser() if arguments.runtime else None
    )
    spool = ShadowSpool(runtime.spool_root)
    if not 1 <= arguments.concurrency <= 4:
        raise ValueError("worker concurrency must be between one and four")
    with ThreadPoolExecutor(max_workers=arguments.concurrency) as executor:
        while True:
            task_ids = tuple(
                task_id
                for task_id in (
                    future.result()
                    for future in (
                        executor.submit(run_one, spool, runtime)
                        for _ in range(arguments.concurrency)
                    )
                )
                if task_id is not None
            )
            for task_id in task_ids:
                print(task_id, flush=True)
            if arguments.once:
                return 0
            if not task_ids:
                time.sleep(arguments.poll_interval)


def status_command(arguments: argparse.Namespace) -> int:
    spool = ShadowSpool(Path(arguments.spool).expanduser())
    print(json.dumps(spool.status(), indent=2, sort_keys=True))
    return 0


def _model_names(value: Any) -> set[str]:
    if not isinstance(value, dict):
        return set()
    rows = value.get("data") or value.get("models") or []
    output = set()
    for row in rows:
        if isinstance(row, dict):
            candidate = row.get("id") or row.get("name")
        else:
            candidate = row
        if candidate:
            output.add(str(candidate))
    return output


def doctor_command(arguments: argparse.Namespace) -> int:
    failures = []
    root = discover_repository(Path(arguments.repository_root))
    policy = load_policy(root)
    if policy is None:
        failures.append(f"missing {root / '.harness-shadow.json'}")
    runtime = load_runtime(
        Path(arguments.runtime).expanduser() if arguments.runtime else None
    )
    ShadowSpool(runtime.spool_root)
    key = (
        os.environ.get(runtime.api_key_env)
        if runtime.api_key_env is not None
        else None
    )
    if runtime.api_key_env is not None and not key:
        failures.append(f"missing environment variable {runtime.api_key_env}")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        response = httpx.get(
            runtime.base_url + "models",
            headers=headers,
            timeout=10,
        )
        if response.status_code >= 400:
            failures.append(f"model endpoint returned HTTP {response.status_code}")
        else:
            names = _model_names(response.json())
            if runtime.model not in names:
                failures.append(
                    f"configured model {runtime.model!r} is not listed by endpoint"
                )
    except Exception as exc:
        failures.append(f"model endpoint probe failed: {type(exc).__name__}")
    result = {
        "ok": not failures,
        "repository": str(root),
        "repository_id": policy.repository_id if policy else None,
        "model": runtime.model,
        "spool": str(runtime.spool_root.expanduser()),
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


def capture_command(arguments: argparse.Namespace) -> int:
    if sys.stdin.isatty():
        raise ValueError("manual shadow capture requires the prompt on stdin")
    prompt = sys.stdin.read(200_001)
    if len(prompt) > 200_000:
        raise ValueError("manual shadow prompt exceeds 200,000 characters")
    task_id = capture_hook_event(
        "beforeSubmitPrompt",
        {
            "session_id": arguments.session,
            "generation_id": arguments.generation,
            "prompt": prompt,
            "source": "manual-stdin",
        },
        repository_root=Path(arguments.repository_root),
        spool_root=Path(arguments.spool).expanduser(),
    )
    print(task_id or "")
    return 0


def replay_command(arguments: argparse.Namespace) -> int:
    from harness.shadow.replay import replay_task

    spool = ShadowSpool(Path(arguments.spool).expanduser())
    report = replay_task(
        spool,
        arguments.task_id,
        candidate_kind=arguments.candidate,
        command_names=tuple(arguments.verification_command) or None,
        work_root=Path(arguments.work_root).expanduser()
        if arguments.work_root
        else None,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if report.verdict != "rejected" else 1


def process_command(arguments: argparse.Namespace) -> int:
    spool_root = Path(arguments.spool).expanduser()
    spool = ShadowSpool(spool_root)
    ledger = LearningLedger(
        Store(
            Path(arguments.learning_db).expanduser()
            if arguments.learning_db
            else spool_root / "learning" / "harness.db"
        ),
        Path(arguments.artifact_root).expanduser()
        if arguments.artifact_root
        else spool_root / "learning" / "artifacts",
    )
    while True:
        task_ids = (
            (arguments.task_id,)
            if arguments.task_id
            else spool.processable_tasks(limit=arguments.limit)
        )
        failed = False
        for task_id in task_ids:
            try:
                result = process_task(spool, ledger, task_id)
                print(
                    json.dumps(
                        result.model_dump(mode="json"),
                        sort_keys=True,
                    )
                )
            except (ValueError, RuntimeError, PermissionError, OSError) as exc:
                failed = True
                _record_hook_error(spool_root, "processor", exc)
                print(
                    f"error processing {task_id}: {redact_text(str(exc))}",
                    file=sys.stderr,
                )
        if arguments.once or arguments.task_id:
            return 1 if failed else 0
        time.sleep(arguments.poll_interval)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m harness.shadow")
    commands = root.add_subparsers(dest="command", required=True)

    hook = commands.add_parser("hook", help="Consume one Cursor hook event from stdin")
    hook.add_argument("event")
    hook.add_argument("--repository-root")
    hook.add_argument("--spool", default=str(default_spool_root()))
    hook.set_defaults(handler=hook_command)

    worker = commands.add_parser("worker", help="Run the local Qwen shadow worker")
    worker.add_argument("--runtime")
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--concurrency", type=int, default=2)
    worker.add_argument("--poll-interval", type=float, default=2.0)
    worker.set_defaults(handler=worker_command)

    status = commands.add_parser("status", help="Show durable local shadow queue counts")
    status.add_argument("--spool", default=str(default_spool_root()))
    status.set_defaults(handler=status_command)

    doctor = commands.add_parser("doctor", help="Validate one enrolled workstation")
    doctor.add_argument("--repository-root", default=".")
    doctor.add_argument("--runtime")
    doctor.set_defaults(handler=doctor_command)

    capture = commands.add_parser(
        "capture",
        help="Capture a manual task safely from standard input",
    )
    capture.add_argument("--repository-root", default=".")
    capture.add_argument("--spool", default=str(default_spool_root()))
    capture.add_argument("--session", default="manual")
    capture.add_argument("--generation", default="manual-1")
    capture.set_defaults(handler=capture_command)

    replay = commands.add_parser(
        "replay",
        help="Mechanically replay one local or Cursor candidate.",
    )
    replay.add_argument("task_id")
    replay.add_argument("--candidate", choices=("local", "frontier"), required=True)
    replay.add_argument(
        "--verification-command",
        action="append",
        default=[],
        help="Run only this named checked-in verification command; repeatable.",
    )
    replay.add_argument("--spool", default=str(default_spool_root()))
    replay.add_argument("--work-root")
    replay.set_defaults(handler=replay_command)

    process = commands.add_parser(
        "process",
        help="Replay, compare, and admit completed shadow tasks.",
    )
    process.add_argument("--task-id")
    process.add_argument("--spool", default=str(default_spool_root()))
    process.add_argument("--learning-db")
    process.add_argument("--artifact-root")
    process.add_argument("--limit", type=int, default=20)
    process.add_argument("--once", action="store_true")
    process.add_argument("--poll-interval", type=float, default=5.0)
    process.set_defaults(handler=process_command)
    return root


def main() -> int:
    arguments = parser().parse_args()
    try:
        return int(arguments.handler(arguments))
    except (ValueError, RuntimeError, PermissionError, OSError) as exc:
        print(f"error: {redact_text(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
