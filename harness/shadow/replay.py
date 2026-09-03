from __future__ import annotations

import hashlib
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from harness.repo_contract import RepoContract, VerificationCommand, build_repo_contract
from harness.shadow.models import (
    RepositorySnapshot,
    RelativePath,
    Sha256,
    StrictModel,
    canonical_json,
)
from harness.shadow.objects import ShadowObjectStore
from harness.shadow.spool import ShadowSpool
from harness.shadow.workspace import (
    ShadowWorkspace,
    materialize_snapshot,
    transition_to_snapshot,
)
from harness.training.security import assert_no_secrets, redact_text


NETWORK_DENY_PROFILE = "(version 1)(allow default)(deny network*)"
_PYTEST_FAILURE = re.compile(r"(?m)^(?:FAILED|ERROR)\s+([^\s]+)")
_UNITTEST_FAILURE = re.compile(
    r"(?m)^(?:FAIL|ERROR):\s+([^\n]+)"
)


class ReplayCommandResult(StrictModel):
    name: str
    argv: tuple[str, ...]
    phase: Literal["parent", "candidate"]
    returncode: int | None = None
    timed_out: bool = False
    duration_ms: float = Field(ge=0)
    stdout_sha256: Sha256
    stderr_sha256: Sha256
    stdout_tail: str = ""
    stderr_tail: str = ""
    failure_fingerprints: tuple[str, ...] = ()


class ReplayReport(StrictModel):
    version: Literal[1] = 1
    replay_id: str
    task_id: str
    candidate_kind: Literal["local", "frontier"]
    repository_id: str
    source_revision: str
    parent_state_sha256: Sha256
    candidate_patch_sha256: Sha256
    candidate_patch_object_path: RelativePath
    candidate_patch_bytes: Annotated[int, Field(ge=0)]
    contract_fingerprint: Sha256
    commands: tuple[ReplayCommandResult, ...]
    verdict: Literal[
        "verified_correction",
        "verified_no_regression",
        "rejected",
        "inconclusive",
    ]
    baseline_failed: bool
    candidate_passed: bool
    candidate_no_regression: bool = False
    candidate_improved: bool = False
    network_isolation: Literal["sandbox-exec", "bubblewrap"]
    created_at: datetime
    evidence_sha256: Sha256


def _tail(data: bytes, limit: int = 4000) -> str:
    return redact_text(data[-limit:].decode("utf-8", errors="replace"))


def _failure_fingerprints(stdout: bytes, stderr: bytes) -> tuple[str, ...]:
    text = (stdout + b"\n" + stderr).decode("utf-8", errors="replace")
    values = {
        redact_text(match.group(1).strip())[:500]
        for pattern in (_PYTEST_FAILURE, _UNITTEST_FAILURE)
        for match in pattern.finditer(text)
        if match.group(1).strip()
    }
    for value in values:
        assert_no_secrets(value, field="replay failure fingerprint")
    return tuple(sorted(values))


def _resolve_argv(command: VerificationCommand, source_root: Path) -> list[str]:
    argv = list(command.argv)
    executable = argv[0]
    if executable.startswith(".venv/"):
        candidate = source_root / executable
        if candidate.is_file() and os.access(candidate, os.X_OK):
            argv[0] = str(candidate)
    elif executable in {"python", "python3"}:
        for candidate in (
            source_root / ".venv/bin/python",
            Path(sys.executable),
            Path(shutil.which(executable) or ""),
        ):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                argv[0] = str(candidate)
                break
    return argv


def _sandbox_argv(argv: list[str], workspace: Path) -> tuple[list[str], str]:
    system = platform.system()
    if system == "Darwin":
        executable = Path("/usr/bin/sandbox-exec")
        if not executable.is_file():
            raise RuntimeError("sandbox-exec is required for replay verification")
        return [
            str(executable),
            "-p",
            NETWORK_DENY_PROFILE,
            *argv,
        ], "sandbox-exec"
    if system == "Linux":
        bubblewrap = shutil.which("bwrap")
        if bubblewrap is None:
            raise RuntimeError("bubblewrap is required for replay verification")
        return [
            bubblewrap,
            "--unshare-net",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--bind",
            str(workspace),
            str(workspace),
            "--tmpfs",
            "/tmp",
            "--chdir",
            str(workspace),
            *argv,
        ], "bubblewrap"
    raise RuntimeError("replay verification requires macOS or Linux isolation")


def _execute(
    command: VerificationCommand,
    *,
    phase: Literal["parent", "candidate"],
    workspace: Path,
    source_root: Path,
) -> tuple[ReplayCommandResult, str]:
    argv = _resolve_argv(command, source_root)
    sandboxed, isolation = _sandbox_argv(argv, workspace)
    started = __import__("time").perf_counter()
    with tempfile.TemporaryDirectory(prefix="harness-shadow-command-") as temporary:
        temporary_root = Path(temporary)
        stdout_path = temporary_root / "stdout"
        stderr_path = temporary_root / "stderr"
        environment = {
            "PATH": os.pathsep.join(
                [
                    str(source_root / "node_modules/.bin"),
                    "/opt/homebrew/bin",
                    "/usr/local/bin",
                    "/usr/bin",
                    "/bin",
                    "/usr/sbin",
                    "/sbin",
                ]
            ),
            "HOME": str(temporary_root),
            "TMPDIR": str(temporary_root),
            "CI": "1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONPATH": str(workspace),
            "PYTHONNOUSERSITE": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                sandboxed,
                cwd=workspace,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            timed_out = False
            try:
                returncode = process.wait(timeout=command.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    returncode = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    returncode = process.wait(timeout=5)
        stdout_data = stdout_path.read_bytes()
        stderr_data = stderr_path.read_bytes()
    stdout_tail = _tail(stdout_data)
    stderr_tail = _tail(stderr_data)
    assert_no_secrets(stdout_tail, field="replay stdout")
    assert_no_secrets(stderr_tail, field="replay stderr")
    return (
        ReplayCommandResult(
            name=command.name,
            argv=tuple(argv),
            phase=phase,
            returncode=returncode,
            timed_out=timed_out,
            duration_ms=(__import__("time").perf_counter() - started) * 1000,
            stdout_sha256=hashlib.sha256(stdout_data).hexdigest(),
            stderr_sha256=hashlib.sha256(stderr_data).hexdigest(),
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            failure_fingerprints=_failure_fingerprints(
                stdout_data,
                stderr_data,
            ),
        ),
        isolation,
    )


def _frontier_snapshot(spool: ShadowSpool, task_id: str) -> RepositorySnapshot:
    snapshots = []
    for event in spool.events(task_id):
        value = event.payload.get("repository_state")
        if event.event_type == "stop" and isinstance(value, dict):
            snapshots.append(RepositorySnapshot.model_validate(value))
    if not snapshots:
        raise ValueError("task has no final Cursor repository snapshot")
    return snapshots[-1]


def _selected_commands(
    contract: RepoContract,
    names: tuple[str, ...] | None,
) -> tuple[VerificationCommand, ...]:
    if not names:
        return tuple(contract.commands)
    requested = set(names)
    selected = tuple(
        command for command in contract.commands if command.name in requested
    )
    missing = requested - {command.name for command in selected}
    if missing:
        raise ValueError(f"unknown verification commands: {sorted(missing)}")
    return selected


def replay_task(
    spool: ShadowSpool,
    task_id: str,
    *,
    candidate_kind: Literal["local", "frontier"],
    command_names: tuple[str, ...] | None = None,
    work_root: Path | None = None,
) -> ReplayReport:
    task, _state = spool.get_task(task_id)
    replay_id = f"replay-{uuid.uuid4().hex}"
    root = (work_root or spool.root / "replays").expanduser()
    store = ShadowObjectStore(spool.root)
    parent_root = materialize_snapshot(
        task.snapshot,
        task.policy,
        work_root=root,
        object_store=store,
        workspace_id=f"{replay_id}-parent",
    )
    contract = build_repo_contract(parent_root)
    if contract is None:
        raise ValueError("repository has no verification contract")
    commands = _selected_commands(contract, command_names)
    candidate_root = materialize_snapshot(
        task.snapshot,
        task.policy,
        work_root=root,
        object_store=store,
        workspace_id=f"{replay_id}-candidate",
    )
    candidate = ShadowWorkspace(candidate_root, task.policy)
    if candidate_kind == "local":
        attempt = spool.get_attempt(task_id)
        if attempt is None:
            raise ValueError("task has no local shadow attempt")
        if attempt.patch:
            candidate.apply_patch(attempt.patch)
    else:
        transition_to_snapshot(
            candidate_root,
            task.policy,
            parent=task.snapshot,
            final=_frontier_snapshot(spool, task_id),
            object_store=store,
        )
    candidate_patch = candidate.diff()
    candidate_patch_sha256 = hashlib.sha256(candidate_patch.encode()).hexdigest()
    patch_object_sha256, patch_bytes, patch_object_path = store.put_text(
        candidate_patch
    )
    if patch_object_sha256 != candidate_patch_sha256:
        raise RuntimeError("candidate patch object digest mismatch")

    results = []
    isolation_values = set()
    for command in commands:
        before, isolation = _execute(
            command,
            phase="parent",
            workspace=parent_root,
            source_root=task.snapshot.repository_root,
        )
        results.append(before)
        isolation_values.add(isolation)
        after, isolation = _execute(
            command,
            phase="candidate",
            workspace=candidate_root,
            source_root=task.snapshot.repository_root,
        )
        results.append(after)
        isolation_values.add(isolation)
    baseline = [row for row in results if row.phase == "parent"]
    after = [row for row in results if row.phase == "candidate"]
    baseline_failed = bool(baseline) and any(
        row.timed_out or row.returncode != 0 for row in baseline
    )
    candidate_passed = bool(after) and all(
        not row.timed_out and row.returncode == 0 for row in after
    )
    command_outcomes = []
    for before, candidate_result in zip(baseline, after, strict=True):
        before_failures = set(before.failure_fingerprints)
        after_failures = set(candidate_result.failure_fingerprints)
        repaired_to_green = (
            before.returncode != 0
            and candidate_result.returncode == 0
            and not candidate_result.timed_out
        )
        reduced_named_failures = (
            before.returncode != 0
            and candidate_result.returncode != 0
            and bool(before_failures)
            and after_failures < before_failures
            and not candidate_result.timed_out
        )
        unchanged_green = (
            before.returncode == 0
            and candidate_result.returncode == 0
            and not candidate_result.timed_out
        )
        command_outcomes.append(
            {
                "no_regression": (
                    unchanged_green
                    or repaired_to_green
                    or reduced_named_failures
                ),
                "improved": repaired_to_green or reduced_named_failures,
            }
        )
    candidate_no_regression = bool(command_outcomes) and all(
        row["no_regression"] for row in command_outcomes
    )
    candidate_improved = any(
        row["improved"] for row in command_outcomes
    )
    if not commands:
        verdict = "inconclusive"
    elif candidate_no_regression and candidate_improved:
        verdict = "verified_correction"
    elif candidate_passed and not baseline_failed:
        verdict = "verified_no_regression"
    else:
        verdict = "rejected"
    if len(isolation_values) != 1:
        raise RuntimeError("replay commands used inconsistent isolation")
    isolation = next(iter(isolation_values)) if isolation_values else (
        "sandbox-exec" if platform.system() == "Darwin" else "bubblewrap"
    )
    created_at = datetime.now(timezone.utc)
    unsigned = {
        "version": 1,
        "replay_id": replay_id,
        "task_id": task_id,
        "candidate_kind": candidate_kind,
        "repository_id": task.policy.repository_id,
        "source_revision": task.snapshot.revision,
        "parent_state_sha256": task.snapshot.state_sha256,
        "candidate_patch_sha256": candidate_patch_sha256,
        "candidate_patch_object_path": patch_object_path,
        "candidate_patch_bytes": patch_bytes,
        "contract_fingerprint": contract.fingerprint,
        "commands": [
            result.model_dump(mode="json", exclude_none=True) for result in results
        ],
        "verdict": verdict,
        "baseline_failed": baseline_failed,
        "candidate_passed": candidate_passed,
        "candidate_no_regression": candidate_no_regression,
        "candidate_improved": candidate_improved,
        "network_isolation": isolation,
        "created_at": created_at.isoformat(),
    }
    provisional = ReplayReport(**unsigned, evidence_sha256="0" * 64)
    report = provisional.model_copy(
        update={
            "evidence_sha256": hashlib.sha256(
                canonical_json(
                    provisional.model_dump(
                        mode="json",
                        exclude={"evidence_sha256"},
                    )
                )
            ).hexdigest()
        }
    )
    spool.record_replay(
        replay_id=replay_id,
        task_id=task_id,
        candidate_kind=candidate_kind,
        report=report.model_dump(mode="json", exclude_none=True),
        report_sha256=hashlib.sha256(
            canonical_json(report.model_dump(mode="json", exclude_none=True))
        ).hexdigest(),
        created_at=created_at,
    )
    return report
