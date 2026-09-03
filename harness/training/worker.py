from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from harness.storage.db import Store
from harness.training.queue import (
    ClaimedJob,
    InvalidTransitionError,
    JobState,
    TrainingQueue,
)


class HandlerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_kind: str = Field(min_length=1)
    argv: tuple[str, ...]
    working_directory: Path
    allowed_nodes: frozenset[str]
    environment_allowlist: frozenset[str] = frozenset()
    timeout_seconds: int = Field(default=86_400, ge=60, le=604_800)

    @field_validator("argv")
    @classmethod
    def command_is_direct_and_absolute(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or not Path(value[0]).is_absolute():
            raise ValueError("handler executable must be an absolute path")
        if Path(value[0]).name in {
            "bash",
            "dash",
            "env",
            "fish",
            "sh",
            "sudo",
            "zsh",
        }:
            raise ValueError("handler must be a direct executable, not a shell")
        return value


@dataclass(frozen=True)
class WorkerResult:
    status: str
    job_id: str | None = None
    attempt: int | None = None
    returncode: int | None = None
    log_path: Path | None = None


def load_handlers(path: Path) -> dict[str, HandlerSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = raw.get("handlers") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise ValueError("worker config requires a handlers list")
    handlers = {
        handler.job_kind: handler
        for handler in (HandlerSpec.model_validate(row) for row in rows)
    }
    if len(handlers) != len(rows):
        raise ValueError("worker config contains duplicate job kinds")
    return handlers


class TrainingWorker:
    def __init__(
        self,
        store: Store,
        *,
        node: str,
        handlers: dict[str, HandlerSpec],
        log_root: Path,
        lease_seconds: int = 1800,
        heartbeat_seconds: int = 60,
    ):
        if heartbeat_seconds < 1 or heartbeat_seconds >= lease_seconds:
            raise ValueError("heartbeat must be positive and shorter than the lease")
        self.queue = TrainingQueue(store)
        self.node = node
        self.handlers = {
            kind: handler
            for kind, handler in handlers.items()
            if node in handler.allowed_nodes
        }
        self.log_root = log_root.expanduser().resolve()
        self.log_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.log_root, 0o700)
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds

    def run_once(self) -> WorkerResult:
        self._reap_expired_handlers()
        claimed = self.queue.claim(
            self.node,
            lease_seconds=self.lease_seconds,
            allowed_job_kinds=frozenset(self.handlers),
        )
        if claimed is None:
            return WorkerResult(status="idle")
        handler = self.handlers[claimed.job_kind]
        raw_executable = Path(handler.argv[0])
        if raw_executable.is_symlink():
            return self._failed(claimed, "invalid executable")
        try:
            executable = raw_executable.resolve(strict=True)
            workdir = handler.working_directory.expanduser().resolve(strict=True)
        except FileNotFoundError:
            return self._failed(claimed, "handler path does not exist")
        if not executable.is_file():
            return self._failed(claimed, "invalid executable")
        if not os.access(executable, os.X_OK):
            return self._failed(claimed, "handler executable is not executable")

        identity = hashlib.sha256(
            f"{claimed.job_id}\0{claimed.attempt}".encode()
        ).hexdigest()[:16]
        prefix = f"{identity}-attempt-{claimed.attempt}"
        payload_path = self.log_root / f"{prefix}.json"
        log_path = self.log_root / f"{prefix}.log"
        result_path = self.log_root / f"{prefix}.result.json"
        public_job = asdict(claimed)
        public_job.pop("lease_token")
        _write_private_json(
            payload_path,
            {
                "schema": "harness.training-worker-payload.v1",
                "job": public_job,
            },
        )
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "HOME",
                "PATH",
                "LANG",
                "LC_ALL",
                "CUDA_VISIBLE_DEVICES",
                *handler.environment_allowlist,
            }
        }
        environment.update(
            {
                "HARNESS_TRAINING_JOB_ID": claimed.job_id,
                "HARNESS_TRAINING_ATTEMPT": str(claimed.attempt),
                "HARNESS_TRAINING_JOB_PAYLOAD": str(payload_path),
                "HARNESS_TRAINING_RESULT": str(result_path),
            }
        )
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            os.chmod(log_path, 0o600)
            process = subprocess.Popen(
                list(handler.argv),
                cwd=workdir,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            handler_pgid = process.pid
            try:
                self.queue.attach_handler(
                    claimed.job_id,
                    node=self.node,
                    attempt=claimed.attempt,
                    lease_token=claimed.lease_token,
                    pid=process.pid,
                    pgid=handler_pgid,
                )
            except InvalidTransitionError:
                _terminate_process_group(process)
                return WorkerResult(
                    status="lease_lost",
                    job_id=claimed.job_id,
                    attempt=claimed.attempt,
                    returncode=process.returncode,
                    log_path=log_path,
                )
            while True:
                try:
                    returncode = process.wait(timeout=self.heartbeat_seconds)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started > handler.timeout_seconds:
                        _terminate_process_group(process)
                        return self._failed(
                            claimed,
                            "handler timed out",
                            log_path=log_path,
                            returncode=process.returncode,
                        )
                    try:
                        self.queue.renew(
                            claimed.job_id,
                            self.node,
                            claimed.attempt,
                            claimed.lease_token,
                            lease_seconds=self.lease_seconds,
                        )
                    except InvalidTransitionError:
                        _terminate_process_group(process)
                        return WorkerResult(
                            status="lease_lost",
                            job_id=claimed.job_id,
                            attempt=claimed.attempt,
                            returncode=process.returncode,
                            log_path=log_path,
                        )
        try:
            self.queue.detach_handler(
                claimed.job_id,
                node=self.node,
                attempt=claimed.attempt,
                lease_token=claimed.lease_token,
                pid=process.pid,
                pgid=handler_pgid,
            )
        except InvalidTransitionError:
            return WorkerResult(
                status="lease_lost",
                job_id=claimed.job_id,
                attempt=claimed.attempt,
                returncode=returncode,
                log_path=log_path,
            )
        if returncode == 0:
            try:
                result = _read_handler_result(result_path)
                self.queue.complete(
                    claimed.job_id,
                    node=self.node,
                    attempt=claimed.attempt,
                    lease_token=claimed.lease_token,
                    checkpoint_uri=result["checkpoint_uri"],
                    checkpoint_sha256=result["checkpoint_sha256"],
                )
            except (InvalidTransitionError, OSError, ValueError) as exc:
                if isinstance(exc, InvalidTransitionError):
                    return WorkerResult(
                        status="lease_lost",
                        job_id=claimed.job_id,
                        attempt=claimed.attempt,
                        returncode=returncode,
                        log_path=log_path,
                    )
                return self._failed(
                    claimed,
                    f"invalid handler result: {exc}",
                    log_path=log_path,
                    returncode=returncode,
                )
            return WorkerResult(
                status="trained",
                job_id=claimed.job_id,
                attempt=claimed.attempt,
                returncode=returncode,
                log_path=log_path,
            )
        return self._failed(
            claimed,
            f"handler exited {returncode}",
            log_path=log_path,
            returncode=returncode,
        )

    def _reap_expired_handlers(self) -> None:
        for handler in self.queue.expired_handlers(self.node):
            if _process_group_exists(handler.handler_pgid):
                if not _handler_identity_matches(
                    handler.handler_pid,
                    handler.job_id,
                    handler.attempt,
                ):
                    continue
                _terminate_process_group_id(handler.handler_pgid)
            try:
                self.queue.release_expired_handler(handler)
            except InvalidTransitionError:
                continue

    def _failed(
        self,
        claimed: ClaimedJob,
        error: str,
        *,
        log_path: Path | None = None,
        returncode: int | None = None,
    ) -> WorkerResult:
        try:
            state = self.queue.fail(
                claimed.job_id,
                error,
                node=self.node,
                attempt=claimed.attempt,
                lease_token=claimed.lease_token,
            )
        except InvalidTransitionError:
            return WorkerResult(
                status="lease_lost",
                job_id=claimed.job_id,
                attempt=claimed.attempt,
                returncode=returncode,
                log_path=log_path,
            )
        return WorkerResult(
            status="retry" if state is JobState.ELIGIBLE else "rejected",
            job_id=claimed.job_id,
            attempt=claimed.attempt,
            returncode=returncode,
            log_path=log_path,
        )


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_handler_result(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("handler did not write a regular result file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("handler result must be a JSON object")
    checkpoint_uri = value.get("checkpoint_uri")
    checkpoint_sha256 = value.get("checkpoint_sha256")
    if not isinstance(checkpoint_uri, str) or not isinstance(
        checkpoint_sha256, str
    ):
        raise ValueError("handler result requires checkpoint URI and SHA-256")
    return {
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": checkpoint_sha256,
    }


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _handler_identity_matches(pid: int, job_id: str, attempt: int) -> bool:
    """Verify Linux process environment before signaling a persisted PID."""

    environ = Path(f"/proc/{pid}/environ")
    try:
        values = set(environ.read_bytes().split(b"\0"))
    except OSError:
        return False
    return {
        f"HARNESS_TRAINING_JOB_ID={job_id}".encode(),
        f"HARNESS_TRAINING_ATTEMPT={attempt}".encode(),
    }.issubset(values)


def _terminate_process_group_id(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not _process_group_exists(pgid):
            return
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
