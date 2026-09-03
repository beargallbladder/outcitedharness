#!/usr/bin/env python3
"""Generate and admit test-killed repair tasks from an owned repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.repo_contract import VerificationCommand, build_repo_contract
from harness.shadow.models import RepositorySnapshot, canonical_json
from harness.shadow.objects import ShadowObjectStore
from harness.shadow.policy import load_policy
from harness.shadow.repository import capture_repository_snapshot
from harness.shadow.replay import _execute
from harness.shadow.workspace import materialize_snapshot
from harness.storage.db import Store
from harness.training.code_curriculum import (
    ADMISSION_POLICY,
    SCHEMA,
    VerificationReceipt,
    apply_mutation,
    capture_verified_task,
    enumerate_mutations,
    make_verified_task,
)
from harness.training.ledger import LearningLedger
from harness.training.security import redact_text


RUN_SCHEMA = "harness.owned-code-curriculum-run.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"curriculum report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o444)
    finally:
        temporary.unlink(missing_ok=True)


def _select_command(
    commands: list[VerificationCommand],
    name: str,
) -> VerificationCommand:
    selected = [command for command in commands if command.name == name]
    if len(selected) != 1:
        available = sorted(command.name for command in commands)
        raise ValueError(
            f"repository requires exactly one {name!r} verifier; available={available}"
        )
    return selected[0]


def _round_robin(points: list[Any], limit: int) -> list[Any]:
    groups: dict[str, deque[Any]] = defaultdict(deque)
    for point in points:
        groups[point.path].append(point)
    selected = []
    while groups and len(selected) < limit:
        for path in sorted(tuple(groups)):
            selected.append(groups[path].popleft())
            if not groups[path]:
                del groups[path]
            if len(selected) >= limit:
                break
    return selected


def _source_datetime(repository: Path, revision: str) -> datetime:
    result = subprocess.run(
        ["git", "-C", str(repository), "show", "-s", "--format=%cI", revision],
        text=True,
        capture_output=True,
        check=True,
    )
    return datetime.fromisoformat(result.stdout.strip())


def _clean_head_snapshot(
    repository: Path,
    repository_id: str,
) -> RepositorySnapshot:
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    empty_digest = _sha256(b"")
    state = {
        "repository_id": repository_id,
        "revision": revision,
        "dirty_patch_sha256": empty_digest,
        "untracked_files": [],
        "omitted_path_count": 0,
    }
    return RepositorySnapshot(
        repository_id=repository_id,
        repository_root=repository,
        revision=revision,
        dirty_patch="",
        dirty_patch_sha256=empty_digest,
        untracked_files=(),
        omitted_path_count=0,
        state_sha256=_sha256(canonical_json(state)),
        captured_at=_source_datetime(repository, revision),
    )


def _clean_tail(value: str, workspace: Path, source_root: Path) -> str:
    return redact_text(value).replace(str(workspace), "<WORKSPACE>").replace(
        str(source_root),
        "<REPOSITORY>",
    )


def build(
    *,
    repository: Path,
    database: Path,
    artifact_root: Path,
    output: Path,
    work_root: Path,
    command_name: str,
    max_mutations: int,
    target_verified: int,
    workers: int,
    per_file_limit: int,
    include_roots: tuple[str, ...],
    source_state: str = "head",
    mutation_offset: int = 0,
) -> dict[str, Any]:
    repository = repository.expanduser().resolve(strict=True)
    policy = load_policy(repository)
    if policy is None or not policy.enabled or policy.owner != "self":
        raise ValueError("repository requires an enabled self-owned shadow policy")
    if max_mutations < 1 or target_verified < 1:
        raise ValueError("mutation and verified-task limits must be positive")
    if target_verified > max_mutations:
        raise ValueError("target verified tasks cannot exceed attempted mutations")
    if mutation_offset < 0:
        raise ValueError("mutation offset cannot be negative")
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between one and 32")

    spool_root = work_root.expanduser().resolve() / "objects"
    workspaces = work_root.expanduser().resolve() / "workspaces"
    object_store = ShadowObjectStore(spool_root)
    if source_state == "head":
        snapshot = _clean_head_snapshot(repository, policy.repository_id)
    elif source_state == "live":
        snapshot = capture_repository_snapshot(repository, policy, object_store)
    else:
        raise ValueError("source_state must be 'head' or 'live'")
    source_time = _source_datetime(repository, snapshot.revision)
    baseline_root = materialize_snapshot(
        snapshot,
        policy,
        work_root=workspaces,
        object_store=object_store,
        workspace_id=f"curriculum-baseline-{snapshot.state_sha256[:20]}",
    )
    try:
        contract = build_repo_contract(baseline_root)
        if contract is None:
            raise ValueError("repository has no verification contract")
        command = _select_command(contract.commands, command_name)
        baseline, isolation = _execute(
            command,
            phase="candidate",
            workspace=baseline_root,
            source_root=repository,
        )
        if baseline.timed_out or baseline.returncode != 0:
            raise ValueError(
                "canonical repository does not pass the selected verifier: "
                + (baseline.stderr_tail or baseline.stdout_tail)[-2000:]
            )
        all_points = enumerate_mutations(
            baseline_root,
            policy,
            include_roots=include_roots,
            per_file_limit=per_file_limit,
        )
        selected = _round_robin(
            all_points,
            mutation_offset + max_mutations,
        )[mutation_offset:]
        if not selected:
            raise ValueError("no safe mutation points were found")

        def validate(spec: Any) -> tuple[Any | None, str]:
            workspace_id = f"curriculum-{spec.mutation_id.removeprefix('mutation-')}"
            workspace = materialize_snapshot(
                snapshot,
                policy,
                work_root=workspaces,
                object_store=object_store,
                workspace_id=workspace_id,
            )
            try:
                path = workspace / spec.path
                source = path.read_text(encoding="utf-8")
                if hashlib.sha256(source.encode()).hexdigest() != spec.source_sha256:
                    return None, "source_drift"
                mutant = apply_mutation(source, spec)
                temporary = path.with_name(f".{path.name}.{spec.mutation_id}.tmp")
                temporary.write_text(mutant, encoding="utf-8")
                os.replace(temporary, path)
                result, result_isolation = _execute(
                    command,
                    phase="parent",
                    workspace=workspace,
                    source_root=repository,
                )
                if result.timed_out:
                    return None, "timeout"
                if result.returncode in {None, 0}:
                    return None, "survived_tests"
                receipt = VerificationReceipt(
                    command_name=command.name,
                    command=command.command,
                    baseline_returncode=int(baseline.returncode),
                    baseline_stdout_sha256=baseline.stdout_sha256,
                    baseline_stderr_sha256=baseline.stderr_sha256,
                    mutant_returncode=int(result.returncode),
                    mutant_stdout_sha256=result.stdout_sha256,
                    mutant_stderr_sha256=result.stderr_sha256,
                    mutant_stdout_tail=_clean_tail(
                        result.stdout_tail,
                        workspace,
                        repository,
                    ),
                    mutant_stderr_tail=_clean_tail(
                        result.stderr_tail,
                        workspace,
                        repository,
                    ),
                    network_isolation=result_isolation,
                )
                task = make_verified_task(
                    snapshot=snapshot,
                    policy=policy,
                    spec=spec,
                    mutant_source=mutant,
                    verification=receipt,
                    created_at=source_time,
                )
                return task, "verified"
            except Exception as exc:
                return None, f"error:{type(exc).__name__}:{redact_text(str(exc))[:300]}"
            finally:
                shutil.rmtree(workspace, ignore_errors=True)

        validated = []
        reasons: Counter[str] = Counter()
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(validate, spec): spec for spec in selected}
            for future in as_completed(futures):
                task, reason = future.result()
                completed += 1
                reasons[reason] += 1
                if task is not None and len(validated) < target_verified:
                    validated.append(task)
                if completed % 10 == 0 or completed == len(selected):
                    print(
                        json.dumps(
                            {
                                "attempted": completed,
                                "selected": len(selected),
                                "verified": len(validated),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

        if not validated:
            raise ValueError("no mutation was killed by the selected verifier")
        ledger = LearningLedger(Store(database), artifact_root)
        event_ids = [
            capture_verified_task(task, policy=policy, ledger=ledger)
            for task in sorted(validated, key=lambda row: row.task_id)
        ]
        records = [
            {
                "task_id": task.task_id,
                "event_id": event_id,
                "lineage_id": task.lineage_id,
                "path": task.mutation.path,
                "operator": task.mutation.operator,
                "source_sha256": task.mutation.source_sha256,
                "mutant_sha256": task.mutation.mutant_sha256,
            }
            for task, event_id in zip(
                sorted(validated, key=lambda row: row.task_id),
                event_ids,
                strict=True,
            )
        ]
        core: dict[str, Any] = {
            "schema": RUN_SCHEMA,
            "curriculum_schema": SCHEMA,
            "admission_policy": ADMISSION_POLICY,
            "repository_id": policy.repository_id,
            "repository_path": str(repository),
            "parent_revision": snapshot.revision,
            "parent_state_sha256": snapshot.state_sha256,
            "source_state": source_state,
            "policy_sha256": hashlib.sha256(canonical_json(policy)).hexdigest(),
            "contract_fingerprint": contract.fingerprint,
            "command": command.command,
            "network_isolation": isolation,
            "available_mutations": len(all_points),
            "attempted_mutations": len(selected),
            "mutation_offset": mutation_offset,
            "verified_tasks": len(records),
            "rejections": dict(sorted(reasons.items())),
            "records": records,
        }
        core["evidence_sha256"] = _sha256(_canonical(core))
        _write_once(output, core)
        return core
    finally:
        shutil.rmtree(baseline_root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("~/.harness/code-curriculum"),
    )
    parser.add_argument("--command", default="unit")
    parser.add_argument("--max-mutations", type=int, default=1000)
    parser.add_argument("--mutation-offset", type=int, default=0)
    parser.add_argument("--target-verified", type=int, default=500)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--per-file-limit", type=int, default=24)
    parser.add_argument(
        "--source-state",
        choices=("head", "live"),
        default="head",
        help="Use immutable HEAD by default; live includes approved dirty files.",
    )
    parser.add_argument(
        "--include-root",
        action="append",
        default=[],
        help="Repository-relative Python source root; repeatable.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    result = build(
        repository=arguments.repository,
        database=arguments.database,
        artifact_root=arguments.artifact_root,
        output=arguments.output,
        work_root=arguments.work_root,
        command_name=arguments.command,
        max_mutations=arguments.max_mutations,
        target_verified=arguments.target_verified,
        workers=arguments.workers,
        per_file_limit=arguments.per_file_limit,
        include_roots=tuple(arguments.include_root or ("harness", "scripts")),
        source_state=arguments.source_state,
        mutation_offset=arguments.mutation_offset,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "repository_id",
                    "available_mutations",
                    "attempted_mutations",
                    "verified_tasks",
                    "evidence_sha256",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
