#!/usr/bin/env python3
"""Administer the durable learning-factory queue."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from harness.storage.db import Store
from harness.training.queue import JobState, PrioritySignals, TrainingQueue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)

    enqueue = commands.add_parser("enqueue")
    enqueue.add_argument("--job-id", required=True)
    enqueue.add_argument("--job-kind", required=True)
    enqueue.add_argument("--dataset-version-id", required=True)
    enqueue.add_argument("--observed-frequency", type=float, required=True)
    enqueue.add_argument("--frontier-cost", type=float, required=True)
    enqueue.add_argument("--local-failure-rate", type=float, required=True)
    enqueue.add_argument("--verification-strength", type=float, required=True)
    enqueue.add_argument("--diversity", type=float, required=True)
    enqueue.add_argument("--expected-gpu-hours", type=float, required=True)
    enqueue.add_argument("--max-attempts", type=int, default=3)
    enqueue.add_argument("--config", type=Path)

    claim = commands.add_parser("claim")
    claim.add_argument("--node", required=True)
    claim.add_argument("--lease-seconds", type=int, default=1800)
    claim.add_argument("--job-kind", action="append", default=[])
    claim.add_argument("--lease-output", type=Path)

    renew = commands.add_parser("renew")
    renew.add_argument("--job-id", required=True)
    renew.add_argument("--lease-file", required=True, type=Path)
    renew.add_argument("--lease-seconds", type=int, default=1800)

    complete = commands.add_parser("complete")
    complete.add_argument("--job-id", required=True)
    complete.add_argument("--lease-file", required=True, type=Path)
    complete.add_argument("--checkpoint-uri", required=True)
    complete.add_argument("--checkpoint-sha256", required=True)

    fail = commands.add_parser("fail")
    fail.add_argument("--job-id", required=True)
    fail.add_argument("--lease-file", required=True, type=Path)
    fail.add_argument("--error", required=True)
    fail.add_argument("--terminal", action="store_true")

    transition = commands.add_parser("transition")
    transition.add_argument("--job-id", required=True)
    transition.add_argument("--target", required=True, choices=[row.value for row in JobState])
    transition.add_argument("--expected", choices=[row.value for row in JobState])

    recover = commands.add_parser("recover")
    recover.set_defaults(command="recover")

    status = commands.add_parser("status")
    status.add_argument("--job-id")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    store = Store(args.database)
    queue = TrainingQueue(store)
    if args.command == "enqueue":
        config = (
            json.loads(args.config.read_text(encoding="utf-8"))
            if args.config
            else {}
        )
        queue.enqueue(
            job_id=args.job_id,
            job_kind=args.job_kind,
            dataset_version_id=args.dataset_version_id,
            signals=PrioritySignals(
                observed_frequency=args.observed_frequency,
                frontier_cost=args.frontier_cost,
                local_failure_rate=args.local_failure_rate,
                verification_strength=args.verification_strength,
                diversity=args.diversity,
                expected_gpu_hours=args.expected_gpu_hours,
            ),
            config=config,
            max_attempts=args.max_attempts,
        )
        print(json.dumps({"enqueued": args.job_id}, sort_keys=True))
    elif args.command == "claim":
        claimed = queue.claim(
            args.node,
            lease_seconds=args.lease_seconds,
            allowed_job_kinds=(
                frozenset(args.job_kind) if args.job_kind else None
            ),
        )
        if claimed is not None and args.lease_output is not None:
            descriptor = os.open(
                args.lease_output,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "job_id": claimed.job_id,
                        "node": claimed.assigned_node,
                        "attempt": claimed.attempt,
                        "lease_token": claimed.lease_token,
                    },
                    handle,
                    sort_keys=True,
                )
                handle.write("\n")
        public = asdict(claimed) if claimed is not None else {"status": "idle"}
        public.pop("lease_token", None)
        print(
            json.dumps(
                public,
                sort_keys=True,
            )
        )
    elif args.command == "renew":
        lease = _load_lease(args.lease_file, args.job_id)
        expires_at = queue.renew(
            args.job_id,
            lease["node"],
            lease["attempt"],
            lease["lease_token"],
            lease_seconds=args.lease_seconds,
        )
        print(
            json.dumps(
                {"job_id": args.job_id, "lease_expires_at": expires_at},
                sort_keys=True,
            )
        )
    elif args.command == "complete":
        lease = _load_lease(args.lease_file, args.job_id)
        queue.complete(
            args.job_id,
            node=lease["node"],
            attempt=lease["attempt"],
            lease_token=lease["lease_token"],
            checkpoint_uri=args.checkpoint_uri,
            checkpoint_sha256=args.checkpoint_sha256,
        )
        print(json.dumps({"job_id": args.job_id, "state": "trained"}))
    elif args.command == "fail":
        lease = _load_lease(args.lease_file, args.job_id)
        state = queue.fail(
            args.job_id,
            args.error,
            node=lease["node"],
            attempt=lease["attempt"],
            lease_token=lease["lease_token"],
            terminal=args.terminal,
        )
        print(json.dumps({"job_id": args.job_id, "state": state.value}))
    elif args.command == "transition":
        queue.transition(
            args.job_id,
            JobState(args.target),
            expected=JobState(args.expected) if args.expected else None,
        )
        print(json.dumps({"job_id": args.job_id, "state": args.target}, sort_keys=True))
    elif args.command == "recover":
        print(json.dumps({"recovered": queue.recover_expired()}, sort_keys=True))
    else:
        with store.connect() as conn:
            if args.job_id:
                rows = conn.execute(
                    "SELECT * FROM training_jobs WHERE job_id = ?",
                    (args.job_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM training_jobs
                    ORDER BY created_at, job_id
                    """
                ).fetchall()
        print(
            json.dumps(
                [
                    {
                        key: value
                        for key, value in dict(row).items()
                        if key != "lease_token"
                    }
                    for row in rows
                ],
                indent=2,
                sort_keys=True,
            )
        )
    return 0


def _load_lease(path: Path, job_id: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("lease file must be a regular file")
    if path.stat().st_mode & 0o077:
        raise ValueError("lease file permissions are too broad")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("job_id") != job_id:
        raise ValueError("lease file belongs to a different job")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
