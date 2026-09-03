#!/usr/bin/env python3
"""Claim and execute resumable datasheet extraction chunks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from harness.electronics.factory_control import (
    ChunkLease,
    ElectronicsFactoryState,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--page-evidence", type=Path, required=True)
    parser.add_argument("--render-dpi", type=int, default=220)
    parser.add_argument("--request-timeout-seconds", type=float, default=900)
    parser.add_argument("--lease-seconds", type=int, default=1800)
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument(
        "--vision-policy",
        choices=("fallback", "always"),
        default="always",
    )
    return parser


def _command(args: argparse.Namespace, lease: ChunkLease) -> list[str]:
    repository = Path(__file__).resolve().parents[1]
    return [
        sys.executable,
        str(repository / "scripts" / "run_datasheet_structural_extraction.py"),
        "--structural-queue",
        str(lease.queue_path),
        "--page-evidence",
        str(args.page_evidence.expanduser().resolve(strict=True)),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--offset",
        str(lease.offset),
        "--limit",
        str(lease.item_count),
        "--render-dpi",
        str(args.render_dpi),
        "--timeout-seconds",
        str(args.request_timeout_seconds),
        "--vision-policy",
        args.vision_policy,
        "--output-directory",
        str(lease.output_directory),
    ]


def _run_one(
    state: ElectronicsFactoryState,
    args: argparse.Namespace,
    lease: ChunkLease,
) -> dict[str, object]:
    if lease.output_directory.exists():
        manifest_sha = state.complete_chunk(lease)
        return {
            "chunk_id": lease.chunk_id,
            "status": "adopted_completed_output",
            "manifest_sha256": manifest_sha,
        }

    logs = state.root / "logs"
    logs.mkdir(mode=0o700, exist_ok=True)
    os.chmod(logs, 0o700)
    log_path = logs / (
        f"{lease.chunk_id}-attempt-{lease.attempt:02d}-{args.node}.log"
    )
    descriptor = os.open(
        log_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    command = _command(args, lease)
    with os.fdopen(descriptor, "wb") as log:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        heartbeat = max(10, args.lease_seconds // 3)
        next_heartbeat = time.monotonic() + heartbeat
        try:
            while True:
                return_code = process.poll()
                if return_code is not None:
                    break
                if time.monotonic() >= next_heartbeat:
                    state.renew_chunk(
                        lease,
                        lease_seconds=args.lease_seconds,
                    )
                    next_heartbeat = time.monotonic() + heartbeat
                time.sleep(1)
            return_code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            state.fail_chunk(
                lease,
                "worker interrupted while extraction process was active",
            )
            raise
        finally:
            log.flush()
            os.fsync(log.fileno())

    if return_code != 0:
        status = state.fail_chunk(
            lease,
            f"extraction process exited with code {return_code}; log={log_path}",
        )
        return {
            "chunk_id": lease.chunk_id,
            "status": status,
            "return_code": return_code,
            "log_path": str(log_path),
        }
    manifest_sha = state.complete_chunk(lease)
    return {
        "chunk_id": lease.chunk_id,
        "status": "completed",
        "manifest_sha256": manifest_sha,
        "log_path": str(log_path),
    }


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.max_jobs <= 10_000:
        raise ValueError("--max-jobs must be within 1..10000")
    state = ElectronicsFactoryState(args.state_root)
    outcomes: list[dict[str, object]] = []
    for _ in range(args.max_jobs):
        lease = state.claim_chunk(
            args.node,
            lease_seconds=args.lease_seconds,
        )
        if lease is None:
            break
        outcomes.append(_run_one(state, args, lease))
    print(
        json.dumps(
            {
                "node": args.node,
                "jobs": len(outcomes),
                "outcomes": outcomes,
                "status": "idle" if not outcomes else "completed_claims",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all(
        outcome["status"] in {"completed", "adopted_completed_output"}
        for outcome in outcomes
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
