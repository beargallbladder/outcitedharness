#!/usr/bin/env python3
"""Administer continuous datasheet intake and extraction state."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from harness.electronics.factory_control import ElectronicsFactoryState


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    discover = commands.add_parser("discover")
    discover.add_argument(
        "--download-root",
        action="append",
        type=Path,
        required=True,
    )
    discover.add_argument("--stability-seconds", type=int, default=120)
    discover.add_argument("--snapshot-output", type=Path)

    seed = commands.add_parser("seed-corpus")
    seed.add_argument("--corpus-registry", type=Path, required=True)

    enqueue = commands.add_parser("enqueue")
    enqueue.add_argument("--structural-queue", type=Path, required=True)
    enqueue.add_argument("--output-root", type=Path, required=True)
    enqueue.add_argument("--chunk-size", type=int, default=10)
    enqueue.add_argument("--max-attempts", type=int, default=3)
    enqueue.add_argument("--start-offset", type=int, default=0)

    claim = commands.add_parser("claim")
    claim.add_argument("--node", required=True)
    claim.add_argument("--lease-seconds", type=int, default=1800)
    claim.add_argument("--lease-output", type=Path)

    recover = commands.add_parser("recover")
    recover.set_defaults(command="recover")

    snapshot = commands.add_parser("snapshot-ready-sources")
    snapshot.add_argument("--output", type=Path, required=True)

    next_cohort = commands.add_parser("seal-next-source-cohort")
    next_cohort.add_argument("--cohort-id", required=True)
    next_cohort.add_argument("--output", type=Path, required=True)
    next_cohort.add_argument("--maximum-documents", type=int, default=5000)

    frontier = commands.add_parser("register-frontier")
    frontier.add_argument("--run-id", required=True)
    frontier.add_argument("--prepared-bundle", type=Path, required=True)
    frontier.add_argument("--submission-state", type=Path, required=True)
    frontier.add_argument("--lifecycle-root", type=Path, required=True)

    commands.add_parser("status")
    return parser


def _write_lease(path: Path, value: dict[str, object]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"lease output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = _parser().parse_args()
    state = ElectronicsFactoryState(args.state_root)
    if args.command == "discover":
        report = asdict(
            state.discover_pdfs(
                args.download_root,
                stability_seconds=args.stability_seconds,
            )
        )
        if args.snapshot_output is not None:
            snapshot = state.seal_ready_source_snapshot(args.snapshot_output)
            report["snapshot"] = {
                "path": str(args.snapshot_output.expanduser().resolve()),
                "evidence_sha256": snapshot["evidence_sha256"],
                "documents": snapshot["counts"]["documents"],
            }
        value: object = report
    elif args.command == "seed-corpus":
        value = state.seed_corpus_registry(args.corpus_registry)
    elif args.command == "enqueue":
        chunks = state.register_queue(
            args.structural_queue,
            args.output_root,
            chunk_size=args.chunk_size,
            max_attempts=args.max_attempts,
            start_offset=args.start_offset,
        )
        value = {"registered": len(chunks), "chunk_ids": chunks}
    elif args.command == "claim":
        lease = state.claim_chunk(
            args.node,
            lease_seconds=args.lease_seconds,
        )
        if lease is None:
            value = {"status": "idle"}
        else:
            private = {
                **asdict(lease),
                "queue_path": str(lease.queue_path),
                "output_directory": str(lease.output_directory),
            }
            if args.lease_output is not None:
                _write_lease(args.lease_output, private)
            private.pop("lease_token")
            value = private
    elif args.command == "recover":
        value = {"recovered": state.recover_expired()}
    elif args.command == "snapshot-ready-sources":
        value = state.seal_ready_source_snapshot(args.output)
    elif args.command == "seal-next-source-cohort":
        value = state.seal_unassigned_source_snapshot(
            args.output,
            cohort_id=args.cohort_id,
            maximum_documents=args.maximum_documents,
        ) or {"status": "idle", "reason": "no_unassigned_ready_sources"}
    elif args.command == "register-frontier":
        created = state.register_frontier_run(
            run_id=args.run_id,
            prepared_bundle=args.prepared_bundle,
            submission_state=args.submission_state,
            lifecycle_root=args.lifecycle_root,
        )
        value = {"run_id": args.run_id, "registered": created}
    else:
        value = state.status()
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
