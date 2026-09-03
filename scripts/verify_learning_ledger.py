#!/usr/bin/env python3
"""Verify every immutable learning-event and external artifact digest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from harness.storage.db import Store
from harness.training.ledger import LearningLedger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _write_once(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return
        raise FileExistsError(f"refusing to overwrite verification: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    store = Store(args.database)
    ledger = LearningLedger(store, args.artifact_root)
    with store.connect() as conn:
        event_rows = conn.execute(
            """
            SELECT event_id, event_sha256
            FROM learning_events ORDER BY event_id
            """
        ).fetchall()
        artifact_rows = conn.execute(
            """
            SELECT artifact_id, event_id, sha256, byte_size
            FROM learning_artifacts ORDER BY artifact_id
            """
        ).fetchall()
        verification_rows = conn.execute(
            """
            SELECT verification_id, event_id, status
            FROM learning_verifications ORDER BY verification_id
            """
        ).fetchall()
        admission_count = conn.execute(
            "SELECT COUNT(*) FROM learning_admissions"
        ).fetchone()[0]
    event_ids = [row["event_id"] for row in event_rows]
    for event_id in event_ids:
        ledger.verify_event(event_id)
    fingerprint_payload = {
        "events": [dict(row) for row in event_rows],
        "artifacts": [dict(row) for row in artifact_rows],
        "verifications": [dict(row) for row in verification_rows],
        "admission_count": admission_count,
    }
    result = {
        "schema": "harness.learning-ledger-verification.v1",
        "events_verified": len(event_ids),
        "artifacts_verified": len(artifact_rows),
        "verifications_checked": len(verification_rows),
        "admissions": admission_count,
        "ledger_fingerprint": hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "passed": True,
    }
    if args.output is not None:
        _write_once(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
