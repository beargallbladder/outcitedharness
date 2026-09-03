#!/usr/bin/env python3
"""Register audited DesignWins train/validation/test lineage in the queue DB."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from harness.storage.db import Store
from harness.training.models import TextPair
from harness.training.queue import DatasetMember, DatasetVersionRegistry
from harness.training.split import Split


EXPECTED_COUNTS = {
    Split.TRAIN: 1101,
    Split.VALIDATION: 127,
    Split.TEST: 141,
}


def _event_id(pair: TextPair) -> str:
    digest = hashlib.sha256(
        f"{pair.pair_id}\0{pair.provenance.content_sha256}".encode("utf-8")
    ).hexdigest()
    return f"designwins-{digest[:32]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--dataset-version-id",
        default="designwins-text-v3-20260829",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = Store(args.database)
    members: list[DatasetMember] = []
    counts: dict[str, int] = {}
    with store.connect() as conn:
        for split in Split:
            path = args.dataset_root / f"{split.value}.jsonl"
            split_count = 0
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    pair = TextPair.model_validate_json(line)
                    event_id = _event_id(pair)
                    artifact = conn.execute(
                        """
                        SELECT artifact_id FROM learning_artifacts
                        WHERE event_id = ? AND kind = 'canonical_response'
                        """,
                        (event_id,),
                    ).fetchone()
                    if artifact is None:
                        raise ValueError(
                            f"missing ledger response for pair {pair.pair_id!r}"
                        )
                    members.append(
                        DatasetMember(
                            event_id=event_id,
                            artifact_id=artifact["artifact_id"],
                            split=split,
                            lineage_id=pair.provenance.lineage_id,
                            source_document_sha256=(
                                pair.provenance.content_sha256
                            ),
                        )
                    )
                    split_count += 1
            if split_count != EXPECTED_COUNTS[split]:
                raise ValueError(
                    f"{split.value} has {split_count} records, "
                    f"expected {EXPECTED_COUNTS[split]}"
                )
            counts[split.value] = split_count
    digest = DatasetVersionRegistry(store).create(
        dataset_version_id=args.dataset_version_id,
        name="designwins-text",
        version="v3-20260829",
        source_revision=args.source_revision,
        split_policy={
            "kind": "preassigned_audited",
            "leakage_keys": [
                "lineage_id",
                "source_document_sha256",
            ],
        },
        members=members,
    )
    print(
        json.dumps(
            {
                "dataset_version_id": args.dataset_version_id,
                "manifest_sha256": digest,
                "counts": counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
