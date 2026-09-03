#!/usr/bin/env python3
"""Build a diverse, GT-leveraged priority queue for local extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from harness.electronics.claims import canonical_json
from harness.electronics.corpus import canonical_vendor, sha256_file, verify_corpus_registry


SCHEMA = "harness.electronics-prioritized-local-work.v1"
CAPABILITY_SCORE = {
    "pin_semantics": 5.0,
    "pin_or_ball": 4.5,
    "parametrics": 4.0,
    "opn_decoder": 3.0,
    "series_summary": 2.0,
}


def _write_new(path: Path, value: bytes) -> None:
    destination = path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"immutable output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic-bundle", type=Path, required=True)
    parser.add_argument("--corpus-registry", type=Path, required=True)
    parser.add_argument("--maximum-work", type=int, default=5000)
    parser.add_argument("--pages-per-document-capability", type=int, default=24)
    parser.add_argument(
        "--capability",
        action="append",
        choices=tuple(sorted(CAPABILITY_SCORE)),
        help="Capability to include; repeat as needed. Defaults to all.",
    )
    parser.add_argument(
        "--partition",
        choices=("factory_candidate", "frozen_evaluation"),
        default="factory_candidate",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.maximum_work <= 100_000:
        raise ValueError("--maximum-work must be within 1..100000")
    if not 1 <= args.pages_per_document_capability <= 64:
        raise ValueError("--pages-per-document-capability must be within 1..64")
    selected_capabilities = set(args.capability or CAPABILITY_SCORE)

    registry_path = args.corpus_registry.expanduser().resolve(strict=True)
    registry = json.loads(registry_path.read_text())
    verify_corpus_registry(registry)
    documents = {
        row["document_sha256"]: row for row in registry["documents"]
    }
    bundle = args.deterministic_bundle.expanduser().resolve(strict=True)
    manifest = json.loads((bundle / "manifest.json").read_text())
    queue_path = bundle / "local-model-queue.jsonl"
    receipt = manifest["artifacts"]["local-model-queue.jsonl"]
    if (
        sha256_file(queue_path) != receipt["sha256"]
        or queue_path.stat().st_size != receipt["bytes"]
    ):
        raise ValueError("local work queue differs from manifest")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with queue_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row["partition"] != args.partition
                or row["capability"] not in selected_capabilities
            ):
                continue
            grouped[(row["document_sha256"], row["capability"])].append(row)

    candidates: list[dict[str, Any]] = []
    collapsed_physical = 0
    for (document_sha, capability), rows in sorted(grouped.items()):
        if capability == "pin_or_ball" and (
            document_sha,
            "pin_semantics",
        ) in grouped:
            semantic_pages = {
                int(row["page_1based"])
                for row in grouped[(document_sha, "pin_semantics")]
            }
            before = len(rows)
            rows = [
                row
                for row in rows
                if int(row["page_1based"]) not in semantic_pages
            ]
            collapsed_physical += before - len(rows)
        rows = sorted(
            rows,
            key=lambda row: (
                int(row["page_1based"]),
                row["work_id"],
            ),
        )[: args.pages_per_document_capability]
        document = documents[document_sha]
        has_gt = bool(document.get("ground_truth"))
        has_published = bool(document.get("published_pinouts"))
        asset_memberships = document.get("asset_memberships") or {}
        exact_opns = sorted(
            {
                str(value).strip()
                for value in asset_memberships.get("manifested_opn", [])
                if str(value).strip()
            }
        )
        raw_vendors = [
            record["vendor"]
            for record in document.get("ground_truth") or []
            if record.get("vendor")
        ] or list(document.get("vendors") or ["unknown"])
        vendors = sorted(
            {
                canonical_vendor(value) or "unknown"
                for value in raw_vendors
            }
        )
        vendor = vendors[0]
        for row in rows:
            score = (
                CAPABILITY_SCORE[capability]
                + (5.0 if has_gt else 0.0)
                + (3.0 if has_published else 0.0)
                + (
                    1.0
                    if row["deterministic_result"]["pin_rows"]
                    or row["deterministic_result"]["parametric_rows"]
                    else 0.0
                )
            )
            candidates.append(
                {
                    **row,
                    "vendor": vendor,
                    "has_ground_truth": has_gt,
                    "has_published_pinout": has_published,
                    "exact_opns": exact_opns,
                    "priority_score": score,
                    "priority_basis": {
                        "capability": CAPABILITY_SCORE[capability],
                        "ground_truth": 5.0 if has_gt else 0.0,
                        "published_pinout": 3.0 if has_published else 0.0,
                        "partial_deterministic_parse": (
                            1.0
                            if row["deterministic_result"]["pin_rows"]
                            or row["deterministic_result"]["parametric_rows"]
                            else 0.0
                        ),
                    },
                }
            )

    by_vendor: dict[str, deque[dict[str, Any]]] = {}
    for vendor in sorted({row["vendor"] for row in candidates}):
        rows = [row for row in candidates if row["vendor"] == vendor]
        rows.sort(
            key=lambda row: (
                -row["priority_score"],
                hashlib.sha256(
                    f"local-priority-v1:{row['work_id']}".encode()
                ).hexdigest(),
            )
        )
        by_vendor[vendor] = deque(rows)
    selected: list[dict[str, Any]] = []
    while by_vendor and len(selected) < args.maximum_work:
        for vendor in list(sorted(by_vendor)):
            queue = by_vendor[vendor]
            if queue:
                selected.append(queue.popleft())
                if len(selected) == args.maximum_work:
                    break
            if not queue:
                del by_vendor[vendor]
    for rank, row in enumerate(selected, 1):
        row["priority_rank"] = rank

    core = {
        "schema": SCHEMA,
        "policy": {
            "partition": args.partition,
            "capabilities": sorted(selected_capabilities),
            "maximum_work": args.maximum_work,
            "pages_per_document_capability": args.pages_per_document_capability,
            "vendor_round_robin": True,
            "ground_truth_leverage": True,
            "pin_semantics_supersedes_duplicate_pin_identity_call": True,
            "frontier_batch_eligible": False,
        },
        "sources": {
            "deterministic_evidence_sha256": manifest["evidence_sha256"],
            "corpus_evidence_sha256": registry["evidence_sha256"],
        },
        "counts": {
            "source_work": manifest["counts"]["local_model_work"],
            "post_dedup_candidates": len(candidates),
            "selected": len(selected),
            "collapsed_duplicate_pin_identity_work": collapsed_physical,
            "vendors": dict(
                sorted(Counter(row["vendor"] for row in selected).items())
            ),
            "capabilities": dict(
                sorted(Counter(row["capability"] for row in selected).items())
            ),
            "with_ground_truth": sum(row["has_ground_truth"] for row in selected),
            "with_published_pinout": sum(
                row["has_published_pinout"] for row in selected
            ),
        },
        "work": selected,
    }
    core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
    payload = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _write_new(args.output, payload)
    print(json.dumps(core["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
