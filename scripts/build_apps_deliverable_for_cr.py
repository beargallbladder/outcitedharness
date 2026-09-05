#!/usr/bin/env python3
"""Assemble the per-node applications deliverable for CR.

Joins CR's staged apps manifest (node_ids, mpns per document) with the
teacher-verified summary claims from the apps run. Emits one JSONL row per
(document, node_id) in the schema promised in the 2026-09-05 reply mail:

  {"node_id", "document_sha256", "mpns", "status": "extracted" | "absent",
   "page_1based", "applications": [...]}

`absent` rows carry an explicit reason. Only verbatim-grounded
`summary.application` claims are emitted; characteristics are not part of
this contract.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    # document_sha256 -> [(application_text, page_1based)]
    applications: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for line in args.claims.read_text(encoding="utf-8").splitlines():
        claim = json.loads(line)
        if claim["field"] != "summary.application":
            continue
        digest = claim["entity"]["canonical_id"]
        page = claim["evidence"][0]["page_1based"]
        applications[digest].append((str(claim["value"]), int(page)))

    rows = []
    documents_extracted = 0
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        digest = entry["sha256"]
        apps = applications.get(digest, [])
        # Deduplicate verbatim strings, keep first page seen, preserve order.
        seen: dict[str, int] = {}
        for text, page in apps:
            seen.setdefault(text, page)
        if seen:
            documents_extracted += 1
        for node_id in entry["node_ids"]:
            if seen:
                rows.append(
                    {
                        "node_id": node_id,
                        "document_sha256": digest,
                        "mpns": entry["mpns_served"],
                        "status": "extracted",
                        "page_1based": min(seen.values()),
                        "applications": list(seen),
                    }
                )
            else:
                rows.append(
                    {
                        "node_id": node_id,
                        "document_sha256": digest,
                        "mpns": entry["mpns_served"],
                        "status": "absent",
                        "reason": "no_applications_block_located",
                        "page_1based": None,
                        "applications": [],
                    }
                )

    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "manifest_documents": sum(
                    1 for _ in args.manifest.open(encoding="utf-8")
                ),
                "documents_with_applications": documents_extracted,
                "rows": len(rows),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
