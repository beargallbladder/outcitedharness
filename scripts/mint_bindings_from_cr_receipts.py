#!/usr/bin/env python3
"""Mint locator package bindings by corroborating printed package tokens.

A binding (document_sha256, package_token) is emitted only when BOTH hold:
  1. the package token (e.g. LQFP48) is printed in the document's own
     extracted page text, and
  2. the token's trailing pin count equals a CR pin-count receipt for an
     MPN the corpus manifest says this document serves (unambiguous
     bucket only).

CR receipts are derived cross-checks, never printed truth; here they only
corroborate a token that is already printed on the page. Every downstream
locator gate (definition-table structure, exact column header match,
JEDEC corroboration, extracted_n == package_n) still applies unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

PACKAGE_TOKEN = re.compile(
    r"\b((?:LQFP|TQFP|QFP|QFN|VQFN|UQFN|UFQFPN|VFQFPN|WLCSP|CSP|UFBGA|"
    r"TFBGA|LFBGA|VFBGA|BGA|LGA|SSOP|TSSOP|HTSSOP|VSSOP|SOIC|PDIP|SDIP|"
    r"HVQFP|PLQFP)[- ]?(\d{2,3}))\b",
    re.IGNORECASE,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-snapshot", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--cr-receipts", type=Path, required=True)
    parser.add_argument("--page-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cohort = {
        d["sha256"]
        for d in json.loads(args.cohort_snapshot.read_text())["documents"]
    }

    # Document sha -> MPNs served, from the intake manifest.
    doc_mpns: dict[str, set[str]] = defaultdict(set)
    for line in args.corpus_manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sha = row.get("sha256")
        if sha in cohort:
            for mpn in row.get("mpns_served") or []:
                doc_mpns[sha].add(str(mpn).upper().strip())

    # Unambiguous receipts: MPN (upper) -> allowed pin counts.
    receipts = json.loads(args.cr_receipts.read_text())
    part_counts: dict[str, set[int]] = defaultdict(set)
    for row in receipts["unambiguous"]:
        part = str(row["part_number"]).upper().strip()
        part_counts[part].update(int(n) for n in row["package_pin_counts"])

    def counts_for(mpns: set[str]) -> set[int]:
        allowed: set[int] = set()
        for mpn in mpns:
            if mpn in part_counts:
                allowed.update(part_counts[mpn])
                continue
            # Receipt parts are often base OPNs of longer orderable MPNs
            # (STM32F103RB prefix of STM32F103RBT6). Deterministic prefix
            # join, minimum 8 characters to avoid accidental stems.
            for part, values in part_counts.items():
                if len(part) >= 8 and mpn.startswith(part):
                    allowed.update(values)
        return allowed

    # Printed package tokens per document from sealed page evidence.
    doc_tokens: dict[str, set[str]] = defaultdict(set)
    with args.page_evidence.open() as handle:
        for line in handle:
            row = json.loads(line)
            sha = row["document_sha256"]
            if sha not in doc_mpns:
                continue
            blob = json.dumps(row.get("blocks") or [])
            for match in PACKAGE_TOKEN.finditer(blob):
                token = match.group(1).upper().replace(" ", "").replace("-", "")
                doc_tokens[sha].add(token)

    minted = 0
    docs = set()
    with args.output.open("w") as out:
        for sha in sorted(doc_tokens):
            allowed = counts_for(doc_mpns[sha])
            if not allowed:
                continue
            for token in sorted(doc_tokens[sha]):
                count = int(re.search(r"(\d{2,3})$", token).group(1))
                if count in allowed:
                    out.write(
                        json.dumps(
                            {
                                "document_sha256": sha,
                                "package": token,
                                "source": "printed_token_cr_receipt_corroborated",
                            }
                        )
                        + "\n"
                    )
                    minted += 1
                    docs.add(sha)

    print(
        json.dumps(
            {
                "cohort_documents": len(cohort),
                "documents_with_manifest_mpns": len(doc_mpns),
                "documents_with_printed_tokens": len(doc_tokens),
                "minted_bindings": minted,
                "documents_bound": len(docs),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
