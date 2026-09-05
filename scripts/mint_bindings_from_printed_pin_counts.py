#!/usr/bin/env python3
"""Mint locator package bindings from pin counts printed on datasheet pages.

TI-style analog/power datasheets rarely encode the pin count in the package
token (SOT-23 is not a 23-pin package), so OPN-decode and CR-receipt minting
both miss them. But these documents print the count directly next to the
package family name (for example "16-Pin TSSOP" or "SOIC (14) 14-pin").

A binding (document_sha256, FAMILY<N>) is emitted only when the family name
appears verbatim in UPPERCASE in the document's own extracted page text,
immediately adjacent to an explicit "<N>-pin" phrase. This is printed truth,
not a derivation. Every downstream locator gate (definition-table structure,
exact column match, extracted_n == package_n) still applies unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# No TO family: "16-pin to ..." false-positives under adjacency matching.
FAMILY = (
    "(LQFP|TQFP|QFP|VQFN|UQFN|WQFN|QFN|DSBGA|WLCSP|UFBGA|NFBGA|BGA|LGA|"
    "HTSSOP|TSSOP|SSOP|VSSOP|MSOP|SOIC|PDIP|DDPAK|WSON|VSON|SON|HSOIC)"
)
# "<N>-Pin FAMILY" — family must be printed uppercase; pin wording may vary.
COUNT_THEN_FAMILY = re.compile(
    r"\b(\d{2,3})\s*[-]?\s*[Pp][Ii][Nn]\s+" + FAMILY + r"\b"
)
# "FAMILY ... <N>-pin" within the same line, short gap.
FAMILY_THEN_COUNT = re.compile(
    r"\b" + FAMILY + r"\b[^\n]{0,30}?\b(\d{2,3})\s*[-]?\s*[Pp][Ii][Nn]"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-evidence", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    tokens: dict[str, set[str]] = defaultdict(set)
    for line in args.page_evidence.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        digest = record["document_sha256"]
        text = " ".join(
            block.get("text", "") for block in record.get("blocks") or []
        )
        for match in COUNT_THEN_FAMILY.finditer(text):
            count, family = int(match.group(1)), match.group(2)
            if 10 <= count <= 100:
                tokens[digest].add(f"{family}{count}")
        for match in FAMILY_THEN_COUNT.finditer(text):
            family, count = match.group(1), int(match.group(2))
            if 10 <= count <= 100:
                tokens[digest].add(f"{family}{count}")

    rows = [
        {"document_sha256": digest, "package": token, "source": args.source_label}
        for digest in sorted(tokens)
        for token in sorted(tokens[digest])
    ]
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "documents": len(tokens),
                "bindings": len(rows),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
