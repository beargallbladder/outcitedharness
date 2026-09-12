#!/usr/bin/env python3
"""Rebuild power-gold fixtures with CR's corrections-layer re-labels.

Applies `fae-power-knife-corrections-v1` (T1 datasheet value supersedes T2
selector label, Truth Merge Policy v1) to the per-document gold fixtures
generated from `power_parametric_knives_v0` selector labels:

- corrected value: `to_a` directly; `to_mohm` is a mOhm numeral and fixtures
  store Ohm, so it is divided by 1000
- `quantity_qualifier`: the grounding document row's own qualifier in the
  reader's vocabulary (resolved from the evidence run's grids via the
  fixture's `source_artifact`), falling back to CR's literal instruction
  `maximum` when the grounding row is not in our extraction
- `source_qualifier`: what the selector export stated, header-derived per
  CR qualifier-encoding-answer-20260912 (rohm.com rds `(Typ)`, ti.com rds
  `(max)`); infineon.com is unstated and stays absent
- per-row `correction` provenance so no audit re-derives the apply

A correction whose `from` value does not match the fixture row is SKIPPED
and reported, never force-applied. Uncorrected rows keep their values; only
`source_qualifier` is added where the domain/field flavor is stated.

    python3 scripts/rebuild_power_gold_fixtures.py \
        --gold-dir /tmp/power-vendor-full/gold \
        --corrections /tmp/power_knife_corrections_v1.json \
        --grids ifx2=results/power-grids-20260909-ifx2/grids.jsonl \
        --grids rohm-si=results/power-grids-20260909-rohm-si/grids.jsonl \
        --out /tmp/power-gold-relabel-20260912/gold
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from power_gold_adjudication import QUANTITY, normalize  # noqa: E402

DOMAIN_PREFIX = {"ti.com": "ti", "infineon.com": "infineon", "rohm.com": "rohm"}
# CR qualifier-encoding-answer-20260912: what the selector column headers
# stated. Only where CR stated it; absence stays absence.
SOURCE_QUALIFIER = {("rohm.com", "rds_on_mohm"): "typical", ("ti.com", "rds_on_mohm"): "maximum"}


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def grounding_qualifier(correction: dict, grids: dict, artifact: str) -> tuple[str | None, bool]:
    """Resolve the correction's grounding document row and its qualifier.

    Returns (qualifier, resolved). The lookup matches the correction's
    `document_row_verbatim` (CR stores the row's label for ROHM prose and the
    row's verbatim for Infineon tables) against the row's label/verbatim, at
    the normalized target value, within the quantity's row vocabulary.
    """
    field = correction["field"]
    to_canon = correction.get("to_a", correction.get("to_mohm"))
    verb = correction.get("document_row_verbatim") or ""
    pattern = QUANTITY[field]
    for row in grids.get(artifact, {}).get("rows", []):
        symbol = (row.get("symbol") or "").replace(" ", "")
        if not ((symbol and pattern.match(symbol)) or pattern.search(row.get("parameter") or "")):
            continue
        unit = row.get("unit")
        # The bounded glyph correction (RDS document rows only).
        if field == "rds_on_mohm" and unit == "mW":
            unit = "mOhm"
        raw = row.get("value")
        vals = raw if isinstance(raw, list) else [raw]
        norms = [normalize(field, unit, v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not any(n is not None and close(n, to_canon) for n in norms):
            continue
        text = f"{row.get('verbatim') or ''} {row.get('label') or ''}"
        if verb[:40] in text or (row.get("label") or "")[:40] in verb:
            return row.get("quantity_qualifier"), True
    return None, False


def apply_corrections(fixtures: dict[str, dict], corrections: list[dict], grids_by_run: dict[str, dict]) -> dict:
    """Apply corrections to the in-memory fixture set. Returns a report."""
    report = Counter()
    qualifier_counts: Counter = Counter()
    by_correction: dict[tuple[str, str, str], dict] = {
        (c["domain"], c["part_number"], c["field"]): c for c in corrections
    }
    for artifact, fixture in fixtures.items():
        prefix, _, rest = artifact.partition("-")
        part = rest[:-4] if rest.endswith(".pdf") else rest
        domain = next((d for d, p in DOMAIN_PREFIX.items() if p == prefix), None)
        if domain is None:
            report["unknown_domain_artifact"] += 1
            continue
        for exp in fixture["expected"]:
            sq = SOURCE_QUALIFIER.get((domain, exp.get("source_field")))
            if sq is not None:
                exp["source_qualifier"] = sq
            correction = by_correction.get((domain, part, exp.get("source_field")))
            if correction is None:
                continue
            from_canon = correction.get("from_a")
            if "from_mohm" in correction:
                from_canon = correction["from_mohm"] / 1000.0  # fixtures store Ohm
                to_value = correction["to_mohm"] / 1000.0
            else:
                to_value = correction["to_a"]
            if not isinstance(exp.get("value"), (int, float)) or not close(exp["value"], from_canon):
                report["skipped_from_mismatch"] += 1
                continue
            run = correction.get("evidence", "").replace(".jsonl", "")
            grids = grids_by_run.get(run, {})
            qualifier, resolved = grounding_qualifier(correction, grids, artifact)
            if not resolved:
                # CR's literal encoding instruction; provenance records that
                # the grounding row was not found in our extraction.
                qualifier = "maximum"
            exp["value"] = to_value
            exp["quantity_qualifier"] = qualifier
            exp["correction"] = {
                "contract": "fae-power-knife-corrections-v1",
                "from": from_canon,
                "to": to_value,
                "document_row_verbatim": correction.get("document_row_verbatim"),
                "evidence": correction.get("evidence"),
                "grounding_qualifier": qualifier,
                "grounding_resolved": resolved,
            }
            report["applied"] += 1
            qualifier_counts[(exp["source_field"], qualifier, resolved)] += 1
    report["corrections_total"] = len(corrections)
    report["qualifiers"] = dict(qualifier_counts)
    return dict(report)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold-dir", type=Path, required=True)
    ap.add_argument("--corrections", type=Path, required=True)
    ap.add_argument("--grids", action="append", default=[], metavar="RUN=PATH", help="evidence run name=path to grids.jsonl (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    corrections_doc = json.loads(args.corrections.read_text())
    corrections = corrections_doc["corrections"] if isinstance(corrections_doc, dict) else corrections_doc
    grids_by_run: dict[str, dict] = {}
    for pair in args.grids:
        run, _, path = pair.partition("=")
        grids = {}
        for line in Path(path).read_text().splitlines():
            if line.strip():
                g = json.loads(line)
                grids[g["_meta"]["source_artifact"]] = g
        grids_by_run[run] = grids

    fixtures = {}
    for path in sorted(args.gold_dir.glob("*.json")):
        fixture = json.loads(path.read_text())
        fixtures[fixture["source_artifact"]] = fixture

    report = apply_corrections(fixtures, corrections, grids_by_run)
    report["documents"] = len(fixtures)

    args.out.mkdir(parents=True, exist_ok=True)
    relabel = {
        "contract": "fae-power-knife-corrections-v1",
        "rule": corrections_doc.get("rule") if isinstance(corrections_doc, dict) else None,
        "corrections_applied": report.get("applied", 0),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    for artifact, fixture in fixtures.items():
        fixture["relabel"] = relabel
        (args.out / f"{artifact}.json").write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n")

    applied = report.pop("applied", 0)
    qualifiers = report.pop("qualifiers", {})
    print(f"documents: {report.pop('documents')}")
    print(f"corrections: {report['corrections_total']} total, {applied} applied, {report.get('skipped_from_mismatch', 0)} skipped (from-value mismatch)")
    for key in sorted(report):
        if key != "corrections_total":
            print(f"  {key}: {report[key]}")
    print("fixture qualifiers (field, qualifier, grounding_resolved):")
    for key in sorted(qualifiers):
        print(f"  {key}: {qualifiers[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
