#!/usr/bin/env python3
"""Refresh the CR drop for the Key Features grid lane.

Copies the grid outputs, the family-document records and the progress ledger
into the export folder, renders GRID-PREVIEW.txt (every document as its ten
groups), splits out the three views CR reads first (package-qualified I/O,
quantity-qualified rows, typed facts), copies the family-grain agreement
report, and writes SHA256SUMS.txt.

    uv run --python 3.11 python scripts/refresh_key_features_drop.py \
        --grid results/key-features-grid-20260908 \
        --records results/family-documents-cr-20260908/records.jsonl \
        --records results/family-documents-20260908/records.jsonl \
        --ledger results/key-features-grid-progress.jsonl \
        --out /Volumes/M5_4TB/exports/family-documents-20260908
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def render_preview(grids: list[dict]) -> str:
    grids = sorted(grids, key=lambda g: (g["_meta"].get("vendor") or "zz", g["_meta"]["source_artifact"]))
    lines: list[str] = []
    for g in grids:
        m = g["_meta"]
        lines.append(f"\n==== {m.get('vendor')} | {(m.get('scope_as_printed') or '')[:60]} | {m['source_artifact']} | groups {g['groups_populated']}/10 | vendor_basis {m.get('vendor_basis')}")
        by: dict[str, list[dict]] = {}
        for r in g["rows"]:
            if r["tier"] == "grid":
                by.setdefault(r["group"], []).append(r)
        for grp in g["groups"]:
            rows = by.get(grp["key"], [])
            if not rows:
                continue
            lines.append(f"  [{grp['label']}]")
            for r in rows[:24]:
                inst = f" ×{r['instances']}" if r.get("instances") else ""
                qq = f" <{r['quantity_qualifier']}>" if r.get("quantity_qualifier") else ""
                vb = " [varies_by_part]" if r.get("varies_by_part") else ""
                typed = ""
                if r.get("typed"):
                    typed = "  typed=" + ", ".join(f"{t['kind']}={t.get('value', t.get('name'))}" for t in r["typed"][:3])
                lines.append(f"     {r['label'][:64]}{inst}{qq}{vb}  p{r['source_pages'][:2]}{typed}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=Path, required=True, help="results/key-features-grid-<date>")
    ap.add_argument("--records", type=Path, action="append", required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "family_document_records").mkdir(exist_ok=True)
    for name in ("grid_rows.jsonl", "grids.jsonl", "summary.json", "gold-report.json"):
        shutil.copy2(args.grid / name, out / name)
    shutil.copy2(args.ledger, out / "progress-ledger.jsonl")
    with (out / "family_document_records" / "records.jsonl").open("w") as f:
        for path in args.records:
            f.write(path.read_text())
    if (args.grid / "agreement").is_dir():
        shutil.copytree(args.grid / "agreement", out / "family-grain-agreement", dirs_exist_ok=True)

    grids = [json.loads(l) for l in (out / "grids.jsonl").read_text().splitlines() if l.strip()]
    (out / "GRID-PREVIEW.txt").write_text(render_preview(grids))

    rows = [json.loads(l) for l in (out / "grid_rows.jsonl").read_text().splitlines() if l.strip()]
    with (out / "io_by_package.jsonl").open("w") as f:
        for r in rows:
            if r.get("section") == "io_by_package":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "quantity_qualified_rows.jsonl").open("w") as f:
        for r in rows:
            if r.get("quantity_qualifier"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    typed = 0
    with (out / "typed_facts.jsonl").open("w") as f:
        for r in rows:
            for t in r.get("typed", []):
                f.write(json.dumps({**t, "group": r["group"], "label": r["label"], "vendor": r.get("vendor"), "scope_as_printed": r.get("scope_as_printed"), "document_sha256": r.get("document_sha256"), "source_pages": r.get("source_pages"), "tier": r["tier"]}, ensure_ascii=False) + "\n")
                typed += 1

    names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    with (out / "SHA256SUMS.txt").open("w") as f:
        for name in names:
            f.write(f"{hashlib.sha256((out / name).read_bytes()).hexdigest()}  {name}\n")
    grid_rows = sum(1 for r in rows if r["tier"] == "grid")
    print(f"documents {len(grids)}  grid rows {grid_rows}  typed facts {typed}  quantity-qualified {sum(1 for r in rows if r.get('quantity_qualifier'))}  io_by_package {sum(1 for r in rows if r.get('section') == 'io_by_package')}  -> {out}")


if __name__ == "__main__":
    main()
