#!/usr/bin/env python3
"""Extract the page-1 vendor one-liner for the power-description queue.

CR power-description-extraction-20260911: verbatim vendor copy from page 1
only — never invented, never paraphrased. One record per part, joined to the
pairs corpus by pdf_sha256 where available. A page whose best line does not
meet the stated floor (32+ chars with a digit, aisle_gold.valid_power_
description) records no_description_line honestly, but the raw best line is
kept so CR's locked validator can rule without re-opening the PDF.

    python3 scripts/extract_power_descriptions.py \
        --queue /Volumes/M5_4TB/exports/cr_requests/power-description-queue-20260911.json \
        --pdf-root /tmp/power-vendor-full/infineon \
        --pdf-root /tmp/power-vendor-full/rohm \
        --pairs /Volumes/M5_4TB/exports/cr_drops/power-pairs-20260909/pairs-infineon.jsonl \
        --pairs /Volumes/M5_4TB/exports/cr_drops/power-pairs-20260909/pairs-rohm.jsonl \
        --out descriptions.jsonl --missing-out missing.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

DOMAIN_PREFIX = {
    "ti.com": "ti", "infineon.com": "infineon", "rohm.com": "rohm", "st.com": "st",
    "analog.com": "analog", "microchip.com": "microchip", "renesas.com": "renesas",
    "onsemi.com": "onsemi", "monolithicpower.com": "monolithicpower", "nxp.com": "nxp",
}

# Page-1 lines that are never the vendor one-liner.
BOILERPLATE = re.compile(
    r"datasheet|data\s*sheet|^\s*features\b|^\s*description$|general\s+description|product\s+description|"
    r"outline|revision|^\s*rev\b|\bwww\.|https?://|copyright|all\s+rights|"
    r"infineon\s+technologies|rohm\s+co|texas\s+instruments|product\s+structure|"
    r"monolithic\s+integrated|sourcing|interconnection|solderable|^\s*[•l○·]\s*|"
    r"product\s+validation|preliminary|engineering\s+sample|"
    r"rds\s*\(\s*on\s*\)|\bvdss\b|\bidm\b|\bvgs\b",
    re.I,
)
MIN_LINE_CHARS = 10
FLOOR_CHARS = 32
TAGLINE_MIN_SIZE = 11.0
TAGLINE_MAX_GAP = 40.0
HEADER_FRACTION = 0.45
DESC_HEADERS = re.compile(r"^(general\s+|product\s+)?description$", re.I)


def meets_floor(line: str) -> bool:
    """CR's stated floor: 32+ chars with a digit (class labels without a
    digit fail). The locked validator is CR's; this is the stated reading."""
    return len(line) >= FLOOR_CHARS and any(c.isdigit() for c in line)


_CTRL_AS_SPACE = re.compile(r"[\x00-\x1f]")


def _is_garbled(text: str) -> bool:
    """CID/Symbol-font pages extract as control-character soup. Broken-but-
    readable encodings use single control chars as spaces (Infineon maps the
    space to \\x03), so controls are normalised to spaces first; a line that
    is still not mostly letters and digits is not vendor copy."""
    cleaned = _CTRL_AS_SPACE.sub(" ", text)
    if not cleaned.strip():
        return True
    sane = sum(1 for c in cleaned if c.isalnum() or c.isspace() or c in ",.-–—/()™®%+±:;'\"&")
    return sane / len(cleaned) < 0.6


def _page_lines(page) -> list[dict]:
    lines = []
    height = page.rect.height
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = _CTRL_AS_SPACE.sub(" ", "".join(s["text"] for s in line["spans"])).strip()
            if not text or _is_garbled(text):
                continue
            y = line["bbox"][1]
            if y > height - 55:  # footer: page numbers, copyright, doc ids
                continue
            lines.append({"text": text, "y": y, "size": max(s["size"] for s in line["spans"])})
    lines.sort(key=lambda l: (l["y"], l["text"]))
    return lines


def _is_part_numberish(text: str, part_number: str) -> bool:
    bare = re.sub(r"[^a-z0-9]", "", text.lower())
    part = re.sub(r"[^a-z0-9]", "", part_number.lower())
    return bool(part) and part in bare and len(bare) <= len(part) + 4


def _merge_tagline(lines: list[dict], start: int) -> str:
    """Merge same-size, vertically-adjacent tagline lines (ROHM prints
    '35V Voltage Resistance' / '1A LDO Regulators' as a two-line tagline)."""
    first = lines[start]
    parts = [first["text"]]
    y = first["y"]
    for nxt in lines[start + 1:]:
        if 0 <= nxt["y"] - y <= TAGLINE_MAX_GAP and abs(nxt["size"] - first["size"]) <= 1.0 and len(nxt["text"]) >= MIN_LINE_CHARS:
            parts.append(nxt["text"])
            y = nxt["y"]
        else:
            break
    return " ".join(parts)


def _description_first_sentence(lines: list[dict]) -> str | None:
    """First sentence of a page-1 Description section, verbatim substring."""
    for i, ln in enumerate(lines):
        if not DESC_HEADERS.match(ln["text"]):
            continue
        paragraph = []
        for nxt in lines[i + 1:]:
            if DESC_HEADERS.match(nxt["text"]) or BOILERPLATE.search(nxt["text"]) or len(nxt["text"]) < 12:
                break
            paragraph.append(nxt["text"])
            if nxt["text"].endswith("."):
                break
            if len(" ".join(paragraph)) > 400:
                break
        text = " ".join(paragraph).strip()
        if not text:
            continue
        m = re.match(r".{20,}?\.(?=\s|$)", text, re.S)
        return (m.group(0) if m else text).strip()
    return None


def page1_description(pdf_path: Path, part_number: str) -> dict:
    """Deterministic page-1 candidates, verbatim only.

    Order: large-font tagline block in the header region, else the
    Description section's first sentence. Everything is recorded verbatim;
    the floor verdict is reported separately so CR's locked validator rules.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        return {"error": "pdf_unreadable"}
    try:
        lines = _page_lines(doc[0])
    finally:
        doc.close()
    candidates = []
    for i, ln in enumerate(lines):
        if ln["size"] < TAGLINE_MIN_SIZE or BOILERPLATE.search(ln["text"]) or _is_part_numberish(ln["text"], part_number):
            continue
        if len(ln["text"]) < MIN_LINE_CHARS:
            continue
        tagline = _merge_tagline(lines, i)
        if not BOILERPLATE.search(tagline):
            candidates.append((ln["size"], ln["y"], tagline))
    if candidates:
        # Largest type wins (the vendor tagline block); ties go to the
        # earliest position on the page.
        best = max(candidates, key=lambda c: (c[0], -c[1]))
        return {"description_verbatim": best[2], "candidate": "tagline"}
    sentence = _description_first_sentence(lines)
    if sentence and not _is_part_numberish(sentence, part_number):
        return {"description_verbatim": sentence, "candidate": "description_section"}
    return {"no_description_line": True}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--pdf-root", action="append", default=[], type=Path)
    ap.add_argument("--pairs", action="append", default=[], type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--missing-out", type=Path, required=True)
    args = ap.parse_args()

    queue = json.loads(args.queue.read_text())
    local: dict[str, Path] = {}
    for root in args.pdf_root:
        for pdf in root.glob("*.pdf"):
            local.setdefault(pdf.name, pdf)
    pairs: dict[tuple[str, str], dict] = {}
    for path in args.pairs:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                pairs[(row["vendor"], row["part_number"])] = row

    out_rows, missing = [], []
    tally: Counter = Counter()
    for item in queue:
        domain, part = item["domain"], item["part_number"]
        filename = f"{DOMAIN_PREFIX[domain]}-{part}.pdf"
        pdf = local.get(filename)
        if pdf is None:
            missing.append({**item, "reason": "pdf_not_local"})
            tally[("missing_pdf", domain)] += 1
            continue
        result = page1_description(pdf, part)
        pair = pairs.get((domain, part))
        local_sha = None
        sha_source, sha_matches = "local", None
        if pair is not None:
            local_sha = local_sha or sha256_of(pdf)
            sha_source = "pairs"
            sha_matches = local_sha == pair.get("pdf_sha256")
        row = {
            "part_number": part,
            "domain": domain,
            "source_url": item.get("datasheet_url"),
            "pdf_sha": (pair or {}).get("pdf_sha256") or local_sha or sha256_of(pdf),
            "pdf_sha_source": sha_source,
            "pdf_sha_matches_pairs": sha_matches,
        }
        if "description_verbatim" in result:
            line = result["description_verbatim"]
            row["candidate_kind"] = result["candidate"]
            row["page1_line_verbatim"] = line
            if meets_floor(line):
                row["description_verbatim"] = line
                tally[("extracted", domain)] += 1
            else:
                row["no_description_line"] = True
                row["floor_failure_reason"] = "under_32_chars" if len(line) < FLOOR_CHARS else "no_digit"
                tally[("floor_fail", domain)] += 1
        else:
            row.update(result)
            tally[(result.get("error", "no_line"), domain)] += 1
        out_rows.append(row)

    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
    args.missing_out.write_text(json.dumps(missing, ensure_ascii=False, indent=1))
    print(f"parts: {len(queue)}  records: {len(out_rows)}  missing-pdf: {len(missing)}")
    for key in sorted(tally):
        print(f"  {key[0]:12} {key[1]:18} {tally[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
