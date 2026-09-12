#!/usr/bin/env python3
"""Fetch the remaining power-description PDFs per CR's manifest.

CR three-drops-ack-20260912: 493 of the 527 missing PDFs have fetchable
URLs; microchip.com (21) and st.com (44) are this network's 403 holdouts —
Samson's click list — and are marked browser_only without a fetch attempt.
Files are saved under the corpus naming ({prefix}-{part}.pdf), validated as
readable PDFs, and sha256-recorded; staged PDFs from CR's tar are copied in
instead of fetched. Resumable: existing valid files are skipped.

    python3 scripts/fetch_power_description_pdfs.py \
        --manifest /Volumes/M5_4TB/exports/cr_requests/power-description-missing-fetch-20260912.json \
        --staged-root "/Volumes/M5_4TB/exports/cr_requests/power-description-pdfs-20260912/Volumes/MACMOBILE/power-datasheet-pairs/ti-20260908/pdf" \
        --out-dir /tmp/power-desc-fetch \
        --status-out /tmp/power-desc-fetch-status.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_power_descriptions import DOMAIN_PREFIX  # noqa: E402

BROWSER_ONLY = {"microchip.com", "st.com"}
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_pdf(path: Path) -> bool:
    try:
        import pymupdf
        doc = pymupdf.open(path)
        ok = doc.page_count > 0
        doc.close()
        return ok and path.read_bytes()[:4] == b"%PDF"
    except Exception:
        return False


def _curl(url: str, dest: Path) -> str:
    """One curl attempt; returns the HTTP status code ('000' on transport
    failure)."""
    r = subprocess.run(
        ["curl", "-sS", "-L", "--max-time", "90",
         "-A", UA, "-o", str(dest), "-w", "%{http_code}", url],
        capture_output=True, text=True,
    )
    return ((r.stdout or "").strip()[-3:]) or "000"


def _pdf_link_in_html(dest: Path) -> str | None:
    """Landing pages (e.g. monolithicpower documentview) carry the real PDF
    link in the body; find the first href ending in .pdf."""
    import re
    try:
        body = dest.read_text(errors="ignore")[:400_000]
    except Exception:
        return None
    m = re.search(r'href="([^"]+\.pdf[^"]*)"', body, re.I)
    if not m:
        m = re.search(r"href='([^']+\.pdf[^']*)'", body, re.I)
    return m.group(1) if m else None


def fetch(url: str, dest: Path, retries: int = 2) -> tuple[bool, str]:
    code = "000"
    for attempt in range(retries + 1):
        code = _curl(url, dest)
        if code == "200" and is_valid_pdf(dest):
            return True, code
        if code == "200" and dest.exists():
            # HTML landing page: follow the real PDF link once.
            link = _pdf_link_in_html(dest)
            if link:
                from urllib.parse import urljoin
                code = _curl(urljoin(url, link), dest)
                if code == "200" and is_valid_pdf(dest):
                    return True, code
        time.sleep(2 * (attempt + 1) ** 2)
    if dest.exists():
        dest.unlink()
    return False, code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--staged-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--status-out", type=Path, required=True)
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    staged = {}
    if args.staged_root is not None and args.staged_root.is_dir():
        for pdf in args.staged_root.glob("*.pdf"):
            staged[pdf.name] = pdf

    status_rows = []
    tally: Counter = Counter()
    for item in manifest:
        domain, part = item["domain"], item["part_number"]
        name = f"{DOMAIN_PREFIX[domain]}-{part}.pdf"
        dest = args.out_dir / name
        row = {"part_number": part, "domain": domain, "url": item["url"]}
        if domain in BROWSER_ONLY:
            row["status"] = "browser_only"
            tally["browser_only"] += 1
        elif name in staged:
            if not dest.exists():
                shutil.copy2(staged[name], dest)
            row["status"] = "staged_by_cr"
            row["pdf_sha"] = sha256_of(dest)
            tally["staged_by_cr"] += 1
        elif dest.exists() and is_valid_pdf(dest):
            row["status"] = "already_fetched"
            row["pdf_sha"] = sha256_of(dest)
            tally["already_fetched"] += 1
        else:
            ok, code = fetch(item["url"], dest)
            if ok:
                row["status"] = "fetched"
                row["pdf_sha"] = sha256_of(dest)
                tally["fetched"] += 1
                time.sleep(args.delay)
            else:
                row["status"] = "fetch_failed"
                row["http_code"] = code
                tally["fetch_failed"] += 1
        status_rows.append(row)

    args.status_out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in status_rows))
    print(f"manifest: {len(manifest)}  {dict(tally)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
