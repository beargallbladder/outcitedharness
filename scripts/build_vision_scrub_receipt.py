#!/usr/bin/env python3
"""Build the vision-scrub completion receipt.

Merges both pipes' results, runs the full grounding audit (with per-pin
word-level location -> evidence-ready bboxes), the package-bogey check, the
GT agreement scoreboard against the 840 sealed records, and spend telemetry.
Writes the sealed receipt CR gates lane eligibility on.
"""
from __future__ import annotations

import json
import multiprocessing
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/Users/samkim/Harnessv1")
SCRUB = Path("/Volumes/M5_4TB/scratch/vision-scrub-v1")
GT_ROOT = Path("/Volumes/M5_4TB/DigiKey_Reference_Designs/claude_ground_truth")

RECEIPT_SCHEMA = "harness.vision-scrub-receipt.v1"

PKG_RE = re.compile(
    r"^(?:LQFP|QFN|TQFN|UFBGA|TFBGA|VFBGA|WLCSP|UFQFPN|LGA|BGA|CSP|QFP|TSSOP|SSOP|MSOP|SOP|SOIC|TQFP|PDIP|DIP)"
    r"(\d{2,4})(?:\+(\d{1,3}))?$",
    re.I,
)


def canon(s):
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


def load_merged():
    by_key = {}
    pipes = {}
    for fname, pipe, pin, pout in (
        ("scrub-results.jsonl", "zai", 0.6, 1.8),
        ("scrub-results-or.jsonl", "openrouter", 1.2, 4.0),
    ):
        n = ti = to = 0
        try:
            handle = open(SCRUB / fname)
        except FileNotFoundError:
            continue
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("deferred"):
                continue
            if "usage" not in r and "error" not in r:
                continue
            key = (r["document_sha256"], r["page_1based"])
            u = r.get("usage") or {}
            ti += int(u.get("prompt_tokens") or 0)
            to += int(u.get("completion_tokens") or 0)
            n += 1
            if key not in by_key:
                r["_pipe"] = pipe
                by_key[key] = r
        pipes[pipe] = {
            "records": n,
            "prompt_tokens": ti,
            "completion_tokens": to,
            "cost_usd": round((ti * pin + to * pout) / 1e6, 2),
        }
    return by_key, pipes


def grounding_job(args):
    sha, page, tables, source = args
    import pymupdf

    try:
        doc = pymupdf.open(source)
        if page > doc.page_count:
            doc.close()
            return (sha, page, None, "page_out_of_range", 0, 0, [])
        text = canon(doc[page - 1].get_text())
        words = [
            (w[4], (w[0], w[1], w[2], w[3])) for w in doc[page - 1].get_text("words")
        ]
        doc.close()
    except Exception:
        return (sha, page, None, "open_failed", 0, 0, [])
    total = grounded = 0
    located = []
    for t in tables or []:
        for p in (t.get("pins") or []):
            total += 1
            name = canon(p.get("name"))
            no = canon(p.get("pin_no"))
            bbox = None
            for wtext, wbox in words:
                if name and canon(wtext) == name:
                    bbox = wbox
                    break
            if (name and name in text) or (no and no in text and len(name) <= 3):
                grounded += 1
                located.append({"pin_no": p.get("pin_no"), "name": p.get("name"),
                                "package": t.get("package"), "name_bbox": bbox})
    rate = grounded / total if total else None
    return (sha, page, rate, "ok", grounded, total, located)


def main():
    merged, pipes = load_merged()
    print(f"merged: {len(merged)} unique pages read", flush=True)

    by_sha_path = {}
    for line in open(ROOT / "results/datasheet-word-columns-corpus-sweep-v3-20260912/extractions.jsonl"):
        r = json.loads(line)
        by_sha_path[r["document_sha256"]] = r["source_path"]

    jobs = []
    for (sha, page), r in merged.items():
        tables = r.get("tables")
        if not tables:
            continue
        src = by_sha_path.get(sha)
        if src:
            jobs.append((sha, page, tables, src))
    print(f"grounding jobs: {len(jobs)} table-bearing pages", flush=True)

    grounding = {}
    located_rows = 0
    with multiprocessing.Pool(8) as pool:
        for i, (sha, page, rate, status, g, t, located) in enumerate(
            pool.imap_unordered(grounding_job, jobs, chunksize=8), 1
        ):
            grounding[(sha, page)] = {"rate": rate, "status": status, "grounded": g, "total": t}
            located_rows += len(located)
            if i % 300 == 0:
                print(f"grounded {i}/{len(jobs)}", flush=True)

    # bogey check over all tables
    bogey = Counter()
    for r in merged.values():
        for t in (r.get("tables") or []):
            m = PKG_RE.match(str(t.get("package") or "").strip().upper().replace(" ", "").replace("-", ""))
            if not m:
                bogey["no_countable_package"] += 1
                continue
            expected = int(m.group(1)) + int(m.group(2) or 0)
            nums = sorted({
                int(str(p.get("pin_no")))
                for p in (t.get("pins") or [])
                if str(p.get("pin_no") or "").strip().isdigit()
            })
            if not nums:
                bogey["nonnumeric_identifiers"] += 1
            elif nums == list(range(1, expected + 1)):
                bogey["exact_full_package"] += 1
            elif len(nums) > expected:
                bogey["over_extraction"] += 1
            elif nums == list(range(nums[0], nums[-1] + 1)):
                bogey["partial_contiguous"] += 1
            else:
                bogey["partial_gaps"] += 1

    # GT scoreboard
    registry = json.loads((ROOT / "results/datasheet-corpus-registry-20260901.json").read_text())
    gt_docs = {d["document_sha256"]: d for d in registry["documents"] if d.get("ground_truth")}
    import hashlib

    def sha_file(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
        return h.hexdigest()

    scoreboard = Counter()
    gt_examples = []
    vision_by_doc = {}
    for (sha, page), r in merged.items():
        if r.get("tables"):
            vision_by_doc.setdefault(sha, []).extend(r["tables"])
    for sha, doc in gt_docs.items():
        entry = doc["ground_truth"][0]
        gt_path = GT_ROOT / entry["path"].split("/")[-1]
        if not gt_path.exists():
            continue
        record = json.loads(gt_path.read_text())
        pinout = record.get("pinout") or {}
        gt_rows = pinout.get("pin_functions_summary") or []
        expected = {
            canon(r_.get("pin_no")): canon(r_.get("name"))
            for r_ in gt_rows
            if r_.get("pin_no") is not None and r_.get("name")
        }
        tables = vision_by_doc.get(sha) or []
        for package in pinout.get("packages") or []:
            scoreboard["packages_total"] += 1
            pkg_key = canon(package)
            # per-package selection: tables whose header matches this package
            pkg_tables = [
                t for t in tables
                if canon(t.get("package")) == pkg_key
            ] or (tables if len(pinout.get("packages") or []) == 1 else [])
            if not pkg_tables:
                scoreboard["not_located"] += 1
                continue
            printed = {}
            for t in pkg_tables:
                for p in (t.get("pins") or []):
                    printed.setdefault(canon(p.get("pin_no")), canon(p.get("name")))
            confirmed = disagree = 0
            for pin, name in expected.items():
                got = printed.get(pin)
                if got is None:
                    continue
                if got == name or name in got or got in name:
                    confirmed += 1
                else:
                    disagree += 1
            if confirmed and not disagree:
                scoreboard["package_confirmed"] += 1
            elif disagree:
                scoreboard["package_disagreement"] += 1
                if len(gt_examples) < 10:
                    gt_examples.append({
                        "doc": doc["paths"][0] if doc.get("paths") else sha[:12],
                        "package": package,
                        "confirmed": confirmed,
                        "disagree": disagree,
                    })
            else:
                scoreboard["not_located"] += 1

    rates = [g["rate"] for g in grounding.values() if g["rate"] is not None]
    pins_total = sum(g["total"] for g in grounding.values())
    pins_grounded = sum(g["grounded"] for g in grounding.values())
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pages_total": 16841,
        "pages_read": len(merged),
        "pages_with_tables": len(jobs),
        "pages_empty_honest": len(merged) - len(jobs),
        "pins_extracted": sum(
            len(p.get("pins") or []) for r in merged.values() for t in (r.get("tables") or [])
        ),
        "pins_located_with_bbox": located_rows,
        "grounding": {
            "pins_total": pins_total,
            "pins_grounded": pins_grounded,
            "rate": round(pins_grounded / max(pins_total, 1), 4),
            "pages_fully_grounded": sum(1 for g in rates if g >= 0.999),
            "pages_audited": len(rates),
        },
        "bogey_check": dict(bogey),
        "gt_scoreboard": dict(scoreboard),
        "gt_disagreement_examples": gt_examples,
        "spend": pipes,
    }
    out = SCRUB / "scrub-receipt.json"
    out.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print(json.dumps(receipt, indent=1, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
