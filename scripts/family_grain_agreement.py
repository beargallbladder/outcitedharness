#!/usr/bin/env python3
"""Second source at family grain: two documents about the same family compared
on the typed facts each Key Features grid carries.

The two-source rule needs a second, independent document for a family fact
to become a knife. At family grain the pair is the reference manual (or user
manual) and the family datasheet for the same line, or two revisions of the
same document. This report pairs documents on a shared family key, compares
the facts both carry, and lists every disagreement. Nothing is promoted here.

    uv run --python 3.11 python scripts/family_grain_agreement.py \
        --grids results/key-features-grid-20260908/grids.jsonl \
        --out results/key-features-grid-20260908/agreement

Compared (overlap only, exact equality):
  core            set of core names on the processing rows (intersection agrees)
  max_freq_mhz    largest MHz on processing rows
  flash_kb_max    largest flash size in KB (code_flash or unqualified flash)
  sram_kb_max     largest SRAM size in KB (total_sram or unqualified)
  <class>_instances  the set of instance names a document lists (TIM1, TIM5, ...)
  <class>_count   largest count the document states in a feature line
  io_<pins>_<pkg> package-qualified I/O, same quantity_qualifier on both sides
  supply_v        rated supply range (lo, hi)
  temp_max_c      highest ambient/unqualified temperature
  pin_counts      package pin counts (intersection agrees)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

# Family keys are the grain a document is written for; a series token
# (STM32F7, RX200, K22) is not one, since two documents in the same series
# can describe disjoint part sets.
FAMILY_KEY = re.compile(
    r"\b(STM32[A-Z]\d{3}|STM32WB\d{2}|STM32[A-Z]\d[A-Z]\d|GD32[A-Z]\d{3}|GD32[A-Z]\d[A-Z]{2}|EFM32[A-Z]{2}\d{1,2}|EFR32[A-Z]{2}\d{1,2}|EFM8[A-Z]{2}\d{1,2}|XMC\d{4}|XMC[147]\d{3}|PIC32[A-Z]{2}|PIC1[68]L?F\d{3,5}|AT?SAM[A-Z]\d{2}|K[LVEW]?\d{2}P\d{2,3}M\d{2,3}SF\d[A-Z0-9]*?(?=RM|\b)|MKL\d{2}|KL\d{2}|MK\d{2}|MKE\d{2}|MKW\d{2}|LPC\d{2,4}|LPC55S\d{2}|MCX[A-Z]\d{2,3}|MCX[A-Z]\d{2}x|i\.?MX\s?RT\d{4}|IMXRT\d{4}|RA\d[A-Z]\d|RX\d{2}[A-Z](?:-[AB])?|RX\d{2}[1-9]|RL78/[A-Z]\d{2}|TM4C\d{3}|MSPM0[GL]\d{4}|MSPM0 [GL]-SERIES|MSP430[A-Z]{1,2}\d{2,4}|F28[0-9P]\d{2,4}[A-Z]?|TMS320F28\d{3}|RM4[68]|TMS570LS\d{2,4}|TC2\d{2}|TC3\d{2}|CY8C6\w{3}|CY8C4\w{3}|PSOC\s?[46]|CEC173\d)(?![0-9])",
    re.I,
)

_CORE = re.compile(r"\b(Cortex-?M\d+\+?|Cortex-?R\d+F?|Cortex-?A\d+|C8051|8051|CIP-51|RISC-V|C28x|RXv[123]|RL78|MIPS32|MIPS M\d[Kk]|AVR|PIC18|PIC16|TriCore|Xtensa)\b", re.I)
_PINS_IN_LABEL = re.compile(r"\b(\d{2,3})\s*-?\s*(?:pin|lead)s?\b", re.I)


_MANUAL = re.compile(r"reference\s+manual|user\s+manual|user'?s\s+guide|technical\s+reference|architecture\s+trm|\bTRM\b|\bRM\d{4}\b|\bUM\d{4}\b|(?:^|[^A-Za-z])(?:S?RM|UM)(?:\s*\(\d\))?\.pdf$|_RM\b|-RM\b|spru[a-z0-9]+|slau\d+|spnu\d+|user_manual|reference-manual", re.I)


def genre(meta: dict[str, Any]) -> str:
    """'manual' (reference/user/technical reference manual) or 'datasheet'."""
    hay = " ".join(str(meta.get(k) or "") for k in ("title_verbatim", "source_artifact", "document_id"))
    return "manual" if _MANUAL.search(hay) else "datasheet"


def scope_tokens(meta: dict[str, Any]) -> set[str]:
    return {t.strip().upper() for t in re.split(r"[,;/]\s*", str(meta.get("scope_as_printed") or "")) if t.strip()}


def norm_core(text: str) -> str:
    return re.sub(r"[\s\-]", "", text).upper().replace("CORTEXM", "CORTEX-M").replace("CORTEXR", "CORTEX-R").replace("CORTEXA", "CORTEX-A")


def family_keys(meta: dict[str, Any]) -> set[str]:
    hay = " ".join(str(meta.get(k) or "") for k in ("scope_as_printed", "title_verbatim", "source_artifact", "document_id"))
    keys = {m.group(1).upper().replace(" ", "").replace(".", "") for m in FAMILY_KEY.finditer(hay)}
    # KL25 also appears inside KL25P80M48SF0; keep the most specific key per prefix.
    keys = {k for k in keys if not any(o != k and o.startswith(k) for o in keys)}
    # A Kinetis RM/DS pair shares the part code; "V2" is a later silicon of the same part.
    return {re.sub(r"(?:RM|DS)$", "", re.sub(r"V\d(?:RM)?$", "", k)) for k in keys}


def kb(value: Any, unit: str | None) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    unit = (unit or "").lower()
    if unit in ("kb", "kbyte", "kbytes", "k"):
        return float(value)
    if unit in ("mb", "mbyte", "mbytes"):
        return float(value) * 1024
    if unit in ("bytes", "byte", "b"):
        return float(value) / 1024
    return None


def typed_facts(grid: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Attribute -> {value, pages, verbatim} for the facts one grid carries."""
    facts: dict[str, dict[str, Any]] = {}
    rows = [r for r in grid["rows"] if r["tier"] == "grid"]

    def put(attribute: str, value: Any, row: dict[str, Any], better: Any = None) -> None:
        cur = facts.get(attribute)
        if cur is None or (better is not None and better(value, cur["value"])):
            facts[attribute] = {"value": value, "pages": row["source_pages"][:3], "verbatim": row["verbatim"][:160]}

    cores: set[str] = set()
    pins: set[int] = set()
    for r in rows:
        group, label, unit, value = r["group"], r["label"], (r.get("unit") or ""), r.get("value")
        qq = r.get("quantity_qualifier")
        if group == "processing":
            for m in _CORE.finditer(label):
                cores.add(norm_core(m.group(1)))
            if unit.lower() == "mhz" and isinstance(value, (int, float)) and 8 <= value <= 3000 and not re.search(r"oscillator|\bRC\b|crystal|\bHSI\b|\bLSI\b|\bIRC\b|flash", label, re.I):
                put("max_freq_mhz", float(value), r, better=lambda a, b: a > b)
        if group == "memory" and isinstance(value, (int, float)) and not r.get("varies_by_part") is None:
            size = kb(value, unit)
            if size is not None and re.search(r"flash|program memory", label, re.I) and qq in ("code_flash", None) and not re.search(r"data flash|eeprom|sector|page|block|bank|boot|option|OTP|information", label, re.I):
                put("flash_kb_max", size, r, better=lambda a, b: a > b)
            if size is not None and re.search(r"\bs?ram\b", label, re.I) and qq in ("total_sram", None) and not re.search(r"bank|retention|backup|cache|tcm|ecc", label, re.I):
                put("sram_kb_max", size, r, better=lambda a, b: a > b)
        # Two count quantities, never mixed: the instances a document names
        # (TIM1, TIM5, ...) and a count it states ("Up to 9 timers", which
        # also counts watchdogs and SysTick). Each compares with its own kind.
        if r.get("instances") and r.get("peripheral_class") and r["peripheral_class"] not in ("gpio", "flash", "sram"):
            if r.get("instance_names"):
                cur = facts.get(f"{r['peripheral_class']}_instances")
                # ST names one block two ways (USB_OTG_FS / OTG_FS; SPI1 doubles as I2S1).
                names = {re.sub(r"^USB_OTG_", "OTG_", n) for n in r["instance_names"]}
                if r["peripheral_class"] == "spi":
                    names = {n for n in names if not n.startswith("I2S")}
                names = sorted(names | set(cur["value"] if cur else []))
                if not names:
                    continue
                facts[f"{r['peripheral_class']}_instances"] = {"value": names, "pages": r["source_pages"][:3], "verbatim": ", ".join(names)[:160]}
            elif not r.get("parent"):
                put(f"{r['peripheral_class']}_count", int(r["instances"]), r, better=lambda a, b: a > b)
        if r.get("section") == "io_by_package" and r.get("package_qualifier"):
            pq = r["package_qualifier"]
            put(f"io_{pq['pin_count']}_{pq['package']}_{qq}", int(value), r)
        if r.get("section") == "supply_range" and isinstance(value, list) and len(value) == 2 and qq in ("rated", None):
            put("supply_v", tuple(value), r)
        if r.get("section") == "temperature_range" and isinstance(value, list) and len(value) == 2 and qq in ("ambient", None):
            put("temp_max_c", float(value[1]), r, better=lambda a, b: a > b)
        if r.get("section") == "packages" and isinstance(r.get("packages"), list):
            for p in r["packages"]:
                m = re.search(r"(\d{2,3})", str(p))
                if m:
                    pins.add(int(m.group(1)))
        elif group == "io_package_environment" and r.get("section") != "io_by_package":
            for m in _PINS_IN_LABEL.finditer(label):
                pins.add(int(m.group(1)))
    if cores:
        facts["core"] = {"value": sorted(cores), "pages": [], "verbatim": ""}
    if pins:
        facts["pin_counts"] = {"value": sorted(pins), "pages": [], "verbatim": ""}
    return facts


def compare(a: dict[str, Any], b: dict[str, Any], attribute: str) -> bool:
    if attribute == "core":
        # Cortex-R4 and Cortex-R4F name the same core with and without the FPU suffix.
        return any(x == y or x.startswith(y) or y.startswith(x) for x in a["value"] for y in b["value"])
    if attribute == "pin_counts":
        return bool(set(a["value"]) & set(b["value"]))
    if attribute.endswith("_instances"):
        # A manual names every instance the family has; a datasheet names
        # the ones its parts have (or the ones its text happens to mention).
        # Consistent when one set contains the other; a conflict is two sets
        # that each name an instance the other lacks.
        sa, sb = set(a["value"]), set(b["value"])
        return sa <= sb or sb <= sa
    if attribute == "supply_v":
        return tuple(a["value"]) == tuple(b["value"])
    return a["value"] == b["value"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grids", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    grids = [json.loads(line) for line in args.grids.read_text().splitlines() if line.strip()]
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for g in grids:
        vendor = g["_meta"].get("vendor") or "?"
        for key in family_keys(g["_meta"]):
            by_key[(vendor, key)].append(g)

    pairs: list[dict[str, Any]] = []
    adjudication: list[dict[str, Any]] = []
    per_attribute: dict[str, Counter] = defaultdict(Counter)
    seen_pairs: set[tuple[str, str]] = set()
    for (vendor, key), docs in sorted(by_key.items()):
        if len(docs) < 2:
            continue
        for ga, gb in combinations(docs, 2):
            sa, sb = ga["_meta"]["document_sha256"], gb["_meta"]["document_sha256"]
            if sa == sb or (min(sa, sb), max(sa, sb)) in seen_pairs:
                continue
            # A second source is a different genre for the same line (datasheet
            # vs manual), or the same genre only when the printed scopes
            # overlap (two revisions, two manuals for one part code). Two
            # per-part datasheets of one series are different parts, not a pair.
            genres = (genre(ga["_meta"]), genre(gb["_meta"]))
            scopes_a, scopes_b = scope_tokens(ga["_meta"]), scope_tokens(gb["_meta"])
            if genres[0] == genres[1] and not (scopes_a & scopes_b):
                continue
            seen_pairs.add((min(sa, sb), max(sa, sb)))
            fa, fb = typed_facts(ga), typed_facts(gb)
            shared = sorted(set(fa) & set(fb))
            if not shared:
                continue
            agree = 0
            rows = []
            manual_side = "a" if genres == ("manual", "datasheet") else "b" if genres == ("datasheet", "manual") else None
            for attribute in shared:
                ok = compare(fa[attribute], fb[attribute], attribute)
                envelope = False
                if not ok and manual_side and attribute in ("flash_kb_max", "sram_kb_max", "max_freq_mhz", "temp_max_c"):
                    # The manual's maximum covers every part it documents; the
                    # datasheet's covers its parts. Manual >= datasheet is the
                    # expected relation, not a conflict.
                    va, vb = fa[attribute]["value"], fb[attribute]["value"]
                    envelope = (va >= vb) if manual_side == "a" else (vb >= va)
                    ok = envelope
                agree += ok
                bucket = re.sub(r"^io_\d+_[A-Z0-9]+_", "io_by_package_", attribute) if attribute.startswith("io_") else attribute
                per_attribute[bucket]["agree" if ok else "disagree"] += 1
                if envelope:
                    per_attribute[bucket]["manual_envelope"] += 1
                relation = None
                if attribute.endswith("_instances"):
                    sa_, sb_ = set(fa[attribute]["value"]), set(fb[attribute]["value"])
                    relation = "equal" if sa_ == sb_ else "a_subset" if sa_ < sb_ else "b_subset" if sb_ < sa_ else "conflict"
                    per_attribute[bucket][relation] += 1
                if envelope:
                    relation = "manual_envelope"
                rows.append({"attribute": attribute, "agree": ok, "relation": relation, "a": fa[attribute]["value"], "b": fb[attribute]["value"]})
                if not ok:
                    adjudication.append({
                        "vendor": vendor, "family_key": key, "attribute": attribute,
                        "a": {"document": ga["_meta"]["source_artifact"], **fa[attribute]},
                        "b": {"document": gb["_meta"]["source_artifact"], **fb[attribute]},
                    })
            pairs.append({
                "vendor": vendor, "family_key": key,
                "a": ga["_meta"]["source_artifact"], "b": gb["_meta"]["source_artifact"],
                "a_pages": ga["_meta"].get("page_count"), "b_pages": gb["_meta"].get("page_count"),
                "genres": list(genres),
                "scope_a": ga["_meta"].get("scope_as_printed"), "scope_b": gb["_meta"].get("scope_as_printed"),
                "scope_relation": "equal" if scopes_a == scopes_b else "overlap" if scopes_a & scopes_b else "disjoint_as_printed",
                "compared": len(shared), "agree": agree,
                "only_a": sorted(set(fa) - set(fb))[:20], "only_b": sorted(set(fb) - set(fa))[:20],
                "rows": rows,
            })

    total_compared = sum(p["compared"] for p in pairs)
    total_agree = sum(p["agree"] for p in pairs)
    attribute_table = {}
    for attribute, c in sorted(per_attribute.items()):
        compared = c["agree"] + c["disagree"]
        attribute_table[attribute] = {"compared": compared, "agree": c["agree"], "disagree": c["disagree"], "agreement_rate": round(c["agree"] / compared, 4) if compared else None, "passes_95": (c["agree"] / compared >= 0.95) if compared else None}
        if attribute.endswith("_instances"):
            attribute_table[attribute]["instance_sets"] = {k: c[k] for k in ("equal", "a_subset", "b_subset", "conflict") if c[k]}
        if c["manual_envelope"]:
            attribute_table[attribute]["agree_as_manual_envelope"] = c["manual_envelope"]
    report = {
        "schema": "harness.electronics-family-grain-agreement-report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grids": str(args.grids),
        "grids_sha256": hashlib.sha256(args.grids.read_bytes()).hexdigest(),
        "documents": len(grids),
        "documents_with_family_key": sum(1 for g in grids if family_keys(g["_meta"])),
        "family_keys_with_two_or_more_documents": sum(1 for docs in by_key.values() if len(docs) >= 2),
        "pairs_compared": len(pairs),
        "overall": {"compared": total_compared, "agree": total_agree, "agreement_rate": round(total_agree / total_compared, 4) if total_compared else None},
        "per_attribute": attribute_table,
        "adjudication_count": len(adjudication),
        "method": "Two documents pair when they share a family key (from scope, title, filename) and vendor, and differ in content hash. Only facts both grids type are compared, exactly; cores and pin counts agree on intersection; instance-name sets agree when one contains the other (a manual names the family's full set, a datasheet its parts'); a scalar maximum agrees when the manual's is at least the datasheet's (reported as manual_envelope); package-qualified I/O compares only under the same package and the same quantity qualifier (stated vs port pins listed). Nothing is promoted by this report.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "family-grain-agreement.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "family-grain-pairs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs))
    (args.out / "family-grain-adjudication.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in adjudication))

    lines = [
        "# Second source at family grain",
        "",
        f"Documents: {len(grids)}; with a family key: {report['documents_with_family_key']}; family keys with two or more documents: {report['family_keys_with_two_or_more_documents']}; pairs compared: {len(pairs)}.",
        "",
        f"Overall: {total_agree}/{total_compared} agree ({report['overall']['agreement_rate']}). Adjudication rows: {len(adjudication)}.",
        "",
        "| attribute | compared | agree | disagree | rate | >=95% | instance sets (equal/subset/conflict) |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for attribute, row in attribute_table.items():
        sets = row.get("instance_sets")
        sets_text = f"{sets.get('equal', 0)}/{sets.get('a_subset', 0) + sets.get('b_subset', 0)}/{sets.get('conflict', 0)}" if sets else (f"manual envelope: {row['agree_as_manual_envelope']}" if row.get("agree_as_manual_envelope") else "")
        lines.append(f"| {attribute} | {row['compared']} | {row['agree']} | {row['disagree']} | {row['agreement_rate']} | {row['passes_95']} | {sets_text} |")
    by_relation = Counter(p["scope_relation"] for p in pairs)
    by_genre = Counter("/".join(sorted(p["genres"])) for p in pairs)
    report["pairs_by_scope_relation"] = dict(by_relation)
    report["pairs_by_genre"] = dict(by_genre)
    (args.out / "family-grain-agreement.json").write_text(json.dumps(report, indent=2) + "\n")
    lines += ["", f"Pairs by genre: {dict(by_genre)}; by printed scope: {dict(by_relation)}. A manual usually covers a superset of the datasheet's parts, so a count it gives is the family maximum; disagreements on counts under 'overlap' or 'disjoint_as_printed' are mostly that.", "", "## Pairs", "", "| vendor | family | a | b | genres | scope | compared | agree |", "|---|---|---|---|---|---|---:|---:|"]
    for p in pairs:
        lines.append(f"| {p['vendor']} | {p['family_key']} | {p['a'][:48]} | {p['b'][:48]} | {'/'.join(p['genres'])} | {p['scope_relation']} | {p['compared']} | {p['agree']} |")
    (args.out / "family-grain-agreement.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print(f"pairs: {len(pairs)}  adjudication: {len(adjudication)}  -> {args.out}")


if __name__ == "__main__":
    main()
