"""Power datasheet reader: electrical characteristics tables to typed facts.

A power part (MOSFET, regulator, gate driver, ...) is described by tables whose
header row names the axes -- Symbol / Parameter / Test conditions / Min / Typ /
Max / Unit -- and whose rows name the quantity. This module reads those tables
with the page geometry, not the text stream, because the text stream loses
exactly the two things that matter:

  * subscripts (``V`` + small ``GS`` -> ``VGS``) fall to the end of the line, so
    ``VGS = 0 V, ID = 250 uA`` reads as ``V = 0 V, I = 250 uA GS D``;
  * merged cells (``1.7 2.0 2.4`` in one MIN cell) hide which column each
    number sat in.

Both are recovered from span positions and font sizes: every span is placed
in the header column whose x-range contains its centre, and spans whose font is
smaller than the row's body font are glued to the token before them.

The emitted grid has the same shape as the MCU Key Features grid
(``harness.electronics-key-features-grid.v1``) so the same gold scorer runs on
it unchanged; the groups follow the device class (CR power-gold-fixtures-
20260909): ``switching``, ``regulation``, ``package_environment``, with
``thermal`` and ``other`` for facts the fixtures do not ask about yet.

Three independent axes on every row, per the agreed contract:
``quantity_qualifier`` (which bound: minimum/typical/maximum/rated/
absolute_maximum), ``qualifier_verbatim`` (the document's own hedge), and
``condition_verbatim`` (the test condition as printed, ``VGS = 10 V``).
Nothing is interpreted; a range is kept as ``[lo, hi]``.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "harness.electronics-key-features-grid.v1"
READER = "power_datasheet.v1"

GROUPS: tuple[tuple[str, str], ...] = (
    ("switching", "Switching"),
    ("regulation", "Regulation"),
    ("package_environment", "Package & environment"),
    ("thermal", "Thermal"),
    ("other", "Other characteristics"),
)

# Header vocabulary -> column role.
_ROLE = (
    (re.compile(r"^\s*(?:test\s+)?conditions?\s*(?:\(\d\))?\s*$|^\s*test\s+condition", re.I), "conditions"),
    (re.compile(r"^\s*parameter", re.I), "parameter"),
    (re.compile(r"^\s*(?:symbol|sym\.?)\s*$", re.I), "symbol"),
    (re.compile(r"^\s*min(?:imum)?\.?\s*(?:\(\d\))?\s*$", re.I), "min"),
    (re.compile(r"^\s*(?:typ(?:ical)?\.?|nom(?:inal)?\.?|typical\s+value)\s*(?:\(\d\))?\s*$", re.I), "typ"),
    (re.compile(r"^\s*max(?:imum)?\.?\s*(?:\(\d\))?\s*$", re.I), "max"),
    (re.compile(r"^\s*(?:value|rating|ratings)s?\s*$", re.I), "value"),
    (re.compile(r"^\s*units?\b", re.I), "unit"),
    (re.compile(r"\blimits?\b", re.I), "limit"),  # "LM2577-12 Limit": a bound whose direction the row decides
    (re.compile(r"^\s*thermal\s+metric", re.I), "parameter"),
)
_VALUE_ROLES = ("min", "typ", "max", "value", "limit", "value_unit")
_VALUE_UNIT = re.compile(r"^[-−–+±]?\s?\d+(?:\.\d+)?\s?[pnuµμmkM]?(?:Ω|Ohm|V|A|W|°C|ºC|C|Hz|s|F|H|J|%)(?:\s*(?:to|–|-|≤\s*[A-Za-z()/_]+\s*≤)\s*[-−–+]?\d+(?:\.\d+)?\s?[pnuµμmkM]?(?:Ω|Ohm|V|A|W|°C|ºC|C|Hz|s|F|H|J|%))?(?:\s*\(\d\))?$")
_VALUE_UNIT_UPPER = re.compile(r"^[A-Za-z()/_]+\s*≤\s*[-−–+]?\d+(?:\.\d+)?\s?[pnuµμmkM]?(?:Ω|Ohm|V|A|W|°C|ºC|C|Hz|s|F|H|J|%)(?:\s*\(\d\))?$")  # "ISWITCH ≤ 3.0A"
_HEADER_CONDITION = re.compile(r"\b(T[AJC]|V[A-Z]{1,3})\s*=\s*[-+]?\d", re.I)

_TITLE_ABS_MAX = re.compile(r"absolute\s+maximum|abs\.?\s*max|maximum\s+ratings", re.I)
_TITLE_RECOMMENDED = re.compile(r"recommended\s+operating|operating\s+conditions|operating\s+range|operating\s+ratings", re.I)
_TITLE_THERMAL = re.compile(r"thermal\s+(?:information|characteristics|resistance|data)", re.I)
_TITLE_SUMMARY = re.compile(r"product\s+summary|key\s+(?:performance\s+)?(?:specifications|parameters)|summary", re.I)
_TITLE_EC = re.compile(r"electrical\s+characteristics|electrical\s+specifications|static\s+characteristics|dynamic\s+characteristics|gate\s+charge\s+characteristics|characteristics", re.I)
_TITLE_ORDER = re.compile(r"ordering|package\s+information|packaging|device\s+information|revision\s+history|pin\s+(?:functions?|configuration)", re.I)

_SYMBOL_LEAD = re.compile(r"^\s*([A-Z][A-Za-z]{0,3}(?:\([A-Za-z0-9. ]{1,8}\))?(?:[A-Za-z0-9]{0,4}))\s+(?=[A-Z(])")
_NUMBER = re.compile(r"^[-–−+±]?\s?\d+(?:[.,]\d+)?$")
_RANGE = re.compile(r"^([-–−+]?\d+(?:\.\d+)?)\s*(?:to|–|-|—|\.\.\.)\s*([-–−+]?\d+(?:\.\d+)?)$")
_UNIT_TOKEN = re.compile(r"^(?:[pnuµμmkMG]?(?:Ω|Ohm|ohm|V|A|C|W|F|H|Hz|s|J|C/W|°C|°C/W|%|nC|dB|ppm/°C|mV/V|V/µs|A/µs))$")

# Symbol -> group. Order matters; first match wins. Symbols are compared with
# subscripts glued and case-folded, e.g. "rds(on)", "v(br)dss", "tj", "vin".
_GROUP_BY_SYMBOL: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:v\(?br\)?dss|vdss|vds|vdsmax|bvdss|vgs|vgs\(th\)|vgsth|vsd|id|idm|ids|is|ism|rds\(?on\)?|rdson|qg|qgs|qgd|qgtot|qoss|qrr|ciss|coss|crss|gfs|td\(?on\)?|td\(?off\)?|tr|tf|trr|eas|iar|ear|pd)$"), "switching"),
    (re.compile(r"^(?:vin|vi|vcc|vdd|vbat|vsupply|vout|vo|vref|vfb|iout|io|iload|iq|ishdn|isd|isupply|fsw|fosc|f|iin|vdo|vdrop|vuvlo|vovp|ilim|iocp|ton|toff|dmax|eff|η|vripple|psrr|vline|vload)$"), "regulation"),
    (re.compile(r"^(?:ta|tj|tstg|tstorage|tl|top|tamb|tc|tcase|θja|rθja|rthja|θjc|rθjc|rthjc|θjb|rθjb|ψjt|ψjb|rθ|rth|rthjc\(?top\)?|rθjc\(?top\)?|rθjc\(?bot\)?)$"), "package_environment"),
)
_PARAM_GROUP: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"drain|gate\s+charge|on[- ]?resistance|avalanche|reverse\s+recovery|capacitance|source|operating\s+current", re.I), "switching"),
    (re.compile(r"input\s+voltage|output\s+voltage|output\s+current|quiescent|switching\s+frequency|dropout|reference|feedback|current\s+limit|undervoltage|overvoltage|efficiency|line\s+regulation|load\s+regulation", re.I), "regulation"),
    (re.compile(r"junction|ambient|storage|operating\s+(?:free-air\s+)?temperature|lead\s+temperature|case\s+temperature", re.I), "package_environment"),
    (re.compile(r"thermal", re.I), "thermal"),
)
_THERMAL_SYMBOL = re.compile(r"^(?:r|ψ|θ)?(?:θ|th|ψ)", re.I)


def _norm_symbol(symbol: str) -> str:
    s = symbol.strip().lower().replace(" ", "").replace("_", "")
    s = re.sub(r"\*\d+", "", s)
    s = s.replace("/", "").replace("*", "")
    s = s.replace("θ", "θ").replace("ψ", "ψ")
    return s


_UNIT_BY_SYMBOL: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:vds|vdss|vgs|vgss|v\(br\)dss)$"), "V"),
    (re.compile(r"^(?:id|idp|idm|is|isp)$"), "A"),
    (re.compile(r"^(?:pd)$"), "W"),
    (re.compile(r"^(?:tj|ta|tstg|tc)$"), "°C"),
    (re.compile(r"^rds"), "mΩ"),
    (re.compile(r"^qg(?:s|d)?$"), "nC"),
)


def _unit_from_symbol(symbol: str) -> str | None:
    sym = _norm_symbol(symbol)
    for pattern, unit in _UNIT_BY_SYMBOL:
        if pattern.match(sym):
            return unit
    return None


_CANONICAL: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"junction\s+temperature|operating\s+junction", re.I), "TJ"),
    (re.compile(r"free-air\s+temperature|ambient\s+temperature|operating\s+temperature", re.I), "TA"),
    (re.compile(r"storage\s+temperature", re.I), "Tstg"),
    (re.compile(r"\binput\s+voltage|supply\s+voltage", re.I), "VIN"),
    (re.compile(r"\boutput\s+voltage", re.I), "VOUT"),
    (re.compile(r"\boutput\s+current|load\s+current", re.I), "IOUT"),
    (re.compile(r"drain[- ]to[- ]source\s+on[- ]?(?:state\s+)?resistance|on[- ]resistance", re.I), "RDS(on)"),
    (re.compile(r"drain[- ]to[- ]source\s+(?:breakdown\s+)?voltage|drain[- ]source\s+voltage|vdss\b", re.I), "VDS"),
    (re.compile(r"continuous\s+(?:dc\s+)?drain(?:[- ]to[- ]drain)?\s+current|drain[- ]to[- ]drain\s+current|operating\s+current", re.I), "ID"),
    (re.compile(r"total\s+gate\s+charge|gate\s+charge\s+total", re.I), "Qg"),
)


def canonical_symbols(parameter: str) -> list[str]:
    """Normalised symbol beside the vendor's words: "Junction Temperature Range" -> TJ."""
    return [sym for pattern, sym in _CANONICAL if pattern.search(parameter or "")]


def classify(symbol: str, parameter: str, section: str) -> str:
    if not symbol.strip() and parameter and " " not in parameter.strip() and len(parameter.strip()) <= 10:
        symbol, parameter = parameter, ""  # the symbol landed in the parameter column
    sym = _norm_symbol(symbol)
    if sym and _THERMAL_SYMBOL.match(sym) and not sym.startswith(("tj", "ta", "tstg", "tc")):
        return "thermal"  # RθJA and friends: resistance, not a rating
    for pattern, group in _GROUP_BY_SYMBOL:
        if sym and pattern.match(sym):
            return group
    # "Operating junction temperature range, TJ": the symbol trails the text.
    trailing = re.search(r",\s*([A-Za-z]{1,2}[A-Za-z()]{0,6})\s*$", symbol + " " + parameter)
    if trailing:
        for pattern, group in _GROUP_BY_SYMBOL:
            if pattern.match(_norm_symbol(trailing.group(1))):
                return group
    hay = f"{symbol} {parameter} {section}"
    for pattern, group in _PARAM_GROUP:
        if pattern.search(hay):
            return group
    return "other"


# ---------------------------------------------------------------------------
# Span geometry
# ---------------------------------------------------------------------------

def _page_spans(page: Any) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"]
                if not text.strip():
                    continue
                x0, y0, x1, y1 = span["bbox"]
                spans.append({"text": text, "size": round(span["size"], 1), "x0": x0, "x1": x1, "y0": y0, "y1": y1, "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2})
    spans.sort(key=lambda s: (round(s["cy"]), s["x0"]))
    return spans


def _lines(spans: list[dict[str, Any]], tol: float = 2.5) -> list[list[dict[str, Any]]]:
    """Cluster spans into visual lines by centre-y; subscripts sit ~2pt low and
    must land on the line of the token they belong to."""
    lines: list[list[dict[str, Any]]] = []
    for span in sorted(spans, key=lambda s: (s["cy"], s["x0"])):
        for line in lines:
            ref = line[0]
            if abs(span["cy"] - ref["cy"]) <= tol or (span["size"] < ref["size"] and ref["y0"] - 1 <= span["cy"] <= ref["y1"] + 2):
                line.append(span)
                break
        else:
            lines.append([span])
    for line in lines:
        line.sort(key=lambda s: s["x0"])
    return lines


def _render(line: list[dict[str, Any]]) -> tuple[str, str]:
    """(glued, marked): 'VGS = 0 V' and 'V_{GS} = 0 V'."""
    if not line:
        return "", ""
    body = max(s["size"] for s in line)
    glued: list[str] = []
    marked: list[str] = []
    prev_x1 = None
    for span in line:
        text = span["text"]
        sub = span["size"] <= body - 1.0
        gap = (span["x0"] - prev_x1) if prev_x1 is not None else 0.0
        if sub:
            glued.append(text.strip())
            marked.append("_{" + text.strip() + "}")
        else:
            if prev_x1 is not None and gap > 1.5 and not text.startswith(" ") and glued and not glued[-1].endswith(" "):
                glued.append(" ")
                marked.append(" ")
            glued.append(text)
            marked.append(text)
        prev_x1 = span["x1"]
    g = re.sub(r"\s+", " ", "".join(glued)).strip()
    m = re.sub(r"\s+", " ", "".join(marked)).strip()
    return g, m


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _roles_for_header(names: list[str], cells: list[tuple | None]) -> tuple[dict[int, str], str | None]:
    """Column index -> role; and a table-level condition printed in the header
    ("TA = 25°C" standing where SYMBOL/PARAMETER would be)."""
    roles: dict[int, str] = {}
    header_condition = None
    for i, name in enumerate(names):
        name = (name or "").replace("\n", " ").strip()
        if not name:
            continue
        if _HEADER_CONDITION.search(name) and not any(p.search(name) for p, _ in _ROLE):
            header_condition = name
            continue
        for pattern, role in _ROLE:
            if pattern.search(name):
                roles[i] = role
                break
    # Dual-FET: one header cell is "Q1 Control FET MIN TYP MAX". The numbers
    # sit in that cell; treat it as the first value group so BVDSS is not lost.
    if "unit" in roles.values() and not (set(roles.values()) & set(_VALUE_ROLES)):
        for i, name in enumerate(names):
            if re.search(r"\bQ1\b|control\s+fet", name or "", re.I) or re.search(r"\bmin\b.*\btyp\b.*\bmax\b", name or "", re.I):
                roles[i] = "min"
                break
    return roles, header_condition


def _is_characteristics_header(roles: dict[int, str]) -> bool:
    values = set(roles.values())
    # ROHM abs-max is Symbol | Value with the unit implied by the symbol (VDSS, ID).
    return ("unit" in values and bool(values & set(_VALUE_ROLES))) or "value_unit" in values or (
        "symbol" in values and "value" in values
    )


_HEADER_WORD = re.compile(r"^(?:parameter|symbol|conditions?|values?|min|typ|max|units?|characteristics)$", re.I)
_DATA_ROW_SYMBOL = re.compile(
    r"^(?:Q/?g(?:s|d)?|C/?(?:iss|oss|rss))",
    re.I,
)


def _header_looks_like_data_row(names: list[str]) -> bool:
    """find_tables promoted the first data row of a continuation table
    (IR Hexfet Qg block, OptiMOS Gate Charge after Ciss) to header."""
    cleaned = [_normalize_pdf_text((n or "").replace("\n", "/")).strip() for n in names]
    cleaned = [c for c in cleaned if c]
    if len(cleaned) < 3 or _HEADER_WORD.match(cleaned[0]):
        return False
    nums = 0
    for n in cleaned:
        parsed = _parse_number(n)
        if parsed is not None:
            nums += 1
        else:
            nums += len(re.findall(r"[-+]?\d+(?:\.\d+)?", n))
    has_sym = any(_DATA_ROW_SYMBOL.search(re.sub(r"[\s\n]+", "", n)) for n in cleaned[:2])
    return has_sym and nums >= 1


def _looks_like_symbol(text: str) -> bool:
    """A cell is a datasheet symbol (Qg, Ciss), not a phrase (Gate charge total)."""
    raw = _normalize_pdf_text((text or "").replace("\n", "/")).strip()
    compact = re.sub(r"[\s/]+", "", raw)
    if _DATA_ROW_SYMBOL.search(compact):
        return True
    return bool(compact) and " " not in raw and len(compact) <= 14 and re.match(r"^[A-Za-z]", compact)


def _headers_align(prev: list[tuple[float, float, str]], curr: list[tuple[float, float, str]], tol: float = 10.0) -> bool:
    """Same x-span as the previous characteristics table on this page."""
    if not prev or not curr:
        return False
    return abs(prev[0][0] - curr[0][0]) <= tol and abs(prev[-1][1] - curr[-1][1]) <= tol


def _header_looks_fragmented(names: list[str]) -> bool:
    """find_tables splits 'PARAMETER' into 'PA'+'RAMETER' and a Q1/Q2 MOSFET
    header into a dozen one- and two-letter cells. Those headers assign the
    wrong x-ranges to MIN/TYP/MAX and drop BVDSS."""
    short = sum(1 for n in names if n and 0 < len(n.strip()) <= 3)
    return len(names) >= 8 and short >= 4


_TITLE_ANY = re.compile(r"ratings?|characteristics|conditions|specifications|summary|thermal|information|parameters", re.I)
_TITLE_ESD = re.compile(r"\bESD\b|electrostatic", re.I)


def _title_above(page_lines: list[list[dict[str, Any]]], bbox: tuple[float, float, float, float]) -> str:
    """Nearest line above the table that reads like a table title; footnotes of
    the previous table are skipped by preferring a line the vocabulary knows.
    Returns the chosen title, with nearer non-title lines appended after ' | '
    so a subtitle such as "(TJ = 25°C unless otherwise noted)" is kept."""
    x0, y0, x1, _ = bbox
    candidates = []
    for line in page_lines:
        cy = line[0]["cy"]
        if y0 - 150 <= cy < y0 - 1 and line[-1]["x1"] > x0 - 5 and line[0]["x0"] < x1 + 5:
            text, _ = _render(line)
            if text and not _NUMBER.match(text):
                candidates.append((cy, text))
    candidates.sort(reverse=True)
    nearest = [t for _, t in candidates[:7]]
    for i, text in enumerate(nearest):
        if _kind_of(text) is not None and len(text) < 160:
            return " | ".join([text, *nearest[:i]])
    return nearest[0] if nearest else ""


def _kind_of(title: str) -> str | None:
    if _TITLE_ESD.search(title):
        return "esd"
    if _TITLE_ABS_MAX.search(title):
        return "absolute_maximum"
    if _TITLE_RECOMMENDED.search(title):
        return "recommended"
    if _TITLE_THERMAL.search(title):
        return "thermal"
    if _TITLE_SUMMARY.search(title):
        return "summary"
    if _TITLE_EC.search(title):
        return "characteristics"
    return None


def _table_kind(title: str) -> str:
    for part in title.split(" | "):
        kind = _kind_of(part)
        if kind is not None:
            return kind
    return "characteristics"


def _header_columns(table: Any, spans: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    """Header columns as (x0, x1, name), merging cells that one header span
    straddles ("TYPICAL VA" + "LUE" -> "TYPICAL VALUE")."""
    cells = [c for c in table.header.cells if c]
    if not cells:
        return []
    hy0 = min(c[1] for c in cells)
    hy1 = max(c[3] for c in cells)
    head_spans = [s for s in spans if hy0 - 1 <= s["cy"] <= hy1 + 1 and table.bbox[0] - 2 <= s["cx"] <= table.bbox[2] + 2]
    groups: list[list[tuple]] = [[c] for c in sorted(cells, key=lambda c: c[0])]
    for s in head_spans:
        hit = [gi for gi, g in enumerate(groups) if min(s["x1"], max(c[2] for c in g)) - max(s["x0"], min(c[0] for c in g)) > 2.0]
        if len(hit) > 1:
            merged = [c for gi in hit for c in groups[gi]]
            groups = [g for gi, g in enumerate(groups) if gi not in hit]
            groups.append(merged)
            groups.sort(key=lambda g: min(c[0] for c in g))
    columns = []
    for g in groups:
        x0, x1 = min(c[0] for c in g), max(c[2] for c in g)
        inside = sorted([s for s in head_spans if x0 - 0.5 <= s["cx"] <= x1 + 0.5], key=lambda s: s["x0"])
        # One cell carrying several role words ("MAX UNIT") is several columns
        # whose rule the table detector missed; split at the span midpoints.
        role_spans = [s for s in inside if any(p.search(s["text"].strip()) for p, _ in _ROLE)]
        if len(role_spans) >= 2 and len(role_spans) == len([s for s in inside if s["text"].strip()]):
            for k, s in enumerate(role_spans):
                left = x0 if k == 0 else (role_spans[k - 1]["x1"] + s["x0"]) / 2
                right = x1 if k == len(role_spans) - 1 else (s["x1"] + role_spans[k + 1]["x0"]) / 2
                columns.append((left, right, s["text"].strip()))
            continue
        name = " ".join(t for t, _ in (_render(l) for l in _lines(inside))).strip()
        columns.append((x0, x1, name))
    return columns


def _column_edges(table: Any) -> list[float]:
    edges: set[float] = set()
    for row in table.rows:
        for cell in row.cells:
            if cell:
                edges.add(round(cell[0], 1))
                edges.add(round(cell[2], 1))
    return sorted(edges)


def _split_values_min_typ_max(
    table: Any,
    spans: list[dict[str, Any]],
    header: list[tuple[float, float, str]],
    roles: dict[int, str],
    data_start: int,
) -> tuple[list[tuple[float, float, str]], dict[int, str], int]:
    """Infineon 'Values' over a second header row Min. / Typ. / Max."""
    value_idxs = [i for i, role in roles.items() if role == "value"]
    if len(value_idxs) != 1:
        return header, roles, data_start
    vi = value_idxs[0]
    x0, x1, name = header[vi]
    if not re.search(r"^values?$", name.strip(), re.I):
        return header, roles, data_start
    sub, nxt = _row_as_header(table, spans, data_start)
    found: dict[str, tuple[float, float, str]] = {}
    for sx0, sx1, subname in sub:
        cx = (sx0 + sx1) / 2
        if not (x0 - 4 <= cx <= x1 + 4):
            continue
        if re.match(r"^\s*min", subname, re.I):
            found["min"] = (sx0, sx1, subname)
        elif re.match(r"^\s*typ", subname, re.I):
            found["typ"] = (sx0, sx1, subname)
        elif re.match(r"^\s*max", subname, re.I):
            found["max"] = (sx0, sx1, subname)
    if "min" not in found or "max" not in found:
        return header, roles, data_start
    pieces = [found[k] for k in ("min", "typ", "max") if k in found]
    new_header = header[:vi] + pieces + header[vi + 1:]
    new_roles, _ = _roles_for_header([n for _, _, n in new_header], [])
    return new_header, new_roles, nxt


_MULTI_ROLE_WORD = re.compile(r"min(?:imum)?\.?|typ(?:ical)?\.?|max(?:imum)?\.?|units?", re.I)

_INLINE_CONDITION = re.compile(
    r"\b(?:VGS|VDS|VCE|VGE|VEE|VCC|VBE|VIN|VOUT|VDD|VSS|VBS"
    r"|ID|IC|IE|TJ|Tvj|TC|TA|fpw|tr|tf|tp|tw)"
    r"(?:\s*\(\s*[A-Za-z0-9.]+\s*\))?"
    r"\s*=\s*[-+]?\d+(?:[.,]\d+)?"
    r"(?:\s*(?:m|k|M|µ|u|n)?\s*(?:V|A|W|°C|C|Hz|kHz|MHz|s|ms|µs|us|ns))?"
    r"(?:\s*[,;]\s*[-+]?\d+(?:[.,]\d+)?"
    r"(?:\s*(?:m|k|M|µ|u|n)?\s*(?:V|A|W|°C|C|Hz|kHz|MHz|s|ms|µs|us|ns))?)?",
    re.I,
)


def _inline_conditions(
    *texts: str | None,
    excluded_symbol: str | None = None,
) -> str | None:
    """Conditions printed inside a parameter cell, not a conditions column.

    Infineon OptiMOS/IAUC EC tables state the test point inside the
    parameter text ("... on-state resistance VGS = 20 V" / "ID = 1 A,
    Tvj = 25 C"). Captured verbatim, deduplicated in print order. A
    condition naming the row's own symbol is the row's value context,
    not a condition, and is skipped.
    """

    own = re.sub(r"[^A-Za-z]", "", excluded_symbol or "").upper()
    found: list[str] = []
    for text in texts:
        if not text:
            continue
        for match in _INLINE_CONDITION.finditer(text):
            token = " ".join(match.group(0).split())
            head = re.sub(r"[^A-Za-z]", "", token.split("=")[0]).upper()
            if own and head and head.startswith(own):
                continue
            if token not in found:
                found.append(token)
    return "; ".join(found) if found else None


def _split_multi_role_header(name: str, x0: float, x1: float) -> list[tuple[float, float, str]]:
    """'Typ. Max. Units' in one cell (IR Hexfet) is three header columns."""
    parts = list(_MULTI_ROLE_WORD.finditer(name or ""))
    if len(parts) < 2:
        return [(x0, x1, name)]
    width = x1 - x0
    n = max(len(name), 1)
    out = []
    for i, match in enumerate(parts):
        left = x0 + width * match.start() / n
        right = x0 + width * (parts[i + 1].start() if i + 1 < len(parts) else n) / n
        out.append((left, right, match.group(0)))
    return out


def _row_as_header(table: Any, spans: list[dict[str, Any]], row_index: int) -> tuple[list[tuple[float, float, str]], int]:
    """Read table.rows[row_index] as a header row: (columns, next data row)."""
    if row_index >= len(table.rows):
        return [], row_index
    row = table.rows[row_index]
    cells = [c for c in row.cells if c]
    if not cells:
        return [], row_index
    ry0, ry1 = row.bbox[1], row.bbox[3]
    in_row = [s for s in spans if ry0 - 1 <= s["cy"] <= ry1 + 1 and table.bbox[0] - 2 <= s["cx"] <= table.bbox[2] + 2]
    columns = []
    for c in sorted(cells, key=lambda c: c[0]):
        inside = [s for s in in_row if c[0] - 0.5 <= s["cx"] <= c[2] + 0.5]
        name = " ".join(t for t, _ in (_render(l) for l in _lines(inside))).strip()
        columns.append((c[0], c[2], name))
    return columns, row_index + 1


def _infer_columns_by_content(table: Any, spans: list[dict[str, Any]], data_start: int) -> tuple[list[tuple[float, float, str]], dict[int, str], int]:
    """No usable header: columns are the union of cell edges, and a column is
    the value column when most of its cells are numbers, the unit column when
    most are unit tokens, the parameter column when it is the leftmost text."""
    edges = _column_edges(table)
    if len(edges) < 3:
        return [], {}, data_start
    columns = [(edges[i], edges[i + 1], "") for i in range(len(edges) - 1)]
    rows = table.rows[data_start:]
    stats = [[0, 0, 0, 0] for _ in columns]  # numeric, unit, text, value+unit
    for row in rows:
        ry0, ry1 = row.bbox[1], row.bbox[3]
        in_row = [s for s in spans if ry0 - 1 <= s["cy"] <= ry1 + 1]
        for ci, (x0, x1, _) in enumerate(columns):
            inside = [s for s in in_row if x0 - 0.5 <= s["cx"] <= x1 + 0.5]
            if not inside:
                continue
            text, _ = _render(sorted(inside, key=lambda s: s["x0"]))
            if _parse_number(text) is not None or all(_parse_number(p) is not None for p in text.split() if p):
                stats[ci][0] += 1
            elif _UNIT_TOKEN.match(text.strip()) or re.fullmatch(r"[pnuµμmkM]?(?:Ω|Ohm|V|A|W|°C|C|Hz|s|F|H|J|%|dB|C/W|°C/W)(?:\s*/\s*[°]?[CW])?", text.strip()):
                stats[ci][1] += 1
            elif _VALUE_UNIT.match(text.strip()) or _VALUE_UNIT_UPPER.match(text.strip()):
                stats[ci][3] += 1
            else:
                stats[ci][2] += 1
    roles: dict[int, str] = {}
    n = max(1, len(rows))
    value_cols = [ci for ci, (num, _, _, _) in enumerate(stats) if num / n >= 0.4]
    unit_cols = [ci for ci, (_, unit, _, _) in enumerate(stats) if unit / n >= 0.3]
    vu_cols = [ci for ci, (_, _, _, vu) in enumerate(stats) if vu / n >= 0.4]
    if not value_cols and vu_cols:
        # "−0.3V to 32V" in one cell: value and unit together, no unit column.
        text_cols = [ci for ci, (num, unit, text, vu) in enumerate(stats) if ci not in vu_cols and text > 0]
        roles[vu_cols[0]] = "value_unit"
        if text_cols:
            roles[text_cols[0]] = "parameter"
            for ci in text_cols[1:]:
                roles[ci] = "conditions"
        return columns, roles, data_start
    if not value_cols:
        return [], {}, data_start
    if not unit_cols:
        # ROHM abs-max: Symbol | Value, unit implied by the symbol (VDSS, ID).
        text_cols = [ci for ci, (num, unit, text, _) in enumerate(stats) if ci not in value_cols and text > 0]
        roles[value_cols[0]] = "value"
        if text_cols:
            roles[text_cols[0]] = "symbol"
        return columns, roles, data_start
    text_cols = [ci for ci, (num, unit, text, _) in enumerate(stats) if ci not in value_cols and ci not in unit_cols and text > 0]
    if len(value_cols) == 1:
        roles[value_cols[0]] = "value"
    elif len(value_cols) == 2:
        roles[value_cols[0]], roles[value_cols[1]] = "min", "max"
    else:
        roles[value_cols[0]], roles[value_cols[1]], roles[value_cols[2]] = "min", "typ", "max"
    roles[unit_cols[-1]] = "unit"
    if text_cols:
        roles[text_cols[0]] = "parameter"
        for ci in text_cols[1:]:
            roles[ci] = "conditions" if ci < min(value_cols) else f"col{ci}"
    return columns, roles, data_start


def _parse_number(text: str) -> float | list[float] | None:
    t = _normalize_pdf_text(text).strip().replace(",", "")
    t = re.sub(r"\(\d\)$", "", t).strip()
    m = _RANGE.match(t)
    if m:
        return [float(m.group(1)), float(m.group(2))]
    if _NUMBER.match(t):
        t = t.replace("±", "").replace(" ", "")
        try:
            return float(t)
        except ValueError:
            return None
    # Infineon Values cell before the Min/Typ/Max split: "‑ ‑ 750".
    parts = [p for p in t.replace("±", "").split() if p not in {"-", "‑", "—", "–", "."}]
    if len(parts) == 1 and _NUMBER.match(parts[0]):
        try:
            return float(parts[0].replace(" ", ""))
        except ValueError:
            return None
    return None


def read_characteristic_tables(document: Any) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for pno in range(document.page_count):
        page = document[pno]
        try:
            tables = page.find_tables().tables
        except Exception:  # pragma: no cover - PyMuPDF internals
            continue
        if not tables:
            continue
        spans = _page_spans(page)
        page_lines = _lines(spans)
        for table in tables:
            if table.header is None or not table.header.names:
                continue
            header = _header_columns(table, spans)
            if not header:
                continue
            roles, header_condition = _roles_for_header([n for _, _, n in header], [(x0, 0, x1, 0) for x0, x1, _ in header])
            title_in_table = None
            data_start = 1 if not table.header.external else 0
            if _header_looks_fragmented([n for _, _, n in header]):
                roles = {}
            if not _is_characteristics_header(roles):
                # Older TI layout: the title is the table's first row
                # ("ABSOLUTE MAXIMUM RATINGS" spanning every column) and the
                # column header is the next row, or there is none and the
                # columns are told apart by what they hold. The same fallback
                # runs when the page title above says this is a
                # characteristics table but the detector's header row is not.
                names = [n for _, _, n in header if n.strip()]
                in_table = len(names) == 1 and (_kind_of(re.sub(r"^\(\d\)\s*", "", names[0])) or re.match(r"over\s+operating", names[0], re.I))
                above = _title_above(page_lines, table.bbox)
                # Infineon OptiMOS/IAUC: find_tables names the header as the
                # part number; the real Parameter/Symbol/Min/Typ/Max row is
                # gone (continuation of Dynamic / Gate Charge). Infer by cell
                # contents instead of skipping the page.
                part_number_header = (
                    len(names) == 1
                    and not _kind_of(names[0])
                    and len(table.rows) >= 8
                    and len(names[0]) <= 48
                )
                # IR Hexfet / OptiMOS page-5: find_tables starts a new table on
                # Qg / Qgs / Ciss; the previous table on this page already had
                # Parameter|Min|Typ|Max or Symbol|Conditions|Values.
                data_row_header = _header_looks_like_data_row(names)
                if in_table:
                    title_in_table = re.sub(r"^\(\d\)\s*", "", names[0])
                elif part_number_header:
                    pass
                elif data_row_header:
                    data_start = 0
                elif not (_kind_of(above.split(" | ")[0]) and len(table.rows) >= 3):
                    continue
                if in_table:
                    header2, next_start = _row_as_header(table, spans, data_start)
                    if header2:
                        roles2, hc2 = _roles_for_header([n for _, _, n in header2], [(x0, 0, x1, 0) for x0, x1, _ in header2])
                        if _is_characteristics_header(roles2):
                            header, roles, header_condition, data_start = header2, roles2, hc2, next_start
                if not _is_characteristics_header(roles):
                    header, roles, data_start = _infer_columns_by_content(table, spans, data_start)
                if not header or not _is_characteristics_header(roles):
                    continue
            title = title_in_table or _title_above(page_lines, table.bbox)
            if _TITLE_ORDER.search(title) and not _TITLE_EC.search(title):
                continue
            header, roles, data_start = _split_values_min_typ_max(table, spans, header, roles, data_start)
            if not _is_characteristics_header(roles):
                continue
            kind = _table_kind(title)
            if kind == "esd":
                continue
            # Columns left of the first value column that the header does not
            # name (or names with a table-level condition such as "TA = 25°C")
            # are the symbol and parameter columns, in that order; a single one
            # is the parameter.
            first_value_x = min([header[i][0] for i in roles if roles[i] in _VALUE_ROLES], default=None)
            if first_value_x is None:
                continue
            unnamed = [i for i, (x0, _, _) in enumerate(header) if x0 < first_value_x and roles.get(i) not in ("symbol", "parameter", "conditions", "unit")]
            has_symbol = "symbol" in roles.values()
            has_parameter = "parameter" in roles.values()
            parameter_x = min([header[i][0] for i in roles if roles[i] == "parameter"], default=None)
            for k, i in enumerate(unnamed):
                cell_x = header[i][0]
                if not has_symbol and k == 0 and (len(unnamed) > 1 or (has_parameter and parameter_x is not None and cell_x < parameter_x)):
                    roles[i] = "symbol"
                    has_symbol = True
                else:
                    roles[i] = "parameter"  # unnamed cells are parameter text
            columns: list[tuple[float, float, str]] = [(x0, x1, roles.get(i) or f"col{i}") for i, (x0, x1, _) in enumerate(header)]

            def column_of(x: float) -> str | None:
                for x0, x1, role in columns:
                    if x0 - 0.5 <= x <= x1 + 0.5:
                        return role
                return None

            section = ""
            last_unit = None
            last_symbol = ""
            last_parameter = ""
            data_rows = table.rows[data_start:]
            consumed: set[int] = set()
            for row in data_rows:
                ry0, ry1 = row.bbox[1], row.bbox[3]
                in_row = [s for s in spans if id(s) not in consumed and ry0 - 1.0 <= s["cy"] <= ry1 + 1.0 and table.bbox[0] - 2 <= s["cx"] <= table.bbox[2] + 2]
                if not in_row:
                    continue
                consumed.update(id(s) for s in in_row)
                by_role: dict[str, list[dict[str, Any]]] = {}
                for s in in_row:
                    role = column_of(s["cx"])
                    if role is None:
                        continue
                    by_role.setdefault(role, []).append(s)
                # A section row: one text run spanning the table with no numbers in value columns.
                has_value = any(r in by_role for r in _VALUE_ROLES)
                non_empty_cells = [c for c in row.cells if c]
                if not has_value and len(non_empty_cells) <= 1:
                    text, _ = _render(sorted(in_row, key=lambda s: s["x0"]))
                    if text and len(text) < 80:
                        section = text
                    continue
                if not has_value:
                    continue

                def col_lines(role: str) -> list[tuple[str, str]]:
                    return [_render(l) for l in _lines(by_role.get(role, []))]

                if "symbol" not in by_role and by_role.get("parameter"):
                    # One merged column holding both. The symbol hugs the left
                    # edge and the parameter text is indented; when both
                    # indentations occur in the row, the left-edge spans are
                    # the symbol.
                    col_x0 = next((x0 for x0, _, role in columns if role == "parameter"), None)
                    if col_x0 is not None:
                        left = [s for s in by_role["parameter"] if s["x0"] <= col_x0 + 8]
                        rest = [s for s in by_role["parameter"] if s["x0"] > col_x0 + 8]
                        body_rest = [s["x0"] for s in rest if s["size"] >= max((x["size"] for x in left), default=0) - 0.5] if left else []
                        if left and body_rest and min(body_rest) - col_x0 >= 25:
                            # Subscripts of the symbol sit just right of it; pull them back.
                            sym_x1 = max(s["x1"] for s in left)
                            subs = [s for s in rest if s["x0"] <= sym_x1 + 3 and s["size"] < max(x["size"] for x in left)]
                            by_role["symbol"] = left + subs
                            by_role["parameter"] = [s for s in rest if s not in subs]
                sym_lines = col_lines("symbol")
                par_lines = col_lines("parameter")
                symbol = " ".join(t for t, _ in sym_lines).strip()
                symbol_marked = " ".join(m for _, m in sym_lines).strip()
                parameter = " ".join(t for t, _ in par_lines).strip()
                if symbol and " " in symbol and len(symbol) > 14 and len(parameter) <= 14:
                    # TI's newer layout puts the parameter text in the first
                    # column and the symbol (or pin list) in the second.
                    symbol, parameter = parameter, symbol
                    symbol_marked = symbol
                if not symbol and parameter:
                    m = _SYMBOL_LEAD.match(parameter)
                    if m:
                        symbol, parameter = m.group(1), parameter[m.end():].strip()
                if not symbol and not parameter:
                    symbol, parameter = last_symbol, last_parameter  # continuation row (second condition line)
                else:
                    last_symbol, last_parameter = symbol, parameter
                # ROHM prints "V/DSS" and "I *1/D"; glue the subscript and drop the footnote.
                symbol = re.sub(r"\*\d+", "", symbol)
                symbol = re.sub(r"[\s/]+", "", symbol)
                unit_tokens = [t for t, _ in col_lines("unit") if t.strip()]
                unit_text = _normalize_pdf_text(" ".join(dict.fromkeys(unit_tokens))).strip()  # "mΩ mΩ" -> "mΩ"; distinct units stay listed
                unit = unit_text or last_unit or _unit_from_symbol(symbol)
                if unit_text:
                    last_unit = unit_text
                cond_lines = col_lines("conditions")
                value_lines = {role: col_lines(role) for role in _VALUE_ROLES if role in by_role}
                n_lines = max([len(cond_lines)] + [len(v) for v in value_lines.values()] + [1])
                # Pair condition lines with value lines when the counts agree; else one fact with all conditions.
                paired = len(cond_lines) == n_lines and n_lines > 1 and all(len(v) in (n_lines, 1) for v in value_lines.values())
                line_params: list[str] | None = None
                if not paired and n_lines > 1 and not cond_lines and len(par_lines) >= n_lines and all(len(v) in (n_lines, 1) for v in value_lines.values()):
                    # No conditions column: the per-line conditions are the
                    # trailing lines of the parameter cell ("Continuous Drain
                    # Current" / "TC = 25°C" / "TA = 25°C" ...).
                    tail = par_lines[len(par_lines) - n_lines:]
                    head = par_lines[: len(par_lines) - n_lines]
                    if all(_HEADER_CONDITION.search(t) or re.search(r"=|\bat\b|pulse|steady|\bt\s*[<≤]", t, re.I) for t, _ in tail):
                        cond_lines = tail
                        parameter = " ".join(t for t, _ in head).strip()
                        paired = True
                    elif not head:
                        line_params = [t for t, _ in tail]  # several parameters share one row: "VIN / PVIN / EN"
                        parameter = ""
                        paired = True
                if not paired and n_lines > 1 and all(len(v) in (n_lines, 1) for v in value_lines.values()):
                    # Several values, one label, no way to tell the lines apart:
                    # one fact per value line, all carrying the full label.
                    # Continuation tables glue Qg/Qgs/Qgd into one find_tables
                    # row; pair when the symbol (or short parameter) line count
                    # matches the value lines.
                    paired = True
                    cond_lines = [(" ; ".join(t for t, _ in cond_lines), "")] * n_lines if cond_lines else []
                for k in range(n_lines if paired else 1):
                    condition = cond_lines[k][0] if paired and cond_lines else (" ; ".join(t for t, _ in cond_lines) if not paired else "")
                    row_parameter = parameter if not line_params else " ".join(p for p in (parameter, line_params[k]) if p)
                    line_symbol = symbol
                    line_marked = symbol_marked
                    if paired and len(sym_lines) == n_lines and _looks_like_symbol(sym_lines[k][0]):
                        line_symbol, line_marked = sym_lines[k]
                    elif paired and not line_params and len(par_lines) == n_lines:
                        cand = par_lines[k][0]
                        row_parameter = cand
                        if _looks_like_symbol(cand):
                            line_symbol, row_parameter = cand, ""
                    line_symbol = re.sub(r"\*\d+", "", line_symbol)
                    line_symbol = re.sub(r"[\s/]+", "", line_symbol)
                    cells: dict[str, Any] = {}
                    verbatim_parts = []
                    for role, lines in value_lines.items():
                        if paired:
                            text = lines[k][0] if len(lines) == n_lines else (lines[0][0] if lines else "")
                        else:
                            text = " ".join(t for t, _ in lines)
                        if not text:
                            continue
                        verbatim_parts.append(text)
                        if role == "value_unit":
                            m_vu = _VALUE_UNIT.match(text.strip()) or _VALUE_UNIT_UPPER.match(text.strip())
                            if not m_vu:
                                continue
                            if "≤" in text and not condition:
                                inner = re.search(r"≤\s*([A-Za-z()/_]+)\s*≤|^([A-Za-z()/_]+)\s*≤", text.strip())
                                if inner and not symbol:
                                    symbol = (inner.group(1) or inner.group(2)).strip()
                            unit_hits = re.findall(r"[pnuµμmkM]?(?:Ω|Ohm|V|A|W|°C|ºC|C|Hz|s|F|H|J|%)(?=\s|$|\s*\(|\s*to|\s*–|\s*-)", text)
                            nums_vu = [float(x.replace("−", "-").replace("–", "-")) for x in re.findall(r"[-−–]?\d+(?:\.\d+)?", re.sub(r"\(\d\)$", "", text))]
                            if unit_hits:
                                unit = unit_hits[-1]
                            if len(nums_vu) == 2:
                                cells["value"] = nums_vu
                            elif len(nums_vu) == 1:
                                cells["value"] = nums_vu[0]
                            continue
                        parsed = _parse_number(text)
                        if parsed is None and role in ("min", "max", "value"):
                            # "1.7 2.0 2.4" or "-40 150" that the geometry could not split:
                            nums = [_parse_number(p) for p in text.split()]
                            nums = [n for n in nums if isinstance(n, float)]
                            if len(nums) == 2 and role in ("min", "value"):
                                parsed = nums
                            elif len(nums) == 3 and role == "min":
                                cells["min"], cells["typ"], cells["max"] = nums
                                continue
                        if parsed is not None:
                            cells[role] = parsed
                        else:
                            cells[f"{role}_text"] = text
                    if not any(k2 in cells for k2 in _VALUE_ROLES):
                        continue
                    line_unit = unit_text or last_unit or _unit_from_symbol(line_symbol) or unit
                    facts.append({
                        "page": pno + 1,
                        "table_title": title,
                        "table_kind": kind,
                        "table_condition": header_condition,
                        "section": section,
                        "symbol": line_symbol,
                        "symbol_as_printed": line_marked or line_symbol,
                        "parameter": row_parameter,
                        "condition_verbatim": (
                            condition
                            or _inline_conditions(
                                row_parameter,
                                line_symbol,
                                excluded_symbol=line_symbol,
                            )
                        ),
                        "unit": line_unit,
                        **cells,
                        "verbatim": " | ".join(p for p in [line_symbol, row_parameter, condition, *verbatim_parts, line_unit or ""] if p),
                    })
    return facts


# ---------------------------------------------------------------------------
# Front-page prose (Features / Description)
# ---------------------------------------------------------------------------
# Regulators state their headline ranges as bullets, not table rows:
# "4.5-V to 17-V input voltage range", "adjustable output voltage from 0.8 V
# to 15 V", "6-A continuous output current". These are family-grain claims
# and the fixtures (vendor parametrics) are built from exactly them.

_NUM = r"(\d+(?:\.\d+)?)"
_SNUM = r"([-+]?\d+(?:\.\d+)?)"
_V = r"\s*-?\s*V(?:olts?)?"
_TO = r"\s*(?:to|–|-|—|\.\.\.)\s*"
# "3V (falling threshold) to 65V"; optional wrapping parens on the pair.
_PAREN = r"(?:\s*\([^)]{0,48}\))?"
_VRANGE = rf"\(?\s*{_NUM}(?:{_V})?{_PAREN}{_TO}{_NUM}{_V}\s*\)?"
# Ampere token that is not µA/nA/kA. mA is allowed and scaled later.
_A = r"\s*-?\s*(?:mA\b|(?<![pnuµμmk])A\b)"
_C = r"\s*°\s*C"
_PROSE_SEP = r"(?:voltage\s*)?(?:range|operating\s+range)?\s*(?:of|from|:)?\s*"
_PROSE_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    # VIN — do not treat a bare trailing "VIN" as the unit of the previous
    # pair; that steals schematic "VOUT1 1.3V-27V VIN 4.5V-30V" as VIN.
    (re.compile(rf"{_VRANGE}\s+(?:wide\s+)?(?:input|supply|operating\s+input)", re.I), "vin_range", "VIN"),
    (re.compile(rf"(?:input|supply|VIN)\s+{_PROSE_SEP}{_VRANGE}", re.I), "vin_range", "VIN"),
    (re.compile(rf"(?:wide\s+)?(?:input|supply)\s+{_PROSE_SEP}{_VRANGE}", re.I), "vin_range", "VIN"),
    (re.compile(rf"operat(?:es|ing|ion)\s+(?:from|over|with)\s+(?:an?\s+)?(?:input\s+)?{_PROSE_SEP}{_VRANGE}", re.I), "vin_range", "VIN"),
    (re.compile(rf"{_NUM}{_V}{_TO}{_NUM}{_V},\s*{_NUM}{_A}", re.I), "vin_range", "VIN"),  # "3V to 65V, 0.3A"
    # VOUT ranges (schematic net names first so they win over a following VIN)
    (re.compile(rf"VOUT\s*[12]?\s*{_VRANGE}", re.I), "vout_range", "VOUT"),
    (re.compile(rf"(?:adjustable\s+)?output\s+{_PROSE_SEP}{_VRANGE}", re.I), "vout_range", "VOUT"),
    (re.compile(rf"{_VRANGE}\s+(?:adjustable\s+)?output", re.I), "vout_range", "VOUT"),
    (re.compile(rf"adjustable\s+(?:from\s+)?{_VRANGE}", re.I), "vout_range", "VOUT"),
    (re.compile(rf"{_NUM}{_V}\s+or\s+{_NUM}{_V}\s+output", re.I), "vout_range", "VOUT"),
    # VOUT / VIN single bounds printed without a numeric max
    (re.compile(rf"(?:outputs?|output\s+voltages?)\s+as\s+low\s+as\s+{_NUM}{_V}", re.I), "vout_min", "VOUT"),
    (re.compile(rf"adjustable\s+output\s+down\s+to\s+{_NUM}{_V}", re.I), "vout_min", "VOUT"),
    (re.compile(rf"output\s+voltage\s+from\s+{_NUM}{_V}\s+to\s+VIN\b", re.I), "vout_min", "VOUT"),
    (re.compile(rf"(?:feedback\s+)?(?:voltage\s+)?reference(?:\s+voltage)?(?:\s+of|:)?\s+{_NUM}{_V}", re.I), "vout_min", "VOUT"),
    (re.compile(rf"{_NUM}{_V}\s*(?:±\s*\d+(?:\.\d+)?\s*%)?\s+voltage reference", re.I), "vout_min", "VOUT"),
    (re.compile(rf"{_NUM}\s*-?\s*V\s+internal\s+voltage\s+reference", re.I), "vout_min", "VOUT"),
    # IOUT
    (re.compile(rf"{_NUM}{_A}\s+(?:continuous\s+|maximum\s+|max\.?\s+|peak\s+)?(?:output|load|rated)\s+current", re.I), "iout_max", "IOUT"),
    (re.compile(rf"(?:up\s+to|supports?|delivers?|provides?|capable\s+of(?:\s+supplying)?|supplying)\s+{_NUM}{_A}", re.I), "iout_max", "IOUT"),
    (re.compile(rf"(?:output|load)\s+current\s*(?:of|up\s+to|:)?\s*{_NUM}{_A}", re.I), "iout_max", "IOUT"),
    (re.compile(rf"{_NUM}{_A}\s+(?:to\s+\S+\s+)?load\s+range", re.I), "iout_max", "IOUT"),
    (re.compile(rf"{_NUM}{_A}\s+(?:synchronous\s+|step-down\s+|buck\s+|boost\s+|LDO\s+|linear\s+|radiation[- ]tolerant\s+|low\s+dropout\s+)*(?:DC/?DC\s+)?(?:converter|regulator)", re.I), "iout_max", "IOUT"),
    (re.compile(rf"\(\s*\d+(?:\.\d+)?{_V},\s*{_NUM}{_A}\s*\)", re.I), "iout_max", "IOUT"),  # "(30V, 1.25A)"
    (re.compile(rf"\d+(?:\.\d+)?{_V}{_TO}\d+(?:\.\d+)?{_V},\s*{_NUM}{_A}", re.I), "iout_max", "IOUT"),
    # Temperature (Features / AEC / junction / ambient). Bare storage is filtered later;
    # "operating and storage" is the Infineon combined rating the fixtures ask for.
    (re.compile(rf"{_SNUM}{_C}{_TO}{_SNUM}{_C}\s+(?:junction|ambient|operating)", re.I), "temp_range", "TA"),
    (re.compile(rf"(?:AEC-Q100\b.{{0,32}}?|(?:device\s+)?temperature\s+grade\s+\d[:\s]+|(?:junction|ambient|operating(?:\s+free-air)?|military)\s+temperature(?:\s+range)?\s*(?:of|from|:)?\s*|rated\s+from\s+)\(?\s*{_SNUM}{_C}{_TO}{_SNUM}{_C}\s*\)?", re.I), "temp_range", "TA"),
    (re.compile(rf"(?:operating\s+and\s+storage|operating\s+junction)\s+temperature(?:\s+range)?\s*(?:T\s*[jv]?\s*,\s*T\s*(?:stg)?|T\s*[jv]|T)?\s*{_SNUM}(?:{_C})?(?:{_TO}|\s+){_SNUM}{_C}", re.I), "temp_range", "TJ"),
    (re.compile(rf"(?:operating\s+and\s+storage\s+temperature|T\s*[jv]\s*,\s*T\s*stg).{{0,200}}?{_SNUM}(?:{_C})?(?:{_TO}|\s+){_SNUM}(?:{_C})?", re.I | re.DOTALL), "temp_range", "TJ"),
    # ROHM outline box + Infineon Features: "VDSS 40V" / "VDSS = 1200 V" / "ID ±24A" / "IDDC = 30 A"
    (re.compile(r"V\s*DSS\s*(?:/?\s*|=)\s*([-+]?\d+(?:\.\d+)?)\s*V", re.I), "vds", "VDS"),
    (re.compile(r"(?:^|[\s/])ID(?:DC)?\s*[±=]?\s*(\d+(?:\.\d+)?)\s*A\b", re.I), "id_max", "ID"),
    (re.compile(r"(?:^|[\s/])I\s*D\s+(\d+(?:\.\d+)?)\s+\d+(?:\.\d+)?\s+A\b", re.I), "id_max", "ID"),
    (re.compile(r"RDS\s*\(?\s*on\s*\)?\s*(?:\(\s*Max\.?\s*\)|=)\s*(\d+(?:\.\d+)?)\s*(m)?\s*[ΩΩωΩohm]?", re.I), "rds_max", "RDS(on)"),
    # OptiMOS Product Summary: "VDS 30 30 V" / "RDS(on),max … 3.7 mW" (mW = mΩ).
    (re.compile(r"\bV\s*DS\s+([-+]?\d+(?:\.\d+)?)\s+(?:\d+(?:\.\d+)?\s+)?V\b", re.I), "vds", "VDS"),
    (re.compile(r"R\s*DS\s*\(?on\)?\s*,?\s*max\b.*?(\d+(?:\.\d+)?)\s*(?:m\s*[WΩΩω]|mΩ)", re.I), "rds_max", "RDS(on)"),
    (re.compile(rf"(?:static|over\s+full\s+T|operating).{{0,48}}T\s*[jvc]?\s*=\s*{_SNUM}(?:{_C})?{_TO}{_SNUM}{_C}", re.I), "temp_range", "TJ"),
    (re.compile(r"Gate\s+charge\s+total\s+Q\s*g\b.{0,32}?(\d+(?:\.\d+)?)", re.I | re.DOTALL), "qg_typ", "Qg"),
    (re.compile(r"\bQg,typ\s+(\d+(?:\.\d+)?)\s*nC", re.I), "qg_typ", "Qg"),
)
_PROSE_STOP = re.compile(r"absolute\s+maximum\s+ratings|pin\s+configuration|electrical\s+characteristics|specifications", re.I)
_PROSE_PARAM = {
    "vin_range": "Input voltage range",
    "vout_range": "Output voltage range",
    "vout_min": "Output voltage",
    "vin_min": "Input voltage",
    "iout_max": "Output current",
    "temp_range": "Operating temperature",
    "vds": "Drain-to-source voltage",
    "id_max": "Drain current",
    "rds_max": "Drain-to-source on-resistance",
    "qg_typ": "Total gate charge",
}
_PROSE_UNIT = {
    "vin_range": "V",
    "vout_range": "V",
    "vout_min": "V",
    "vin_min": "V",
    "iout_max": "A",
    "temp_range": "°C",
    "vds": "V",
    "id_max": "A",
    "rds_max": "Ohm",
    "qg_typ": "nC",
}
_FIXED_VOUT_HEAD = re.compile(r"output voltage options|available in output voltage|fixed output voltages?\b", re.I)
_FIXED_VOUT_VALUE = re.compile(r"(\d+(?:\.\d+)?)\s*-?V\b", re.I)


# Older ROHM PDFs map Symbol-font ± / − / Ω through the private-use area.
_PDF_CHAR = {
    "\uf0b1": "±",
    "\uf02d": "-",
    "\uf02b": "+",
    "\uf0b0": "°",
    "\uf057": "Ω",
    "\u2011": "-",  # non-breaking hyphen; Infineon "‑55 ‑ 150"
    "\u00ad": "",
}


def _normalize_pdf_text(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2212", "-").replace("\u00a0", " ")
    for src, dst in _PDF_CHAR.items():
        text = text.replace(src, dst)
    return text


def _flatten_front_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = _normalize_pdf_text(text)
    # Infineon Min/Typ/Max: "T\n-55 -\n175 °C" and "T -55 -\n175 °C".
    text = re.sub(r"\n(?=\s*[-+]\d)", " ", text)
    text = re.sub(r"-\s*\n(?=\s*[-+]?\d)", " ", text)
    # Bullets and sentences; join soft line breaks inside a bullet.
    return re.sub(r"\n(?![•\u2022\-\u25a0\u25aa\u2013]|\d+\s)", " ", text)


def _numeric_groups(match: re.Match[str]) -> list[float]:
    out: list[float] = []
    for g in match.groups():
        if g is None:
            continue
        try:
            out.append(float(g))
        except ValueError:
            continue
    return out


def _prose_value(kind: str, nums: list[float], matched: str) -> Any | None:
    if kind == "temp_range":
        if len(nums) < 2 or nums[0] >= nums[1] or nums[0] < -80 or nums[1] > 220 or nums[1] < 40:
            return None
        return [nums[0], nums[1]]
    if kind.endswith("_range"):
        if len(nums) < 2:
            return None
        lo, hi = (nums[0], nums[1]) if nums[0] <= nums[1] else (nums[1], nums[0])
        if lo == hi or hi > 2000:
            return None
        return [lo, hi]
    if kind == "vds":
        if not nums or abs(nums[0]) > 5000:
            return None
        return nums[0]
    if kind == "rds_max":
        if not nums:
            return None
        value = nums[0]
        if re.search(r"\bm\s*[ΩΩωohmW]|mΩ", matched, re.I):
            value = value / 1000.0
        return value
    if kind == "qg_typ":
        if not nums or nums[0] <= 0 or nums[0] > 5000:
            return None
        return nums[0]
    if not nums or nums[0] > 1000:
        return None
    value = nums[0]
    if kind == "iout_max" and re.search(r"mA\b", matched):
        value = value / 1000.0
    return value


def _fixed_vout_from_options(flat: str) -> list[float] | None:
    """Family of fixed outputs listed as 'LM140LA-5.0 5V … LM140LA-15 15V'."""
    head = _FIXED_VOUT_HEAD.search(flat)
    if not head:
        return None
    window = flat[head.start():head.start() + 400]
    vals = sorted({float(m.group(1)) for m in _FIXED_VOUT_VALUE.finditer(window) if 0.5 <= float(m.group(1)) <= 60})
    if len(vals) < 2:
        return None
    return [vals[0], vals[-1]]


def _facts_from_front_text(flat: str, page: int, seen: set[tuple]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []

    def add(kind: str, symbol: str, value: Any, verbatim: str) -> None:
        key = (kind.split("_")[0], str(value) if not isinstance(value, list) else f"{value[0]}:{value[1]}")
        # Dedup VIN 4.5-18 whether it came from vin_range or another spelling.
        if key in seen:
            return
        seen.add(key)
        facts.append({
            "page": page,
            "table_title": "front-page prose",
            "table_kind": "prose",
            "table_condition": None,
            "section": "features",
            "symbol": symbol,
            "symbol_as_printed": symbol,
            "parameter": _PROSE_PARAM[kind],
            "condition_verbatim": None,
            "unit": _PROSE_UNIT[kind],
            "value": value,
            "prose_kind": kind,
            "verbatim": verbatim[:240],
        })

    for pattern, kind, symbol in _PROSE_PATTERNS:
        for m in pattern.finditer(flat):
            window = flat[max(0, m.start() - 48):m.end() + 24]
            if kind == "temp_range" and re.search(r"solder|lead\s+temp", window, re.I):
                continue
            if kind == "temp_range" and re.search(r"storage", window, re.I) and not re.search(r"operating", window, re.I):
                continue
            nums = _numeric_groups(m)
            value = _prose_value(kind, nums, m.group(0))
            if value is None:
                continue
            add(kind, symbol, value, flat[max(0, m.start() - 40):m.end() + 20].strip())

    options = _fixed_vout_from_options(flat)
    if options:
        head = _FIXED_VOUT_HEAD.search(flat)
        add("vout_range", "VOUT", options, flat[head.start():head.start() + 160].strip() if head else "")
    return facts


def read_front_page_facts(document: Any, max_pages: int = 3) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for pno in range(min(max_pages, document.page_count)):
        facts.extend(_facts_from_front_text(_flatten_front_text(document[pno].get_text()), pno + 1, seen))
    return facts


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

_SYMBOL_IN_TEXT = re.compile(
    r"\b(V\s*\(?\s*BR\s*\)?\s*DSS|VDSS|VDS|RDS\s*\(?\s*on\s*\)?|IDDC|ID,pulse|IDM|ID|"
    r"QG(?:\([^)]{0,16}\))?|Qg,typ|Q\s*g(?![a-z]))\b",
    re.I,
)


def _recover_symbol(symbol: str, parameter: str, verbatim: str = "") -> str:
    if (symbol or "").strip():
        return symbol
    aliases = canonical_symbols(parameter or "")
    if aliases:
        return aliases[0]
    match = _SYMBOL_IN_TEXT.search(f"{parameter} {verbatim}")
    if not match:
        return symbol
    raw = re.sub(r"[\s,]+", "", match.group(1)).upper()
    return {
        "VBRDSS": "V(BR)DSS",
        "VDSS": "VDSS",
        "VDS": "VDS",
        "RDSON": "RDS(on)",
        "RDS(ON)": "RDS(on)",
        "IDDC": "ID",
        "IDPULSE": "ID,pulse",
        "IDM": "IDM",
        "ID": "ID",
        "QG": "Qg",
        "QG,TYP": "Qg",
        "QGTYP": "Qg",
    }.get(raw if not raw.startswith("QG(") else "QG", raw)


def _rows_from_fact(fact: dict[str, Any], base: dict[str, Any]) -> list[dict[str, Any]]:
    if not (fact.get("symbol") or "").strip():
        recovered = _recover_symbol("", fact.get("parameter") or "", fact.get("verbatim") or "")
        if recovered:
            fact = {**fact, "symbol": recovered, "symbol_as_printed": fact.get("symbol_as_printed") or recovered}
    group = classify(fact["symbol"], fact["parameter"] or (fact.get("condition_verbatim") or "" if not fact["symbol"] else ""), fact["section"])
    label_head = " ".join(p for p in (fact["symbol"], fact["parameter"]) if p).strip()
    kind = fact["table_kind"]
    rows: list[dict[str, Any]] = []

    def row(value: Any, qq: str | None) -> dict[str, Any]:
        unit = fact.get("unit")
        shown = f"{value[0]} to {value[1]}" if isinstance(value, list) else f"{value:g}"
        label = f"{label_head} {shown}{' ' + unit if unit else ''}"
        if fact.get("condition_verbatim"):
            label += f" ({fact['condition_verbatim']})"
        return {
            **base,
            "group": group,
            "peripheral_class": None,
            "label": label,
            "instances": None,
            "value": value,
            "unit": unit,
            "qualifier_verbatim": None,
            "quantity_qualifier": qq,
            "condition_verbatim": fact.get("condition_verbatim"),
            "table_condition": fact.get("table_condition"),
            "symbol": fact["symbol"],
            "symbol_as_printed": fact["symbol_as_printed"],
            "parameter": fact["parameter"],
            "section": fact["section"] or None,
            "table_title": fact["table_title"] or None,
            "table_kind": kind,
            "verbatim": fact["verbatim"][:240],
            "source_pages": [fact["page"]],
            "varies_by_part": False,
            "tier": "grid",
            "also_printed": [c for c in canonical_symbols(f"{fact['parameter']} {fact.get('condition_verbatim') or '' if not fact['parameter'] else fact['parameter']}") if _norm_symbol(c) != _norm_symbol(fact["symbol"])],
            "typed": [],
        }

    if kind == "prose":
        # Gold VIN/VOUT/temp want "rated"; IOUT wants "maximum". A range is
        # always a rated headline; a single current is a max load.
        pk = fact.get("prose_kind") or ""
        if isinstance(fact["value"], list) or pk != "iout_max":
            qq = "rated"
        else:
            qq = "maximum"
        rows.append(row(fact["value"], qq))
    elif kind == "absolute_maximum":
        if "min" in fact and "max" in fact and not isinstance(fact["min"], list) and not isinstance(fact.get("max"), list):
            rows.append(row([fact["min"], fact["max"]], "absolute_maximum"))
        else:
            for role in ("value", "max", "min"):
                if role in fact:
                    rows.append(row(fact[role], "absolute_maximum"))
                    break
    elif kind == "recommended":
        if "min" in fact and "max" in fact and not isinstance(fact["min"], list):
            rows.append(row([fact["min"], fact["max"]], "rated"))
        else:
            for role in ("value", "min", "max", "typ"):
                if role in fact:
                    rows.append(row(fact[role], "rated" if role != "typ" else "typical"))
                    break
    else:
        if "value" in fact:
            rows.append(row(fact["value"], "typical" if kind == "summary" else None))
        for role, qq in (("min", "minimum"), ("typ", "typical"), ("max", "maximum"), ("limit", "limit")):
            if role in fact:
                rows.append(row(fact[role], qq))
    return rows


def build_power_grid(path: Path, *, vendor: str | None = None, part_numbers: list[str] | None = None, aisle: str | None = None, sha256: str | None = None) -> dict[str, Any]:
    import pymupdf

    data = path.read_bytes()
    sha = sha256 or hashlib.sha256(data).hexdigest()
    document = pymupdf.open(stream=data, filetype="pdf")
    facts = read_characteristic_tables(document)
    facts.extend(read_front_page_facts(document))
    base = {"vendor": vendor, "vendor_basis": "manifest" if vendor else None, "grain": "part", "scope_as_printed": ", ".join(part_numbers or []) or None, "source_artifact": path.name, "document_sha256": sha}
    rows: list[dict[str, Any]] = []
    for fact in facts:
        rows.extend(_rows_from_fact(fact, base))
    populated = {r["group"] for r in rows}
    return {
        "schema": SCHEMA,
        "_meta": {**base, "aisle": aisle, "page_count": document.page_count, "reader": READER, "extracted_at": datetime.now(timezone.utc).isoformat(), "models_used": [], "tables_read": len({(f["page"], f["table_title"]) for f in facts}), "facts": len(facts)},
        "groups": [{"key": k, "label": l} for k, l in GROUPS],
        "groups_populated": len(populated),
        "rows": rows,
        "rows_by_group": dict(Counter(r["group"] for r in rows)),
        "typed_facts_by_group": {},
    }
