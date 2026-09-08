"""Above-OPN document census: grain, parts named, pin bogeys, device tables.

Answers, per PDF and with page receipts, the questions that must be settled
BEFORE any model (local or frontier) reads a family/series document:

  grain              : above_opn | single_opn | not_device_family |
                       unknown | extraction_failed. A family datasheet
                       (STM32C011x4/x6, GD32F405xx) or a reference manual is
                       above_opn; a document that names exactly one orderable
                       part is single_opn. Only above_opn documents feed the
                       family-content and knife lanes.
  parts_named        : concrete orderable part tokens the document itself
                       prints (ordering information, device summary), each
                       with the page it was read from. Never expanded from a
                       wildcard, never taken from a taxonomy.
  series_tokens      : wildcard series tokens as printed (STM32C011x4,
                       GD32F405xx). Verbatim; expansion is not our call.
  pin_bogeys         : expected pin counts derived from printed package
                       tokens (LQFP64 -> 64). These are the denominators the
                       pin lane is scored against; they are held by the
                       verifier and never shown to an extractor.
  device_tables      : located device-comparison / feature tables: page,
                       shape, orientation, the attribute rows they address
                       (qualifier-anchored: code flash is not data flash),
                       and how many variant columns/rows they carry.
  denominator        : parts_or_variants x attributes_addressed. What the
                       document can be asked for; what a later extraction is
                       scored against.

Deterministic PyMuPDF only. Nothing here is a value extraction; it is the
census that makes later extraction measurable.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from harness.electronics.document_scope import SERIES_TOKEN, read_scope
from harness.electronics.locator import PACKAGE_KEY

CENSUS_SCHEMA = "harness.electronics-family-census.v1"

FRONT_PAGES = 3
KEYWORD_SCAN_PAGES = 40
MAX_TABLE_PAGES = 20

# Pages worth scanning for a device table or an ordering list, by printed
# heading. Matched on page text, whitespace-normalized.
DEVICE_TABLE_KEYWORDS = re.compile(
    r"device\s+(?:summary|information|overview|table|comparison|list|"
    r"selection|features?)|ordering\s+information|product\s+(?:selection|"
    r"line|matrix|table|overview)|family\s+(?:overview|comparison|features?)|"
    r"part\s+number\s+(?:table|list|matrix)|feature\s+(?:comparison|summary)|"
    r"memory\s+(?:size|configuration)|product\s+(?:list|lineup|line-up)|"
    r"peripheral\s+counts?|features?\s+and\s+peripheral",
    re.IGNORECASE,
)

# Knife/content attributes and the printed row/column labels that address
# them. Order matters where labels overlap: data flash and code flash are
# both "flash", and the RA2E2 2 KB read came from ignoring the qualifier.
ATTRIBUTE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("data_flash_kb", re.compile(r"\bdata\s+(?:flash|area)\b", re.I)),
    ("eeprom_kb", re.compile(r"\beeprom\b", re.I)),
    (
        "code_flash_kb",
        re.compile(
            r"\b(?:code|program)\s+(?:flash|area|memory)\b|\bflash\b(?!\s+data)",
            re.I,
        ),
    ),
    ("sram_kb", re.compile(r"\bs?ram\b", re.I)),
    (
        "freq_mhz",
        re.compile(
            r"\b(?:max(?:imum|\.)?\s+)?(?:cpu\s+|core\s+|system\s+)?"
            r"(?:frequency|clock\s+(?:speed|frequency|rate)|speed)\b|\bmhz\b|\bf\s*max\b",
            re.I,
        ),
    ),
    ("core", re.compile(r"\b(?:core|cortex|cpu)\b", re.I)),
    ("package", re.compile(r"\bpackages?\b", re.I)),
    (
        "pin_count",
        re.compile(r"\b(?:pins?|pin\s+count|gpios?|i/os?|i/o\s+(?:pins|count))\b", re.I),
    ),
    ("adc_channels", re.compile(r"\badcs?\b", re.I)),
    ("dac_channels", re.compile(r"\bdacs?\b", re.I)),
    ("timers", re.compile(r"\btimers?\b", re.I)),
    ("pwm_channels", re.compile(r"\bpwm\b", re.I)),
    (
        "operating_voltage",
        re.compile(
            r"\b(?:operating|supply|vdd|power\s+supply)\s*(?:voltage|range)?\b|\bvdd\b",
            re.I,
        ),
    ),
    (
        "temp_range",
        re.compile(r"\b(?:operating\s+)?temperature\b|\btemp\.?\s+range\b", re.I),
    ),
    (
        "standby_current_ua",
        re.compile(
            r"\b(?:standby|stop|sleep|shutdown|deep\s*sleep|power[\s-]?down)\b.*?"
            r"\b(?:current|consumption|[µu]a)\b",
            re.I,
        ),
    ),
    ("usart_count", re.compile(r"\bu?s?arts?\b|\buart\b", re.I)),
    ("spi_count", re.compile(r"\bspi\b", re.I)),
    ("i2c_count", re.compile(r"\bi2c\b", re.I)),
    ("can_count", re.compile(r"\bcan(?:\s*fd)?\b", re.I)),
    ("usb", re.compile(r"\busb\b", re.I)),
    ("ethernet", re.compile(r"\bethernet\b|\beth\b", re.I)),
)
ATTRIBUTE_NAMES = tuple(name for name, _ in ATTRIBUTE_PATTERNS)

# Tokens that look like part numbers but are document or standard ids.
_NON_PART_PREFIX = re.compile(
    r"^(?:RM|UM|DS|AN|PM|ES|TN|DOC|REV|ISO|IEC|JESD|IEEE|ARM|SLA|SLV|SPR|"
    r"SBO|SNV|SLU|SLL|SNO|DDI|IHI|MIL|EIA|JEDEC|ROHS|UL|CE|FCC|USB|PCI|"
    r"CAN|SPI|I2C|I2S|SDIO|GPIO|ADC|DAC|DMA|RTC|PWM|UART|USART|LQFP|TQFP|"
    r"QFN|BGA|WLCSP|TSSOP|SOIC|DFN|MHZ|KHZ|KB|MB|GB|V|A)\d",
)
_STEM = re.compile(r"^[A-Z]{1,10}\d{1,3}")
_WILDCARD = re.compile(r"x")
# Concrete orderable tokens may start with a single letter (R7FA2E2A72DNK,
# S32K144, R5F100LEA); SERIES_TOKEN's two-letter floor is for wildcards.
PART_TOKEN = re.compile(r"\b([A-Z]{1,10}\d{1,4}[A-Z0-9]*(?:-[A-Z0-9]{1,6})?)\b")
# Data-column labels that mark a ratings/characteristics table rather than a
# device matrix: MIN/MAX/UNIT, bare units, or a value range.
_UNIT_LABEL = re.compile(
    r"^(?:[-–]?\d[\d.,]*\s*)?(?:v|mv|a|ma|[µu]a|na|w|mw|°c|c|mhz|khz|hz|"
    r"ohm|ω|kω|ms|us|ns|%|value/units?|units?|min|max|typ|nom)"
    r"(?:\s+to\s+.*)?$|.*\bunits?\b",
    re.I,
)


def _normalize(text: str) -> str:
    return " ".join((text or "").split())


def is_wildcard_token(token: str) -> bool:
    """STM32C011x4, GD32F405xx, STM32G0x1: lowercase x inside a series token."""
    return bool(_WILDCARD.search(token))


_NON_PART_TOKEN = re.compile(
    r"^(?:X?OSC\d|SERNUM|CRC\d|SHA\d|AES\d|RSA\d|MD\d|UTF\d|ISO\d|RS\d{3}|"
    r"IEEE\d|CAT\d|CLASS\d|GRADE\d|LEVEL\d|TYPE\d|TIM\d|CH\d|IO\d|"
    r"P[A-K]\d{1,2}$|VDD\d|VSS\d|VBAT|NRST|BOOT\d|OPT\d|FIG\d|TAB\d|"
    r"REV\d|PAGE\d|NOTE\d|STEP\d|CASE\d|BIT\d|WORD\d|BYTE\d|"
    r"YUV\d|RGB\d|JPEG|MPEG|H26\d|"
    # Renesas document ids (R01DS0427EJ0120) and package codes (PLQP0032GB-A)
    r"R\d{2}[A-Z]{2}\d{4}|P[A-Z]{3}\d{4}[A-Z]{2}(?:-[A-Z])?$)",
)


def is_concrete_part_token(token: str) -> bool:
    if is_wildcard_token(token):
        return False
    if not 6 <= len(token) <= 24:
        return False
    if sum(ch.isdigit() for ch in token) < 2:
        return False
    if _NON_PART_PREFIX.match(token) or _NON_PART_TOKEN.match(token):
        return False
    if PACKAGE_KEY.search(token):
        return False
    if token.endswith("/"):
        return False
    return True


# Electrical-characteristics tables (Symbol/Parameter/Conditions/Min/Typ/Max/
# Unit) mention frequency and temperature but are not device tables.
_PARAMETRIC_HEADER = re.compile(
    r"^(?:symbol|parameter|conditions?|min\.?|typ\.?|max\.?|unit|units|"
    r"nom\.?|description|comments?|test\s+conditions?)$",
    re.I,
)


def stems_from_tokens(tokens: Iterable[str]) -> set[str]:
    stems: set[str] = set()
    for token in tokens:
        match = _STEM.match(token.split("/")[0])
        if match:
            stems.add(match.group(0))
    return stems


def parts_from_text(
    text: str, stems: set[str] | None
) -> list[str]:
    """Concrete part tokens in ``text``; when ``stems`` is given, only tokens
    sharing one of the document's own family stems."""
    found: list[str] = []
    for token in PART_TOKEN.findall(_normalize(text)):
        if not is_concrete_part_token(token):
            continue
        if stems and not any(token.startswith(stem) for stem in stems):
            continue
        if token not in found:
            found.append(token)
    return found


def wildcards_from_text(text: str) -> list[str]:
    found: list[str] = []
    for token in SERIES_TOKEN.findall(_normalize(text)):
        if is_wildcard_token(token) and token not in found:
            found.append(token)
    return found


def packages_from_text(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for match in PACKAGE_KEY.finditer(_normalize(text).upper()):
        family = match.group(1) or match.group(4)
        count = match.group(2) or match.group(3)
        if not family or not count:
            continue
        pin_count = int(count)
        if pin_count < 3 or pin_count > 2000:
            continue
        key = (family.upper(), pin_count)
        if key in seen:
            continue
        seen.add(key)
        out.append({"package_family": family.upper(), "pin_count": pin_count})
    return out


def attributes_in_label(label: str) -> list[str]:
    """Attribute names addressed by one printed row/column label, with
    qualifier precedence: 'Data area (KB)' is data flash, not code flash."""
    text = _normalize(label)
    if not text:
        return []
    hits: list[str] = []
    for name, pattern in ATTRIBUTE_PATTERNS:
        if pattern.search(text):
            hits.append(name)
    if "data_flash_kb" in hits and "code_flash_kb" in hits:
        hits.remove("code_flash_kb")
    if "eeprom_kb" in hits and "code_flash_kb" in hits and "flash" not in text.lower():
        hits.remove("code_flash_kb")
    return hits


def _cell(value: Any) -> str:
    return _normalize(str(value)) if value is not None else ""


_TITLE_ROW = re.compile(r"^(?:table|tab\.)\s*\d", re.I)


def strip_title_rows(rows: list[list[Any]]) -> list[list[Any]]:
    """Drop leading rows that are a table caption merged into the grid
    ("Table 1.12 Product list (1 of 2)" spanning every column)."""
    out = list(rows)
    while out:
        cells = [_cell(c) for c in out[0]]
        filled = [c for c in cells if c]
        if len(filled) == 1 and _TITLE_ROW.match(filled[0]):
            out = out[1:]
            continue
        break
    return out


def classify_table(rows: list[list[Any]]) -> dict[str, Any] | None:
    """Return a device-table description or None when the table is not one.

    A device table carries >=2 distinct attribute labels down its label
    column(s) (parts_as_columns: GD32/ST/Microchip style) or across its
    header row (parts_as_rows: selection-guide style) and >=2 variants.
    """
    rows = strip_title_rows(rows)
    if len(rows) < 3 or not rows[0] or len(rows[0]) < 3:
        return None
    width = max(len(row) for row in rows)
    grid = [[_cell(c) for c in row] + [""] * (width - len(row)) for row in rows]

    label_cols = min(2, width - 1)
    row_attributes: list[str] = []
    row_hits = 0
    carried = ""
    for row in grid[1:]:
        label = " ".join(part for part in row[:label_cols] if part)
        if row[0]:
            carried = row[0]
        elif label_cols > 1 and row[1]:
            label = f"{carried} {row[1]}"
        hits = attributes_in_label(label)
        if hits:
            row_hits += 1
            for hit in hits:
                if hit not in row_attributes:
                    row_attributes.append(hit)

    header_attributes: list[str] = []
    for cell in grid[0][1:]:
        for hit in attributes_in_label(cell):
            if hit not in header_attributes:
                header_attributes.append(hit)

    header_cells = [c for c in grid[0] if c]
    parametric_header_cells = sum(
        1 for c in grid[0] + grid[1] if c and _PARAMETRIC_HEADER.match(c)
    )
    if parametric_header_cells >= 2:
        return None
    header_parts = parts_from_text(" ".join(header_cells), None)
    header_wildcards = wildcards_from_text(" ".join(header_cells))
    second_row_codes = [
        c for c in grid[1][1:] if c and re.fullmatch(r"[A-Z0-9]{1,6}", c)
    ]

    if len(row_attributes) >= 2 and row_hits >= 2:
        data_columns = width - label_cols
        variant_labels = header_parts or second_row_codes or [
            c for c in grid[0][label_cols:] if c
        ]
        # Ratings/characteristics tables whose data columns are MIN/MAX/UNIT
        # or bare units address voltage and temperature but carry no device
        # variants; they belong to the parametric lane, not here.
        if any(
            _PARAMETRIC_HEADER.match(label) or _UNIT_LABEL.match(label)
            for label in variant_labels
        ):
            return None
        return {
            "orientation": "parts_as_columns",
            "rows": len(rows),
            "columns": width,
            "variant_count": max(data_columns, len(variant_labels)),
            "variant_labels": variant_labels[:64],
            "series_tokens": header_wildcards,
            "attributes_addressed": row_attributes,
            "attribute_rows": row_hits,
        }
    if len(header_attributes) >= 2:
        first_col_parts = parts_from_text(
            " ".join(row[0] for row in grid[1:]), None
        )
        first_col_wildcards = wildcards_from_text(
            " ".join(row[0] for row in grid[1:])
        )
        if len(first_col_parts) >= 2 or len(first_col_wildcards) >= 2:
            return {
                "orientation": "parts_as_rows",
                "rows": len(rows),
                "columns": width,
                "variant_count": len(rows) - 1,
                "variant_labels": first_col_parts[:64],
                "series_tokens": wildcards_from_text(
                    " ".join(row[0] for row in grid[1:])
                ),
                "attributes_addressed": header_attributes,
                "attribute_rows": len(header_attributes),
            }
    return None


def _candidate_pages(
    document: Any,
    lane_pages: Mapping[str, Iterable[int]] | None,
    page_texts: dict[int, str],
) -> list[int]:
    """1-based pages to inspect for device tables, ordering lists, packages."""
    pages: list[int] = list(range(1, min(document.page_count, FRONT_PAGES) + 1))
    if lane_pages:
        for lane in ("series_summary", "opn_decoder"):
            for page in lane_pages.get(lane) or []:
                if isinstance(page, int) and 1 <= page <= document.page_count:
                    pages.append(page)
    for index in range(min(document.page_count, KEYWORD_SCAN_PAGES)):
        text = page_texts.get(index + 1)
        if text is None:
            text = document[index].get_text()
            page_texts[index + 1] = text
        normalized = _normalize(text)
        if DEVICE_TABLE_KEYWORDS.search(normalized):
            pages.append(index + 1)
        elif len(parts_from_text(normalized, None)) >= 3:
            # A page naming three or more concrete parts is a device table,
            # an ordering list or a product list whatever its heading says.
            pages.append(index + 1)
    ordered: list[int] = []
    for page in pages:
        if page not in ordered:
            ordered.append(page)
    return ordered[:MAX_TABLE_PAGES]


def classify_grain(
    *,
    scope_status: str,
    document_subject: str,
    document_class: str,
    series_tokens: list[str],
    parts_named: list[str],
    device_tables: list[dict[str, Any]],
    registry_stems: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if scope_status != "read_ok":
        return "extraction_failed", [scope_status]
    if document_subject in {"cpu_core", "peripheral"}:
        return "not_device_family", [f"document_subject={document_subject}"]
    if series_tokens:
        reasons.append("wildcard_series_tokens_printed")
    if len(parts_named) >= 2:
        reasons.append(f"parts_named={len(parts_named)}")
    if any(t.get("variant_count", 0) >= 2 for t in device_tables):
        reasons.append("device_table_with_variants")
    if document_class in {"reference_manual", "user_manual", "programming_manual"}:
        reasons.append(f"document_class={document_class}")
    if registry_stems >= 2:
        reasons.append(f"registry_stems={registry_stems}")
    if reasons:
        return "above_opn", reasons
    if len(parts_named) == 1 or registry_stems == 1:
        return "single_opn", [f"parts_named={len(parts_named)}", f"registry_stems={registry_stems}"]
    return "unknown", ["no_grain_signal"]


def above_opn_kind(
    *,
    document_class: str,
    series_tokens: list[str],
    parts_named: list[str],
    device_tables: list[dict[str, Any]],
) -> str | None:
    """What kind of above-OPN document this is, for scoping the lanes.

    family_matrix     : a device table with >=2 variants and >=3 attributes;
                        the knife lane's primary input.
    manual            : reference/user/programming manual (shared-content
                        lane; variant matrix usually lives in the datasheet).
    ordering_variants : >=2 orderable parts printed but no attribute matrix
                        (package/temperature/reel variants of one device).
    wildcard_only     : a printed series wildcard and nothing else.
    """
    if any(
        t.get("variant_count", 0) >= 2 and len(t.get("attributes_addressed") or []) >= 3
        for t in device_tables
    ):
        return "family_matrix"
    if document_class in {"reference_manual", "user_manual", "programming_manual"}:
        return "manual"
    if len(parts_named) >= 2:
        return "ordering_variants"
    if series_tokens:
        return "wildcard_only"
    return None


def census_document(
    path: Path,
    *,
    lane_pages: Mapping[str, Iterable[int]] | None = None,
    registry_stems: int = 0,
    vendor: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    import pymupdf

    scope = read_scope(path)
    row: dict[str, Any] = {
        "schema": CENSUS_SCHEMA,
        "source_path": str(path),
        "document_sha256": scope.get("document_sha256"),
        "vendor": vendor,
        "category": category,
        "page_count": scope.get("page_count"),
        "registry_stems": registry_stems,
        "scope_status": scope.get("status"),
        "extraction_failed_reason": scope.get("extraction_failed_reason"),
        "document_class": scope.get("document_class"),
        "document_subject": scope.get("document_subject"),
        "title_verbatim": scope.get("title_verbatim"),
        "governs_series": list(scope.get("governs_series") or []),
        "scope_statement": scope.get("scope_statement"),
        "scope_page": scope.get("scope_page"),
        "series_tokens": [],
        "family_stems": [],
        "parts_named": [],
        "parts_named_count": 0,
        "pin_bogeys": [],
        "packages": [],
        "device_tables": [],
        "device_table_located": False,
        "attributes_addressed": [],
        "variant_count": 0,
        "denominator_cells": 0,
        "pages_inspected": [],
        "grain": "unknown",
        "grain_reasons": [],
        "above_opn_kind": None,
    }
    if scope.get("status") != "read_ok":
        row["grain"], row["grain_reasons"] = classify_grain(
            scope_status=str(scope.get("status")),
            document_subject="unknown",
            document_class="unknown",
            series_tokens=[],
            parts_named=[],
            device_tables=[],
            registry_stems=registry_stems,
        )
        return row

    document = pymupdf.open(path)
    page_texts: dict[int, str] = {}
    pages = _candidate_pages(document, lane_pages, page_texts)
    row["pages_inspected"] = pages

    title_text = " ".join(
        part for part in (scope.get("title_verbatim"), scope.get("scope_statement")) if part
    )
    series_tokens = list(scope.get("governs_series") or [])
    for token in wildcards_from_text(title_text):
        if token not in series_tokens:
            series_tokens.append(token)

    parts: list[dict[str, Any]] = []
    packages: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    page_parts_pending: list[tuple[int, str]] = []

    for page_number in pages:
        page = document[page_number - 1]
        text = page_texts.get(page_number)
        if text is None:
            text = page.get_text()
            page_texts[page_number] = text
        for token in wildcards_from_text(text):
            if token not in series_tokens:
                series_tokens.append(token)
        page_parts_pending.append((page_number, text))
        for package in packages_from_text(text):
            entry = dict(package, page=page_number)
            if not any(
                p["package_family"] == entry["package_family"]
                and p["pin_count"] == entry["pin_count"]
                for p in packages
            ):
                packages.append(entry)
        try:
            found = page.find_tables()
        except Exception:  # noqa: BLE001 - a table finder crash is not a verdict
            continue
        for table in found.tables:
            try:
                rows = table.extract()
            except Exception:  # noqa: BLE001
                continue
            description = classify_table(rows)
            if description is None:
                continue
            description["page"] = page_number
            tables.append(description)
            for token in description.get("series_tokens") or []:
                if token not in series_tokens:
                    series_tokens.append(token)

    # Family stems anchor parts_named to the document's own family: printed
    # wildcards first, then concrete parts the device tables label, then the
    # title. Without a stem, parts are still collected but unanchored.
    stems = stems_from_tokens(series_tokens)
    if not stems:
        stems = stems_from_tokens(
            label
            for table in tables
            for label in table.get("variant_labels") or []
            if is_concrete_part_token(label)
        )
    if not stems:
        stems = stems_from_tokens(
            t for t in PART_TOKEN.findall(_normalize(title_text)) if is_concrete_part_token(t)
        )
    for page_number, text in page_parts_pending:
        for token in parts_from_text(text, stems or None):
            if not any(p["token"] == token for p in parts):
                parts.append({"token": token, "page": page_number})
    for table in tables:
        for token in table.get("variant_labels") or []:
            if is_concrete_part_token(token) and (
                not stems or any(token.startswith(s) for s in stems)
            ) and not any(p["token"] == token for p in parts):
                parts.append({"token": token, "page": table["page"]})

    attributes: list[str] = []
    for table in tables:
        for name in table["attributes_addressed"]:
            if name not in attributes:
                attributes.append(name)
    variant_count = max([t["variant_count"] for t in tables], default=0)
    parts_or_variants = max(len(parts), variant_count)

    row.update(
        {
            "series_tokens": series_tokens,
            "family_stems": sorted(stems),
            "parts_named": parts,
            "parts_named_count": len(parts),
            "pin_bogeys": sorted({p["pin_count"] for p in packages}),
            "packages": packages,
            "device_tables": tables,
            "device_table_located": bool(tables),
            "attributes_addressed": attributes,
            "variant_count": variant_count,
            "denominator_cells": parts_or_variants * len(attributes),
        }
    )
    row["grain"], row["grain_reasons"] = classify_grain(
        scope_status=str(scope.get("status")),
        document_subject=str(scope.get("document_subject")),
        document_class=str(scope.get("document_class")),
        series_tokens=series_tokens,
        parts_named=[p["token"] for p in parts],
        device_tables=tables,
        registry_stems=registry_stems,
    )
    if row["grain"] == "above_opn":
        row["above_opn_kind"] = above_opn_kind(
            document_class=str(scope.get("document_class")),
            series_tokens=series_tokens,
            parts_named=[p["token"] for p in parts],
            device_tables=tables,
        )
    return row


def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-vendor and overall counts for the census report."""
    rows = list(rows)
    overall: dict[str, Any] = {
        "documents": len(rows),
        "grain": Counter(),
        "document_class": Counter(),
        "above_opn": {
            "documents": 0,
            "kind": Counter(),
            "device_table_located": 0,
            "scope_statement_printed": 0,
            "parts_named_total": 0,
            "documents_with_parts_named": 0,
            "documents_with_pin_bogeys": 0,
            "denominator_cells_total": 0,
            "attributes_addressed": Counter(),
        },
    }
    by_vendor: dict[str, dict[str, Any]] = {}

    def bucket() -> dict[str, Any]:
        return {
            "documents": 0,
            "grain": Counter(),
            "above_opn": 0,
            "above_opn_kind": Counter(),
            "above_opn_device_table_located": 0,
            "above_opn_parts_named_total": 0,
            "above_opn_denominator_cells": 0,
            "attributes_addressed": Counter(),
        }

    for row in rows:
        grain = row.get("grain") or "unknown"
        overall["grain"][grain] += 1
        overall["document_class"][row.get("document_class") or "unknown"] += 1
        vendor = row.get("vendor") or "unknown"
        b = by_vendor.setdefault(vendor, bucket())
        b["documents"] += 1
        b["grain"][grain] += 1
        if grain != "above_opn":
            continue
        a = overall["above_opn"]
        a["documents"] += 1
        b["above_opn"] += 1
        kind = row.get("above_opn_kind") or "unclassified"
        a["kind"][kind] += 1
        b["above_opn_kind"][kind] += 1
        if row.get("device_table_located"):
            a["device_table_located"] += 1
            b["above_opn_device_table_located"] += 1
        if row.get("scope_statement"):
            a["scope_statement_printed"] += 1
        n_parts = int(row.get("parts_named_count") or 0)
        a["parts_named_total"] += n_parts
        b["above_opn_parts_named_total"] += n_parts
        if n_parts:
            a["documents_with_parts_named"] += 1
        if row.get("pin_bogeys"):
            a["documents_with_pin_bogeys"] += 1
        cells = int(row.get("denominator_cells") or 0)
        a["denominator_cells_total"] += cells
        b["above_opn_denominator_cells"] += cells
        for name in row.get("attributes_addressed") or []:
            a["attributes_addressed"][name] += 1
            b["attributes_addressed"][name] += 1

    def plain(value: Any) -> Any:
        if isinstance(value, Counter):
            return dict(sorted(value.items(), key=lambda kv: (-kv[1], kv[0])))
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        return value

    return {
        "schema": "harness.electronics-family-census-summary.v1",
        "overall": plain(overall),
        "by_vendor": {
            vendor: plain(b)
            for vendor, b in sorted(
                by_vendor.items(), key=lambda kv: -kv[1]["documents"]
            )
        },
    }
