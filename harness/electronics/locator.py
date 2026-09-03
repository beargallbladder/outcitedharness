"""Fail-closed pin-definition table localization for technical datasheets.

The locator selects definition tables, never package drawings or ballout figures.
Ambiguous documents are withheld with a machine-readable reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JEDEC = re.compile(r"^[A-HJ-NP-WY]\d{1,2}$")
PACKAGE_FAMILIES = (
    r"TFBGA|UFBGA|LFBGA|XFBGA|FBGA|BGA|WLCSP|DSBGA|CABGA|CSBGA|"
    r"LQFP|TQFP|QFP|VFQFPN|UFQFPN|VQFN|WQFN|TQFN|UTQFN|"
    r"XQFN|QFN|DFN|LGA|WLP|CSP|"
    r"TSSOP|SSOP|SOIC|SPDIP|PDIP"
)
PACKAGE_KEY = re.compile(
    rf"(?i)\b({PACKAGE_FAMILIES})\s*[- ]?\s*(\d{{1,4}})\b"
    rf"|\b(\d{{1,4}})\s*(?:[- ]\s*)?(?:PIN|BALL)?\s*"
    rf"({PACKAGE_FAMILIES})\b"
)
PACKAGE_CELL = PACKAGE_KEY
PIN_SECTION_TITLES = (
    "PIN DESCRIPTIONS",
    "PIN DESCRIPTION",
    "PINOUTS AND PIN DESCRIPTION",
    "PINOUT, PIN DESCRIPTION",
    "PINOUTS/BALLOUTS, PIN DESCRIPTION",
    "PINOUT/BALLOUT SCHEMATICS",
    # Vendor section conventions beyond ST. These only widen the TOC search
    # window; every candidate page must still contain a structural
    # definition table (package column headers plus pin-name markers), so
    # the fail-closed guarantee is unchanged.
    "PIN CONFIGURATION AND FUNCTIONS",  # Texas Instruments
    "PINNING INFORMATION",  # NXP
    "PIN ASSIGNMENT",  # Renesas, Nordic, Infineon (covers "PIN ASSIGNMENTS")
    "PIN FUNCTIONS",  # Renesas
    "PIN DIAGRAMS",  # Microchip
    "PINOUT DESCRIPTION",  # Microchip
    "PIN ALLOCATION TABLE",  # Microchip
    "PIN CONFIGURATION",  # Infineon, Silicon Labs
    "PIN DEFINITIONS",  # GigaDevice, Espressif, Silicon Labs sections
)
SECTION_STOPS = (
    "ELECTRICAL",
    "PACKAGE INFORMATION",
    "MEMORY MAPPING",
    "FUNCTIONAL OVERVIEW",
    "FUNCTIONAL DESCRIPTION",
    "LIST OF TABLES",
    "LIST OF FIGURES",
    "REVISION HISTORY",
)


@dataclass(frozen=True)
class TocEntry:
    level: int
    title: str
    page: int


@dataclass(frozen=True)
class LocateResult:
    document_id: str
    requested_package: str
    expected_package_pins: int
    source_path: str
    status: str
    reason: str
    toc_class: str | None
    toc_title: str | None
    pages_1based: tuple[int, ...]
    column_header: str | None
    column_headers_all: tuple[str, ...]
    corroborating_jedec_ids: int | None


def classify_title(title: str) -> str | None:
    normalized = " ".join((title or "").upper().split())
    if any(
        marker in normalized
        for marker in ("NRST PIN", "PIN INPUT", "PIN LOADING", "PIN CHARACTERISTIC")
    ):
        return "electrical"
    if any(
        marker in normalized
        for marker in (
            "PACKAGE INFORMATION",
            "OUTLINE",
            "MECHANICAL",
            "FOOTPRINT",
            "MARKING",
            "THERMAL",
        )
    ):
        return "outline"
    if "ALTERNATE FUNCTION" in normalized:
        return "af"

    is_table = normalized.startswith("TABLE") or " TABLE " in f" {normalized} "
    is_figure = normalized.startswith(("FIGURE", "FIG."))
    definition = any(
        marker in normalized
        for marker in (
            "PIN AND BALL",
            "PIN/BALL",
            "PIN DEFINITION",
            "BALL DEFINITION",
            "PIN DEFINITIONS",
            "BALL DEFINITIONS",
            "PIN AND BALL DESCRIPTION",
        )
    )
    if is_table and definition:
        return "other_table" if "FMC PIN" in normalized else "definition_table"
    if (
        is_table
        and re.search(r"\bPIN DEFINITIONS?\b", normalized)
        and "ALTERNATE" not in normalized
    ):
        return "definition_table"
    if is_figure and "BALLOUT" in normalized:
        return "ballout_figure"
    if is_figure and "PINOUT" in normalized:
        return "pinout_figure"
    if any(marker in normalized for marker in PIN_SECTION_TITLES) or re.search(
        r"\bPIN(OUTS)? AND PIN DESCRIPTION\b", normalized
    ):
        return "pin_section"
    return None


def _cell(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str("" if value is None else value),
    ).strip()


def _table_blob(rows: list[list[Any]], limit: int = 4) -> str:
    return " | ".join(_cell(cell) for row in rows[:limit] for cell in row).upper()


def _is_legend(rows: list[list[Any]]) -> bool:
    blob = _table_blob(rows, 3)
    return all(marker in blob for marker in ("ABBREVIATION", "DEFINITION", "PIN NAME"))


def _is_alternate_function_table(rows: list[list[Any]]) -> bool:
    blob = _table_blob(rows, 3)
    return (
        bool(re.search(r"\bAF0\b", blob))
        or "PORT A ALTERNATE" in blob
        or ("ALTERNATE FUNCTION" in blob and not PACKAGE_CELL.search(blob))
    )


def package_headers(rows: list[list[Any]]) -> list[str]:
    headers: list[str] = []
    for row in rows[:4]:
        for value in row:
            candidate = _cell(value)
            if (
                not candidate
                or len(candidate) > 48
                or not PACKAGE_CELL.search(candidate)
                or not PACKAGE_KEY.search(candidate)
            ):
                continue
            if candidate not in headers:
                headers.append(candidate)
    return headers


def is_definition_table(rows: list[list[Any]]) -> bool:
    if not rows or _is_legend(rows) or _is_alternate_function_table(rows):
        return False
    if not package_headers(rows):
        return False
    blob = _table_blob(rows)
    return any(
        marker in blob
        for marker in (
            "PIN NAME",
            "PIN/BALL NAME",
            "PIN NUMBER",
            "PIN/BALL",
            "FUNCTION AFTER RESET",
        )
    )


def _definition_tables(page: Any) -> list[list[list[Any]]]:
    tables = page.find_tables()
    if tables is None:
        return []
    return [
        rows
        for table in tables.tables
        if (rows := table.extract() or []) and is_definition_table(rows)
    ]


def _package_identity(value: str) -> tuple[str, str, bool] | None:
    normalized = (
        (value or "").upper().replace("WITH SMPS", "SMPS").replace("_", " ")
    )
    match = PACKAGE_KEY.search(normalized)
    if not match:
        return None
    family = match.group(1) or match.group(4)
    count = match.group(2) or match.group(3)
    return family.upper(), count, "SMPS" in normalized


def package_pin_count(value: str) -> int | None:
    identity = _package_identity(value)
    return int(identity[1]) if identity is not None else None


def match_package_column(requested: str, headers: list[str]) -> str | None:
    requested_identity = _package_identity(requested)
    if requested_identity is None:
        return None
    requested_text = _cell(requested).upper()
    family, count, requested_smps = requested_identity
    matches: list[tuple[int, str]] = []
    for header in headers:
        identity = _package_identity(header)
        if identity is None or identity[:2] != (family, count):
            continue
        score = 2 + int(identity[2] == requested_smps)
        score += int(not identity[2] and not requested_smps)
        matches.append((score, header))
    if not matches:
        return None
    best_score = max(score for score, _ in matches)
    best_headers = [header for score, header in matches if score == best_score]
    if len(best_headers) == 1:
        return best_headers[0]

    exact = [
        header
        for header in best_headers
        if _cell(header).upper() == requested_text
    ]
    canonical = f"{family}{count}{'SMPS' if requested_smps else ''}"
    requested_key = re.sub(r"[^A-Z0-9]", "", requested_text)
    if requested_key != canonical and len(exact) == 1:
        return exact[0]
    return None


def validate_physical_pin_truth(
    pins: Any,
    *,
    package: str,
    expected_package_pins: int,
) -> None:
    if not isinstance(pins, list) or not pins:
        raise ValueError("physical pins must be a non-empty list")
    identifiers: list[str] = []
    records: list[tuple[str, str]] = []
    for row in pins:
        if not isinstance(row, dict):
            raise ValueError("physical pin row must be an object")
        raw_number = row.get("pin_no")
        name = row.get("name")
        if (
            isinstance(raw_number, bool)
            or not isinstance(raw_number, (str, int))
            or not isinstance(name, str)
            or not name.strip()
        ):
            raise ValueError("physical pin row has an invalid identity")
        identifier = re.sub(r"\s+", "", str(raw_number).upper())
        identifier = re.sub(r"\(\d+\)$", "", identifier)
        if not identifier:
            raise ValueError("physical pin row has an empty identifier")
        identifiers.append(identifier)
        records.append(
            (
                identifier,
                re.sub(r"[^A-Z0-9]+", "", name.upper()),
            )
        )

    if len(set(records)) != len(records):
        raise ValueError("physical pin records contain exact duplicates")

    package_identity = _package_identity(package)
    if package_identity is None or package_identity[0] != "LQFP":
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("physical pin identifiers are not unique")
        return
    if any(not identifier.isdigit() for identifier in identifiers):
        raise ValueError("LQFP physical pin identifiers must be numeric")
    actual = {int(identifier) for identifier in identifiers}
    expected = set(range(1, expected_package_pins + 1))
    if actual != expected:
        raise ValueError(
            "LQFP physical pins do not match the package range"
        )


def _toc_entries(document: Any) -> list[TocEntry]:
    return [
        TocEntry(level=int(row[0]), title=str(row[1]).strip(), page=int(row[2]))
        for row in document.get_toc() or []
    ]


def _consecutive_definition_pages(
    document: Any,
    start_page: int,
    final_page: int,
) -> list[int]:
    pages: list[int] = []
    found_first = False
    for page_number in range(
        max(1, start_page),
        min(int(document.page_count), final_page) + 1,
    ):
        if _definition_tables(document[page_number - 1]):
            found_first = True
            pages.append(page_number)
        elif found_first:
            break
    return pages


def _walk_back_continuation(document: Any, start_page: int) -> int:
    page_number = start_page
    while page_number > 1 and _definition_tables(document[page_number - 2]):
        page_number -= 1
    return page_number


def _pin_section(entries: list[TocEntry]) -> tuple[int, TocEntry] | None:
    for index, entry in enumerate(entries):
        if entry.level <= 2 and classify_title(entry.title) == "pin_section":
            return index, entry
    return None


def _section_end(
    entries: list[TocEntry],
    start_index: int,
    document_pages: int,
) -> int:
    start_level = entries[start_index].level
    for entry in entries[start_index + 1 :]:
        normalized = entry.title.upper()
        if entry.level <= start_level and any(
            marker in normalized for marker in SECTION_STOPS
        ):
            return entry.page - 1
    return document_pages


def _jedec_cell(value: Any) -> str | None:
    normalized = re.sub(r"\(\d+\)$", "", _cell(value).upper().replace(" ", ""))
    return normalized if JEDEC.fullmatch(normalized) else None


def _corroborating_jedec_ids(
    document: Any,
    pages: list[int],
    selected_column: str,
) -> int:
    selected_identity = _package_identity(selected_column)
    identifiers: set[str] = set()
    for page_number in pages:
        for rows in _definition_tables(document[page_number - 1]):
            column_index = None
            for row in rows[:4]:
                for index, value in enumerate(row):
                    candidate = _cell(value)
                    identity = _package_identity(candidate)
                    if candidate == selected_column or identity == selected_identity:
                        column_index = index
            if column_index is None:
                continue
            for row in rows[1:]:
                if column_index < len(row):
                    if identifier := _jedec_cell(row[column_index]):
                        identifiers.add(identifier)
    return len(identifiers)


def _withheld(
    *,
    document_id: str,
    requested_package: str,
    expected_package_pins: int,
    source_path: str,
    reason: str,
    toc_class: str | None = None,
    toc_title: str | None = None,
    pages: list[int] | None = None,
    headers: list[str] | None = None,
) -> LocateResult:
    return LocateResult(
        document_id=document_id,
        requested_package=requested_package,
        expected_package_pins=expected_package_pins,
        source_path=source_path,
        status="withhold",
        reason=reason,
        toc_class=toc_class,
        toc_title=toc_title,
        pages_1based=tuple(pages or ()),
        column_header=None,
        column_headers_all=tuple(headers or ()),
        corroborating_jedec_ids=None,
    )


def locate_pin_definition_pages(
    document: Any,
    *,
    document_id: str,
    requested_package: str,
    expected_package_pins: int,
    source_path: str,
) -> LocateResult:
    if not document_id or not requested_package or expected_package_pins < 1:
        raise ValueError("invalid locator request")
    entries = _toc_entries(document)
    table_hits = [
        (index, entry)
        for index, entry in enumerate(entries)
        if classify_title(entry.title) == "definition_table"
    ]

    if table_hits:
        index, entry = table_hits[0]
        start_page = entry.page
        if "(continued)" in entry.title.casefold():
            start_page = _walk_back_continuation(document, start_page)
        final_page = (
            entries[index + 1].page - 1
            if index + 1 < len(entries)
            else int(document.page_count)
        )
        toc_class = "definition_table"
        toc_title = entry.title
    else:
        section = _pin_section(entries)
        if section is None:
            return _withheld(
                document_id=document_id,
                requested_package=requested_package,
                expected_package_pins=expected_package_pins,
                source_path=source_path,
                reason="toc_has_no_definition_table_and_no_pin_section",
            )
        index, entry = section
        start_page = entry.page
        final_page = _section_end(entries, index, int(document.page_count))
        toc_class = "pin_section"
        toc_title = entry.title

    pages = _consecutive_definition_pages(document, start_page, final_page)
    if not pages:
        return _withheld(
            document_id=document_id,
            requested_package=requested_package,
            expected_package_pins=expected_package_pins,
            source_path=source_path,
            reason="definition_table_pages_empty_find_tables",
            toc_class=toc_class,
            toc_title=toc_title,
        )

    headers: list[str] = []
    for page_number in pages[:3]:
        for rows in _definition_tables(document[page_number - 1]):
            for header in package_headers(rows):
                if header not in headers:
                    headers.append(header)
    selected_column = match_package_column(requested_package, headers)
    if selected_column is None:
        return _withheld(
            document_id=document_id,
            requested_package=requested_package,
            expected_package_pins=expected_package_pins,
            source_path=source_path,
            reason="package_column_not_in_table_headers",
            toc_class=toc_class,
            toc_title=toc_title,
            pages=pages,
            headers=headers,
        )

    return LocateResult(
        document_id=document_id,
        requested_package=requested_package,
        expected_package_pins=expected_package_pins,
        source_path=source_path,
        status="send",
        reason="toc_definition_table_and_header_match",
        toc_class=toc_class,
        toc_title=toc_title,
        pages_1based=tuple(pages),
        column_header=selected_column,
        column_headers_all=tuple(headers),
        corroborating_jedec_ids=_corroborating_jedec_ids(
            document,
            pages,
            selected_column,
        ),
    )


def locate_pdf(
    path: Path,
    *,
    document_id: str,
    requested_package: str,
    expected_package_pins: int,
) -> LocateResult:
    if path.is_symlink() or not path.is_file():
        raise ValueError("datasheet must be a regular file")
    with path.open("rb") as handle:
        signature = handle.read(5)
    if signature != b"%PDF-":
        raise ValueError("datasheet does not have a PDF signature")
    import pymupdf as fitz

    document = fitz.open(path)
    try:
        if document.page_count < 1 or document.needs_pass:
            raise ValueError("datasheet is empty or encrypted")
        return locate_pin_definition_pages(
            document,
            document_id=document_id,
            requested_package=requested_package,
            expected_package_pins=expected_package_pins,
            source_path=str(path),
        )
    finally:
        document.close()
