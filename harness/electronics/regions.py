"""Structural datasheet-table discovery and focused evidence rendering."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from harness.electronics.locator import (
    PACKAGE_KEY,
    is_definition_table,
    package_headers,
)
from harness.electronics.table_extractors import (
    PARAMETER_HEADER,
    pin_semantic_roles,
)


PIN_HEADER = re.compile(
    r"\b(?:PIN|BALL|TERMINAL)(?:\s*/\s*(?:PIN|BALL))?\s*"
    r"(?:NUMBER|NO\.?|NAME|ASSIGNMENT)?\b",
    re.IGNORECASE,
)
NAME_HEADER = re.compile(
    r"\b(?:(?:PIN|BALL|SIGNAL|TERMINAL)\s*(?:/BALL\s*)?)?NAME\b",
    re.IGNORECASE,
)
SEMANTIC_HEADER = re.compile(
    r"\b(?:DESCRIPTION|FUNCTION|TYPE|DIRECTION)\b",
    re.IGNORECASE,
)
REGISTER_VETO = re.compile(
    r"\b(?:OFFSET|RESET\s+VALUE|BIT\s+FIELD|REGISTER)\b|0x[0-9A-F]{2,}",
    re.IGNORECASE,
)
ORDERING_VETO = re.compile(
    r"\b(?:ORDERING|ORDERABLE|PART\s+NUMBER|PACKAGE\s+OPTION)\b",
    re.IGNORECASE,
)
SUMMARY_SIGNAL = re.compile(
    r"\b(?:FEATURES?|APPLICATIONS?|DESCRIPTION|PRODUCT\s+(?:OVERVIEW|"
    r"SUMMARY)|DEVICE\s+(?:OVERVIEW|INFORMATION))\b",
    re.IGNORECASE,
)
OPN_SIGNAL = re.compile(
    r"\b(?:ORDERING\s+(?:INFORMATION|GUIDE)|ORDERABLE\s+PART\s+NUMBER|"
    r"PART\s+NUMBERING|DEVICE\s+NAMING|NOMENCLATURE|PACKAGE\s+OPTION)\b",
    re.IGNORECASE,
)
PARAMETRIC_ROLE_HEADER = re.compile(
    r"\b(?:MIN(?:IMUM)?|TYP(?:ICAL)?|MAX(?:IMUM)?|VALUE|"
    r"UNITS?|CONDITIONS?|TEST\s+CONDITIONS?)\b",
    re.IGNORECASE,
)
PHYSICAL_ID = re.compile(
    r"^(?:\d{1,4}|[A-HJ-NP-Z]{1,3}\d{1,3}|(?:EP|PAD|TAB|DAP)\d*)$",
    re.IGNORECASE,
)
PIN_COLUMN_HEADER = re.compile(
    r"^(?:PIN|BALL|TERMINAL)(?:\s*(?:NUMBER|NO\.?))?$",
    re.IGNORECASE,
)


def _cell(value: Any) -> str:
    return " ".join(str("" if value is None else value).split())


def pin_table_score(rows: Sequence[Sequence[Any]]) -> tuple[int, tuple[str, ...]]:
    """Score table structure without relying on vendor-specific pin names."""

    if len(rows) < 3:
        return 0, ("too_few_rows",)
    header = " | ".join(
        _cell(cell) for row in rows[:4] for cell in row if _cell(cell)
    )
    all_text = " | ".join(
        _cell(cell) for row in rows for cell in row if _cell(cell)
    )
    if REGISTER_VETO.search(all_text) and not (
        NAME_HEADER.search(header) and PIN_HEADER.search(header)
    ):
        return 0, ("register_table_veto",)

    maximum_columns = max((len(row) for row in rows), default=0)
    structural_columns = 0
    for column in range(maximum_columns):
        values = [
            _cell(row[column])
            for row in rows[1:]
            if column < len(row) and _cell(row[column])
        ]
        physical = [value for value in values if PHYSICAL_ID.fullmatch(value)]
        if len(physical) >= 3 and len(physical) / max(1, len(values)) >= 0.45:
            structural_columns += 1

    reasons: list[str] = []
    score = 0
    if is_definition_table([list(row) for row in rows]):
        score += 8
        reasons.append("locator_definition_table")
    if NAME_HEADER.search(header):
        score += 3
        reasons.append("pin_name_header")
    if PIN_HEADER.search(header):
        score += 2
        reasons.append("pin_identity_header")
    if SEMANTIC_HEADER.search(header):
        score += 1
        reasons.append("semantic_header")
    headers = package_headers([list(row) for row in rows])
    if headers:
        score += 2
        reasons.append("package_header")
    if structural_columns:
        score += min(3, structural_columns + 1)
        reasons.append("physical_identifier_column")
    if not {
        "locator_definition_table",
        "pin_name_header",
        "pin_identity_header",
    }.intersection(reasons):
        return 0, ("missing_pin_definition_header",)
    return score, tuple(reasons or ("no_structural_signal",))


def structural_pin_regions(
    page_evidence: Mapping[str, Any],
    *,
    minimum_score: int = 5,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for fallback_index, table in enumerate(page_evidence.get("tables") or []):
        if not isinstance(table, Mapping):
            continue
        rows = table.get("rows")
        bbox = table.get("bbox")
        if not isinstance(rows, list) or not (
            isinstance(bbox, list) and len(bbox) == 4
        ):
            continue
        score, reasons = pin_table_score(rows)
        if score < minimum_score:
            continue
        regions.append(
            {
                "table_index": int(table.get("table_index", fallback_index)),
                "bbox": [float(value) for value in bbox],
                "score": score,
                "reasons": list(reasons),
                "package_headers": package_headers(rows),
                "semantic_roles": list(pin_semantic_roles(table)),
                "rows": len(rows),
            }
        )
    return regions


def parametric_table_score(
    rows: Sequence[Sequence[Any]],
) -> tuple[int, tuple[str, ...]]:
    """Require an explicit parameter field and multiple value-role headers."""

    if len(rows) < 2:
        return 0, ("too_few_rows",)
    header_index = max(
        range(min(6, len(rows))),
        key=lambda index: sum(
            bool(
                PARAMETER_HEADER.search(_cell(cell))
                or PARAMETRIC_ROLE_HEADER.search(_cell(cell))
            )
            for cell in rows[index]
        ),
    )
    header = " | ".join(
        _cell(cell)
        for row in rows[: header_index + 1]
        for cell in row
        if _cell(cell)
    )
    all_text = " | ".join(
        _cell(cell) for row in rows for cell in row if _cell(cell)
    )
    if REGISTER_VETO.search(all_text):
        return 0, ("register_table_veto",)
    if ORDERING_VETO.search(header):
        return 0, ("ordering_table_veto",)
    if not PARAMETER_HEADER.search(header):
        return 0, ("missing_parameter_header",)

    roles = {
        match.group(0).casefold()
        for match in PARAMETRIC_ROLE_HEADER.finditer(header)
    }
    if len(roles) < 2:
        return 0, ("insufficient_value_headers",)

    reasons = ["parameter_header", "multiple_value_headers"]
    score = 7
    populated_rows = sum(
        sum(bool(_cell(cell)) for cell in row) >= 2 for row in rows[1:]
    )
    if populated_rows >= 2:
        score += 1
        reasons.append("multiple_data_rows")
    return score, tuple(reasons)


def structural_parametric_regions(
    page_evidence: Mapping[str, Any],
    *,
    minimum_score: int = 5,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for fallback_index, table in enumerate(page_evidence.get("tables") or []):
        if not isinstance(table, Mapping):
            continue
        rows = table.get("rows")
        bbox = table.get("bbox")
        if not isinstance(rows, list) or not (
            isinstance(bbox, list) and len(bbox) == 4
        ):
            continue
        score, reasons = parametric_table_score(rows)
        if score < minimum_score:
            continue
        regions.append(
            {
                "table_index": int(table.get("table_index", fallback_index)),
                "bbox": [float(value) for value in bbox],
                "score": score,
                "reasons": list(reasons),
                "package_headers": [],
                "rows": len(rows),
            }
        )
    return regions


def structural_text_regions(
    page_evidence: Mapping[str, Any],
    capability: str,
) -> list[dict[str, Any]]:
    """Return one source-bounded page region for summary or OPN evidence."""

    signals = {
        "series_summary": SUMMARY_SIGNAL,
        "opn_decoder": OPN_SIGNAL,
    }
    signal = signals.get(capability)
    if signal is None:
        raise ValueError(f"unsupported text-region capability: {capability}")
    blocks = [
        block
        for block in page_evidence.get("blocks") or []
        if isinstance(block, Mapping)
        and isinstance(block.get("bbox"), list)
        and len(block["bbox"]) == 4
        and _cell(block.get("text"))
    ]
    tables = [
        table
        for table in page_evidence.get("tables") or []
        if isinstance(table, Mapping)
        and isinstance(table.get("bbox"), list)
        and len(table["bbox"]) == 4
    ]
    visible_text = "\n".join(
        [
            *(_cell(block.get("text")) for block in blocks),
            *(
                " | ".join(
                    _cell(cell)
                    for row in table.get("rows") or []
                    if isinstance(row, list)
                    for cell in row
                    if _cell(cell)
                )
                for table in tables
            ),
        ]
    )
    if not signal.search(visible_text):
        return []
    bboxes = [
        *([float(value) for value in block["bbox"]] for block in blocks),
        *([float(value) for value in table["bbox"]] for table in tables),
    ]
    if not bboxes:
        return []
    return [
        {
            "table_index": None,
            "bbox": [
                min(bbox[0] for bbox in bboxes),
                min(bbox[1] for bbox in bboxes),
                max(bbox[2] for bbox in bboxes),
                max(bbox[3] for bbox in bboxes),
            ],
            "score": 5,
            "reasons": [
                (
                    "visible_summary_evidence"
                    if capability == "series_summary"
                    else "visible_ordering_evidence"
                )
            ],
            "package_headers": [],
            "semantic_roles": [],
            "rows": sum(
                len(table.get("rows") or [])
                for table in tables
            ),
        }
    ]


def printed_package_mentions(
    page_evidence: Mapping[str, Any],
) -> tuple[str, ...]:
    mentions: list[str] = []
    for block in page_evidence.get("blocks") or []:
        if not isinstance(block, Mapping):
            continue
        text = _cell(block.get("text"))
        for match in PACKAGE_KEY.finditer(text):
            value = _cell(match.group(0))
            if value and value.casefold() not in {
                item.casefold() for item in mentions
            }:
                mentions.append(value)
    return tuple(mentions)


def pdftotext_layout_page(
    source_path: Path,
    page_1based: int,
    *,
    maximum_characters: int = 100_000,
    timeout_seconds: float = 15,
) -> dict[str, Any]:
    executable = shutil.which("pdftotext")
    if timeout_seconds <= 0:
        raise ValueError("pdftotext timeout must be positive")
    if executable is None:
        return {
            "extractor": "pdftotext-layout",
            "available": False,
            "text": "",
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
    try:
        completed = subprocess.run(
            [
                executable,
                "-f",
                str(page_1based),
                "-l",
                str(page_1based),
                "-layout",
                str(source_path),
                "-",
            ],
            check=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "extractor": "pdftotext-layout",
            "available": False,
            "text": "",
            "error": f"{type(exc).__name__}: {exc}",
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
    text = completed.stdout.decode("utf-8", errors="replace")
    truncated = len(text) > maximum_characters
    text = text[:maximum_characters]
    payload = text.encode("utf-8")
    return {
        "extractor": "pdftotext-layout",
        "available": True,
        "text": text,
        "truncated": truncated,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def render_focused_regions(
    source_path: Path,
    page_1based: int,
    regions: Sequence[Mapping[str, Any]],
    destination: Path,
    *,
    dpi: int,
    margin_points: float = 6.0,
) -> dict[str, Any]:
    if not regions:
        raise ValueError("focused rendering requires structural regions")
    import pymupdf

    with pymupdf.open(source_path) as document:
        page = document[page_1based - 1]
        x0 = max(0.0, min(float(region["bbox"][0]) for region in regions) - margin_points)
        y0 = max(0.0, min(float(region["bbox"][1]) for region in regions) - margin_points)
        x1 = min(
            float(page.rect.width),
            max(float(region["bbox"][2]) for region in regions) + margin_points,
        )
        y1 = min(
            float(page.rect.height),
            max(float(region["bbox"][3]) for region in regions) + margin_points,
        )
        clip = pymupdf.Rect(x0, y0, x1, y1)
        if clip.is_empty or clip.is_infinite:
            raise ValueError("structural region produced an invalid crop")
        pixmap = page.get_pixmap(dpi=dpi, alpha=False, clip=clip)
        pixmap.save(destination)
    return {
        "rendering": "focused_structural_regions",
        "page_1based": page_1based,
        "source_bboxes": [list(region["bbox"]) for region in regions],
        "clip_bbox": [x0, y0, x1, y1],
        "dpi": dpi,
        "width": pixmap.width,
        "height": pixmap.height,
    }


def render_full_page(
    source_path: Path,
    page_1based: int,
    destination: Path,
    *,
    dpi: int = 120,
) -> dict[str, Any]:
    """Render one complete source page without reconstructed table text."""

    import pymupdf

    with pymupdf.open(source_path) as document:
        page = document[page_1based - 1]
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        pixmap.save(destination)
        page_bbox = [
            float(page.rect.x0),
            float(page.rect.y0),
            float(page.rect.x1),
            float(page.rect.y1),
        ]
    return {
        "rendering": "full_source_page",
        "page_1based": page_1based,
        "source_bbox": page_bbox,
        "dpi": dpi,
        "width": pixmap.width,
        "height": pixmap.height,
    }


def render_package_columns(
    source_path: Path,
    page_1based: int,
    column_header: str,
    destination: Path,
    *,
    dpi: int,
    include_semantics: bool = False,
) -> dict[str, Any]:
    """Render the selected package IDs beside shared name/semantic columns."""

    import pymupdf

    selected = _cell(column_header).casefold()
    with pymupdf.open(source_path) as document:
        page = document[page_1based - 1]
        matches = []
        tables = page.find_tables()
        for table in tables.tables if tables is not None else ():
            rows = table.extract() or []
            score, _reasons = pin_table_score(rows)
            if score < 5:
                continue
            package_hits = []
            name_hits = []
            printed_packages = set(package_headers(rows))
            all_package_columns = {
                column_index
                for row in rows[:4]
                for column_index, raw in enumerate(row)
                if _cell(raw) in printed_packages
            }
            for row_index, row in enumerate(rows[:4]):
                for column_index, raw in enumerate(row):
                    value = _cell(raw)
                    if value.casefold() == selected:
                        package_hits.append((row_index, column_index))
                    if (
                        NAME_HEADER.search(value)
                        and column_index not in all_package_columns
                    ):
                        name_hits.append((row_index, column_index))
            package_columns = {column for _, column in package_hits}
            if len(package_columns) != 1:
                continue
            name_columns = {column for _, column in name_hits}
            if len(name_columns) == 1:
                name_column = next(iter(name_columns))
                matches.append(
                    (
                        table,
                        package_hits[0],
                        next(
                            position
                            for position in name_hits
                            if position[1] == name_column
                        ),
                    )
                )
        if len(matches) != 1:
            raise ValueError(
                "selected package column is not uniquely resolvable"
            )
        table, package_position, name_position = matches[0]
        pin_cell = table.rows[package_position[0]].cells[package_position[1]]
        name_cell = table.rows[name_position[0]].cells[name_position[1]]
        if pin_cell is None or name_cell is None:
            raise ValueError("selected header cell has no geometry")
        table_bbox = tuple(float(value) for value in table.bbox)
        pin_bbox = (
            float(pin_cell[0]),
            table_bbox[1],
            float(pin_cell[2]),
            table_bbox[3],
        )
        name_bbox = (
            float(name_cell[0]),
            table_bbox[1],
            table_bbox[2] if include_semantics else float(name_cell[2]),
            table_bbox[3],
        )
        height = table_bbox[3] - table_bbox[1]
        pin_width = max(90.0, (pin_bbox[2] - pin_bbox[0]) * 2.5)
        name_width = max(
            220.0,
            (name_bbox[2] - name_bbox[0])
            * (1.25 if include_semantics else 2.0),
        )
        gap = 6.0
        focused = pymupdf.open()
        try:
            target = focused.new_page(
                width=pin_width + gap + name_width,
                height=height,
            )
            target.show_pdf_page(
                pymupdf.Rect(0, 0, pin_width, height),
                document,
                page.number,
                clip=pymupdf.Rect(pin_bbox),
                keep_proportion=False,
            )
            target.show_pdf_page(
                pymupdf.Rect(
                    pin_width + gap,
                    0,
                    pin_width + gap + name_width,
                    height,
                ),
                document,
                page.number,
                clip=pymupdf.Rect(name_bbox),
                keep_proportion=False,
            )
            target.draw_line(
                pymupdf.Point(pin_width + gap / 2, 0),
                pymupdf.Point(pin_width + gap / 2, height),
                color=(0, 0, 0),
                width=1,
            )
            pixmap = target.get_pixmap(dpi=dpi, alpha=False)
            pixmap.save(destination)
        finally:
            focused.close()
    return {
        "rendering": (
            "focused_package_name_and_semantic_columns"
            if include_semantics
            else "focused_package_and_name_columns"
        ),
        "page_1based": page_1based,
        "column_header": column_header,
        "source_table_bbox": list(table_bbox),
        "source_pin_bbox": list(pin_bbox),
        "source_pin_name_bbox": list(name_bbox),
        "dpi": dpi,
        "width": pixmap.width,
        "height": pixmap.height,
    }


__all__ = [
    "parametric_table_score",
    "pdftotext_layout_page",
    "pin_table_score",
    "printed_package_mentions",
    "render_focused_regions",
    "render_full_page",
    "render_package_columns",
    "structural_parametric_regions",
    "structural_pin_regions",
    "structural_text_regions",
]
