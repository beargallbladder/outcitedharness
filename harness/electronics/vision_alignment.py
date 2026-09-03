"""Evidence-based alignment of pinout labels to datasheet definition tables."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


_PRIMARY_TOC = re.compile(
    r"(?i)(pin(?:/ball)?\s+definitions?|ball\s+definitions?|"
    r"terminal\s+functions?|signal\s+descriptions?|pin\s+list)"
)
_SECONDARY_TOC = re.compile(
    r"(?i)(pinout.*pin\s+description|pin\s+descriptions?|"
    r"terminal\s+configuration.*functions)"
)
_EXCLUDED_TOC = re.compile(
    r"(?i)(figure|characteristic|loading|voltage|protection|timing|"
    r"package\s+information|mechanical|outline|alternate\s+function)"
)
_PACKAGE_NOISE = {
    "BALL",
    "BALLS",
    "LEAD",
    "LEADS",
    "PACKAGE",
    "PACKAGES",
    "PIN",
    "PINS",
    "NAME",
    "NUMBER",
    "TERMINAL",
}
_PACKAGE_FAMILIES = {
    "BGA",
    "FBGA",
    "LFBGA",
    "MAPBGA",
    "NFBGA",
    "TFBGA",
    "UFBGA",
    "VFBGA",
    "WLCSP",
    "LQFP",
    "QFP",
    "TQFP",
    "QFN",
    "VQFN",
    "VFQFPN",
    "UFQFPN",
    "DIP",
    "PDIP",
    "SOIC",
    "TSSOP",
}


@dataclass(frozen=True)
class TableEvidence:
    page_1based: int
    table_index: int
    bbox: tuple[float, float, float, float]
    number_column: int
    name_column: int
    package_header: str
    package_candidate: str
    header_bbox: tuple[float, float, float, float]
    matched_rows: tuple[int, ...]
    matched_target_indices: tuple[int, ...]
    matched_row_bboxes: tuple[tuple[float, float, float, float], ...]


def normalize_pin_number(value: Any) -> str:
    normalized = re.sub(
        r"\s+",
        "",
        str("" if value is None else value).upper(),
    )
    return re.sub(r"\(\d+\)$", "", normalized)


def normalize_pin_name(value: Any) -> str:
    return re.sub(
        r"[^A-Z0-9]+",
        "",
        str("" if value is None else value).upper(),
    )


def target_identities(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], int]:
    identities: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        identity = (
            normalize_pin_number(row.get("pin_no")),
            normalize_pin_name(row.get("name")),
        )
        if not all(identity):
            raise ValueError("target contains an invalid pin identity")
        if identity in identities:
            raise ValueError("target contains a duplicate pin identity")
        identities[identity] = index
    if len(identities) < 8:
        raise ValueError("target contains fewer than eight pin identities")
    return identities


def _toc_score(title: str) -> int:
    if _EXCLUDED_TOC.search(title):
        return 0
    if _PRIMARY_TOC.search(title):
        return 2
    if _SECONDARY_TOC.search(title):
        return 1
    return 0


def definition_pages(
    document: Any,
    *,
    hinted_pages_1based: Iterable[int] = (),
    maximum_span_pages: int = 64,
) -> tuple[int, ...]:
    """Return conservative definition-table page candidates from PDF structure."""

    entries = [
        (int(row[0]), str(row[1]).strip(), int(row[2]))
        for row in document.get_toc() or []
        if len(row) >= 3
    ]
    scored = [
        (index, level, title, page, _toc_score(title))
        for index, (level, title, page) in enumerate(entries)
        if _toc_score(title)
    ]
    if scored:
        best_score = max(row[4] for row in scored)
        scored = [row for row in scored if row[4] == best_score]

    pages: set[int] = {
        int(page)
        for page in hinted_pages_1based
        if 1 <= int(page) <= int(document.page_count)
    }
    for entry_index, level, _title, start, _score in scored:
        end = min(int(document.page_count), start + maximum_span_pages - 1)
        for next_level, _next_title, next_page in entries[entry_index + 1 :]:
            if next_page > start and next_level <= level:
                end = min(end, next_page - 1)
                break
        pages.update(range(max(1, start), max(1, end) + 1))

    if pages:
        return tuple(sorted(pages))

    header = re.compile(
        r"(?i)(pin/ball\s+name|signal\s+name.{0,40}(?:pin|ball)|"
        r"(?:pin|ball)\s+(?:number|name).{0,40}(?:function|description))",
        re.DOTALL,
    )
    for page_index in range(int(document.page_count)):
        text = document[page_index].get_text() or ""
        if header.search(text):
            pages.add(page_index + 1)
    return tuple(sorted(pages))


def extract_page_tables(
    document: Any,
    pages_1based: Iterable[int],
) -> dict[int, list[dict[str, Any]]]:
    """Extract ruled tables while retaining page and bounding-box evidence."""

    output: dict[int, list[dict[str, Any]]] = {}
    for page_number in sorted(set(pages_1based)):
        page = document[page_number - 1]
        found = page.find_tables()
        tables: list[dict[str, Any]] = []
        for table_index, table in enumerate(found.tables if found is not None else []):
            rows = table.extract() or []
            column_count = max((len(row) for row in rows), default=0)
            if not 3 <= len(rows) <= 500 or not 2 <= column_count <= 32:
                continue
            tables.append(
                {
                    "index": table_index,
                    "bbox": tuple(float(value) for value in table.bbox),
                    "rows": rows,
                    "row_bboxes": [
                        tuple(float(value) for value in row.bbox)
                        for row in table.rows
                    ],
                }
            )
        if tables:
            output[page_number] = tables
    return output


def _package_tokens(value: str) -> tuple[set[str], set[str], set[str]]:
    compact = re.sub(r"[^A-Z0-9]+", "", value.upper())
    raw = set(re.findall(r"[A-Z]+[A-Z0-9]*|\d+", value.upper()))
    numbers = set(re.findall(r"\d+", value))
    families = {
        family
        for family in _PACKAGE_FAMILIES
        if family in compact
    }
    tokens = {
        token
        for token in raw
        if token not in _PACKAGE_NOISE and token not in families
    }
    return tokens, numbers, families


def resolve_package_header(
    header: str,
    package_candidates: Sequence[str],
) -> str | None:
    """Resolve a table column header to exactly one declared package."""

    header_tokens, header_numbers, header_families = _package_tokens(header)
    scored: list[tuple[int, str]] = []
    for package in package_candidates:
        tokens, numbers, families = _package_tokens(package)
        code_overlap = {
            token
            for token in tokens & header_tokens
            if not token.isdigit() and len(token) >= 2
        }
        number_overlap = numbers & header_numbers
        family_overlap = families & header_families
        score = 4 * len(code_overlap) + 3 * len(family_overlap) + len(number_overlap)
        if code_overlap or (family_overlap and number_overlap):
            scored.append((score, package))
    if not scored:
        return None
    best = max(score for score, _package in scored)
    winners = [package for score, package in scored if score == best]
    return winners[0] if len(winners) == 1 else None


def _column_evidence(
    *,
    page_number: int,
    table: Mapping[str, Any],
    number_column: int,
    name_column: int,
    identities: Mapping[tuple[str, str], int],
    package_candidates: Sequence[str],
) -> TableEvidence | None:
    rows = table["rows"]
    matched_rows: list[int] = []
    matched_target_indices: list[int] = []
    for row_index, row in enumerate(rows):
        if number_column >= len(row) or name_column >= len(row):
            continue
        identity = (
            normalize_pin_number(row[number_column]),
            normalize_pin_name(row[name_column]),
        )
        target_index = identities.get(identity)
        if target_index is not None:
            matched_rows.append(row_index)
            matched_target_indices.append(target_index)
    if len(set(matched_target_indices)) < 2:
        return None

    first_data_row = min(matched_rows)
    header_cells = []
    for row in rows[: min(first_data_row, 4)]:
        if number_column < len(row):
            value = re.sub(r"\s+", " ", str(row[number_column] or "")).strip()
            if value and value not in header_cells:
                header_cells.append(value)
    package_header = " | ".join(header_cells)
    package = resolve_package_header(package_header, package_candidates)
    if package is None:
        return None
    row_bboxes = table["row_bboxes"]
    header_rows = row_bboxes[:first_data_row]
    if not header_rows or any(index >= len(row_bboxes) for index in matched_rows):
        return None
    table_bbox = tuple(float(value) for value in table["bbox"])
    header_bbox = (
        table_bbox[0],
        min(row[1] for row in header_rows),
        table_bbox[2],
        max(row[3] for row in header_rows),
    )
    return TableEvidence(
        page_1based=page_number,
        table_index=int(table["index"]),
        bbox=tuple(table["bbox"]),
        number_column=number_column,
        name_column=name_column,
        package_header=package_header,
        package_candidate=package,
        header_bbox=header_bbox,
        matched_rows=tuple(matched_rows),
        matched_target_indices=tuple(matched_target_indices),
        matched_row_bboxes=tuple(
            tuple(row_bboxes[index]) for index in matched_rows
        ),
    )


def _row_crop_chunks(
    evidence: Sequence[TableEvidence],
    *,
    maximum_rows: int = 8,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    used_target_indices: set[int] = set()
    for item in sorted(
        evidence,
        key=lambda value: (value.page_1based, value.table_index),
    ):
        pending: list[
            tuple[int, int, tuple[float, float, float, float]]
        ] = []
        for row, target_index, bbox in zip(
            item.matched_rows,
            item.matched_target_indices,
            item.matched_row_bboxes,
            strict=True,
        ):
            if target_index in used_target_indices:
                continue
            if pending and (
                row != pending[-1][0] + 1 or len(pending) >= maximum_rows
            ):
                chunks.append(_row_crop_chunk(item, pending))
                pending = []
            pending.append((row, target_index, bbox))
            used_target_indices.add(target_index)
        if pending:
            chunks.append(_row_crop_chunk(item, pending))
    return chunks


def _row_crop_chunk(
    item: TableEvidence,
    rows: Sequence[tuple[int, int, tuple[float, float, float, float]]],
) -> dict[str, Any]:
    return {
        "page_1based": item.page_1based,
        "table_index": item.table_index,
        "package_candidate": item.package_candidate,
        "package_header": item.package_header,
        "number_column": item.number_column,
        "name_column": item.name_column,
        "header_bbox": item.header_bbox,
        "body_bbox": (
            min(row[2][0] for row in rows),
            min(row[2][1] for row in rows),
            max(row[2][2] for row in rows),
            max(row[2][3] for row in rows),
        ),
        "source_rows": [row[0] for row in rows],
        "target_indices": [row[1] for row in rows],
    }


def align_record(
    *,
    tables_by_page: Mapping[int, Sequence[Mapping[str, Any]]],
    target_rows: Sequence[Mapping[str, Any]],
    package_candidates: Sequence[str],
    minimum_coverage: float = 0.9,
) -> dict[str, Any]:
    """Align one frontier target to one unambiguous package table column."""

    identities = target_identities(target_rows)
    by_package: dict[str, list[TableEvidence]] = defaultdict(list)
    for page_number, tables in tables_by_page.items():
        for table in tables:
            column_count = max((len(row) for row in table["rows"]), default=0)
            best_for_package: dict[str, TableEvidence] = {}
            for number_column in range(column_count):
                for name_column in range(column_count):
                    if name_column == number_column:
                        continue
                    evidence = _column_evidence(
                        page_number=page_number,
                        table=table,
                        number_column=number_column,
                        name_column=name_column,
                        identities=identities,
                        package_candidates=package_candidates,
                    )
                    if evidence is None:
                        continue
                    current = best_for_package.get(evidence.package_candidate)
                    if current is None or len(evidence.matched_target_indices) > len(
                        current.matched_target_indices
                    ):
                        best_for_package[evidence.package_candidate] = evidence
            for package, evidence in best_for_package.items():
                by_package[package].append(evidence)

    package_results: list[tuple[int, str, set[int], list[TableEvidence]]] = []
    for package, evidence in by_package.items():
        matched = {
            target_index
            for item in evidence
            for target_index in item.matched_target_indices
        }
        package_results.append((len(matched), package, matched, evidence))
    package_results.sort(key=lambda row: (-row[0], row[1]))
    if not package_results:
        return {
            "status": "withhold",
            "reason": "no_package_column_with_exact_pin_identity_matches",
            "target_rows": len(identities),
            "matched_rows": 0,
            "coverage": 0.0,
            "package_candidates_scored": {},
            "row_crop_status": "withhold",
            "row_crop_examples": 0,
            "row_crop_target_rows": 0,
            "row_crop_chunks": [],
            "tables": [],
        }

    matched_count, package, matched, evidence = package_results[0]
    runner_up = package_results[1][0] if len(package_results) > 1 else 0
    required_margin = max(2, math.ceil(len(identities) * 0.02))
    coverage = matched_count / len(identities)
    package_unambiguous = not runner_up or (
        matched_count - runner_up >= required_margin
    )
    if matched_count < 8:
        status, reason = "withhold", "fewer_than_eight_exact_visible_rows"
    elif not package_unambiguous:
        status, reason = "withhold", "package_column_alignment_is_ambiguous"
    elif coverage < minimum_coverage:
        status, reason = "withhold", "exact_visible_row_coverage_below_threshold"
    else:
        status, reason = "aligned", "package_column_and_pin_rows_exactly_matched"

    selected = [
        item
        for item in evidence
        if set(item.matched_target_indices) & matched
    ]
    row_crop_chunks = (
        _row_crop_chunks(selected)
        if package_unambiguous and matched_count >= 8
        else []
    )
    return {
        "status": status,
        "reason": reason,
        "target_rows": len(identities),
        "matched_rows": matched_count,
        "coverage": round(coverage, 6),
        "selected_package": package,
        "runner_up_matched_rows": runner_up,
        "required_margin": required_margin,
        "matched_target_indices": sorted(matched),
        "row_crop_status": "eligible" if row_crop_chunks else "withhold",
        "row_crop_examples": len(row_crop_chunks),
        "row_crop_target_rows": len(
            {
                index
                for chunk in row_crop_chunks
                for index in chunk["target_indices"]
            }
        ),
        "row_crop_chunks": row_crop_chunks,
        "package_candidates_scored": {
            candidate: count
            for count, candidate, _matched, _evidence in package_results
        },
        "tables": [asdict(item) for item in selected],
    }
