"""Deterministic PyMuPDF evidence extraction from indexed datasheet pages."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable, Mapping

from harness.electronics.claims import canonical_json


PAGE_EVIDENCE_SCHEMA = "harness.electronics-page-evidence.v1"


def selected_pages(
    profile: Mapping[str, Any],
    *,
    maximum_pages_per_lane: int,
) -> dict[int, set[str]]:
    if maximum_pages_per_lane < 1:
        raise ValueError("maximum_pages_per_lane must be positive")
    output: dict[int, set[str]] = defaultdict(set)
    for lane, pages in sorted((profile.get("lane_pages") or {}).items()):
        for page in sorted({int(value) for value in pages})[:maximum_pages_per_lane]:
            output[page].add(str(lane))
    for location in profile.get("exact_pin_locations") or []:
        if location.get("status") != "send":
            continue
        for page in location.get("pages_1based") or []:
            output[int(page)].update({"pin_or_ball", "pin_semantics"})
    return output


def _blocks(page: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in page.get_text("blocks") or []:
        if len(raw) < 5:
            continue
        text = str(raw[4] or "").strip()
        if not text:
            continue
        output.append(
            {
                "bbox": [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])],
                "text": text,
            }
        )
    return output


def _tables(page: Any) -> list[dict[str, Any]]:
    finder = page.find_tables()
    if finder is None:
        return []
    output: list[dict[str, Any]] = []
    for index, table in enumerate(finder.tables):
        rows = table.extract() or []
        if not rows:
            continue
        bbox = getattr(table, "bbox", None)
        output.append(
            {
                "table_index": index,
                "bbox": [float(value) for value in bbox] if bbox else None,
                "rows": [
                    [
                        None if cell is None else str(cell).strip()
                        for cell in row
                    ]
                    for row in rows
                ],
            }
        )
    return output


def extract_profile_evidence(
    document: Any,
    profile: Mapping[str, Any],
    *,
    maximum_pages_per_lane: int,
) -> Iterable[dict[str, Any]]:
    document_sha256 = str(profile["document_sha256"])
    for page_1based, lanes in sorted(
        selected_pages(
            profile,
            maximum_pages_per_lane=maximum_pages_per_lane,
        ).items()
    ):
        if not 1 <= page_1based <= int(document.page_count):
            raise ValueError(
                f"indexed page {page_1based} is outside document bounds"
            )
        page = document[page_1based - 1]
        blocks = _blocks(page)
        tables = (
            _tables(page)
            if lanes & {"pin_or_ball", "pin_semantics", "parametrics"}
            else []
        )
        core = {
            "schema": PAGE_EVIDENCE_SCHEMA,
            "document_sha256": document_sha256,
            "source_path": profile["source_path"],
            "page_1based": page_1based,
            "page_size": {
                "width": float(page.rect.width),
                "height": float(page.rect.height),
            },
            "lanes": sorted(lanes),
            "blocks": blocks,
            "tables": tables,
            "extractor": {
                "name": "pymupdf",
                "network_used": False,
                "ocr_used": False,
            },
        }
        core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
        yield core


__all__ = [
    "PAGE_EVIDENCE_SCHEMA",
    "extract_profile_evidence",
    "selected_pages",
]
