"""Deterministic PyMuPDF evidence extraction from indexed datasheet pages."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable, Mapping

from harness.electronics.claims import canonical_json


PAGE_EVIDENCE_SCHEMA = "harness.electronics-page-evidence.v2"


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
    for index, raw in enumerate(page.get_text("blocks") or []):
        if len(raw) < 5:
            continue
        text = str(raw[4] or "").strip()
        if not text:
            continue
        output.append(
            {
                "bbox": [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])],
                "text": text,
                "block_id": f"b{index}",
                "reading_order": index,
            }
        )
    return output


def _layout_ir(page: Any, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Emit a compact page-structure artifact for deterministic retries.

    Uses PyMuPDF's dict structure when available. Falls back to a block-only
    layout when a page has no text-layer or dict extraction fails.
    """

    output_blocks: list[dict[str, Any]] = []
    output_images: list[dict[str, Any]] = []
    try:
        raw = page.get_text("dict") or {}
    except Exception:  # noqa: BLE001
        raw = {}
    for index, block in enumerate(raw.get("blocks") or []):
        block_type = int(block.get("type", 0))
        bbox = [float(value) for value in block.get("bbox", ())] if block.get("bbox") else None
        block_id = f"b{index}"
        if block_type != 0:
            # non-text block (image/vector); retain geometry for downstream
            if bbox:
                output_images.append(
                    {
                        "block_id": block_id,
                        "kind": "visual_object",
                        "bbox": bbox,
                    }
                )
            continue
        lines_out: list[dict[str, Any]] = []
        line_index = 0
        for line in block.get("lines") or []:
            spans = line.get("spans") or []
            line_text = "".join(str(span.get("text") or "") for span in spans).strip()
            if not line_text:
                continue
            line_bbox = (
                [float(value) for value in line.get("bbox", ())]
                if line.get("bbox")
                else None
            )
            lines_out.append(
                {
                    "line_id": f"{block_id}.l{line_index}",
                    "text": line_text,
                    "bbox": line_bbox,
                }
            )
            line_index += 1
        if not lines_out or not bbox:
            continue
        output_blocks.append(
            {
                "block_id": block_id,
                "bbox": bbox,
                "reading_order": len(output_blocks),
                "lines": lines_out,
            }
        )
    if output_blocks:
        return {
            "source": "text_layer",
            "reading_order": "pymupdf_native",
            "text_blocks": output_blocks,
            "visual_blocks": output_images,
        }
    # Fallback artifact: preserve normalized block text + bbox even when full
    # line geometry is unavailable.
    return {
        "source": "normalized_blocks",
        "reading_order": "pymupdf_native",
        "text_blocks": [
            {
                "block_id": block["block_id"],
                "bbox": block["bbox"],
                "reading_order": block["reading_order"],
                "lines": [{"line_id": f"{block['block_id']}.l0", "text": block["text"], "bbox": block["bbox"]}],
            }
            for block in blocks
        ],
        "visual_blocks": output_images,
    }


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
    enable_ocr_fallback: bool = False,
    ocr_language: str = "eng",
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
        ocr_attempted = False
        ocr_used = False
        ocr_error: str | None = None
        extraction_status = "text_layer"
        if not blocks:
            extraction_status = "no_text_layer"
            if enable_ocr_fallback:
                ocr_attempted = True
                try:
                    # PyMuPDF OCR path: no network, fail-closed if unavailable.
                    textpage = page.get_textpage_ocr(language=ocr_language)
                    ocr_blocks_raw = page.get_text("blocks", textpage=textpage) or []
                    for index, raw in enumerate(ocr_blocks_raw):
                        if len(raw) < 5:
                            continue
                        text = str(raw[4] or "").strip()
                        if not text:
                            continue
                        blocks.append(
                            {
                                "bbox": [
                                    float(raw[0]),
                                    float(raw[1]),
                                    float(raw[2]),
                                    float(raw[3]),
                                ],
                                "text": text,
                                "block_id": f"ocr_b{index}",
                                "reading_order": index,
                            }
                        )
                    if blocks:
                        ocr_used = True
                        extraction_status = "ocr_fallback"
                    else:
                        extraction_status = "no_text_after_ocr"
                except Exception as exc:  # noqa: BLE001
                    ocr_error = f"{type(exc).__name__}: {exc}"
                    extraction_status = "ocr_failed"
        tables = (
            _tables(page)
            if lanes & {"pin_or_ball", "pin_semantics", "parametrics"}
            else []
        )
        layout_ir = _layout_ir(page, blocks)
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
            "layout_ir": layout_ir,
            "extraction_status": extraction_status,
            "needs_visual_verification": extraction_status in {
                "no_text_layer",
                "no_text_after_ocr",
                "ocr_failed",
            },
            "extractor": {
                "name": "pymupdf",
                "network_used": False,
                "ocr_used": ocr_used,
                "ocr_attempted": ocr_attempted,
                "ocr_language": ocr_language if ocr_attempted else None,
                "ocr_error": ocr_error,
            },
        }
        core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
        yield core


__all__ = [
    "PAGE_EVIDENCE_SCHEMA",
    "extract_profile_evidence",
    "selected_pages",
]
