"""PyMuPDF-first section indexing for electronics datasheets."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from harness.electronics.locator import locate_pin_definition_pages


PAGE_INDEX_SCHEMA = "harness.electronics-page-index.v1"
# Longer tokens first so \b alternation prefers UFBGA over BGA, WSON over SON.
_PACKAGE_FAMILY = re.compile(
    r"\b(UFQFPN|VFQFPN|HTSSOP|DSBGA|WLCSP|UFBGA|TFBGA|LFBGA|VFBGA|NFBGA|"
    r"HSOIC|TSSOP|VSSOP|LQFP|TQFP|VQFN|UQFN|WQFN|SSOP|MSOP|SOIC|PDIP|SDIP|"
    r"WSON|VSON|QFP|QFN|BGA|LGA|SOT|SON)\b"
)
_PACKAGE_SCAN_PAGES = 30


def _printed_package_families(document: Any) -> set[str]:
    """Distinct uppercase package family tokens printed in the front matter.

    Used to decide whether a document plausibly ships a single package
    variant. Multi-package MCU datasheets (for example LQFP plus QFN) print
    several family names; attesting single-package from a lone pin-count
    binding alone sent wrong-package pages to the teacher in the mcupin4
    run and wasted most of that batch.
    """

    families: set[str] = set()
    for page_number in range(min(int(document.page_count), _PACKAGE_SCAN_PAGES)):
        for match in _PACKAGE_FAMILY.finditer(document[page_number].get_text()):
            families.add(match.group(1))
    return families
LANE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "pin_or_ball": (
        re.compile(r"\bPIN(?:OUT|OUTS)?\b", re.IGNORECASE),
        re.compile(r"\bBALL(?:OUT|OUTS)?\b", re.IGNORECASE),
        re.compile(r"\bPIN\s*(?:AND|/)\s*BALL\b", re.IGNORECASE),
        re.compile(r"\bTERMINAL\s+(?:FUNCTIONS?|ASSIGNMENTS?)\b", re.IGNORECASE),
    ),
    "pin_semantics": (
        re.compile(r"\bPIN\s+(?:DESCRIPTION|FUNCTION)", re.IGNORECASE),
        re.compile(r"\bBALL\s+(?:DESCRIPTION|FUNCTION)", re.IGNORECASE),
        re.compile(r"\bALTERNATE\s+FUNCTION", re.IGNORECASE),
        re.compile(r"\bMULTIPLEX", re.IGNORECASE),
    ),
    "parametrics": (
        re.compile(r"\bELECTRICAL\s+CHARACTERISTICS?\b", re.IGNORECASE),
        re.compile(r"\bABSOLUTE\s+MAXIMUM\b", re.IGNORECASE),
        re.compile(r"\bRECOMMENDED\s+OPERATING\b", re.IGNORECASE),
        re.compile(r"\bCLOCK", re.IGNORECASE),
        re.compile(r"\bMEMORY\b", re.IGNORECASE),
        re.compile(r"\bPOWER\s+(?:CONSUMPTION|MODES?)\b", re.IGNORECASE),
        re.compile(r"\bTHERMAL\s+CHARACTERISTICS?\b", re.IGNORECASE),
    ),
    "series_summary": (
        re.compile(r"\bFEATURES?\b", re.IGNORECASE),
        re.compile(r"\bDESCRIPTION\b", re.IGNORECASE),
        re.compile(r"\bOVERVIEW\b", re.IGNORECASE),
        re.compile(r"\bAPPLICATIONS?\b", re.IGNORECASE),
        re.compile(r"\bPRODUCT\s+SUMMARY\b", re.IGNORECASE),
    ),
    "opn_decoder": (
        re.compile(r"\bORDERING\s+INFORMATION\b", re.IGNORECASE),
        re.compile(r"\bPART\s+NUMBERING\b", re.IGNORECASE),
        re.compile(r"\bDEVICE\s+INFORMATION\b", re.IGNORECASE),
        re.compile(r"\bNOMENCLATURE\b", re.IGNORECASE),
        re.compile(r"\bORDERABLE\s+(?:PART|DEVICE)", re.IGNORECASE),
    ),
}
HEADING_MAX_LENGTH = 180


def classify_section(title: str) -> tuple[str, ...]:
    normalized = " ".join(str(title or "").split())
    output = [
        lane
        for lane, patterns in LANE_PATTERNS.items()
        if any(pattern.search(normalized) for pattern in patterns)
    ]
    return tuple(output)


def _toc(document: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in document.get_toc() or []:
        if len(raw) < 3:
            continue
        try:
            level, title, page = int(raw[0]), str(raw[1]).strip(), int(raw[2])
        except (TypeError, ValueError):
            continue
        if not title or page < 1 or page > int(document.page_count):
            continue
        lanes = classify_section(title)
        if lanes:
            output.append(
                {
                    "level": level,
                    "title": title,
                    "page_1based": page,
                    "lanes": list(lanes),
                    "source": "toc",
                }
            )
    return output


def _heading_lines(text: str) -> Iterable[str]:
    for line in text.splitlines():
        normalized = " ".join(line.split())
        if (
            3 <= len(normalized) <= HEADING_MAX_LENGTH
            and classify_section(normalized)
        ):
            yield normalized


def _fallback_headings(
    document: Any,
    *,
    existing_lanes: set[str],
    maximum_pages: int | None,
) -> list[dict[str, Any]]:
    missing = set(LANE_PATTERNS) - existing_lanes
    if not missing:
        return []
    page_limit = int(document.page_count)
    if maximum_pages is not None:
        page_limit = min(page_limit, maximum_pages)
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for page_index in range(page_limit):
        text = document[page_index].get_text("text") or ""
        for title in _heading_lines(text):
            lanes = [lane for lane in classify_section(title) if lane in missing]
            if not lanes:
                continue
            identity = (page_index + 1, title.casefold())
            if identity in seen:
                continue
            seen.add(identity)
            output.append(
                {
                    "level": None,
                    "title": title,
                    "page_1based": page_index + 1,
                    "lanes": lanes,
                    "source": "page_text",
                }
            )
    return output


def index_document(
    document: Any,
    *,
    document_sha256: str,
    source_path: Path,
    package_requests: Iterable[tuple[str, int]] = (),
    fallback_maximum_pages: int | None = None,
) -> dict[str, Any]:
    if len(document_sha256) != 64:
        raise ValueError("document_sha256 is invalid")
    toc_entries = _toc(document)
    toc_lanes = {
        lane for entry in toc_entries for lane in entry.get("lanes") or []
    }
    fallback = _fallback_headings(
        document,
        existing_lanes=toc_lanes,
        maximum_pages=fallback_maximum_pages,
    )
    entries = [*toc_entries, *fallback]
    pages: dict[str, set[int]] = defaultdict(set)
    for entry in entries:
        for lane in entry["lanes"]:
            pages[lane].add(int(entry["page_1based"]))
    exact: list[dict[str, Any]] = []
    unique_requests: list[tuple[str, int]] = []
    for package, expected_pins in package_requests:
        request = (str(package).strip(), int(expected_pins))
        if request[0] and request[1] >= 1 and request not in unique_requests:
            unique_requests.append(request)
    # The borderless-table relaxation needs no package-column projection, so
    # it is only safe when the document binds exactly one package request AND
    # its printed front matter names at most one package family, matching the
    # request. A lone binding is a weak proxy on multi-package documents.
    single_package = False
    if len(unique_requests) == 1:
        requested_family = re.sub(
            r"[-\s]?\d+$", "", unique_requests[0][0].upper()
        )
        single_package = bool(requested_family) and _printed_package_families(
            document
        ) <= {requested_family}
    for request in unique_requests:
        result = locate_pin_definition_pages(
            document,
            document_id=document_sha256,
            requested_package=request[0],
            expected_package_pins=request[1],
            source_path=str(source_path),
            text_section_single_package=single_package,
        )
        exact.append(
            {
                "package": result.requested_package,
                "expected_package_pins": result.expected_package_pins,
                "status": result.status,
                "reason": result.reason,
                "pages_1based": list(result.pages_1based),
                "column_header": result.column_header,
                "column_headers_all": list(result.column_headers_all),
                "corroborating_jedec_ids": result.corroborating_jedec_ids,
            }
        )
    core = {
        "schema": PAGE_INDEX_SCHEMA,
        "document_sha256": document_sha256,
        "source_path": str(source_path),
        "page_count": int(document.page_count),
        "section_entries": entries,
        "lane_pages": {
            lane: sorted(lane_pages) for lane, lane_pages in sorted(pages.items())
        },
        "exact_pin_locations": exact,
    }
    core["evidence_sha256"] = hashlib.sha256(
        json_bytes(core)
    ).hexdigest()
    return core


def json_bytes(value: Mapping[str, Any]) -> bytes:
    import json

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def package_requests_from_ground_truth(
    records: Iterable[Mapping[str, Any]],
) -> tuple[tuple[str, int], ...]:
    requests: set[tuple[str, int]] = set()
    for value in records:
        pinout = value.get("pinout")
        if not isinstance(pinout, Mapping):
            continue
        packages = pinout.get("packages")
        declared = pinout.get("declared_pin_total")
        if not isinstance(packages, list):
            continue
        for package in packages:
            text = str(package or "").strip()
            matches = [int(item) for item in re.findall(r"\b(\d{2,4})\b", text)]
            expected = matches[0] if matches else None
            if expected is None and isinstance(declared, int):
                expected = int(declared)
            if text and expected:
                requests.add((text, expected))
    return tuple(sorted(requests))


__all__ = [
    "PAGE_INDEX_SCHEMA",
    "classify_section",
    "index_document",
    "package_requests_from_ground_truth",
]
