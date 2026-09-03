#!/usr/bin/env python3
"""Compare one local model on PDF text versus rendered pin-table pages."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from harness.electronics import locator
from harness.electronics.locator import (
    locate_pin_definition_pages,
    validate_physical_pin_truth,
)


FIXTURE_SCHEMA = "harness.datasheet-modality-fixture.v1"
RESULT_SCHEMA = "harness.datasheet-modality-evaluation.v4"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MODES = ("table", "text", "image", "image_focused", "image_rows")


SYSTEM_PROMPT = """You extract physical pin identity rows from one datasheet
pin-definition table page. Read ONLY the requested package column and its pin-name
column. Ignore alternate-function tables, package drawings, and other packages.
Return only rows visibly present on this page. Omit blank, dash, and not-applicable
package cells. Include every physical power, ground, reserved, exposed-pad, NC,
and "not connected" row. Never infer missing rows or continue a sequence.

Return JSON only:
{"pins":[{"pin_no":"<exact cell>","name":"<exact pin name>"}]}
"""

PIN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "physical_pin_rows",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "pins": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pin_no": {"type": "string"},
                            "name": {"type": "string"},
                        },
                        "required": ["pin_no", "name"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["pins"],
            "additionalProperties": False,
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _regular_file(path: Path, kind: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{kind} must be a regular file: {path}")
    return path.resolve()


def _fixture_path(root: Path, value: Any, kind: str) -> Path:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise ValueError(f"{kind} path must be relative")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{kind} path escapes the fixture") from error
    return _regular_file(path, kind)


def load_fixture(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = _regular_file(path, "fixture manifest")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != FIXTURE_SCHEMA:
        raise ValueError("unsupported fixture manifest")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("fixture cases must be a non-empty list")
    root = manifest_path.parent.resolve()
    cases: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError("fixture case must be an object")
        identifier = str(raw.get("id") or "")
        if not identifier or identifier in identifiers:
            raise ValueError("fixture case identifiers must be non-empty and unique")
        identifiers.add(identifier)
        pdf = _fixture_path(root, raw.get("pdf"), "datasheet")
        truth_path = _fixture_path(
            root,
            raw.get("ground_truth"),
            "ground truth",
        )
        for artifact, field, kind in (
            (pdf, "pdf_sha256", "datasheet"),
            (truth_path, "ground_truth_sha256", "ground truth"),
        ):
            expected = raw.get(field)
            if not isinstance(expected, str) or not SHA256.fullmatch(expected):
                raise ValueError(f"{identifier}: invalid {kind} digest")
            if sha256_file(artifact) != expected:
                raise ValueError(f"{identifier}: {kind} digest mismatch")
        with pdf.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise ValueError(f"{identifier}: invalid PDF signature")
        truth = json.loads(truth_path.read_text())
        pins = truth.get("pins") if isinstance(truth, dict) else None
        expected_rows = int(raw.get("expected_ground_truth_rows", -1))
        if (
            not isinstance(pins, list)
            or len(pins) != expected_rows
            or int(truth.get("n_pins", -1)) != expected_rows
        ):
            raise ValueError(f"{identifier}: inconsistent ground truth")
        package = raw.get("requested_package")
        package_pins = int(raw.get("expected_package_pins", 0))
        if not isinstance(package, str) or not package or package_pins < 1:
            raise ValueError(f"{identifier}: invalid package request")
        validate_physical_pin_truth(
            pins,
            package=package,
            expected_package_pins=package_pins,
        )
        cases.append(
            {
                **raw,
                "pdf_path": pdf,
                "truth_path": truth_path,
                "truth": truth,
            }
        )
    return manifest, cases


def validate_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("endpoint must be an HTTP(S) URL without credentials")
    return value.rstrip("/")


def discover_model(
    client: httpx.Client,
    base_url: str,
    *,
    configured_model: str = "",
    allowed_companion_models: tuple[str, ...] = (),
) -> str:
    response = client.get(f"{base_url}/models")
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list) or not models:
        raise ValueError("endpoint returned no model identities")
    identities = [
        row.get("id") if isinstance(row, dict) else None for row in models
    ]
    if any(not isinstance(model, str) or not model for model in identities):
        raise ValueError("endpoint returned an invalid model identity")
    if len(set(identities)) != len(identities):
        raise ValueError("endpoint returned duplicate model identities")
    if configured_model:
        expected = {configured_model, *allowed_companion_models}
        if set(identities) != expected:
            raise ValueError(
                f"endpoint models {sorted(identities)!r} != "
                f"declared models {sorted(expected)!r}"
            )
        return configured_model
    if allowed_companion_models or len(identities) != 1:
        raise ValueError("endpoint must expose exactly one model")
    return identities[0]


def discover_runtime_version(client: httpx.Client, base_url: str) -> str:
    parsed = urlsplit(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    candidates = [f"{base_url}/version"]
    root_version = f"{origin}/version"
    if root_version not in candidates:
        candidates.append(root_version)

    for candidate in candidates:
        response = client.get(candidate)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        payload = response.json()
        version = payload.get("version") if isinstance(payload, dict) else None
        if not isinstance(version, str) or not version:
            raise ValueError("endpoint returned an invalid runtime version")
        return version
    raise ValueError("endpoint does not expose a runtime version")


def extract_json(text: str) -> dict[str, Any] | None:
    value = (text or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(value[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _pin_number(value: Any) -> str:
    if value is None:
        return ""
    normalized = re.sub(r"\s+", "", str(value).upper())
    return re.sub(r"\(\d+\)$", "", normalized)


def _pin_name(value: Any) -> str:
    normalized = re.sub(r"\(.*?\)", "", str(value or "").upper())
    compact = re.sub(r"[^A-Z0-9]+", "", normalized)
    if compact in {"NOCONNECT", "NOTCONNECTED"}:
        return "NC"
    return compact


def prediction_pins(
    value: dict[str, Any] | None,
    *,
    maximum_rows: int,
) -> tuple[list[dict[str, str]], str | None]:
    if value is None or not isinstance(value.get("pins"), list):
        return [], "response_is_not_pin_json"
    raw_pins = value["pins"]
    if len(raw_pins) > maximum_rows:
        return [], "response_exceeds_pin_row_bound"
    pins: list[dict[str, str]] = []
    for raw in raw_pins:
        if not isinstance(raw, dict):
            return [], "pin_row_is_not_an_object"
        number = raw.get("pin_no")
        name = raw.get("name")
        if not isinstance(number, (str, int)) or not isinstance(name, str):
            return [], "pin_row_has_invalid_types"
        normalized_number = _pin_number(number)
        normalized_name = _pin_name(name)
        if not normalized_number or not normalized_name:
            return [], "pin_row_is_empty"
        if normalized_number in {"-", "–", "—", "N/A", "NA"}:
            return [], "pin_row_has_non_physical_identifier"
        pins.append({"pin_no": str(number).strip(), "name": name.strip()})
    return pins, None


def merge_pins(rows: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for pins in rows:
        for pin in pins:
            identity = (_pin_number(pin["pin_no"]), _pin_name(pin["name"]))
            merged.setdefault(identity, pin)
    return sorted(
        merged.values(),
        key=lambda pin: (_pin_number(pin["pin_no"]), _pin_name(pin["name"])),
    )


def score_prediction(
    truth: dict[str, Any],
    prediction: list[dict[str, str]],
) -> dict[str, Any]:
    truth_pins = truth["pins"]
    truth_pairs = {
        (_pin_number(pin.get("pin_no")), _pin_name(pin.get("name")))
        for pin in truth_pins
        if isinstance(pin, dict)
    }
    predicted_pairs = {
        (_pin_number(pin["pin_no"]), _pin_name(pin["name"]))
        for pin in prediction
    }
    truth_numbers = {number for number, _ in truth_pairs if number}
    predicted_numbers = {number for number, _ in predicted_pairs if number}
    truth_names = {name for _, name in truth_pairs if name}
    predicted_names = {name for _, name in predicted_pairs if name}

    def metrics(expected: set[Any], actual: set[Any]) -> tuple[float, float, float]:
        overlap = len(expected & actual)
        precision = overlap / len(actual) if actual else 0.0
        recall = overlap / len(expected) if expected else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return precision, recall, f1

    pair_precision, pair_recall, pair_f1 = metrics(
        truth_pairs,
        predicted_pairs,
    )
    _, number_recall, _ = metrics(truth_numbers, predicted_numbers)
    _, name_recall, _ = metrics(truth_names, predicted_names)
    expected_count = len(truth_pins)
    count_error = abs(len(prediction) - expected_count) / max(1, expected_count)
    return {
        "expected_rows": expected_count,
        "predicted_rows": len(prediction),
        "count_relative_error": count_error,
        "count_within_tolerance": count_error <= 0.10
        or abs(len(prediction) - expected_count) <= 2,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "pair_f1": pair_f1,
        "pin_number_recall": number_recall,
        "pin_name_recall": name_recall,
    }


def _selected_definition_table(
    page: Any,
    selected_column: str,
) -> tuple[Any, list[list[Any]], int, int, int, int]:
    selected_text = re.sub(r"\s+", " ", selected_column).strip().casefold()
    candidates: list[tuple[Any, list[list[Any]], int, int, int, int]] = []
    tables = page.find_tables()
    for table in tables.tables if tables is not None else ():
        rows = table.extract() or []
        if not locator.is_definition_table(rows):
            continue

        package_hits: list[tuple[int, int]] = []
        name_hits: list[tuple[int, int]] = []
        for row_index, row in enumerate(rows[:4]):
            for column_index, raw in enumerate(row):
                value = re.sub(r"\s+", " ", str(raw or "")).strip()
                if value.casefold() == selected_text:
                    package_hits.append((row_index, column_index))
                normalized = value.upper()
                if "PIN NAME" in normalized or "PIN/BALL NAME" in normalized:
                    name_hits.append((row_index, column_index))

        package_columns = {column for _, column in package_hits}
        name_columns = {column for _, column in name_hits}
        if len(package_columns) != 1 or len(name_columns) != 1:
            continue
        package_row, package_column = package_hits[0]
        name_row, name_column = name_hits[0]
        candidates.append(
            (
                table,
                rows,
                package_row,
                package_column,
                name_row,
                name_column,
            )
        )

    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one definition table with the selected package column"
        )
    return candidates[0]


def _focused_table_geometry(
    page: Any,
    selected_column: str,
) -> tuple[tuple[float, float, float, float], ...]:
    (
        table,
        _,
        package_row,
        package_column,
        name_row,
        name_column,
    ) = _selected_definition_table(page, selected_column)
    package_cell = table.rows[package_row].cells[package_column]
    name_cell = table.rows[name_row].cells[name_column]
    if package_cell is None or name_cell is None:
        raise ValueError("selected table header cells have no geometry")

    table_rect = tuple(float(value) for value in table.bbox)
    package_rect = (
        float(package_cell[0]),
        table_rect[1],
        float(package_cell[2]),
        table_rect[3],
    )
    name_rect = (
        float(name_cell[0]),
        table_rect[1],
        float(name_cell[2]),
        table_rect[3],
    )
    if (
        package_rect[2] <= package_rect[0]
        or name_rect[2] <= name_rect[0]
        or table_rect[3] <= table_rect[1]
    ):
        raise ValueError("selected table has invalid column geometry")
    return table_rect, package_rect, name_rect


def _selected_physical_rows(
    rows: list[list[Any]],
    *,
    package_row: int,
    package_column: int,
    name_row: int,
    name_column: int,
) -> tuple[list[tuple[int, str, str]], list[list[str]]]:
    selected: list[tuple[int, str, str]] = []
    evidence_rows: list[list[str]] = []
    start = max(package_row, name_row) + 1
    for row_index, row in enumerate(rows[start:], start=start):
        if max(package_column, name_column) >= len(row):
            raise ValueError("selected table row is shorter than its headers")
        number = re.sub(r"\s+", " ", str(row[package_column] or "")).strip()
        name = re.sub(r"\s+", " ", str(row[name_column] or "")).strip()
        evidence_rows.append([number, name])
        if not number or number in {"-", "–", "—"}:
            continue
        if not name:
            raise ValueError("selected table has a pin without a name")
        selected.append((row_index, number, name))
    return selected, evidence_rows


def _extract_table_page(
    page: Any,
    *,
    selected_column: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    (
        table,
        rows,
        package_row,
        package_column,
        name_row,
        name_column,
    ) = _selected_definition_table(page, selected_column)
    selected, evidence_rows = _selected_physical_rows(
        rows,
        package_row=package_row,
        package_column=package_column,
        name_row=name_row,
        name_column=name_column,
    )
    pins = [
        {"pin_no": number, "name": name}
        for _, number, name in selected
    ]

    evidence = json.dumps(
        {
            "selected_column": selected_column,
            "rows": evidence_rows,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return pins, {
        "rendering": "table_cells",
        "input_sha256": hashlib.sha256(evidence).hexdigest(),
        "input_bytes": len(evidence),
        "source_table_bbox": [float(value) for value in table.bbox],
        "package_column_index": package_column,
        "pin_name_column_index": name_column,
    }


def _focused_table_image(
    page: Any,
    *,
    selected_column: str,
    dpi: int,
) -> tuple[bytes, dict[str, Any]]:
    table_rect, package_rect, name_rect = _focused_table_geometry(
        page,
        selected_column,
    )
    import pymupdf as fitz

    height = table_rect[3] - table_rect[1]
    package_width = max(90.0, (package_rect[2] - package_rect[0]) * 2.5)
    name_width = max(220.0, (name_rect[2] - name_rect[0]) * 2.0)
    gap = 6.0
    focused = fitz.open()
    try:
        target = focused.new_page(
            width=package_width + gap + name_width,
            height=height,
        )
        target.show_pdf_page(
            fitz.Rect(0, 0, package_width, height),
            page.parent,
            page.number,
            clip=fitz.Rect(package_rect),
            keep_proportion=False,
        )
        target.show_pdf_page(
            fitz.Rect(package_width + gap, 0, package_width + gap + name_width, height),
            page.parent,
            page.number,
            clip=fitz.Rect(name_rect),
            keep_proportion=False,
        )
        target.draw_line(
            fitz.Point(package_width + gap / 2, 0),
            fitz.Point(package_width + gap / 2, height),
            color=(0, 0, 0),
            width=1,
        )
        pixmap = target.get_pixmap(dpi=dpi, alpha=False)
        image = pixmap.tobytes("png")
    finally:
        focused.close()
    return image, {
        "rendering": "focused_columns",
        "source_table_bbox": list(table_rect),
        "source_package_bbox": list(package_rect),
        "source_pin_name_bbox": list(name_rect),
        "width": pixmap.width,
        "height": pixmap.height,
        "dpi": dpi,
    }


def _row_packed_table_image(
    page: Any,
    *,
    selected_column: str,
    dpi: int,
) -> tuple[bytes, dict[str, Any]]:
    (
        table,
        rows,
        package_row,
        package_column,
        name_row,
        name_column,
    ) = _selected_definition_table(page, selected_column)
    selected, _ = _selected_physical_rows(
        rows,
        package_row=package_row,
        package_column=package_column,
        name_row=name_row,
        name_column=name_column,
    )
    if not selected:
        raise ValueError("selected table page has no physical pin rows")

    package_cell = table.rows[package_row].cells[package_column]
    name_cell = table.rows[name_row].cells[name_column]
    if package_cell is None or name_cell is None:
        raise ValueError("selected table header cells have no geometry")
    package_x = (float(package_cell[0]), float(package_cell[2]))
    name_x = (float(name_cell[0]), float(name_cell[2]))

    def row_cell(
        row_index: int,
        column_index: int,
        header_cell: Any,
    ) -> Any:
        direct = table.rows[row_index].cells[column_index]
        if direct is not None:
            return direct
        center = (float(header_cell[0]) + float(header_cell[2])) / 2
        covering = [
            cell
            for cell in table.rows[row_index].cells
            if cell is not None and float(cell[0]) <= center <= float(cell[2])
        ]
        if len(covering) != 1:
            raise ValueError("selected physical row has ambiguous merged cells")
        return covering[0]

    first_data_index = max(package_row, name_row) + 1
    first_data_cells = [
        cell
        for cell in table.rows[first_data_index].cells
        if cell is not None
    ]
    if not first_data_cells:
        raise ValueError("selected table has no data-row geometry")
    segments: list[
        tuple[
            float,
            float,
            tuple[float, float],
            tuple[float, float],
        ]
    ] = [
        (
            float(table.bbox[1]),
            min(float(cell[1]) for cell in first_data_cells),
            package_x,
            name_x,
        )
    ]
    for row_index, _, _ in selected:
        cells = [
            cell for cell in table.rows[row_index].cells if cell is not None
        ]
        if not cells:
            raise ValueError("selected physical row has no geometry")
        row_package_cell = row_cell(
            row_index,
            package_column,
            package_cell,
        )
        row_name_cell = row_cell(
            row_index,
            name_column,
            name_cell,
        )
        segments.append(
            (
                min(float(cell[1]) for cell in cells),
                max(float(cell[3]) for cell in cells),
                (
                    float(row_package_cell[0]),
                    float(row_package_cell[2]),
                ),
                (
                    float(row_name_cell[0]),
                    float(row_name_cell[2]),
                ),
            )
        )
    if any(end <= start for start, end, _, _ in segments):
        raise ValueError("selected table has invalid row geometry")

    import pymupdf as fitz

    package_width = max(90.0, (package_x[1] - package_x[0]) * 2.5)
    name_width = max(220.0, (name_x[1] - name_x[0]) * 2.0)
    gap = 6.0
    height = sum(end - start for start, end, _, _ in segments)
    packed = fitz.open()
    try:
        target = packed.new_page(
            width=package_width + gap + name_width,
            height=height,
        )
        target_y = 0.0
        for source_y0, source_y1, source_package_x, source_name_x in segments:
            segment_height = source_y1 - source_y0
            target.show_pdf_page(
                fitz.Rect(0, target_y, package_width, target_y + segment_height),
                page.parent,
                page.number,
                clip=fitz.Rect(
                    source_package_x[0],
                    source_y0,
                    source_package_x[1],
                    source_y1,
                ),
                keep_proportion=False,
            )
            target.show_pdf_page(
                fitz.Rect(
                    package_width + gap,
                    target_y,
                    package_width + gap + name_width,
                    target_y + segment_height,
                ),
                page.parent,
                page.number,
                clip=fitz.Rect(
                    source_name_x[0],
                    source_y0,
                    source_name_x[1],
                    source_y1,
                ),
                keep_proportion=False,
            )
            target_y += segment_height
        target.draw_line(
            fitz.Point(package_width + gap / 2, 0),
            fitz.Point(package_width + gap / 2, height),
            color=(0, 0, 0),
            width=1,
        )
        pixmap = target.get_pixmap(dpi=dpi, alpha=False)
        image = pixmap.tobytes("png")
    finally:
        packed.close()
    return image, {
        "rendering": "physical_rows",
        "source_table_bbox": [float(value) for value in table.bbox],
        "source_row_indices": [row_index for row_index, _, _ in selected],
        "physical_rows": len(selected),
        "width": pixmap.width,
        "height": pixmap.height,
        "dpi": dpi,
    }


def _request_content(
    *,
    mode: str,
    page: Any,
    page_number: int,
    document_id: str,
    package: str,
    column_header: str,
    dpi: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    instruction = (
        f"Document: {document_id}\nPDF page: {page_number}\n"
        f"Requested package: {package}\nExact matched column: {column_header}\n"
    )
    if mode == "text":
        text = page.get_text("text")
        content = [
            {
                "type": "text",
                "text": f"{instruction}\nEXTRACTED PAGE TEXT:\n{text}",
            }
        ]
        return content, {
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "input_bytes": len(text.encode()),
        }
    if mode not in {"image", "image_focused", "image_rows"}:
        raise ValueError(f"unsupported mode: {mode}")
    if mode == "image_rows":
        image, image_evidence = _row_packed_table_image(
            page,
            selected_column=column_header,
            dpi=dpi,
        )
        instruction += (
            "The image contains only physical rows from the selected package "
            "column on the left and aligned pin names on the right. Extract "
            f"every displayed row exactly once. This image contains exactly "
            f"{image_evidence['physical_rows']} physical rows, including any "
            "NC or not-connected rows.\n"
        )
    elif mode == "image_focused":
        image, image_evidence = _focused_table_image(
            page,
            selected_column=column_header,
            dpi=dpi,
        )
        instruction += (
            "The image places the selected package column on the left and the "
            "aligned pin-name column on the right.\n"
        )
    else:
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        image = pixmap.tobytes("png")
        image_evidence = {
            "rendering": "full_page",
            "width": pixmap.width,
            "height": pixmap.height,
            "dpi": dpi,
        }
    encoded = base64.b64encode(image).decode()
    content = [
        {"type": "text", "text": instruction},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        },
    ]
    return content, {
        "input_sha256": hashlib.sha256(image).hexdigest(),
        "input_bytes": len(image),
        **image_evidence,
    }


def chat(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    content: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": PIN_RESPONSE_FORMAT,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = client.post(f"{base_url}/chat/completions", json=payload)
            response.raise_for_status()
            value = response.json()
            choices = value.get("choices") if isinstance(value, dict) else None
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("completion response has invalid choices")
            message = choices[0].get("message")
            text = message.get("content") if isinstance(message, dict) else None
            if not isinstance(text, str):
                raise ValueError("completion response has no text content")
            return text, {
                "elapsed_seconds": time.monotonic() - started,
                "attempts": attempt,
                "usage": value.get("usage"),
                "finish_reason": choices[0].get("finish_reason"),
            }
        except (httpx.HTTPError, ValueError) as error:
            last_error = error
            if attempt == 3:
                break
            time.sleep(attempt)
    raise RuntimeError(f"model request failed after retries: {last_error}")


def _physical_identity_error(
    case: dict[str, Any],
    pins: list[dict[str, str]],
) -> str | None:
    try:
        validate_physical_pin_truth(
            pins,
            package=str(case["requested_package"]),
            expected_package_pins=int(case["expected_package_pins"]),
        )
    except ValueError as failure:
        return str(failure)
    return None


def _evaluate_table_mode(
    *,
    document: Any,
    located: locator.LocateResult,
    case: dict[str, Any],
) -> dict[str, Any]:
    page_results = []
    page_pins: list[list[dict[str, str]]] = []
    for page_number in located.pages_1based:
        try:
            pins, input_evidence = _extract_table_page(
                document[page_number - 1],
                selected_column=str(located.column_header),
            )
            input_error = None
            parse_error = None
        except ValueError as failure:
            pins = []
            input_evidence = {}
            input_error = str(failure)
            parse_error = "table_extraction_failed"
        page_pins.append(pins)
        page_results.append(
            {
                "page": page_number,
                **input_evidence,
                "input_error": input_error,
                "parse_error": parse_error,
                "request_error": None,
                "pins": pins,
                "raw_response": "",
            }
        )
    merged = merge_pins(page_pins)
    complete = all(row["parse_error"] is None for row in page_results)
    identity_error = _physical_identity_error(case, merged)
    return {
        "mode": "table",
        "complete": complete,
        "contract_valid": complete and identity_error is None,
        "physical_identity_error": identity_error,
        "pages": page_results,
        "prediction": {"pins": merged},
        "score": score_prediction(case["truth"], merged),
    }


def _evaluate_mode(
    *,
    document: Any,
    located: locator.LocateResult,
    case: dict[str, Any],
    mode: str,
    client: httpx.Client,
    base_url: str,
    model: str,
    dpi: int,
    max_tokens: int,
) -> dict[str, Any]:
    page_results = []
    page_pins: list[list[dict[str, str]]] = []
    maximum_rows = max(32, int(case["expected_package_pins"]) * 2)
    for page_number in located.pages_1based:
        try:
            content, input_evidence = _request_content(
                mode=mode,
                page=document[page_number - 1],
                page_number=page_number,
                document_id=str(case["id"]),
                package=str(case["requested_package"]),
                column_header=str(located.column_header),
                dpi=dpi,
            )
        except ValueError as failure:
            page_pins.append([])
            page_results.append(
                {
                    "page": page_number,
                    "input_error": str(failure),
                    "parse_error": "input_preparation_failed",
                    "request_error": None,
                    "pins": [],
                    "raw_response": "",
                }
            )
            continue
        try:
            response, request_evidence = chat(
                client,
                base_url=base_url,
                model=model,
                content=content,
                max_tokens=max_tokens,
            )
            parsed = extract_json(response)
            pins, error = prediction_pins(parsed, maximum_rows=maximum_rows)
            request_error = None
        except RuntimeError as failure:
            response = ""
            request_evidence = {}
            pins = []
            error = "model_request_failed"
            request_error = str(failure)
        page_pins.append(pins)
        page_results.append(
            {
                "page": page_number,
                **input_evidence,
                **request_evidence,
                "input_error": None,
                "parse_error": error,
                "request_error": request_error,
                "pins": pins,
                "raw_response": response,
            }
        )
    merged = merge_pins(page_pins)
    complete = all(
        row["parse_error"] is None and row.get("finish_reason") == "stop"
        for row in page_results
    )
    identity_error = _physical_identity_error(case, merged)
    return {
        "mode": mode,
        "complete": complete,
        "contract_valid": complete and identity_error is None,
        "physical_identity_error": identity_error,
        "pages": page_results,
        "prediction": {"pins": merged},
        "score": score_prediction(case["truth"], merged),
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "cases": len(cases),
        "sent": sum(case["locator"]["status"] == "send" for case in cases),
        "withheld": sum(
            case["locator"]["status"] == "withhold" for case in cases
        ),
        "withhold_reasons": {},
        "modes": {},
    }
    for case in cases:
        located = case["locator"]
        if located["status"] == "withhold":
            reason = located["reason"]
            summary["withhold_reasons"][reason] = (
                summary["withhold_reasons"].get(reason, 0) + 1
            )
    for mode in MODES:
        rows = [
            case["modalities"][mode]
            for case in cases
            if mode in case["modalities"]
        ]
        scores = [row["score"] for row in rows]
        summary["modes"][mode] = {
            "cases": len(rows),
            "complete_cases": sum(row["complete"] for row in rows),
            "contract_valid_cases": sum(
                bool(row.get("contract_valid")) for row in rows
            ),
            "count_within_tolerance": sum(
                score["count_within_tolerance"] for score in scores
            ),
            "mean_pair_f1": (
                sum(score["pair_f1"] for score in scores) / len(scores)
                if scores
                else None
            ),
            "mean_pin_number_recall": (
                sum(score["pin_number_recall"] for score in scores) / len(scores)
                if scores
                else None
            ),
            "mean_pin_name_recall": (
                sum(score["pin_name_recall"] for score in scores) / len(scores)
                if scores
                else None
            ),
        }
    text_f1 = summary["modes"]["text"]["mean_pair_f1"]
    image_f1 = summary["modes"]["image"]["mean_pair_f1"]
    summary["image_minus_text_pair_f1"] = (
        image_f1 - text_f1
        if isinstance(image_f1, float) and isinstance(text_f1, float)
        else None
    )
    focused_f1 = summary["modes"]["image_focused"]["mean_pair_f1"]
    summary["image_focused_minus_text_pair_f1"] = (
        focused_f1 - text_f1
        if isinstance(focused_f1, float) and isinstance(text_f1, float)
        else None
    )
    summary["image_focused_minus_image_pair_f1"] = (
        focused_f1 - image_f1
        if isinstance(focused_f1, float) and isinstance(image_f1, float)
        else None
    )
    table_f1 = summary["modes"]["table"]["mean_pair_f1"]
    summary["image_focused_minus_table_pair_f1"] = (
        focused_f1 - table_f1
        if isinstance(focused_f1, float) and isinstance(table_f1, float)
        else None
    )
    row_image_f1 = summary["modes"]["image_rows"]["mean_pair_f1"]
    summary["image_rows_minus_image_focused_pair_f1"] = (
        row_image_f1 - focused_f1
        if isinstance(row_image_f1, float) and isinstance(focused_f1, float)
        else None
    )
    summary["image_rows_minus_table_pair_f1"] = (
        row_image_f1 - table_f1
        if isinstance(row_image_f1, float) and isinstance(table_f1, float)
        else None
    )
    return summary


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest, cases = load_fixture(args.fixture)
    selected = cases[: args.max_cases] if args.max_cases else cases
    selected_modes = tuple(getattr(args, "modes", None) or MODES)
    base_url = validate_base_url(args.endpoint)
    model_manifest_sha256 = sha256_file(
        _regular_file(args.model_manifest, "model manifest")
    )
    import pymupdf as fitz

    with httpx.Client(timeout=args.timeout) as client:
        discovered = discover_model(
            client,
            base_url,
            configured_model=args.model,
            allowed_companion_models=tuple(args.allowed_companion_models),
        )
        runtime_version = discover_runtime_version(client, base_url)
        if args.model and args.model != discovered:
            raise ValueError(
                f"configured model {args.model!r} != endpoint model {discovered!r}"
            )
        model = args.model or discovered
        results = []
        for case in selected:
            document = fitz.open(case["pdf_path"])
            try:
                if document.needs_pass or document.page_count < 1:
                    raise ValueError(f"{case['id']}: encrypted or empty datasheet")
                located = locate_pin_definition_pages(
                    document,
                    document_id=str(case["id"]),
                    requested_package=str(case["requested_package"]),
                    expected_package_pins=int(case["expected_package_pins"]),
                    source_path=str(case["pdf_path"]),
                )
                modalities = {}
                if located.status == "send":
                    for mode in selected_modes:
                        if mode == "table":
                            modalities[mode] = _evaluate_table_mode(
                                document=document,
                                located=located,
                                case=case,
                            )
                        else:
                            modalities[mode] = _evaluate_mode(
                                document=document,
                                located=located,
                                case=case,
                                mode=mode,
                                client=client,
                                base_url=base_url,
                                model=model,
                                dpi=args.dpi,
                                max_tokens=args.max_tokens,
                            )
                result = {
                    "id": case["id"],
                    "vendor": case["vendor"],
                    "bucket": case["bucket"],
                    "requested_package": case["requested_package"],
                    "expected_package_pins": case["expected_package_pins"],
                    "locator": asdict(located),
                    "modalities": modalities,
                }
                results.append(result)
                print(
                    json.dumps(
                        {
                            "id": case["id"],
                            "locator": located.status,
                            "table_f1": modalities.get("table", {})
                            .get("score", {})
                            .get("pair_f1"),
                            "text_f1": modalities.get("text", {})
                            .get("score", {})
                            .get("pair_f1"),
                            "image_f1": modalities.get("image", {})
                            .get("score", {})
                            .get("pair_f1"),
                            "image_focused_f1": modalities.get(
                                "image_focused",
                                {},
                            )
                            .get("score", {})
                            .get("pair_f1"),
                            "image_rows_f1": modalities.get("image_rows", {})
                            .get("score", {})
                            .get("pair_f1"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            finally:
                document.close()

    core = {
        "schema": RESULT_SCHEMA,
        "model": model,
        "runtime_version": runtime_version,
        "runtime_image_id": args.runtime_image_id,
        "model_manifest_sha256": model_manifest_sha256,
        "fixture_sha256": sha256_file(args.fixture.resolve()),
        "configuration": {
            "dpi": args.dpi,
            "max_tokens": args.max_tokens,
            "max_cases": args.max_cases,
            "modes": list(selected_modes),
            "allowed_companion_models": list(args.allowed_companion_models),
        },
        "cases": results,
        "summary": summarize(results),
    }
    core["identity"] = {
        "schema": "harness.datasheet-modality-evaluation-identity.v1",
        "core_sha256": hashlib.sha256(_canonical(core)).hexdigest(),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "locator_sha256": sha256_file(Path(locator.__file__).resolve()),
        "source_gold_set_sha256": manifest["source_gold_set_sha256"],
    }
    return core


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError(f"output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--allowed-companion-model",
        action="append",
        dest="allowed_companion_models",
        default=[],
    )
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--runtime-image-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--mode",
        action="append",
        choices=MODES,
        dest="modes",
        default=[],
    )
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if (
        args.max_cases < 0
        or not 72 <= args.dpi <= 300
        or not 256 <= args.max_tokens <= 16384
        or args.timeout < 10
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", args.runtime_image_id)
        or any(
            not model or model != model.strip()
            for model in args.allowed_companion_models
        )
        or len(set(args.allowed_companion_models))
        != len(args.allowed_companion_models)
        or args.model in args.allowed_companion_models
    ):
        parser.error("invalid evaluation limits")
    return args


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    write_new_json(args.output, result)
    print(json.dumps(result["summary"], sort_keys=True))
    print("DATASHEET_MODALITY_EVALUATION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
