"""Conservative evidence and ground-truth checks for local extraction output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from harness.electronics.local_model import focused_page_context
from harness.electronics.table_extractors import (
    PARAMETER_HEADER,
    VALUE_HEADER,
    physical_pin_identifiers,
    pin_identity_rows,
    pin_semantic_values,
)

PARAMETRIC_ROLE_HEADER = re.compile(
    r"\b(?:MIN(?:IMUM)?|TYP(?:ICAL)?|MAX(?:IMUM)?|VALUE)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LocalVerification:
    passed: bool
    terminal_status: str | None
    reason: str | None
    checks: tuple[str, ...]
    metrics: dict[str, float | int]


def _string(value: Any) -> str:
    return str("" if value is None else value)


def _normalized(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", _string(value).upper())


def _source_segments(
    capability: str,
    page: Mapping[str, Any],
) -> list[str]:
    context = focused_page_context(capability, page)
    values = [
        str(block.get("text") or "")
        for block in context.get("blocks") or []
        if isinstance(block, Mapping)
    ]
    for table in context.get("tables") or []:
        if not isinstance(table, Mapping):
            continue
        values.extend(
            " | ".join(_string(cell) for cell in row)
            for row in table.get("rows") or []
            if isinstance(row, list)
        )
    digital_text = context.get("digital_text")
    if isinstance(digital_text, Mapping):
        values.extend(str(digital_text.get("text") or "").splitlines())
    return [value for value in values if value.strip()]


def _source_text(capability: str, page: Mapping[str, Any]) -> str:
    return "\n".join(_source_segments(capability, page))


def _printed(value: str, source: str) -> bool:
    raw = " ".join(value.split()).upper()
    if not raw:
        return False
    source_upper = source.upper()
    normalized = _normalized(raw)
    if len(normalized) >= 4 and normalized in _normalized(source_upper):
        return True
    return bool(
        re.search(
            rf"(?<![A-Z0-9]){re.escape(raw)}(?![A-Z0-9])",
            source_upper,
        )
    )


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        return None
    return [" ".join(item.split()) for item in value]


def _leaf_values(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [
            leaf
            for child in value.values()
            for leaf in _leaf_values(child)
        ]
    if isinstance(value, list):
        return [leaf for child in value for leaf in _leaf_values(child)]
    if value is None or isinstance(value, bool):
        return []
    text = " ".join(str(value).split())
    return [text] if text else []


def _parametric_rows(
    page: Mapping[str, Any],
) -> list[dict[str, Any]]:
    context = focused_page_context("parametrics", page)
    rows: list[dict[str, Any]] = []
    for table in context.get("tables") or []:
        if not isinstance(table, Mapping):
            continue
        table_rows = [
            row for row in table.get("rows") or [] if isinstance(row, list)
        ]
        if not table_rows:
            continue
        header_index = max(
            range(min(6, len(table_rows))),
            key=lambda index: sum(
                bool(
                    PARAMETER_HEADER.search(str(cell or ""))
                    or VALUE_HEADER.search(str(cell or ""))
                )
                for cell in table_rows[index]
            ),
        )
        parameter_rows = [
            index
            for index in range(header_index + 1)
            if any(
                PARAMETER_HEADER.search(str(cell or ""))
                for cell in table_rows[index]
            )
        ]
        structure_start = parameter_rows[0] if parameter_rows else 0
        header_rows = table_rows[structure_start : header_index + 1]
        header_quote = "\n".join(
            " | ".join(
                " ".join(str(cell or "").split()) for cell in header_row
            )
            for header_row in header_rows
        )
        width = max(len(row) for row in table_rows)
        spread_headers: list[list[str]] = []
        for header_row in header_rows:
            spread: list[str] = []
            previous = ""
            for column in range(width):
                value = (
                    " ".join(str(header_row[column] or "").split())
                    if column < len(header_row)
                    else ""
                )
                if value:
                    previous = value
                spread.append(previous)
            spread_headers.append(spread)
        column_headers = [
            " | ".join(
                value
                for value in (
                    spread[column] for spread in spread_headers
                )
                if value
            )
            for column in range(width)
        ]
        value_columns = {
            index
            for index in range(width)
            if any(
                index < len(header_row)
                and PARAMETRIC_ROLE_HEADER.search(
                    str(header_row[index] or "")
                )
                for header_row in header_rows
            )
        }
        parameter_columns = [
            index
            for index in range(width)
            if any(
                index < len(header_row)
                and PARAMETER_HEADER.search(str(header_row[index] or ""))
                for header_row in header_rows
            )
        ]
        primary_parameter_column = (
            min(parameter_columns) if parameter_columns else 0
        )
        inherited = [""] * width
        group_rows: list[str] = []
        for table_row in table_rows[header_index + 1 :]:
            raw = [
                (
                    " ".join(str(table_row[index] or "").split())
                    if index < len(table_row)
                    else ""
                )
                for index in range(width)
            ]
            if raw[primary_parameter_column]:
                inherited = [""] * width
                group_rows = []
            raw_row_text = " | ".join(value for value in raw if value)
            group_rows.append(raw_row_text)
            effective = list(raw)
            for index, value in enumerate(raw):
                if index in value_columns:
                    continue
                if value and value not in {"-", "—"}:
                    inherited[index] = value
                elif inherited[index]:
                    effective[index] = inherited[index]
            rows.append(
                {
                    "column_headers": tuple(column_headers),
                    "raw_cells": tuple(raw),
                    "row_context": " | ".join(
                        value for value in effective if value
                    ),
                    "evidence_span": "\n".join(
                        [header_quote, *group_rows]
                    ),
                    "value_columns": tuple(sorted(value_columns)),
                }
            )
    return rows


def _matching_parametric_cells(
    fact: Mapping[str, Any],
    page: Mapping[str, Any],
) -> list[tuple[dict[str, Any], str]]:
    field = str(fact["field"])
    value = str(fact["value"])
    unit = str(fact.get("unit") or "").strip()
    conditions = _leaf_values(fact.get("conditions"))
    matching: list[tuple[dict[str, Any], str]] = []
    for row in _parametric_rows(page):
        row_context = str(row["row_context"])
        if not _printed(field, row_context):
            continue
        if unit and not _printed(unit, row_context):
            continue
        raw_cells = row["raw_cells"]
        column_headers = row["column_headers"]
        for column in row["value_columns"]:
            if column >= len(raw_cells) or not _printed(
                value, str(raw_cells[column])
            ):
                continue
            header = str(column_headers[column])
            if all(
                _printed(condition, header)
                or _printed(condition, row_context)
                for condition in conditions
            ):
                matching.append((row, header))
    return matching


def quoted_parametric_evidence(
    fact: Mapping[str, Any],
    page: Mapping[str, Any],
) -> str | None:
    """Return only joined verbatim table rows that ground one parametric fact."""

    matching = _matching_parametric_cells(fact, page)
    return str(matching[0][0]["evidence_span"]) if matching else None


def _focused_values(
    page: Mapping[str, Any],
    *,
    require_semantics: bool,
) -> tuple[
    str,
    list[
        tuple[
            str,
            tuple[str, ...],
            tuple[str, ...],
            Mapping[str, Any],
            int,
        ]
    ],
]:
    context = focused_page_context(
        "pin_semantics" if require_semantics else "pin_or_ball",
        page,
    )
    values = [
        str(block.get("text") or "")
        for block in context.get("blocks") or []
        if isinstance(block, Mapping)
    ]
    row_values: list[
        tuple[
            str,
            tuple[str, ...],
            tuple[str, ...],
            Mapping[str, Any],
            int,
        ]
    ] = []
    for table in context.get("tables") or []:
        if not isinstance(table, Mapping):
            continue
        for row_index, row in enumerate(table.get("rows") or []):
            if not isinstance(row, list):
                continue
            cells = tuple(_normalized(cell) for cell in row if _normalized(cell))
            physical_tokens = tuple(
                token
                for cell in row
                for token in (
                    _normalized(part)
                for part in re.split(r"[,;\s]+", _string(cell))
                )
                if token
            )
            row_text = "".join(cells)
            row_values.append(
                (row_text, cells, physical_tokens, table, row_index)
            )
            values.extend(_string(cell) for cell in row)
    return "\n".join(values), row_values


def _verify_pin_rows(
    result: Mapping[str, Any],
    page: Mapping[str, Any],
    ground_truth_rows: Iterable[Mapping[str, Any]] = (),
    *,
    require_semantics: bool,
) -> LocalVerification:
    """Require identities and semantics to be traceable to printed evidence."""

    payload = result.get("result")
    pins = payload.get("pins") if isinstance(payload, Mapping) else None
    if not isinstance(pins, list) or not pins:
        return LocalVerification(
            passed=False,
            terminal_status="no_answer",
            reason="local result contains no pin rows",
            checks=("nonempty_pin_rows",),
            metrics={"pins": 0},
        )
    if not all(isinstance(pin, Mapping) for pin in pins):
        return LocalVerification(
            passed=False,
            terminal_status="low_confidence",
            reason="local result contains a malformed pin row",
            checks=("pin_row_objects",),
            metrics={"pins": len(pins)},
        )

    focused_text, evidence_rows = _focused_values(
        page,
        require_semantics=require_semantics,
    )
    text = _normalized(focused_text)
    identities: list[tuple[str, str]] = []
    names_grounded = 0
    numbers_grounded = 0
    combined_identifiers = 0
    semantic_rows = 0
    functions = 0
    grounded_functions = 0
    scalar_semantics = 0
    grounded_scalar_semantics = 0
    row_grounded_identities = 0
    literal_nulls = 0
    nonvalue_scalar_markers = 0
    unexpected_identity_lane_semantics = 0
    empty_identities = 0
    names_by_number: dict[str, set[str]] = {}
    for pin in pins:
        number = _normalized(pin.get("pin_no"))
        name = _normalized(pin.get("name"))
        raw_number = _string(pin.get("pin_no"))
        identities.append((number, name))
        empty_identities += not number or not name
        names_by_number.setdefault(number, set()).add(name)
        names_grounded += bool(name and name in text)
        numbers_grounded += bool(number and number in text)
        matching_rows = [
            (row_text, cells, physical_tokens, table, row_index)
            for row_text, cells, physical_tokens, table, row_index in evidence_rows
            if (number in cells or number in physical_tokens) and name in cells
        ]
        if (
            not evidence_rows
            and not require_semantics
            and number in text
            and name in text
        ):
            matching_rows = [
                (text, (number, name), (number, name), {}, -1)
            ]
        row_grounded_identities += bool(matching_rows)
        combined_identifiers += bool(
            re.search(r"[,;]|\b(?:AND|TO)\b", raw_number, re.IGNORECASE)
            or len(physical_pin_identifiers(raw_number)) > 1
        )
        literal_nulls += sum(
            isinstance(pin.get(field), str)
            and str(pin[field]).strip().casefold() in {"null", "none", "n/a"}
            for field in ("type", "dir", "supply_domain")
        )
        nonvalue_scalar_markers += sum(
            isinstance(pin.get(field), str)
            and str(pin[field]).strip().casefold()
            in {"-", "—", "–", "na", "n/a"}
            for field in ("type", "dir", "supply_domain")
        )
        unexpected_identity_lane_semantics += bool(
            pin.get("type") is not None
            or pin.get("dir") is not None
            or pin.get("supply_domain") is not None
            or pin.get("functions")
        )
        pin_functions = [
            str(value)
            for value in pin.get("functions") or []
            if str(value).strip()
        ]
        normalized_functions = [_normalized(value) for value in pin_functions]
        semantic_sources = [
            pin_semantic_values(
                table,
                row_index,
                pin_no=pin.get("pin_no"),
                name=pin.get("name"),
            )
            for _row_text, _cells, _tokens, table, row_index in matching_rows
        ]
        functions += len(normalized_functions)
        grounded_functions += sum(
            bool(
                value
                and any(
                    value in _normalized(source)
                    for semantic in semantic_sources
                    for source in semantic["functions"]
                )
            )
            for value in normalized_functions
        )
        scalar_values = {
            field: _normalized(pin.get(field))
            for field in ("type", "dir", "supply_domain")
            if pin.get(field) is not None
            and _normalized(pin.get(field))
            and not (
                isinstance(pin.get(field), str)
                and str(pin[field]).strip().casefold()
                in {"-", "—", "–", "na", "n/a"}
            )
        }
        scalar_semantics += len(scalar_values)
        grounded_scalar_semantics += sum(
            any(
                value == _normalized(source)
                for semantic in semantic_sources
                for source in semantic[field]
            )
            for field, value in scalar_values.items()
        )
        if any(
            value
            and len(value) >= 4
            and value not in {number, name}
            and value not in name
            for value in normalized_functions
        ) or bool(scalar_values):
            semantic_rows += 1

    conflicts = sum(
        len({name for name in names if name}) > 1
        for names in names_by_number.values()
    )
    name_grounding_rate = names_grounded / len(pins)
    number_grounding_rate = numbers_grounded / len(pins)
    semantic_rate = semantic_rows / len(pins)
    row_grounding_rate = row_grounded_identities / len(pins)
    function_grounding_rate = (
        grounded_functions / functions if functions else 0.0
    )
    scalar_grounding_rate = (
        grounded_scalar_semantics / scalar_semantics
        if scalar_semantics
        else 0.0
    )
    expected = {
        (_normalized(row.get("pin_no")), _normalized(row.get("name")))
        for row in ground_truth_rows
        if _normalized(row.get("pin_no")) and _normalized(row.get("name"))
    }
    predicted = {
        identity for identity in identities if all(identity)
    }
    matched = len(predicted & expected)
    gt_precision = matched / len(predicted) if predicted and expected else 0.0
    metrics: dict[str, float | int] = {
        "pins": len(pins),
        "conflicting_pin_numbers": conflicts,
        "literal_null_fields": literal_nulls,
        "nonvalue_scalar_markers": nonvalue_scalar_markers,
        "empty_pin_identities": empty_identities,
        "unexpected_identity_lane_semantics": (
            unexpected_identity_lane_semantics
        ),
        "combined_pin_identifiers": combined_identifiers,
        "name_grounding_rate": name_grounding_rate,
        "number_grounding_rate": number_grounding_rate,
        "row_identity_grounding_rate": row_grounding_rate,
        "semantic_row_rate": semantic_rate,
        "function_grounding_rate": function_grounding_rate,
        "scalar_semantic_grounding_rate": scalar_grounding_rate,
        "ground_truth_identities": len(expected),
        "ground_truth_matches": matched,
        "ground_truth_precision": gt_precision,
    }
    identity_checks = (
        "nonempty_pin_rows",
        "nonempty_pin_identity_fields",
        "unique_pin_number_meaning",
        "one_physical_identifier_per_row",
        "null_value_contract",
        "nonvalue_scalar_contract",
        "printed_name_grounding",
        "same_row_identity_grounding",
        "ground_truth_identity_precision_when_available",
    )
    checks = (
        identity_checks
        + (
            "nonidentity_semantics",
            "semantic_column_alignment",
            "printed_function_grounding",
        )
        if require_semantics
        else identity_checks
    )

    if conflicts:
        return LocalVerification(
            False,
            "cross_source_disagreement",
            "one or more pin numbers map to conflicting names",
            checks,
            metrics,
        )
    if empty_identities:
        return LocalVerification(
            False,
            "low_confidence",
            "one or more pin rows has an empty number or name",
            checks,
            metrics,
        )
    if combined_identifiers:
        return LocalVerification(
            False,
            "cross_source_disagreement",
            "one or more rows combine multiple physical pin identifiers",
            checks,
            metrics,
        )
    if literal_nulls:
        return LocalVerification(
            False,
            "low_confidence",
            "model emitted string null markers instead of JSON null",
            checks,
            metrics,
        )
    if nonvalue_scalar_markers:
        return LocalVerification(
            False,
            "low_confidence",
            "model emitted a non-value marker instead of JSON null",
            checks,
            metrics,
        )
    if not require_semantics and unexpected_identity_lane_semantics:
        return LocalVerification(
            False,
            "low_confidence",
            "identity-only extraction emitted unsupported semantic fields",
            checks,
            metrics,
        )
    if name_grounding_rate < 0.90:
        return LocalVerification(
            False,
            "low_confidence",
            "fewer than 90% of pin names are grounded in page evidence",
            checks,
            metrics,
        )
    if number_grounding_rate < 0.90:
        return LocalVerification(
            False,
            "low_confidence",
            "fewer than 90% of pin numbers are grounded in page evidence",
            checks,
            metrics,
        )
    if row_grounding_rate < 0.90:
        return LocalVerification(
            False,
            "low_confidence",
            "fewer than 90% of identities are grounded in one table row",
            checks,
            metrics,
        )
    if expected and gt_precision < 0.90:
        return LocalVerification(
            False,
            "cross_source_disagreement",
            "pin identity precision against owned ground truth is below 90%",
            checks,
            metrics,
        )
    if require_semantics and semantic_rate < 0.50:
        return LocalVerification(
            False,
            "low_confidence",
            "fewer than 50% of rows contain semantics beyond the pin identity",
            checks,
            metrics,
        )
    if require_semantics and functions and function_grounding_rate < 0.80:
        return LocalVerification(
            False,
            "low_confidence",
            "fewer than 80% of extracted functions are printed on the page",
            checks,
            metrics,
        )
    if (
        require_semantics
        and scalar_semantics
        and scalar_grounding_rate < 0.80
    ):
        return LocalVerification(
            False,
            "low_confidence",
            "fewer than 80% of scalar semantics match their printed column",
            checks,
            metrics,
        )
    return LocalVerification(
        True,
        None,
        None,
        checks,
        metrics,
    )


def grounded_pin_rows(
    result: Mapping[str, Any],
    page: Mapping[str, Any],
    *,
    require_semantics: bool,
) -> list[dict[str, Any]]:
    """Salvage only independently row- and column-grounded teacher pin rows."""

    payload = result.get("result")
    pins = payload.get("pins") if isinstance(payload, Mapping) else None
    if not isinstance(pins, list):
        return []
    _focused_text, evidence_rows = _focused_values(
        page,
        require_semantics=require_semantics,
    )
    admitted: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for pin in pins:
        if not isinstance(pin, Mapping):
            continue
        number = _normalized(pin.get("pin_no"))
        name = _normalized(pin.get("name"))
        raw_number = _string(pin.get("pin_no"))
        if (
            not number
            or not name
            or re.search(
                r"[,;]|\b(?:AND|TO)\b",
                raw_number,
                re.IGNORECASE,
            )
            or len(physical_pin_identifiers(raw_number)) > 1
        ):
            continue
        matching_rows = [
            (table, row_index)
            for _text, cells, tokens, table, row_index in evidence_rows
            if (number in cells or number in tokens) and name in cells
        ]
        if not matching_rows:
            continue
        source_identities = [
            identity
            for table, row_index in matching_rows
            for identity in pin_identity_rows(table)
            if identity["row_index"] == row_index
            and _normalized(identity["pin_no"]) == number
            and _normalized(identity["name"]) == name
        ]
        source_identity = (
            source_identities[0]
            if source_identities
            else {
                "pin_no": pin["pin_no"],
                "name": pin["name"],
            }
        )
        identity = (number, name)
        if identity in seen:
            continue
        seen.add(identity)
        sanitized: dict[str, Any] = {
            "pin_no": source_identity["pin_no"],
            "name": source_identity["name"],
        }
        if require_semantics:
            sources = [
                pin_semantic_values(
                    table,
                    row_index,
                    pin_no=source_identity["pin_no"],
                    name=source_identity["name"],
                )
                for table, row_index in matching_rows
            ]
            for field in ("type", "dir", "supply_domain"):
                by_normalized: dict[str, str] = {}
                for source in sources:
                    for value in source[field]:
                        by_normalized.setdefault(_normalized(value), value)
                by_normalized.pop("", None)
                sanitized[field] = (
                    next(iter(by_normalized.values()))
                    if len(by_normalized) == 1
                    else None
                )
            function_sources = [
                _normalized(value)
                for source in sources
                for value in source["functions"]
                if _normalized(value)
            ]
            sanitized["functions"] = list(
                dict.fromkeys(
                    str(value)
                    for value in pin.get("functions") or []
                    if _normalized(value)
                    and any(
                        _normalized(value) in source
                        for source in function_sources
                    )
                )
            )
        admitted.append(sanitized)

    names_by_number: dict[str, set[str]] = {}
    for pin in admitted:
        names_by_number.setdefault(
            _normalized(pin["pin_no"]),
            set(),
        ).add(_normalized(pin["name"]))
    conflicts = {
        number for number, names in names_by_number.items() if len(names) > 1
    }
    return [
        pin
        for pin in admitted
        if _normalized(pin["pin_no"]) not in conflicts
    ]


def verify_pin_or_ball(
    result: Mapping[str, Any],
    page: Mapping[str, Any],
    ground_truth_rows: Iterable[Mapping[str, Any]] = (),
) -> LocalVerification:
    return _verify_pin_rows(
        result,
        page,
        ground_truth_rows,
        require_semantics=False,
    )


def verify_pin_semantics(
    result: Mapping[str, Any],
    page: Mapping[str, Any],
    ground_truth_rows: Iterable[Mapping[str, Any]] = (),
) -> LocalVerification:
    return _verify_pin_rows(
        result,
        page,
        ground_truth_rows,
        require_semantics=True,
    )


def verify_parametrics(
    result: Mapping[str, Any],
    page: Mapping[str, Any],
) -> LocalVerification:
    """Require each parametric value to stay in one printed table row."""

    payload = result.get("result")
    facts = payload.get("facts") if isinstance(payload, Mapping) else None
    checks = (
        "nonempty_parametric_facts",
        "parametric_fact_schema",
        "null_value_contract",
        "nonvalue_marker_contract",
        "unique_parametric_facts",
        "same_row_parametric_grounding",
        "printed_value_role_header",
    )
    if not isinstance(facts, list) or not facts:
        return LocalVerification(
            False,
            "no_answer",
            "local result contains no parametric facts",
            checks,
            {"facts": 0},
        )
    valid = all(
        isinstance(fact, Mapping)
        and isinstance(fact.get("field"), str)
        and fact["field"].strip()
        and isinstance(fact.get("value"), (str, int, float))
        and not isinstance(fact.get("value"), bool)
        and (
            fact.get("value_role") is None
            or isinstance(fact.get("value_role"), str)
        )
        and (
            fact.get("unit") is None
            or isinstance(fact.get("unit"), str)
        )
        and isinstance(fact.get("conditions"), Mapping)
        for fact in facts
    )
    if not valid:
        return LocalVerification(
            False,
            "schema_failed",
            "local result contains a malformed parametric fact",
            checks,
            {"facts": len(facts)},
        )

    literal_nulls = sum(
        str(value).strip().casefold() in {"null", "none", "n/a"}
        for fact in facts
        for value in (
            fact.get("value"),
            fact.get("value_role"),
            fact.get("unit"),
            *_leaf_values(fact.get("conditions")),
        )
        if isinstance(value, str)
    )
    absent_value_markers = sum(
        isinstance(fact.get("value"), str)
        and str(fact["value"]).strip().casefold()
        in {"-", "—", "–", "na", "n/a"}
        for fact in facts
    )
    identities = [
        (
            _normalized(fact["field"]),
            _normalized(fact["value"]),
            _normalized(fact.get("value_role")),
            _normalized(fact.get("unit")),
            tuple(
                sorted(
                    _normalized(value)
                    for value in _leaf_values(fact.get("conditions"))
                )
            ),
        )
        for fact in facts
    ]
    duplicate_facts = len(identities) - len(set(identities))
    row_grounded = 0
    role_grounded = 0
    for fact in facts:
        matching = _matching_parametric_cells(fact, page)
        row_grounded += bool(matching)
        role = fact.get("value_role")
        role_grounded += bool(
            role is None
            or (
                str(role).strip()
                and any(
                    _printed(str(role), header)
                    for _row, header in matching
                )
            )
        )
    row_grounding_rate = row_grounded / len(facts)
    role_grounding_rate = role_grounded / len(facts)
    metrics: dict[str, float | int] = {
        "facts": len(facts),
        "literal_null_fields": literal_nulls,
        "absent_value_markers": absent_value_markers,
        "duplicate_facts": duplicate_facts,
        "same_row_grounded_facts": row_grounded,
        "same_row_grounding_rate": row_grounding_rate,
        "role_grounded_facts": role_grounded,
        "value_role_grounding_rate": role_grounding_rate,
    }
    if literal_nulls:
        return LocalVerification(
            False,
            "low_confidence",
            "model emitted string null markers instead of JSON null",
            checks,
            metrics,
        )
    if absent_value_markers:
        return LocalVerification(
            False,
            "low_confidence",
            "model emitted a non-value table marker as a parametric fact",
            checks,
            metrics,
        )
    if duplicate_facts:
        return LocalVerification(
            False,
            "cross_source_disagreement",
            "local result contains duplicate parametric facts",
            checks,
            metrics,
        )
    if row_grounding_rate < 1.0:
        return LocalVerification(
            False,
            "low_confidence",
            "one or more parametric facts are not grounded in one table row",
            checks,
            metrics,
        )
    if role_grounding_rate < 1.0:
        return LocalVerification(
            False,
            "low_confidence",
            "one or more value roles are not printed in the table header",
            checks,
            metrics,
        )
    return LocalVerification(True, None, None, checks, metrics)


def verify_series_summary(
    result: Mapping[str, Any],
    page: Mapping[str, Any],
) -> LocalVerification:
    """Require summary facts and applications to be visibly printed."""

    payload = result.get("result")
    if not isinstance(payload, Mapping):
        return LocalVerification(
            False,
            "no_answer",
            "local result contains no summary object",
            ("summary_object",),
            {"facts": 0},
        )
    summary = payload.get("summary")
    characteristics = _string_list(payload.get("characteristics"))
    applications = _string_list(payload.get("applications"))
    checks = (
        "summary_schema",
        "nonempty_summary_facts",
        "printed_characteristics",
        "printed_applications",
        "summary_token_grounding",
        "no_competitor_or_recommendation_claims",
    )
    if (
        not isinstance(summary, str)
        or not summary.strip()
        or characteristics is None
        or applications is None
    ):
        return LocalVerification(
            False,
            "schema_failed",
            "summary, characteristics, or applications are malformed",
            checks,
            {"facts": 0},
        )
    facts = [*characteristics, *applications]
    if not facts:
        return LocalVerification(
            False,
            "no_answer",
            "summary contains no explicit characteristics or applications",
            checks,
            {"facts": 0},
        )
    forbidden = re.compile(
        r"\b(?:compet(?:es?|itor|itors|itive)|alternative\s+to|"
        r"better\s+than|recommended|recommendation)\b",
        re.IGNORECASE,
    )
    if forbidden.search(" ".join([summary, *facts])):
        return LocalVerification(
            False,
            "low_confidence",
            "summary emitted competitor, positioning, or recommendation text",
            checks,
            {"facts": len(facts), "forbidden_claims": 1},
        )

    source_segments = _source_segments("series_summary", page)
    source = "\n".join(source_segments)
    grounded_characteristics = sum(
        any(_printed(value, segment) for segment in source_segments)
        for value in characteristics
    )
    grounded_applications = sum(
        any(_printed(value, segment) for segment in source_segments)
        for value in applications
    )
    grounded_facts = grounded_characteristics + grounded_applications
    fact_grounding_rate = grounded_facts / len(facts)
    stopwords = {
        "AND",
        "ARE",
        "FOR",
        "FROM",
        "HAS",
        "THE",
        "THIS",
        "WITH",
    }
    summary_tokens = {
        token
        for token in re.findall(r"[A-Z0-9]+", summary.upper())
        if len(token) >= 3 and token not in stopwords
    }
    source_tokens = set(re.findall(r"[A-Z0-9]+", source.upper()))
    grounded_summary_tokens = len(summary_tokens & source_tokens)
    summary_grounding_rate = (
        grounded_summary_tokens / len(summary_tokens)
        if summary_tokens
        else 0.0
    )
    metrics: dict[str, float | int] = {
        "facts": len(facts),
        "characteristics": len(characteristics),
        "applications": len(applications),
        "grounded_characteristics": grounded_characteristics,
        "grounded_applications": grounded_applications,
        "fact_grounding_rate": fact_grounding_rate,
        "summary_tokens": len(summary_tokens),
        "summary_token_grounding_rate": summary_grounding_rate,
        "forbidden_claims": 0,
    }
    if fact_grounding_rate < 1.0:
        return LocalVerification(
            False,
            "low_confidence",
            "one or more summary facts are not printed on the page",
            checks,
            metrics,
        )
    if summary_grounding_rate < 0.80:
        return LocalVerification(
            False,
            "low_confidence",
            "fewer than 80% of summary content tokens are printed on the page",
            checks,
            metrics,
        )
    return LocalVerification(True, None, None, checks, metrics)


def verify_opn_decoder(
    result: Mapping[str, Any],
    page: Mapping[str, Any],
) -> LocalVerification:
    """Require every decoded OPN segment to be printed on the page."""

    payload = result.get("result")
    checks = (
        "opn_schema",
        "nonempty_decoder_fields",
        "null_value_contract",
        "unique_suffix_codes",
        "printed_decoder_grounding",
    )
    if not isinstance(payload, Mapping):
        return LocalVerification(
            False,
            "no_answer",
            "local result contains no OPN decoder object",
            checks,
            {"decoder_values": 0},
        )
    scalar_fields = ("series", "base_part", "package_code")
    if any(
        payload.get(field) is not None
        and not isinstance(payload.get(field), str)
        for field in scalar_fields
    ):
        return LocalVerification(
            False,
            "schema_failed",
            "OPN decoder scalar fields are malformed",
            checks,
            {"decoder_values": 0},
        )
    suffixes = payload.get("suffixes")
    if not isinstance(suffixes, list) or not all(
        isinstance(item, Mapping)
        and isinstance(item.get("code"), str)
        and item["code"].strip()
        and (
            item.get("meaning") is None
            or isinstance(item.get("meaning"), str)
        )
        for item in suffixes
    ):
        return LocalVerification(
            False,
            "schema_failed",
            "OPN suffixes must contain code and nullable meaning fields",
            checks,
            {"decoder_values": 0},
        )
    literal_nulls = sum(
        isinstance(payload.get(field), str)
        and str(payload[field]).strip().casefold() in {"null", "none", "n/a"}
        for field in scalar_fields
    ) + sum(
        isinstance(item.get("meaning"), str)
        and str(item["meaning"]).strip().casefold() in {"null", "none", "n/a"}
        for item in suffixes
    )
    if literal_nulls:
        return LocalVerification(
            False,
            "low_confidence",
            "model emitted string null markers instead of JSON null",
            checks,
            {"decoder_values": 0, "literal_null_fields": literal_nulls},
        )
    codes = [_normalized(item["code"]) for item in suffixes]
    if len(codes) != len(set(codes)):
        return LocalVerification(
            False,
            "cross_source_disagreement",
            "OPN decoder contains duplicate suffix codes",
            checks,
            {"decoder_values": 0, "duplicate_suffix_codes": 1},
        )

    values = [
        str(payload[field]).strip()
        for field in scalar_fields
        if isinstance(payload.get(field), str) and str(payload[field]).strip()
    ]
    for item in suffixes:
        values.append(str(item["code"]).strip())
        if isinstance(item.get("meaning"), str) and item["meaning"].strip():
            values.append(str(item["meaning"]).strip())
    if not values:
        return LocalVerification(
            False,
            "no_answer",
            "OPN decoder contains no explicit printed fields",
            checks,
            {"decoder_values": 0},
        )
    source_segments = _source_segments("opn_decoder", page)
    grounded = sum(
        any(_printed(value, segment) for segment in source_segments)
        for value in values
    )
    grounding_rate = grounded / len(values)
    grounded_suffix_associations = sum(
        any(
            _printed(str(item["code"]), segment)
            and (
                item.get("meaning") is None
                or not str(item.get("meaning") or "").strip()
                or _printed(str(item["meaning"]), segment)
            )
            for segment in source_segments
        )
        for item in suffixes
    )
    suffix_association_rate = (
        grounded_suffix_associations / len(suffixes)
        if suffixes
        else 1.0
    )
    metrics = {
        "decoder_values": len(values),
        "grounded_decoder_values": grounded,
        "decoder_grounding_rate": grounding_rate,
        "suffixes": len(suffixes),
        "grounded_suffix_associations": grounded_suffix_associations,
        "suffix_association_rate": suffix_association_rate,
        "literal_null_fields": literal_nulls,
        "duplicate_suffix_codes": 0,
    }
    if grounding_rate < 1.0:
        return LocalVerification(
            False,
            "low_confidence",
            "one or more decoded OPN values are not printed on the page",
            checks,
            metrics,
        )
    if suffix_association_rate < 1.0:
        return LocalVerification(
            False,
            "low_confidence",
            "one or more suffix meanings are not printed with their code",
            checks,
            metrics,
        )
    return LocalVerification(True, None, None, checks, metrics)


__all__ = [
    "LocalVerification",
    "grounded_pin_rows",
    "verify_opn_decoder",
    "verify_parametrics",
    "verify_pin_or_ball",
    "verify_pin_semantics",
    "verify_series_summary",
]
