"""Conservative deterministic parsers for extracted datasheet tables."""

from __future__ import annotations

import re
from typing import Any, Mapping

from harness.electronics.locator import package_pin_count


PIN_HEADER = re.compile(
    r"^(?:PIN|BALL)(?:\s*(?:\(S\)|S|NO\.?|NUMBER|#))?$",
    re.I,
)
NAME_HEADER = re.compile(
    r"^(?:(?:PIN|BALL|SIGNAL|PIN\s*/\s*BALL)\s*)?NAME\b",
    re.I,
)
TYPE_HEADER = re.compile(r"\bTYPE\b", re.I)
DIRECTION_HEADER = re.compile(r"\b(?:DIRECTION|DIR)\b|^I\s*/\s*O$", re.I)
IO_STRUCTURE_HEADER = re.compile(
    r"\bI\s*/?\s*O\s+(?:STRUCTURE|LEVEL)\b"
    r"|\bELECTRICAL\s+(?:TYPE|STRUCTURE|CHARACTERISTICS?)\b",
    re.I,
)
SUPPLY_DOMAIN_HEADER = re.compile(
    r"\b(?:I\s*/?\s*O|POWER|SUPPLY)\s+(?:SUPPLY\s+)?DOMAIN\b",
    re.I,
)
DESCRIPTION_HEADER = re.compile(r"\bDESCRIPTIONS?\b", re.I)
FUNCTION_HEADER = re.compile(
    r"\b(?:FUNCTIONS?|MULTIPLEX|RESET STATE)\b",
    re.I,
)
FUNCTION_OR_DESCRIPTION_HEADER = re.compile(
    r"\b(?:FUNCTIONS?|DESCRIPTIONS?|MULTIPLEX|RESET STATE)\b",
    re.I,
)
PARAMETER_HEADER = re.compile(r"\b(?:PARAMETER|CHARACTERISTIC|SYMBOL)\b", re.I)
VALUE_HEADER = re.compile(r"\b(?:MIN|TYP|MAX|VALUE|UNIT|CONDITION)\b", re.I)
PARAMETRIC_VALUE_ROLES = {
    "min": "min",
    "minimum": "minimum",
    "typ": "typ",
    "typical": "typical",
    "max": "max",
    "maximum": "maximum",
    "value": "value",
}
NON_VALUE_MARKERS = {"-", "—", "–", "na", "n/a"}
DIRECTION_CODES = {
    "a",
    "ai",
    "ao",
    "i",
    "io",
    "o",
    "p",
    "s",
}


def _cell(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str("" if value is None else value),
    ).strip()


def _normalized_cell(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", _cell(value).upper())


def _header_score(row: list[Any], patterns: tuple[re.Pattern[str], ...]) -> int:
    cells = [_cell(value) for value in row]
    return sum(any(pattern.search(cell) for pattern in patterns) for cell in cells)


def _column(row: list[Any], pattern: re.Pattern[str]) -> int | None:
    matches = [
        index for index, value in enumerate(row) if pattern.search(_cell(value))
    ]
    return matches[0] if len(matches) == 1 else None


def _value(row: list[Any], index: int | None) -> str | None:
    if index is None or index >= len(row):
        return None
    value = _cell(row[index])
    return value or None


def _physical(value: str | None) -> str | None:
    if value is None:
        return None
    raw = _cell(value).upper()
    if raw in {"-", "—", "N/A", "NA", "NC"}:
        return None
    if len(raw) > 24:
        return None
    if re.fullmatch(r"\d{1,4}(?:\(\d+\))?", raw):
        return raw
    ball = re.fullmatch(r"([A-HJ-NP-Z]{1,3})\s*(\d{1,3})", raw)
    return "".join(ball.groups()) if ball else None


def physical_pin_identifiers(value: Any) -> tuple[str, ...]:
    """Return one or more explicit physical IDs without concatenating them."""

    raw = _cell(value)
    single = _physical(raw)
    if single is not None:
        return (single,)
    values = [
        physical
        for token in re.split(r"[,;\s]+", raw)
        if (physical := _physical(token)) is not None
    ]
    return tuple(dict.fromkeys(values)) if len(values) > 1 else ()


def _pin_table_layout(
    table: Mapping[str, Any],
) -> tuple[tuple[str, ...], int]:
    rows = table.get("rows")
    if not isinstance(rows, list) or not rows:
        return (), 0
    header_signal_seen = False
    data_start = min(6, len(rows))
    for index, row in enumerate(rows[:6]):
        if not isinstance(row, list):
            continue
        physicals = sum(
            _physical(_cell(value)) is not None for value in row
        )
        if physicals >= 2 or (physicals >= 1 and header_signal_seen):
            data_start = index
            break
        header_signal_seen = header_signal_seen or any(
            pattern.search(_cell(value))
            for value in row
            for pattern in (
                PIN_HEADER,
                NAME_HEADER,
                TYPE_HEADER,
                FUNCTION_OR_DESCRIPTION_HEADER,
            )
        )
    width = max(
        (len(row) for row in rows[:data_start] if isinstance(row, list)),
        default=0,
    )
    return (
        tuple(
            " ".join(
                _cell(row[column])
                for row in rows[:data_start]
                if isinstance(row, list)
                and column < len(row)
                and _cell(row[column])
            )
            for column in range(width)
        ),
        data_start,
    )


def _pin_header_cells(table: Mapping[str, Any]) -> tuple[str, ...]:
    return _pin_table_layout(table)[0]


def _physical_header(value: str) -> bool:
    return bool(
        package_pin_count(value) is not None
        or PIN_HEADER.search(_cell(value))
    )


def pin_identity_rows(table: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return only visibly printed pin/ball identities from a table."""

    rows = table.get("rows")
    headers, data_start = _pin_table_layout(table)
    if not isinstance(rows, list) or not headers:
        return []
    physical_columns = [
        index
        for index, header in enumerate(headers)
        if _physical_header(header)
    ]
    name_columns = [
        index
        for index, header in enumerate(headers)
        if NAME_HEADER.search(header) and index not in physical_columns
    ]
    if not physical_columns or not name_columns:
        return []
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row_index, row in enumerate(rows[data_start:], start=data_start):
        if not isinstance(row, list):
            continue
        for name_column in name_columns:
            if name_column >= len(row):
                continue
            name = _cell(row[name_column])
            if (
                not name
                or name.casefold() in NON_VALUE_MARKERS
                or len(name) > 160
            ):
                continue
            physical_column = min(
                physical_columns,
                key=lambda index: abs(index - name_column),
            )
            physicals = physical_pin_identifiers(
                _value(row, physical_column)
            )
            if not physicals:
                continue
            for physical in physicals:
                identity = (physical, re.sub(r"\s+", "", name.upper()))
                if identity in seen:
                    continue
                seen.add(identity)
                output.append(
                    {
                        "pin_no": physical,
                        "name": name,
                        "row_index": row_index,
                        "table": table,
                    }
                )
    return output


def _semantic_role(header: str, value: str) -> str | None:
    if SUPPLY_DOMAIN_HEADER.search(header):
        return "supply_domain"
    if IO_STRUCTURE_HEADER.search(header):
        return "type"
    if DIRECTION_HEADER.search(header):
        return "dir"
    if TYPE_HEADER.search(header):
        normalized = re.sub(r"[^a-z0-9]+", "", value.casefold())
        return "dir" if normalized in DIRECTION_CODES else "type"
    if FUNCTION_HEADER.search(header) or DESCRIPTION_HEADER.search(header):
        return "functions"
    return None


def pin_semantic_values(
    table: Mapping[str, Any],
    row_index: int,
    *,
    pin_no: Any,
    name: Any,
) -> dict[str, tuple[str, ...]]:
    """Return field-aligned, verbatim semantics for one printed pin row."""

    empty = {
        "type": (),
        "dir": (),
        "supply_domain": (),
        "functions": (),
    }
    rows = table.get("rows")
    headers = _pin_header_cells(table)
    if (
        not isinstance(rows, list)
        or row_index < 0
        or row_index >= len(rows)
        or not isinstance(rows[row_index], list)
        or not headers
    ):
        return empty
    row = rows[row_index]
    wanted_name = _normalized_cell(name)
    wanted_number = _normalized_cell(pin_no)
    name_columns = [
        index
        for index, value in enumerate(row)
        if _normalized_cell(value) == wanted_name
    ]
    number_columns = [
        index
        for index, value in enumerate(row)
        if wanted_number
        and (
            _normalized_cell(value) == wanted_number
            or wanted_number
            in {
                _normalized_cell(part)
                for part in re.split(r"[,;\s]+", _cell(value))
                if part
            }
        )
    ]
    if not name_columns or not number_columns:
        return empty
    name_column = name_columns[0]
    number_column = min(
        number_columns,
        key=lambda index: abs(index - name_column),
    )
    group_start = min(name_column, number_column)
    later_name_headers = [
        index
        for index, header in enumerate(headers)
        if index > name_column and NAME_HEADER.search(header)
    ]
    group_end = min(later_name_headers, default=len(row))

    values: dict[str, list[str]] = {
        "type": [],
        "dir": [],
        "supply_domain": [],
        "functions": [],
    }
    for index in range(group_start, min(group_end, len(row), len(headers))):
        if index in {name_column, number_column}:
            continue
        value = _cell(row[index])
        if not value or value.casefold() in NON_VALUE_MARKERS:
            continue
        role = _semantic_role(headers[index], value)
        if role is not None and value not in values[role]:
            values[role].append(value)
    return {key: tuple(value) for key, value in values.items()}


def pin_semantic_roles(table: Mapping[str, Any]) -> tuple[str, ...]:
    """Return semantic fields with at least one printed, non-value-filtered cell."""

    rows = table.get("rows")
    headers, data_start = _pin_table_layout(table)
    if not isinstance(rows, list) or not headers:
        return ()
    roles: set[str] = set()
    for row in rows[data_start:]:
        if not isinstance(row, list):
            continue
        for index in range(min(len(row), len(headers))):
            value = _cell(row[index])
            if not value or value.casefold() in NON_VALUE_MARKERS:
                continue
            role = _semantic_role(headers[index], value)
            if role is not None:
                roles.add(role)
    return tuple(sorted(roles))


def project_pin_table(
    table: Mapping[str, Any],
    selected_header: str,
) -> dict[str, Any]:
    """Remove unselected package columns while preserving shared semantics."""

    rows = table.get("rows")
    if not isinstance(rows, list):
        return dict(table)
    normalized_header = _cell(selected_header).casefold()
    matches = {
        column
        for row in rows[:4]
        if isinstance(row, list)
        for column, value in enumerate(row)
        if _cell(value).casefold() == normalized_header
    }
    if len(matches) != 1:
        return dict(table)
    selected = next(iter(matches))
    width = max(
        (len(row) for row in rows if isinstance(row, list)),
        default=0,
    )
    package_columns = {
        column
        for row in rows[:4]
        if isinstance(row, list)
        for column, value in enumerate(row)
        if package_pin_count(_cell(value)) is not None
    }
    if selected in package_columns and len(package_columns) > 1:
        columns = [
            column
            for column in range(width)
            if column == selected or column not in package_columns
        ]
    else:
        columns = [
            column
            for column in (selected, selected + 1)
            if column < width
        ]
    return {
        **table,
        "rows": [
            [row[column] if column < len(row) else None for column in columns]
            for row in rows
            if isinstance(row, list)
        ],
        "projected_source_columns": columns,
    }


def parse_pin_table(
    table: Mapping[str, Any],
    *,
    document_sha256: str,
    page_1based: int,
) -> list[dict[str, Any]]:
    rows = table.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return []
    header_candidates = [
        (index, row)
        for index, row in enumerate(rows[:6])
        if isinstance(row, list)
    ]
    if not header_candidates:
        return []
    header_index, header = max(
        header_candidates,
        key=lambda item: _header_score(
            item[1],
            (
                PIN_HEADER,
                NAME_HEADER,
                TYPE_HEADER,
                FUNCTION_OR_DESCRIPTION_HEADER,
            ),
        ),
    )
    name_index = _column(header, NAME_HEADER)
    if name_index is None:
        return []
    package_by_column: dict[int, str] = {}
    for candidate_row in rows[: max(4, header_index + 1)]:
        if not isinstance(candidate_row, list):
            continue
        for index, value in enumerate(candidate_row):
            cell = _cell(value)
            if package_pin_count(cell) is not None:
                package_by_column[index] = cell
    package_columns = sorted(package_by_column.items())
    generic_pin_index = _column(header, PIN_HEADER)
    if not package_columns and generic_pin_index is not None:
        package_columns = [(generic_pin_index, "UNRESOLVED")]
    if not package_columns:
        return []
    type_index = _column(header, TYPE_HEADER)
    direction_index = _column(header, DIRECTION_HEADER)
    function_index = _column(header, FUNCTION_OR_DESCRIPTION_HEADER)
    output: list[dict[str, Any]] = []
    for package_index, package in package_columns:
        for row_index, row in enumerate(rows[header_index + 1 :], header_index + 1):
            if not isinstance(row, list):
                continue
            physical = _physical(_value(row, package_index))
            name = _value(row, name_index)
            if physical is None or name is None or len(name) > 160:
                continue
            functions = _value(row, function_index)
            output.append(
                {
                    "document_sha256": document_sha256,
                    "page_1based": page_1based,
                    "table_index": table.get("table_index"),
                    "row_index": row_index,
                    "package": package,
                    "expected_package_pins": package_pin_count(package),
                    "pin_no": physical,
                    "name": name,
                    "type": _value(row, type_index),
                    "direction": _value(row, direction_index),
                    "functions_verbatim": functions,
                    "method": "pymupdf_deterministic_table",
                }
            )
    return output


def parse_parametric_table(
    table: Mapping[str, Any],
    *,
    document_sha256: str,
    page_1based: int,
) -> list[dict[str, Any]]:
    rows = table.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return []
    header_candidates = [
        (index, row)
        for index, row in enumerate(rows[:6])
        if isinstance(row, list)
    ]
    if not header_candidates:
        return []
    header_index, header = max(
        header_candidates,
        key=lambda item: _header_score(
            item[1],
            (PARAMETER_HEADER, VALUE_HEADER),
        ),
    )
    if _header_score(header, (PARAMETER_HEADER,)) < 1:
        return []
    if _header_score(header, (VALUE_HEADER,)) < 2:
        return []
    names = [
        _cell(value).casefold().replace(" ", "_") or f"column_{index}"
        for index, value in enumerate(header)
    ]
    output: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows[header_index + 1 :], header_index + 1):
        if not isinstance(row, list):
            continue
        values = {
            names[index]: _cell(value)
            for index, value in enumerate(row[: len(names)])
            if _cell(value)
        }
        if len(values) < 2:
            continue
        output.append(
            {
                "document_sha256": document_sha256,
                "page_1based": page_1based,
                "table_index": table.get("table_index"),
                "row_index": row_index,
                "values_verbatim": values,
                "method": "pymupdf_deterministic_table",
            }
        )
    return output


def _header_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _cell(value).casefold()).strip("_")


def normalize_parametric_facts(
    row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand one verbatim table row into value-level, conversion-free facts."""

    raw = row.get("values_verbatim", row)
    if not isinstance(raw, Mapping):
        return []
    values = {
        _header_key(key): _cell(value)
        for key, value in raw.items()
        if _cell(value)
    }
    if not values:
        return []

    field_key = next(
        (
            key
            for key in (
                "characteristic",
                "description",
                "parameter_name",
                "column_1",
                "parameter",
                "symbol",
            )
            if values.get(key)
        ),
        None,
    )
    if field_key is None:
        return []
    unit_key = next(
        (key for key in ("unit", "units") if values.get(key)),
        None,
    )
    ignored = {
        field_key,
        *(key for key in ("unit", "units") if key in values),
        *PARAMETRIC_VALUE_ROLES,
    }
    conditions = {
        key: value
        for key, value in values.items()
        if key not in ignored
    }
    return [
        {
            "field": values[field_key],
            "value": value,
            "value_role": PARAMETRIC_VALUE_ROLES[key],
            "unit": values.get(unit_key) if unit_key else None,
            "conditions": conditions,
        }
        for key, value in values.items()
        if key in PARAMETRIC_VALUE_ROLES
        and value.strip().casefold() not in NON_VALUE_MARKERS
    ]


__all__ = [
    "PARAMETER_HEADER",
    "VALUE_HEADER",
    "normalize_parametric_facts",
    "parse_parametric_table",
    "parse_pin_table",
    "physical_pin_identifiers",
    "pin_identity_rows",
    "pin_semantic_values",
    "project_pin_table",
]
