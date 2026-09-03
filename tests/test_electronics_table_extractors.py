from __future__ import annotations

from harness.electronics.table_extractors import (
    normalize_parametric_facts,
    parse_parametric_table,
    parse_pin_table,
    physical_pin_identifiers,
    pin_identity_rows,
    pin_semantic_values,
)


def test_pin_parser_preserves_package_and_verbatim_semantics():
    table = {
        "table_index": 2,
        "rows": [
            ["", "LQFP64", "LQFP100", "", "", ""],
            [
                "Pin name",
                "Pin number",
                "Pin number",
                "Type",
                "Direction",
                "Function",
            ],
            ["PA0", "1", "2", "GPIO", "I/O", "ADC0; USART2"],
            ["PA1", "-", "3", "GPIO", "I/O", "ADC1"],
        ],
    }

    rows = parse_pin_table(
        table,
        document_sha256="a" * 64,
        page_1based=10,
    )

    assert len(rows) == 3
    lqfp64 = [row for row in rows if row["package"] == "LQFP64"]
    assert lqfp64 == [
        {
            "document_sha256": "a" * 64,
            "page_1based": 10,
            "table_index": 2,
            "row_index": 2,
            "package": "LQFP64",
            "expected_package_pins": 64,
            "pin_no": "1",
            "name": "PA0",
            "type": "GPIO",
            "direction": "I/O",
            "functions_verbatim": "ADC0; USART2",
            "method": "pymupdf_deterministic_table",
        }
    ]


def test_pin_parser_does_not_treat_pin_name_as_pin_number():
    table = {
        "table_index": 0,
        "rows": [
            ["Pin name", "Type", "Description"],
            ["PA0", "GPIO", "ADC0"],
        ],
    }

    assert (
        parse_pin_table(
            table,
            document_sha256="a" * 64,
            page_1based=1,
        )
        == []
    )


def test_pin_helpers_preserve_numeric_zero_identifier():
    table = {
        "table_index": 0,
        "rows": [
            ["Pin Name", "Pin(s)", "Description"],
            ["VSS", 0, "Ground"],
        ],
    }

    assert physical_pin_identifiers(0) == ("0",)
    assert pin_identity_rows(table)[0]["pin_no"] == "0"
    assert pin_semantic_values(
        table,
        1,
        pin_no=0,
        name="VSS",
    )["functions"] == ("Ground",)


def test_parametric_parser_preserves_printed_columns_without_normalizing():
    table = {
        "table_index": 1,
        "rows": [
            ["Parameter", "Condition", "Min", "Typ", "Max", "Unit"],
            ["Supply voltage", "TA = 25 C", "1.7", "", "3.6", "V"],
        ],
    }

    rows = parse_parametric_table(
        table,
        document_sha256="b" * 64,
        page_1based=42,
    )

    assert rows[0]["values_verbatim"]["max"] == "3.6"
    assert rows[0]["values_verbatim"]["unit"] == "V"
    assert rows[0]["values_verbatim"]["condition"] == "TA = 25 C"


def test_parametric_normalizer_expands_values_without_conversion():
    facts = normalize_parametric_facts(
        {
            "values_verbatim": {
                "parameter": "Supply voltage",
                "condition": "TA = 25 C",
                "min": "1.7",
                "max": "3.6",
                "unit": "V",
            }
        }
    )

    assert facts == [
        {
            "field": "Supply voltage",
            "value": "1.7",
            "value_role": "min",
            "unit": "V",
            "conditions": {"condition": "TA = 25 C"},
        },
        {
            "field": "Supply voltage",
            "value": "3.6",
            "value_role": "max",
            "unit": "V",
            "conditions": {"condition": "TA = 25 C"},
        },
    ]


def test_parametric_normalizer_prefers_unnamed_description_over_symbol():
    facts = normalize_parametric_facts(
        {
            "values_verbatim": {
                "parameter": "I PVDDQ",
                "column_1": "PVDD sleep mode current",
                "typ.": "2.25",
                "unit": "uA",
            }
        }
    )

    assert facts[0]["field"] == "PVDD sleep mode current"
    assert facts[0]["conditions"] == {"parameter": "I PVDDQ"}
    assert facts[0]["value_role"] == "typ"


def test_parametric_normalizer_omits_nonvalue_markers():
    facts = normalize_parametric_facts(
        {
            "values_verbatim": {
                "parameter": "Supply voltage",
                "min": "—",
                "typ": "N/A",
                "max": "3.6",
                "unit": "V",
            }
        }
    )

    assert [(fact["value_role"], fact["value"]) for fact in facts] == [
        ("max", "3.6")
    ]
