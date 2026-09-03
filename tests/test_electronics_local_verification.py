from harness.electronics.local_verification import (
    grounded_pin_rows,
    quoted_parametric_evidence,
    verify_opn_decoder,
    verify_parametrics,
    verify_pin_or_ball,
    verify_pin_semantics,
    verify_series_summary,
)


def _result(pins):
    return {"result": {"pins": pins}}


def _page(text: str):
    return {
        "document_sha256": "a" * 64,
        "page_1based": 1,
        "blocks": [{"text": text}],
        "tables": [],
    }


def _table_page(rows):
    return {
        "document_sha256": "a" * 64,
        "page_1based": 1,
        "blocks": [],
        "tables": [{"table_index": 0, "rows": rows}],
    }


def test_pin_semantics_passes_grounded_gt_consistent_rows() -> None:
    pins = [
        {
            "pin_no": 1,
            "name": "PA0",
            "type": "gpio",
            "dir": "I/O",
            "functions": ["ADC input channel zero"],
        },
        {
            "pin_no": 2,
            "name": "VDD",
            "type": "power",
            "dir": "P",
            "functions": ["Positive supply voltage"],
        },
    ]
    verdict = verify_pin_semantics(
        _result(pins),
        _table_page(
            [
                ["Pin", "Name", "Type", "Direction", "Description"],
                ["1", "PA0", "GPIO", "I/O", "ADC input channel zero"],
                ["2", "VDD", "POWER", "P", "Positive supply voltage"],
            ]
        ),
        pins,
    )
    assert verdict.passed is True
    assert verdict.metrics["ground_truth_precision"] == 1


def test_pin_semantics_keeps_description_out_of_type() -> None:
    page = _table_page(
        [
            ["Pin Name", "Pin(s)", "Description"],
            ["PC00", "1", "GPIO"],
        ]
    )
    correct = verify_pin_semantics(
        _result(
            [
                {
                    "pin_no": 1,
                    "name": "PC00",
                    "type": None,
                    "dir": None,
                    "supply_domain": None,
                    "functions": ["GPIO"],
                }
            ]
        ),
        page,
    )
    mislabeled = verify_pin_semantics(
        _result(
            [
                {
                    "pin_no": 1,
                    "name": "PC00",
                    "type": "GPIO",
                    "dir": None,
                    "supply_domain": None,
                    "functions": ["GPIO"],
                }
            ]
        ),
        page,
    )

    assert correct.passed is True
    assert mislabeled.passed is False
    assert "match their printed column" in mislabeled.reason


def test_pin_semantics_preserves_numeric_zero_identifier() -> None:
    page = _table_page(
        [
            ["Pin Name", "Pin(s)", "Description"],
            ["VSS", 0, "Ground"],
        ]
    )
    result = _result(
        [
            {
                "pin_no": 0,
                "name": "VSS",
                "type": None,
                "dir": None,
                "supply_domain": None,
                "functions": ["Ground"],
            }
        ]
    )

    assert verify_pin_semantics(result, page).passed is True
    assert grounded_pin_rows(
        result,
        page,
        require_semantics=True,
    ) == [
        {
            "pin_no": "0",
            "name": "VSS",
            "type": None,
            "dir": None,
            "supply_domain": None,
            "functions": ["Ground"],
        }
    ]


def test_pin_semantics_maps_pin_type_and_io_structure_by_value() -> None:
    verdict = verify_pin_semantics(
        _result(
            [
                {
                    "pin_no": 1,
                    "name": "PE2",
                    "type": "FT_h",
                    "dir": "I/O",
                    "supply_domain": None,
                    "functions": ["TRACECLK"],
                }
            ]
        ),
        _table_page(
            [
                [
                    "LQFP144",
                    "Pin name",
                    "Pin type",
                    "I/O structure",
                    "Alternate functions",
                ],
                ["1", "PE2", "I/O", "FT_h", "TRACECLK"],
            ]
        ),
    )

    assert verdict.passed is True
    assert verdict.metrics["scalar_semantic_grounding_rate"] == 1


def test_pin_semantics_rejects_nonvalue_scalar_markers() -> None:
    verdict = verify_pin_semantics(
        _result(
            [
                {
                    "pin_no": 1,
                    "name": "VDD",
                    "type": "-",
                    "dir": "S",
                    "supply_domain": None,
                    "functions": [],
                }
            ]
        ),
        _table_page(
            [
                ["Pin", "Name", "Pin type", "I/O structure"],
                ["1", "VDD", "S", "-"],
            ]
        ),
    )

    assert verdict.passed is False
    assert "non-value marker" in verdict.reason


def test_grounded_pin_rows_repairs_field_mapping_and_drops_separators() -> None:
    rows = grounded_pin_rows(
        _result(
            [
                {
                    "pin_no": 1,
                    "name": "PE2",
                    "type": "I/O",
                    "dir": "I/O",
                    "supply_domain": "FT_h",
                    "functions": ["TRACECLK", "NOT_PRINTED"],
                },
                {
                    "pin_no": "-",
                    "name": "VSS",
                    "type": "-",
                    "dir": "S",
                    "supply_domain": None,
                    "functions": [],
                },
            ]
        ),
        _table_page(
            [
                [
                    "Pin",
                    "Name",
                    "Pin type",
                    "I/O structure",
                    "Alternate functions",
                ],
                ["1", "PE2", "I/O", "FT_h", "TRACECLK"],
                ["-", "VSS", "S", "-", "-"],
            ]
        ),
        require_semantics=True,
    )

    assert rows == [
        {
            "pin_no": "1",
            "name": "PE2",
            "type": "FT_h",
            "dir": "I/O",
            "supply_domain": None,
            "functions": ["TRACECLK"],
        }
    ]


def test_pin_semantics_rejects_empty_schema_valid_response() -> None:
    verdict = verify_pin_semantics(_result([]), _page("PA0"))
    assert verdict.passed is False
    assert verdict.terminal_status == "no_answer"


def test_pin_semantics_rejects_empty_identity_fields() -> None:
    verdict = verify_pin_semantics(
        _result(
            [
                {
                    "pin_no": "",
                    "name": "PA0",
                    "type": "GPIO",
                    "dir": "I/O",
                    "functions": ["ADC input"],
                }
            ]
        ),
        _page("PA0 GPIO I/O ADC input"),
    )

    assert verdict.passed is False
    assert "empty number or name" in verdict.reason


def test_pin_semantics_rejects_identity_only_and_string_nulls() -> None:
    verdict = verify_pin_semantics(
        _result(
            [
                {
                    "pin_no": 1,
                    "name": "PA0",
                    "type": "GPIO",
                    "dir": "null",
                    "functions": ["PA0"],
                }
            ]
        ),
        _page("1 PA0 GPIO"),
    )
    assert verdict.passed is False
    assert verdict.terminal_status == "low_confidence"
    assert "string null" in verdict.reason


def test_pin_semantics_rejects_conflicting_pin_numbers() -> None:
    verdict = verify_pin_semantics(
        _result(
            [
                {
                    "pin_no": 1,
                    "name": "PA0",
                    "type": "GPIO",
                    "dir": "I/O",
                    "functions": ["ADC input"],
                },
                {
                    "pin_no": 1,
                    "name": "PB0",
                    "type": "GPIO",
                    "dir": "I/O",
                    "functions": ["Timer output"],
                },
            ]
        ),
        _page("1 PA0 ADC input 1 PB0 Timer output"),
    )
    assert verdict.passed is False
    assert verdict.terminal_status == "cross_source_disagreement"


def test_pin_semantics_rejects_combined_physical_identifiers() -> None:
    verdict = verify_pin_semantics(
        _result(
            [
                {
                    "pin_no": "6, 7, 8",
                    "name": "BATT",
                    "type": "power",
                    "dir": None,
                    "functions": ["Connection with battery"],
                }
            ]
        ),
        _page("6 7 8 BATT Connection with battery"),
    )
    assert verdict.passed is False
    assert verdict.terminal_status == "cross_source_disagreement"
    assert "multiple physical pin" in verdict.reason


def test_pin_identity_does_not_invent_required_semantics() -> None:
    verdict = verify_pin_or_ball(
        _result(
            [
                {
                    "pin_no": "A1",
                    "name": "VDD",
                    "type": None,
                    "dir": None,
                    "functions": [],
                }
            ]
        ),
        _page("A1 VDD"),
    )
    assert verdict.passed is True


def test_pin_identity_grounding_uses_only_selected_package_columns() -> None:
    page = {
        "document_sha256": "a" * 64,
        "page_1based": 1,
        "blocks": [],
        "tables": [
            {
                "rows": [
                    ["PKG-A", None, "PKG-B", None],
                    ["Pin", "Name", "Pin", "Name"],
                    ["1", "PA0", "9", "PB9"],
                ]
            }
        ],
        "structural_evidence": {
            "regions": [{"table_index": 0, "bbox": [0, 0, 100, 100]}],
            "package_scope": {
                "package": "PKG-A",
                "column_header": "PKG-A",
            },
        },
    }
    verdict = verify_pin_or_ball(
        _result(
            [
                {
                    "pin_no": "9",
                    "name": "PB9",
                    "type": None,
                    "dir": None,
                    "functions": [],
                }
            ]
        ),
        page,
    )
    assert verdict.passed is False
    assert verdict.metrics["name_grounding_rate"] == 0


def test_pin_identity_rejects_semantics_not_requested_by_lane() -> None:
    verdict = verify_pin_or_ball(
        _result(
            [
                {
                    "pin_no": "1",
                    "name": "PA0",
                    "type": "I/O",
                    "dir": "I/O",
                    "functions": ["ADC input"],
                }
            ]
        ),
        _page("1 PA0 ADC input"),
    )
    assert verdict.passed is False
    assert "unsupported semantic" in verdict.reason


def test_pin_identity_allows_explicit_grouped_source_cell_to_expand() -> None:
    pins = [
        {
            "pin_no": number,
            "name": "BATT",
            "type": None,
            "dir": None,
            "functions": [],
        }
        for number in ("6", "7", "8")
    ]
    verdict = verify_pin_or_ball(
        _result(pins),
        _table_page(
            [
                ["Pin", "Name"],
                ["6, 7, 8", "BATT"],
            ]
        ),
    )
    assert verdict.passed is True


def test_parametrics_passes_same_row_printed_fact() -> None:
    verdict = verify_parametrics(
        {
            "result": {
                "facts": [
                    {
                        "field": "Supply voltage",
                        "value": "1.7",
                        "value_role": "Min",
                        "unit": "V",
                        "conditions": {"test": "TA = 25 C"},
                    }
                ]
            }
        },
        _table_page(
            [
                ["Parameter", "Test Conditions", "Min", "Typ", "Max", "Unit"],
                ["Supply voltage", "TA = 25 C", "1.7", "1.8", "3.6", "V"],
            ]
        ),
    )

    assert verdict.passed is True
    assert verdict.metrics["same_row_grounding_rate"] == 1


def test_parametrics_associates_grouped_headers_with_value_column() -> None:
    page = _table_page(
        [
            ["Data flash characteristics", "", "", "", "", "", "", "", "", ""],
            [
                "Parameter",
                "",
                "Symbol",
                "FCLK = 4 MHz",
                "",
                "",
                "20 MHz <= FCLK <= 50 MHz",
                "",
                "",
                "Unit",
            ],
            ["", "", "", "Min", "Typ", "Max", "Min", "Typ", "Max", ""],
            [
                "Programming time",
                "4-byte",
                "tDP4",
                "—",
                "0.36",
                "3.8",
                "—",
                "0.16",
                "1.7",
                "ms",
            ],
        ]
    )
    fact = {
        "field": "tDP4",
        "value": "0.36",
        "value_role": "typ",
        "unit": "ms",
        "conditions": {"FCLK": "4 MHz"},
    }
    correct = verify_parametrics(
        {
            "result": {
                "facts": [fact]
            }
        },
        page,
    )
    wrong_group = verify_parametrics(
        {
            "result": {
                "facts": [
                    {
                        "field": "tDP4",
                        "value": "0.36",
                        "value_role": "typ",
                        "unit": "ms",
                        "conditions": {"FCLK": "20 MHz <= FCLK <= 50 MHz"},
                    }
                ]
            }
        },
        page,
    )

    assert correct.passed is True
    quote = quoted_parametric_evidence(fact, page)
    assert quote is not None
    assert "FCLK = 4 MHz" in quote
    assert "tDP4" in quote
    assert "0.36" in quote
    assert wrong_group.passed is False
    assert "one table row" in wrong_group.reason


def test_parametrics_rejects_cross_row_value_pairing() -> None:
    verdict = verify_parametrics(
        {
            "result": {
                "facts": [
                    {
                        "field": "Supply voltage",
                        "value": "4",
                        "value_role": "Max",
                        "unit": "uA",
                        "conditions": {"test": "VDD = 3.3 V"},
                    }
                ]
            }
        },
        _table_page(
            [
                ["Parameter", "Condition", "Min", "Typ", "Max", "Unit"],
                ["Supply voltage", "TA = 25 C", "1.7", "1.8", "3.6", "V"],
                ["Sleep current", "VDD = 3.3 V", "", "2", "4", "uA"],
            ]
        ),
    )

    assert verdict.passed is False
    assert "one table row" in verdict.reason


def test_parametrics_rejects_string_nulls() -> None:
    verdict = verify_parametrics(
        {
            "result": {
                "facts": [
                    {
                        "field": "Sleep current",
                        "value": "2",
                        "value_role": "Typ",
                        "unit": "null",
                        "conditions": {},
                    }
                ]
            }
        },
        _table_page(
            [
                ["Parameter", "Typ", "Unit"],
                ["Sleep current", "2", "uA"],
            ]
        ),
    )

    assert verdict.passed is False
    assert "string null" in verdict.reason


def test_parametrics_rejects_nonvalue_table_markers() -> None:
    verdict = verify_parametrics(
        {
            "result": {
                "facts": [
                    {
                        "field": "Sleep current",
                        "value": "—",
                        "value_role": "Min",
                        "unit": "uA",
                        "conditions": {},
                    }
                ]
            }
        },
        _table_page(
            [
                ["Parameter", "Min", "Typ", "Unit"],
                ["Sleep current", "—", "2", "uA"],
            ]
        ),
    )

    assert verdict.passed is False
    assert "non-value" in verdict.reason


def test_series_summary_passes_only_printed_facts() -> None:
    verdict = verify_series_summary(
        {
            "result": {
                "summary": "Ultra-low-power MCU for wearable devices.",
                "characteristics": ["Ultra-low-power MCU"],
                "applications": ["Wearable devices"],
            }
        },
        _page(
            "The ABC family is an ultra-low-power MCU. "
            "Applications include wearable devices."
        ),
    )

    assert verdict.passed is True
    assert verdict.metrics["fact_grounding_rate"] == 1


def test_series_summary_rejects_unprinted_competitor_positioning() -> None:
    verdict = verify_series_summary(
        {
            "result": {
                "summary": "Recommended alternative to Vendor X.",
                "characteristics": ["Ultra-low-power MCU"],
                "applications": ["Wearable devices"],
            }
        },
        _page("Ultra-low-power MCU for wearable devices."),
    )

    assert verdict.passed is False
    assert "competitor" in verdict.reason


def test_series_summary_rejects_unprinted_application() -> None:
    verdict = verify_series_summary(
        {
            "result": {
                "summary": "Low-power MCU for medical equipment.",
                "characteristics": ["Low-power MCU"],
                "applications": ["Medical equipment"],
            }
        },
        _page("Low-power MCU for wearable devices."),
    )

    assert verdict.passed is False
    assert "not printed" in verdict.reason


def test_opn_decoder_passes_printed_short_codes_and_meanings() -> None:
    verdict = verify_opn_decoder(
        {
            "result": {
                "series": "ABC",
                "base_part": "ABC123",
                "package_code": "QFN",
                "suffixes": [
                    {"code": "T", "meaning": "Tape and reel"},
                ],
            }
        },
        _page(
            "Ordering information: ABC series, base part ABC123. "
            "QFN package. T = Tape and reel."
        ),
    )

    assert verdict.passed is True
    assert verdict.metrics["decoder_grounding_rate"] == 1


def test_opn_decoder_rejects_inferred_suffix_meaning() -> None:
    verdict = verify_opn_decoder(
        {
            "result": {
                "series": "ABC",
                "base_part": "ABC123",
                "package_code": None,
                "suffixes": [
                    {"code": "T", "meaning": "Automotive qualified"},
                ],
            }
        },
        _page("Ordering information: ABC series, ABC123-T."),
    )

    assert verdict.passed is False
    assert "not printed" in verdict.reason
