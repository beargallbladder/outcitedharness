from harness.electronics.regions import (
    parametric_table_score,
    pin_table_score,
    printed_package_mentions,
    structural_parametric_regions,
    structural_pin_regions,
    structural_text_regions,
)


def test_single_package_pin_table_is_structural() -> None:
    rows = [
        ["Pin Number", "Pin Name", "Type", "Description"],
        ["1", "VIN", "Power", "Input supply"],
        ["2", "GND", "Power", "Ground"],
        ["3", "EN", "Input", "Enable input"],
    ]
    score, reasons = pin_table_score(rows)

    assert score >= 5
    assert "pin_name_header" in reasons
    assert "physical_identifier_column" in reasons


def test_register_map_is_not_a_pin_table() -> None:
    rows = [
        ["Register", "Offset", "Reset Value"],
        ["CTRL", "0x00", "0x0000"],
        ["STATUS", "0x04", "0x0000"],
        ["DATA", "0x08", "0x0000"],
    ]
    score, reasons = pin_table_score(rows)

    assert score == 0
    assert reasons == ("register_table_veto",)


def test_ordering_guide_is_not_a_pin_definition_table() -> None:
    rows = [
        [
            "Ordering Part Number",
            "Flash Memory (kB)",
            "Digital Port I/Os (Total)",
            "Package",
        ],
        ["EFM8BB10F8G-A-QSOP24", "8", "18", "QSOP24"],
        ["EFM8BB10F8G-A-QFN20", "8", "16", "QFN20"],
        ["EFM8BB10F8G-A-SOIC16", "8", "13", "SOIC16"],
    ]
    score, reasons = pin_table_score(rows)

    assert score == 0
    assert reasons == ("missing_pin_definition_header",)


def test_regions_preserve_table_geometry_and_package_headers() -> None:
    page = {
        "tables": [
            {
                "table_index": 2,
                "bbox": [10, 20, 300, 500],
                "rows": [
                    ["LQFP 64", "Pin Name", "Description"],
                    ["1", "PA0", "GPIO"],
                    ["2", "PA1", "GPIO"],
                    ["3", "VDD", "Supply"],
                ],
            }
        ]
    }
    regions = structural_pin_regions(page)

    assert regions[0]["table_index"] == 2
    assert regions[0]["bbox"] == [10.0, 20.0, 300.0, 500.0]
    assert regions[0]["package_headers"] == ["LQFP 64"]
    assert regions[0]["semantic_roles"] == ["functions"]


def test_large_bga_ball_coordinates_are_physical_identifiers() -> None:
    score, reasons = pin_table_score(
        [
            ["Ball", "Signal Name", "Description"],
            ["AA10", "DDR_D0", "Data"],
            ["AB11", "DDR_D1", "Data"],
            ["AC12", "DDR_D2", "Data"],
        ]
    )

    assert score >= 5
    assert "physical_identifier_column" in reasons


def test_regions_report_actual_semantic_field_coverage() -> None:
    regions = structural_pin_regions(
        {
            "tables": [
                {
                    "table_index": 0,
                    "bbox": [0, 0, 100, 100],
                    "rows": [
                        [
                            "LQFP64",
                            "Pin Name",
                            "Pin Type",
                            "I/O Structure",
                            "Alternate Functions",
                        ],
                        ["1", "PA0", "I/O", "FT_h", "USART0"],
                        ["2", "VDD", "S", "-", "Supply"],
                    ],
                }
            ]
        }
    )

    assert regions[0]["semantic_roles"] == ["dir", "functions", "type"]


def test_package_scope_can_come_from_printed_page_heading() -> None:
    page = {
        "blocks": [
            {"text": "Pin Description — 40-pin TQFN Package"},
            {"text": "Table 4. Pin Functions"},
        ]
    }
    assert printed_package_mentions(page) == ("40-pin TQFN",)


def test_electrical_characteristics_table_is_parametric_structure() -> None:
    rows = [
        ["Parameter", "Test Conditions", "Min", "Typ", "Max", "Unit"],
        ["Supply voltage", "TA = 25 C", "1.7", "1.8", "3.6", "V"],
        ["Sleep current", "VDD = 3.3 V", "", "2", "4", "uA"],
    ]

    score, reasons = parametric_table_score(rows)
    regions = structural_parametric_regions(
        {
            "tables": [
                {
                    "table_index": 3,
                    "bbox": [10, 20, 400, 500],
                    "rows": rows,
                }
            ]
        }
    )

    assert score >= 5
    assert "parameter_header" in reasons
    assert regions[0]["table_index"] == 3
    assert regions[0]["bbox"] == [10.0, 20.0, 400.0, 500.0]


def test_register_map_is_not_parametric_structure() -> None:
    rows = [
        ["Register", "Parameter", "Offset", "Reset Value"],
        ["CTRL", "Enable", "0x00", "0x0000"],
        ["STATUS", "Ready", "0x04", "0x0000"],
    ]

    score, reasons = parametric_table_score(rows)

    assert score == 0
    assert reasons == ("register_table_veto",)


def test_series_summary_region_preserves_visible_page_evidence() -> None:
    regions = structural_text_regions(
        {
            "blocks": [
                {"bbox": [10, 20, 300, 40], "text": "1 Features"},
                {
                    "bbox": [10, 45, 400, 180],
                    "text": "Low-power MCU with 64 KB flash",
                },
                {"bbox": [10, 190, 300, 220], "text": "2 Applications"},
            ],
            "tables": [],
        },
        "series_summary",
    )

    assert regions == [
        {
            "table_index": None,
            "bbox": [10.0, 20.0, 400.0, 220.0],
            "score": 5,
            "reasons": ["visible_summary_evidence"],
            "package_headers": [],
            "semantic_roles": [],
            "rows": 0,
        }
    ]


def test_opn_decoder_region_requires_visible_ordering_signal() -> None:
    ordering = structural_text_regions(
        {
            "blocks": [
                {
                    "bbox": [20, 30, 300, 60],
                    "text": "Ordering Information",
                }
            ],
            "tables": [
                {
                    "bbox": [20, 70, 500, 300],
                    "rows": [
                        ["Orderable Part Number", "Package"],
                        ["ABC123RGT", "QFN"],
                    ],
                }
            ],
        },
        "opn_decoder",
    )
    unrelated = structural_text_regions(
        {
            "blocks": [
                {"bbox": [0, 0, 100, 20], "text": "Electrical ratings"}
            ],
            "tables": [],
        },
        "opn_decoder",
    )

    assert ordering[0]["bbox"] == [20.0, 30.0, 500.0, 300.0]
    assert ordering[0]["reasons"] == ["visible_ordering_evidence"]
    assert unrelated == []
