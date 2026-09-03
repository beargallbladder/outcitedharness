from __future__ import annotations

import pytest

from harness.electronics.locator import (
    classify_title,
    is_definition_table,
    locate_pin_definition_pages,
    match_package_column,
    validate_physical_pin_truth,
)


class _Table:
    def __init__(self, rows):
        self._rows = rows

    def extract(self):
        return self._rows


class _Tables:
    def __init__(self, rows):
        self.tables = [_Table(rows)] if rows else []


class _Page:
    def __init__(self, rows=None):
        self._rows = rows

    def find_tables(self):
        return _Tables(self._rows)


class _Document:
    def __init__(self, toc, pages):
        self._toc = toc
        self._pages = pages
        self.page_count = len(pages)

    def get_toc(self):
        return self._toc

    def __getitem__(self, index):
        return self._pages[index]


DEFINITION_ROWS = [
    ["Pin name", "LQFP64", "LQFP100", "Function after reset"],
    ["PA0", "1", "2", "GPIO"],
    ["PA1", "3", "4", "GPIO"],
]


def test_title_classifier_rejects_convincing_non_definition_sources():
    assert classify_title("Table 4. Pin and ball definitions") == "definition_table"
    assert classify_title("Figure 4. BGA ballout") == "ballout_figure"
    assert classify_title("Table 9. Alternate function mapping") == "af"
    assert classify_title("Package information and outline") == "outline"


def test_definition_table_requires_package_and_pin_headers():
    assert is_definition_table(DEFINITION_ROWS) is True
    assert (
        is_definition_table(
            [
                ["Abbreviation", "Definition", "Pin name"],
                ["I/O", "Input/output", "PA0"],
            ]
        )
        is False
    )
    assert (
        is_definition_table(
            [
                ["Pin name", "AF0", "AF1"],
                ["PA0", "USART2_CTS", "TIM2_CH1"],
            ]
        )
        is False
    )


def test_package_match_never_substitutes_a_cousin_package():
    headers = ["TFBGA216", "TFBGA240 with SMPS", "LQFP144"]
    assert match_package_column("TFBGA216", headers) == "TFBGA216"
    assert match_package_column("TFBGA240 SMPS", headers) == "TFBGA240 with SMPS"
    assert match_package_column("TFBGA225", headers) is None


def test_package_match_supports_count_before_family_names():
    headers = ["144-Pin QFP (PGE)", "337-Ball BGA (ZWT)"]
    assert match_package_column("QFP144", headers) == "144-Pin QFP (PGE)"
    assert match_package_column("337-Ball BGA", headers) == "337-Ball BGA (ZWT)"


def test_package_match_withholds_generic_request_across_qualified_variants():
    headers = ["UFQFPN28", "UFQFPN28 (STM32L031GxUxS only)"]
    assert match_package_column("UFQFPN28", headers) is None
    assert (
        match_package_column(
            "UFQFPN28 (STM32L031GxUxS only)",
            headers,
        )
        == "UFQFPN28 (STM32L031GxUxS only)"
    )
    assert match_package_column("UFQFPN28 unspecified variant", headers) is None


def test_locator_returns_consecutive_definition_pages_for_exact_package():
    document = _Document(
        [
            [1, "Table 3. Pin and ball definitions", 1],
            [1, "Electrical characteristics", 3],
        ],
        [_Page(DEFINITION_ROWS), _Page(DEFINITION_ROWS), _Page()],
    )

    result = locate_pin_definition_pages(
        document,
        document_id="stm32-test",
        requested_package="LQFP64",
        expected_package_pins=64,
        source_path="/fixtures/stm32-test.pdf",
    )

    assert result.status == "send"
    assert result.pages_1based == (1, 2)
    assert result.column_header == "LQFP64"


def test_locator_withholds_when_exact_package_column_is_missing():
    document = _Document(
        [[1, "Table 3. Pin definitions", 1]],
        [_Page(DEFINITION_ROWS)],
    )

    result = locate_pin_definition_pages(
        document,
        document_id="stm32-test",
        requested_package="UFBGA144",
        expected_package_pins=144,
        source_path="/fixtures/stm32-test.pdf",
    )

    assert result.status == "withhold"
    assert result.reason == "package_column_not_in_table_headers"


def test_locator_withholds_instead_of_density_fallback():
    document = _Document(
        [[1, "Figure 3. Pinout", 1], [1, "Electrical characteristics", 2]],
        [_Page(), _Page()],
    )

    result = locate_pin_definition_pages(
        document,
        document_id="stm32-test",
        requested_package="LQFP64",
        expected_package_pins=64,
        source_path="/fixtures/stm32-test.pdf",
    )

    assert result.status == "withhold"
    assert result.reason == "toc_has_no_definition_table_and_no_pin_section"


def test_physical_truth_requires_unique_pin_identifiers():
    with pytest.raises(ValueError, match="identifiers are not unique"):
        validate_physical_pin_truth(
            [
                {"pin_no": "A1", "name": "PA0"},
                {"pin_no": "A1", "name": "PA1"},
            ],
            package="UFBGA2",
            expected_package_pins=2,
        )


def test_lqfp_truth_allows_distinct_aliases_for_the_same_physical_pin():
    validate_physical_pin_truth(
        [
            {"pin_no": 1, "name": "OSC_IN"},
            {"pin_no": 1, "name": "PD0"},
            {"pin_no": 2, "name": "VSS"},
        ],
        package="LQFP2",
        expected_package_pins=2,
    )


def test_lqfp_truth_allows_numeric_pin_footnotes():
    validate_physical_pin_truth(
        [
            {"pin_no": "1 (3)", "name": "PA0"},
            {"pin_no": 2, "name": "VSS"},
        ],
        package="LQFP2",
        expected_package_pins=2,
    )


def test_lqfp_truth_must_cover_the_exact_package_range():
    with pytest.raises(ValueError, match="do not match the package range"):
        validate_physical_pin_truth(
            [
                {"pin_no": 1, "name": "PA0"},
                {"pin_no": 3, "name": "PA1"},
            ],
            package="LQFP2",
            expected_package_pins=2,
        )
