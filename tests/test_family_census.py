from __future__ import annotations

from harness.electronics.family_census import (
    above_opn_kind,
    attributes_in_label,
    classify_grain,
    classify_table,
    is_concrete_part_token,
    is_wildcard_token,
    packages_from_text,
    parts_from_text,
    stems_from_tokens,
    summarize,
    wildcards_from_text,
)


def test_wildcards_are_series_not_parts() -> None:
    assert is_wildcard_token("GD32F405xx")
    assert is_wildcard_token("STM32G0x1")
    assert not is_wildcard_token("STM32C011F4")
    assert wildcards_from_text("For GD32F405xx, GD32F407xx and GD32F450xx") == [
        "GD32F405xx",
        "GD32F407xx",
        "GD32F450xx",
    ]
    assert parts_from_text("For GD32F405xx microcontrollers", None) == []


def test_concrete_part_tokens_reject_ids_packages_and_noise() -> None:
    assert is_concrete_part_token("STM32C011F4")
    assert is_concrete_part_token("R7FA0E1073CFJ")
    assert is_concrete_part_token("S32K144")
    for junk in (
        "RM0090",
        "DS40002183B",
        "LQFP64",
        "VQFN48",
        "OSC32K",
        "XOSC32K",
        "SERNUM15",
        "YUV444",
        "R01DS0427EJ0120",
        "PLQP0032GB-A",
        "REV1",
        "PA10",
    ):
        assert not is_concrete_part_token(junk), junk


def test_parts_are_anchored_to_family_stems() -> None:
    text = "Ordering: STM32C011F4, STM32C011J6 and companion LDO XC6206P332MR"
    assert parts_from_text(text, {"STM32"}) == ["STM32C011F4", "STM32C011J6"]
    assert stems_from_tokens(["STM32C011x4", "GD32F405xx", "AVR128DA28"]) == {
        "STM32",
        "GD32",
        "AVR128",
    }


def test_packages_yield_pin_bogeys() -> None:
    found = packages_from_text("Available in LQFP64, LQFP100 and UFBGA144 packages; 48-pin QFN")
    assert {(p["package_family"], p["pin_count"]) for p in found} == {
        ("LQFP", 64),
        ("LQFP", 100),
        ("UFBGA", 144),
        ("QFN", 48),
    }


def test_attribute_labels_are_qualifier_anchored() -> None:
    assert attributes_in_label("Flash Data area (KB)") == ["data_flash_kb"]
    assert "code_flash_kb" in attributes_in_label("Flash Code area (KB)")
    assert "code_flash_kb" in attributes_in_label("Flash memory (Kbyte)")
    assert "code_flash_kb" not in attributes_in_label("Data flash memory")
    assert attributes_in_label("EEPROM") == ["eeprom_kb"]
    assert "standby_current_ua" in attributes_in_label("Standby mode current (uA)")
    assert attributes_in_label("") == []


def test_parts_as_columns_device_table_is_recognised() -> None:
    rows = [
        ["Part Number", "", "GD32F405xx", "", ""],
        ["", "", "RE", "RG", "RK"],
        ["Flash", "Code area (KB)", "512", "512", "512"],
        ["", "Data area (KB)", "0", "512", "2560"],
        ["SRAM (KB)", "", "192", "192", "192"],
        ["Timers", "General timer", "8", "8", "8"],
        ["Package", "", "LQFP64", "LQFP64", "LQFP64"],
    ]
    table = classify_table(rows)
    assert table is not None
    assert table["orientation"] == "parts_as_columns"
    assert table["variant_count"] == 3
    assert table["variant_labels"] == ["RE", "RG", "RK"]
    assert table["series_tokens"] == ["GD32F405xx"]
    assert table["attributes_addressed"][:3] == ["code_flash_kb", "data_flash_kb", "sram_kb"]


def test_parts_as_rows_selection_table_is_recognised() -> None:
    rows = [
        ["Device", "Flash (KB)", "SRAM (KB)", "Pins", "Max freq (MHz)"],
        ["S32K142", "256", "32", "64", "112"],
        ["S32K144", "512", "64", "100", "112"],
        ["S32K146", "1024", "128", "144", "112"],
    ]
    table = classify_table(rows)
    assert table is not None
    assert table["orientation"] == "parts_as_rows"
    assert table["variant_count"] == 3
    assert table["variant_labels"] == ["S32K142", "S32K144", "S32K146"]
    assert set(table["attributes_addressed"]) >= {"code_flash_kb", "sram_kb", "pin_count", "freq_mhz"}


def test_electrical_characteristics_table_is_not_a_device_table() -> None:
    rows = [
        ["Symbol", "Parameter", "Conditions", "Min", "Typ", "Max", "Unit"],
        ["fHSE", "External clock frequency", "", "4", "", "48", "MHz"],
        ["TA", "Ambient temperature", "", "-40", "", "85", "C"],
        ["VDD", "Operating voltage", "", "2.0", "", "3.6", "V"],
    ]
    assert classify_table(rows) is None


def test_absolute_maximum_ratings_table_is_not_a_device_table() -> None:
    # TI-style ratings table: attribute-looking rows, but the data columns are
    # MIN / MAX / UNIT, not device variants.
    rows = [
        ["", "", "MIN", "MAX", "UNIT"],
        ["Input voltage", "VBUS", "-0.3", "18", "V"],
        ["Operating junction temperature", "TJ", "-40", "125", "°C"],
        ["Pin voltage", "PG, CE", "-0.3", "7", "V"],
        ["Timer", "TMR", "", "10", "s"],
    ]
    assert classify_table(rows) is None


def test_small_or_single_attribute_tables_are_ignored() -> None:
    assert classify_table([["a", "b", "c"], ["d", "e", "f"]]) is None
    assert classify_table([["Pin", "Name", "Type"], ["1", "VDD", "S"], ["2", "PA0", "I/O"]]) is None


def test_grain_rules() -> None:
    grain, reasons = classify_grain(
        scope_status="read_ok",
        document_subject="device_family",
        document_class="datasheet",
        series_tokens=["STM32C011x4"],
        parts_named=["STM32C011F4", "STM32C011J6"],
        device_tables=[{"variant_count": 5}],
        registry_stems=3,
    )
    assert grain == "above_opn"
    assert "wildcard_series_tokens_printed" in reasons

    assert classify_grain(
        scope_status="read_ok",
        document_subject="device_family",
        document_class="datasheet",
        series_tokens=[],
        parts_named=["BQ2000T"],
        device_tables=[],
        registry_stems=1,
    )[0] == "single_opn"

    assert classify_grain(
        scope_status="read_ok",
        document_subject="cpu_core",
        document_class="programming_manual",
        series_tokens=["STM32F4xx"],
        parts_named=[],
        device_tables=[],
        registry_stems=1,
    )[0] == "not_device_family"

    assert classify_grain(
        scope_status="extraction_failed",
        document_subject="unknown",
        document_class="unknown",
        series_tokens=[],
        parts_named=[],
        device_tables=[],
        registry_stems=0,
    )[0] == "extraction_failed"

    # A reference manual is above-OPN even when it prints no wildcard on the
    # cover: the class alone places it above any single orderable part.
    assert classify_grain(
        scope_status="read_ok",
        document_subject="device_family",
        document_class="reference_manual",
        series_tokens=[],
        parts_named=[],
        device_tables=[],
        registry_stems=0,
    )[0] == "above_opn"


def test_above_opn_kind_prefers_matrix_then_manual_then_variants() -> None:
    matrix = [{"variant_count": 4, "attributes_addressed": ["code_flash_kb", "sram_kb", "pin_count"]}]
    assert above_opn_kind(document_class="datasheet", series_tokens=[], parts_named=[], device_tables=matrix) == "family_matrix"
    assert above_opn_kind(document_class="reference_manual", series_tokens=["STM32F4xx"], parts_named=[], device_tables=[]) == "manual"
    assert above_opn_kind(document_class="datasheet", series_tokens=[], parts_named=["BQ2000TPN", "BQ2000TSN"], device_tables=[]) == "ordering_variants"
    assert above_opn_kind(document_class="datasheet", series_tokens=["GD32F405xx"], parts_named=[], device_tables=[]) == "wildcard_only"
    thin = [{"variant_count": 4, "attributes_addressed": ["freq_mhz", "temp_range"]}]
    assert above_opn_kind(document_class="datasheet", series_tokens=[], parts_named=[], device_tables=thin) is None


def test_summary_counts_by_vendor_and_attribute() -> None:
    rows = [
        {
            "vendor": "st",
            "grain": "above_opn",
            "above_opn_kind": "family_matrix",
            "document_class": "datasheet",
            "device_table_located": True,
            "scope_statement": None,
            "parts_named_count": 6,
            "pin_bogeys": [20],
            "denominator_cells": 78,
            "attributes_addressed": ["code_flash_kb", "sram_kb"],
        },
        {
            "vendor": "st",
            "grain": "single_opn",
            "document_class": "datasheet",
        },
        {
            "vendor": "ti",
            "grain": "above_opn",
            "above_opn_kind": "ordering_variants",
            "document_class": "datasheet",
            "device_table_located": False,
            "scope_statement": "For TPS7A47xx",
            "parts_named_count": 3,
            "pin_bogeys": [],
            "denominator_cells": 0,
            "attributes_addressed": [],
        },
    ]
    summary = summarize(rows)
    overall = summary["overall"]
    assert overall["documents"] == 3
    assert overall["grain"] == {"above_opn": 2, "single_opn": 1}
    assert overall["above_opn"]["documents"] == 2
    assert overall["above_opn"]["kind"] == {"family_matrix": 1, "ordering_variants": 1}
    assert overall["above_opn"]["device_table_located"] == 1
    assert overall["above_opn"]["scope_statement_printed"] == 1
    assert overall["above_opn"]["parts_named_total"] == 9
    assert overall["above_opn"]["documents_with_pin_bogeys"] == 1
    assert overall["above_opn"]["denominator_cells_total"] == 78
    assert overall["above_opn"]["attributes_addressed"] == {"code_flash_kb": 1, "sram_kb": 1}
    assert list(summary["by_vendor"]) == ["st", "ti"]
    assert summary["by_vendor"]["st"]["above_opn"] == 1
    assert summary["by_vendor"]["st"]["above_opn_kind"] == {"family_matrix": 1}


def test_title_row_is_stripped_before_classification() -> None:
    from harness.electronics.family_census import classify_table, strip_title_rows

    rows = [
        ["Table 1.12 Product list (1 of 2)", None, None, None],
        ["Product part number", "Code flash", "SRAM", "Operating temperature"],
        ["R7FA2E1A93CFM", "128", "16", "-40 to +105°C"],
        ["R7FA2E1A72DFL", "64", "16", "-40 to +85°C"],
    ]
    assert len(strip_title_rows(rows)) == 3
    described = classify_table(rows)
    assert described is not None and described["orientation"] == "parts_as_rows"
