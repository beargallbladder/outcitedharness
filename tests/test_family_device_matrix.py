"""Unit tests for the deterministic family device-matrix reader."""

from __future__ import annotations

from harness.electronics.family_device_matrix import (
    _split_glued_runs,
    bind_columns,
    parse_value,
    read_matrix_table,
    row_attributes,
    st_flash_code_kb,
)


def test_parse_plain_integer_and_footnote():
    leaf = parse_value("256(1)", "code_flash_kb", "KB")
    assert leaf["status"] == "typed" and leaf["typ"] == 256 and leaf["unit"] == "KB"


def test_parse_breakdown_and_megabytes():
    assert parse_value("128 (112+16)", "sram_kb", "KB")["typ"] == 128
    assert parse_value("1 M bytes", "code_flash_kb", None)["typ"] == 1024
    assert parse_value("1 M", "code_flash_kb", None)["typ"] == 1024


def test_parse_bitwidth_sum_only_for_counts():
    leaf = parse_value("5 (16-bit) 2 (32-bit)", "timer_general_purpose", None)
    assert leaf["status"] == "typed" and leaf["typ"] == 7 and leaf["note"] == "sum of bit-width groups"
    leaf = parse_value("1 (16-bit) high frequency", "timer_advanced", None)
    assert leaf["typ"] == 1


def test_parse_explicit_none_and_presence():
    assert parse_value("-", "spi_count", None)["typ"] == 0
    leaf = parse_value("Yes", "rtc", None)
    assert leaf["status"] == "boolean" and leaf["value"] is True


def test_parse_alternatives_stay_unknown():
    leaf = parse_value("16 (parity-protected) or 18 (not parity-protected)", "sram_kb", "KB")
    assert leaf["status"] == "unknown" and leaf["reason"] == "alternative_values"


def test_parse_ranges_and_grades():
    leaf = parse_value("1.8 to 3.6 V(4)", "operating_voltage", None)
    assert (leaf["min"], leaf["max"]) == (1.8, 3.6)
    leaf = parse_value("Ambient: - 40 to 85 °C / -40 to 105 °C Junction: -40 to 125 °C", "temp_range", None)
    assert (leaf["min"], leaf["max"]) == (-40.0, 85.0)
    assert leaf["grades"] == [[-40.0, 85.0], [-40.0, 105.0]]
    assert parse_value("Junction temperature: –40 to +130 °C", "temp_range", None)["status"] == "unknown"


def test_row_attributes_label_grammar():
    assert row_attributes("GPIOs", "") == [("gpio_count", None)]
    assert row_attributes("Tamper pins", "") == []
    assert row_attributes("GPIOs", "Normal I/Os (TC, TTa)") == []
    assert row_attributes("Maximum CPU frequency", "") == [("freq_mhz", None)]
    assert row_attributes("FMC", "NOR flash memory/RAM controller") == []
    assert row_attributes("SRAM in Kbytes", "SRAM mapped onto AXI bus") == []
    assert row_attributes("Communication interfaces", "SPI / I2S") == [("spi_count", 0), ("i2s_count", 1)]
    assert row_attributes("Comm. interfaces", "USART/ UART") == [("usart_count", 0), ("uart_count", 1)]
    assert row_attributes("12-bit ADC", "Number of channels") == [("adc_channels", None)]
    assert row_attributes("Timers", "Advanced-control (PWM)") == [("timer_advanced", None)]


def test_bind_header_tokens_with_newlines():
    grid = [["Peripherals", "", "STM32 F446MC", "STM32 F446ME"], ["Flash", "", "256", "512"]]
    rows, bindings = bind_columns(grid, 2, {})
    assert rows == 1
    assert [b["part_number"] for b in bindings] == ["STM32F446MC", "STM32F446ME"]


def test_bind_header_code_list():
    grid = [["Peripheral", "", "STM32G050 _ F6 K6 K8 C6 C8", "", "", "", ""], ["Flash", "", "32", "32", "64", "32", "64"]]
    _, bindings = bind_columns(grid, 2, {})
    assert [b["part_number"] for b in bindings] == ["STM32G050F6", "STM32G050K6", "STM32G050K8", "STM32G050C6", "STM32G050C8"]
    assert {b["binding"] for b in bindings} == {"header_codes"}


def test_bind_wildcard_join_with_device_summary():
    grid = [["Peripheral", "", "STM32F101Tx", "", "STM32F101Cx", ""], ["Flash - Kbytes", "", "64", "128", "64", "128"]]
    summary = {"STM32F101x8": ["STM32F101T8", "STM32F101C8"], "STM32F101xB": ["STM32F101TB", "STM32F101CB"]}
    _, bindings = bind_columns(grid, 2, summary)
    assert [b["part_number"] for b in bindings] == ["STM32F101T8", "STM32F101TB", "STM32F101C8", "STM32F101CB"]
    assert {b["binding"] for b in bindings} == {"device_summary_join"}


def test_bind_wildcard_without_summary_stays_unbound():
    grid = [["Peripheral", "", "STM32F101Tx", ""], ["Flash", "", "64", "128"]]
    _, bindings = bind_columns(grid, 2, {})
    assert all(b["part_number"] is None for b in bindings)
    assert all(b["binding"] == "unbound" for b in bindings)


def test_bind_header_list_names_several_parts():
    grid = [["Peripheral", "", "STM32L552CE,STM32L552CC/STM32L552CExxP", "STM32L552RE"], ["Flash", "", "512", "512"]]
    _, bindings = bind_columns(grid, 2, {})
    assert bindings[0]["part_numbers"] == ["STM32L552CE", "STM32L552CC"]
    assert bindings[0]["also_covers"] == ["STM32L552CExxP"]


def test_glued_runs_split_only_when_counts_match():
    assert _split_glued_runs([("2 2", False), ("", True)]) == [("2", False), ("2", False)]
    assert _split_glued_runs([("1 10", False)]) == [("1 10", False)]
    assert _split_glued_runs([("3 2 1", False), ("", True)]) == [("3 2 1", False), ("", True)]


def test_read_matrix_table_shared_vs_variant_and_merged_cells():
    rows = [
        ["Peripherals", None, "STM32F446MC", "STM32F446ME", "STM32F446RC"],
        ["Flash memory in Kbytes", None, "256", "512", "256"],
        ["SRAM in Kbytes", "System", "128 (112+16)", None, None],
        ["Timers", "General-purpose", "10", None, None],
        ["Communication interfaces", "SPI / I2S", "4/3 (simplex)(2)", None, None],
        ["GPIOs", None, "63", "63", "50"],
    ]
    table = read_matrix_table(rows, page=14, device_summary={})
    assert table is not None
    assert [b["part_number"] for b in table["bindings"]] == ["STM32F446MC", "STM32F446ME", "STM32F446RC"]
    labels = [r["label"] for r in table["attribute_rows"]]
    assert "SRAM in Kbytes System" in labels and "GPIOs" in labels
    sram = next(r for r in table["attribute_rows"] if r["label"].startswith("SRAM"))
    assert sram["values"] == [("128 (112+16)", False), ("128 (112+16)", True), ("128 (112+16)", True)]


def test_non_part_header_is_not_a_device_table():
    rows = [
        ["Mode", None, "Sleep", "Stop", "Standby"],
        ["Flash", None, "on", "off", "off"],
        ["SRAM", None, "on", "on", "off"],
        ["Timers", None, "yes", "no", "no"],
    ]
    assert read_matrix_table(rows, page=3, device_summary={}) is None


def test_st_flash_code():
    assert st_flash_code_kb("STM32F446MC") == 256
    assert st_flash_code_kb("STM32G0B1RE") == 512
    assert st_flash_code_kb("STM32F410RBI") == 128
    assert st_flash_code_kb("STM32WB15CC") is None
