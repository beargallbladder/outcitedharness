"""Unit tests for the family-level document reader (pure-text paths)."""

from __future__ import annotations

from harness.electronics.family_document import (
    _bullets,
    _feature,
    classify_chapter_title,
    read_chapter_features,
    read_instances,
)


def test_chapter_titles_map_to_classes_with_exclusions() -> None:
    assert classify_chapter_title("30 Universal synchronous/asynchronous receiver transmitter (USART/UART)") == "usart"
    assert classify_chapter_title("6 NVM (Flash)") == "flash"
    assert classify_chapter_title("31 CAN-FD") == "can"
    assert classify_chapter_title("11 TinyEngineTM NPU") == "npu"
    # exclusions: look-alikes that are not the peripheral
    assert classify_chapter_title("9 Dual-Clock Comparator (DCC) Module") != "comparator"
    assert classify_chapter_title("22.10 Trip-Zone (TZ) Submodule") != "trustzone"
    assert classify_chapter_title("25.14 Message RAM") != "sram"
    assert classify_chapter_title("3.9 32-Bit CPU Timers 0/1/2") == "timer"


def test_instances_count_only_multi_page_names() -> None:
    pages = {
        1: "SPI1 SPI2 SPI3 TIM1 TIM2 USART1 GPIOA GPIOB FDCAN1 USB_OTG_HS",
        2: "SPI1, SPI2 and SPI3 support I2S. TIM1 TIM2 TIM15. USART1 GPIOA GPIOB FDCAN1",
        3: "SPI6 is quoted once here only. TIM17 too.",
    }
    out = read_instances(pages)
    assert [i["instance"] for i in out["SPI"]["instances"] if not i["weak"]] == ["SPI1", "SPI2", "SPI3"]
    assert out["SPI"]["count_asserted"] == 3
    weak = {i["instance"] for i in out["SPI"]["instances"] if i["weak"]}
    assert weak == {"SPI6"}
    assert out["GPIO"]["count_asserted"] == 2
    assert out["FDCAN"]["count_asserted"] == 1
    assert [i["instance"] for i in out["USB"]["instances"]] == ["USB_OTG_HS"]


def test_bullets_keep_wrapped_lines_and_stop_at_next_heading() -> None:
    text = "• Full-duplex synchronous transfers on three lines\n• 4-bit to 16-bit data size\nselection\n• Up to 8 slaves\n30.4 SPI implementation\n• not part of features\n"
    assert _bullets(text) == [
        "Full-duplex synchronous transfers on three lines",
        "4-bit to 16-bit data size selection",
        "Up to 8 slaves",
    ]


def test_feature_types_numbers_and_qualifier_verbatim() -> None:
    row = _feature("Up to 2 Mbytes of Flash memory with ECC", 1)
    assert row["qualifier_verbatim"] == "Up to"
    assert row["numbers"][0] == {"value": 2, "unit": "Mbytes"}
    assert "flash" in row["classes"]
    bare = _feature("640 Kbytes of SRAM", 1)
    assert bare["qualifier_verbatim"] is None
    assert bare["numbers"][0] == {"value": 640, "unit": "Kbytes"}


def test_renesas_cover_hierarchy_headers_bullets_subbullets_and_counts() -> None:
    from harness.electronics.family_document import _bullet_items, split_count

    text = (
        "Features\n■ Memory\n\uf0b7512-KB code flash memory\n\uf0b796-KB SRAM\n■ Connectivity\n"
        "\uf0b7Serial Communications Interface (SCI) × 4\n- UART\n- Simple SPI\n\uf0b7I2C bus interface (IIC) × 2\n"
    )
    items = _bullet_items(text)
    assert [(i["section"], i["text"], i["level"]) for i in items] == [
        ("Memory", "512-KB code flash memory", 1),
        ("Memory", "96-KB SRAM", 1),
        ("Connectivity", "Serial Communications Interface (SCI) × 4", 1),
        ("Connectivity", "UART", 2),
        ("Connectivity", "Simple SPI", 2),
        ("Connectivity", "I2C bus interface (IIC) × 2", 1),
    ]
    assert items[3]["parent"] == "Serial Communications Interface (SCI) × 4"
    assert split_count("Serial Communications Interface (SCI) × 4") == ("Serial Communications Interface (SCI)", 4)
    assert split_count("8-bit D/A Converter (DAC8) ×2 (for ACMPLP)") == ("8-bit D/A Converter (DAC8) (for ACMPLP)", 2)
    assert split_count("DMA x8") == ("DMA", 8)
    assert split_count("Temperature Sensor") == ("Temperature Sensor", None)


def test_single_glyph_style_has_no_header_level() -> None:
    from harness.electronics.family_document import _bullet_items

    items = _bullet_items("• Up to 2 Mbytes of Flash\n• 640 Kbytes of SRAM\n")
    assert [i["section"] for i in items] == [None, None]
    assert [i["text"] for i in items] == ["Up to 2 Mbytes of Flash", "640 Kbytes of SRAM"]


def test_prose_facts_collapse_core_spellings_and_pick_widest_supply() -> None:
    from harness.electronics.family_document import read_prose_facts

    pages = {
        1: "Arm® Cortex®-M4 with FPU running up to 168 MHz\nCortex®-M4\n",
        2: "ARM Cortex-M4 core. Operating voltage VDD: 1.8 to 3.6 V. ADC range 2.4 to 3.6 V (not a supply line).",
        3: "The supply voltage range is 2.7 to 3.6 V for the USB block. -40 to +105 °C\n",
    }
    facts = read_prose_facts(pages)
    kinds = [(f["kind"], f["verbatim"]) for f in facts]
    assert sum(1 for k, _ in kinds if k == "core") == 1
    assert ("max_frequency", "up to 168 MHz") in kinds
    supply = [f for f in facts if f["kind"] == "supply_range"]
    assert len(supply) == 1 and supply[0]["value"] == [1.8, 3.6]
    assert [f["value"] for f in facts if f["kind"] == "temperature_range"] == [[-40, 105]]


def test_quantity_qualifier_names_the_quantity_not_the_bound() -> None:
    from harness.electronics.key_features_grid import quantity_qualifier

    def row(label, value, unit, section=None, cls=None, qualifier=None, context=None):
        return {"label": label, "verbatim": label, "value": value, "unit": unit, "section": section, "peripheral_class": cls, "qualifier_verbatim": qualifier, "context": context}

    assert quantity_qualifier(row("512-KB code flash memory", 512, "KB", cls="flash")) == "code_flash"
    assert quantity_qualifier(row("8-KB data flash memory (100,000 erase/write cycles)", 8, "KB", cls="flash")) == "data_flash"
    assert quantity_qualifier(row("Up to 2 Mbytes of Flash memory", 2, "Mbytes", cls="flash", qualifier="Up to")) is None  # bare flash: not called code
    assert quantity_qualifier(row("96-KB SRAM", 96, "KB", cls="sram")) == "total_sram"
    assert quantity_qualifier(row("Main internal SRAM1 (112 KB)", 112, "KB", cls="sram")) == "sram_bank"
    assert quantity_qualifier(row("Ta = –40°C to +85°C", [-40, 85], "°C", section="temperature_range")) == "ambient"
    assert quantity_qualifier(row("–40 to +125 °C", [-40, 125], "°C", section="temperature_range", context="Junction temperature Tj –40 to +125 °C")) == "junction"
    assert quantity_qualifier(row("–40 to +125 °C", [-40, 125], "°C", section="temperature_range")) is None
    assert quantity_qualifier(row("1.8 to 3.6 V", [1.8, 3.6], "V", section="supply_range", context="The device requires a 1.8 to 3.6 V operating voltage supply (VDD)")) == "rated"
    assert quantity_qualifier(row("1.8 to 4.0 V", [1.8, 4.0], "V", section="supply_range", context="Absolute maximum ratings VDD 1.8 to 4.0 V")) == "absolute_maximum"
    assert quantity_qualifier(row("Maximum operating frequency: 48 MHz", 48, "MHz", cls="cpu_core", qualifier="Maximum")) == "maximum"
    assert quantity_qualifier(row("SCI", None, None)) is None


def test_sram_banks_must_sum_to_total_or_total_is_refused() -> None:
    from harness.electronics.key_features_grid import _sram_bank_check

    def r(label, value, qq, tier="grid"):
        return {"label": label, "value": value, "unit": "KB", "quantity_qualifier": qq, "tier": tier}

    ok = [r("SRAM1 112 KB", 112, "sram_bank"), r("SRAM2 16 KB", 16, "sram_bank"), r("128 KB SRAM", 128, "total_sram")]
    assert _sram_bank_check(ok)["sram_banks_sum_to_total"] is True and ok[2]["tier"] == "grid"
    bad = [r("SRAM1 112 KB", 112, "sram_bank"), r("SRAM2 16 KB", 16, "sram_bank"), r("256 KB SRAM", 256, "total_sram")]
    res = _sram_bank_check(bad)
    assert res["sram_banks_sum_to_total"] is False and bad[2]["tier"] == "below_grid" and bad[2]["refused"]


def test_chapter_features_tag_section_and_enclosing_chapter() -> None:
    chapters = {"all_entries": [
        {"level": 1, "title": "30 Serial peripheral interface (SPI)", "page": 100},
        {"level": 1, "title": "31 Inter-integrated circuit (I2C)", "page": 140},
    ]}
    pages = {
        101: "30.1 Introduction\ntext\n30.3 SPI main features\n• Full-duplex synchronous transfers\n• Up to 8 slaves\n30.4 SPI implementation\n",
        141: "Refer to the data sheet for device-specific features\n• stray bullet\n",
    }
    rows = read_chapter_features(pages, chapters, 200)
    assert len(rows) == 2
    assert {r["chapter_class"] for r in rows} == {"spi"}
    assert rows[0]["section"] == "SPI main" or rows[0]["section"] == "SPI"
    assert rows[1]["qualifier_verbatim"] == "Up to"
