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
    # Bare "Flash": the document did not say which; CR wants the null, not a guess.
    assert quantity_qualifier(row("Up to 2 Mbytes of Flash memory", 2, "Mbytes", cls="flash", qualifier="Up to")) is None
    # A multi-valued line has no scalar value but still names its quantity.
    assert quantity_qualifier({**row("8 KB or 16 KB data flash", None, None, cls="flash"), "group": "memory"}) == "data_flash"
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


def test_description_paragraph_reads_counted_sized_and_named_facts() -> None:
    from harness.electronics.family_document import read_description_facts

    paragraph = (
        "1. \nGeneral description \n"
        "The GD32F403xx device belongs to the performance line of GD32 MCU Family. It is a new 32-bit general-purpose "
        "microcontroller based on the Arm® Cortex®-M4 RISC core with best cost-performance ratio in terms of enhanced "
        "processing capacity. The GD32F403xx device incorporates the Arm® Cortex®-M4 32-bit processor core operating at "
        "168 MHz frequency with Flash accesses zero wait states to obtain maximum efficiency. It provides up to 3072 KB "
        "on-chip Flash memory and 128 KB SRAM memory. The devices offer up to three 12-bit 2.6M MSPS ADCs, two 12-bit DACs, "
        "up to eight general-purpose 16-bit timers, as well as standard and advanced communication interfaces: up to three "
        "SPIs, two I2Cs, two CANs, a SDIO, and an USBFS. The device operates from a 2.6 to 3.6 V power supply and available "
        "in –40 to +85 °C temperature range. Each TC can be configured to perform frequency generation. The above features "
        "make GD32F403xx devices suitable for a wide range of applications."
    )
    rows = read_description_facts({1: paragraph})
    by_label = {r["label"]: r for r in rows}
    assert by_label["up to 3072 KB on-chip Flash memory"]["qualifier_verbatim"] == "up to"
    assert "flash" in by_label["up to 3072 KB on-chip Flash memory"]["classes"]
    assert by_label["12-bit 2.6M MSPS ADCs"]["count"] == 3
    assert by_label["general-purpose 16-bit timers"]["count"] == 8
    assert by_label["SPIs"]["count"] == 3 and "spi" in by_label["SPIs"]["classes"]
    assert by_label["CANs"]["count"] == 2 and "can" in by_label["CANs"]["classes"]
    assert "SDIO" in by_label and "USBFS" in by_label
    assert "2.6 to 3.6 V power supply" in by_label
    assert any(l.startswith("–40 to +85") for l in by_label)
    # The verb "can" is not the bus; marketing sentences are not facts.
    assert not any("configured" in l or "suitable" in l for l in by_label)


def test_includes_list_reads_rzg_product_card() -> None:
    from harness.electronics.family_document import read_includes_list

    pages = {
        9: (
            "1.1 Introduction\n"
            "The RZ/G1H includes:\n"
            "•\n"
            "Four 1.4-GHz ARM Cortex®-A15 MPCore® cores,\n"
            "•\n"
            "Four 780-MHz ARM Cortex®-A7 MPCore® cores,\n"
            "•\n"
            "3 channels Display Output,\n"
            "1.2 System Configuration Diagram\n"
        )
    }
    rows = read_includes_list(pages)
    texts = [r["verbatim"] for r in rows]
    assert any("1.4-GHz" in t and "Cortex" in t and "A15" in t for t in texts)
    assert any("780-MHz" in t and "A7" in t for t in texts)
    assert any("Display Output" in t for t in texts)
    assert all(r["source"] == "includes_list" for r in rows)


def test_item_description_specs_read_cores_cache_and_gpio() -> None:
    from harness.electronics.family_document import read_item_description_specs

    pages = {
        7: (
            "Contents\n"
            "1.3 List of Specifications ......................................................... 1-3\n"
            "1.4 Power Supply Voltages and Temperature Range ............. 1-24\n"
        ),
        11: (
            "1.3 List of Specifications\n"
            "1.3.1 ARM Core\n"
            "Item\n"
            "Description\n"
            "System CPU Cortex-A15\n"
            "• ARM Cortex-A15 Quad MPCore 1.4 GHz\n"
            "• L1 I/D cache 32/32 KBytes, L2 cache 2 MBytes\n"
            "System CPU Cortex-A7\n"
            "• ARM Cortex-A7 Quad MPCore 780 MHz\n"
            "• L1 I/D cache 32/32 KBytes, L2 cache 512 KBytes\n"
        ),
        12: (
            "Item\n"
            "Description\n"
            "General-purpose I/O (GPIO)\n"
            "• General-purpose I/O ports: 188 ports\n"
            "• Supports GPIO interrupts.\n"
        ),
        31: (
            "Item\n"
            "Description\n"
            "Process\n"
            "28-nm Si-CMOS\n"
            "Package\n"
            "FC-BGA2727-831\n"
        ),
        32: "1.4 Power Supply Voltages and Temperature Range\n• Temperature range\n",
    }
    rows = read_item_description_specs(pages)
    texts = [r["verbatim"] for r in rows]
    assert any("Cortex-A15 Quad MPCore 1.4 GHz" in t for t in texts)
    assert any("L2 cache 2 MBytes" in t for t in texts)
    assert any("Cortex-A7 Quad MPCore 780 MHz" in t for t in texts)
    assert any("L2 cache 512 KBytes" in t for t in texts)
    assert any("188 ports" in t for t in texts)
    assert any("FC-BGA2727-831" in t for t in texts)
    a15 = next(r for r in rows if "Cortex-A15 Quad" in r["verbatim"])
    assert a15["vendor_section"] == "System CPU Cortex-A15"
    assert "cpu_core" in a15["classes"]


def test_stated_gpio_pairs_with_fc_bga_not_ballout() -> None:
    from harness.electronics.family_document import read_io_by_package

    pages = {
        12: "General-purpose I/O (GPIO)\n• General-purpose I/O ports: 188 ports\n",
        31: "Package\nFC-BGA2727-831\n",
        35: (
            "3.1 Top View (Left)\n"
            "A1 A2 A3 B1 B2 B3\n"
            "VSS VDD GPIO0 GPIO1\n"
        ),
    }
    rows = read_io_by_package(pages)
    assert len(rows) == 1
    assert rows[0]["pattern"] == "stated_gpio_and_fcbga"
    assert rows[0]["io_count"] == 188
    assert rows[0]["pin_count"] == 831
    assert rows[0]["package"] == "FCBGA"
