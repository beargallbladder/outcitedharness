"""Ordering-code decoders: the part number as a second printed source."""

from __future__ import annotations

from harness.electronics.ordering_codes import decode, source_label


def test_st_and_gd_share_letter_tables():
    assert decode("STM32F446MC") == {"scheme": "stm32", "flash_kb": 256, "pin_count_nominal": 81}
    assert decode("GD32F405RG") == {"scheme": "gd32", "flash_kb": 1024, "pin_count": 64}
    assert decode("STM32WB55RG") == {}  # wireless line uses another table


def test_vendor_schemes():
    assert decode("EFM32GG12B110F1024GM64-A") == {"scheme": "silabs_efm32_efr32", "flash_kb": 1024, "pin_count": 64}
    assert decode("EFM8BB31F16A-D-4QFN24") == {"scheme": "silabs_efm8", "flash_kb": 16, "pin_count": 24}
    assert decode("AVR128DA28") == {"scheme": "avr_dx", "flash_kb": 128, "pin_count": 28}
    assert decode("ATtiny1624") == {"scheme": "tinyavr", "flash_kb": 16, "pin_count": 14}
    assert decode("ATmega4809") == {"scheme": "megaavr0", "flash_kb": 48}
    assert decode("PIC32MX320F064H") == {"scheme": "pic32", "flash_kb": 64, "pin_count": 64}
    assert decode("PIC24FJ128GA106") == {"scheme": "pic24fj", "flash_kb": 128}
    assert decode("R7FA2E1A93CFM") == {"scheme": "renesas_ra", "flash_kb": 128}


def test_undecodable_is_empty_not_guessed():
    assert decode("MSP430AFE221IPW") == {}
    assert decode("PIC18F6527") == {}
    assert decode(None) == {}
    assert source_label(decode("GD32F405RG"), "flash_kb") == "ordering_code:gd32:flash_kb"
