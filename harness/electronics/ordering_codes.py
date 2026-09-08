"""Vendor ordering-code decoders: the part number as a second printed source.

Every MCU vendor prints a part-numbering figure in the datasheet ("Ordering
information", "Part numbering", "Device nomenclature"). Decoding a part
number through that scheme is the vendor stating flash size, pin count and
package a second time, through a different process than the device table.
That makes it an independent source for the two-source rule (Tier B).

Only schemes that decode unambiguously from the part number alone are here.
A family whose letters mean different things across lines (TI MSP430 memory
digits, Microchip PIC16/18 numerics, ATmega pin digit) is deliberately left
out: returning None is the honest answer.

Each decoder returns {"flash_kb": int, "pin_count": int} with whichever keys
it can decode, or {} when the part does not match. Values are checked
against the documents by ``scripts/validate_ordering_codes.py``; a decoder
that disagrees with its own vendor's tables gets fixed or removed, never
softened.
"""

from __future__ import annotations

import re

# ST and GigaDevice share the memory-letter and pin-letter alphabets (GD32
# mirrors STM32 numbering).
ST_MEMORY_LETTER_KB = {
    "4": 16, "6": 32, "8": 64, "B": 128, "C": 256, "D": 384, "E": 512,
    "F": 768, "G": 1024, "H": 1536, "I": 2048, "J": 4096, "K": 3072,
}
ST_PIN_LETTER = {
    "D": 14, "Y": 20, "F": 20, "G": 28, "K": 32, "T": 36, "S": 44, "C": 48,
    "U": 63, "R": 64, "M": 81, "O": 90, "V": 100, "Q": 132, "Z": 144,
    "I": 176, "A": 169, "B": 208, "N": 216, "X": 240,
}
_ST_PART = re.compile(
    r"^STM32(?:[A-Z]\d[A-Z0-9]\d|[A-Z]{2}\d{2}|[A-Z]{3}\d{1,2}|[A-Z]{3}\d)([A-Z])([0-9A-Z])(?:[A-Z0-9]*)$"
)
_GD_PART = re.compile(r"^GD32(?:[A-Z]{1,2}\d{3})([A-Z])([0-9A-Z])(?:[A-Z]\d)?$")

# Silicon Labs EFM32/EFR32: ...F<flash KB><temp><package><pins>-<rev>.
_SILABS_32 = re.compile(r"^EF[MR]32[A-Z]{2}\d{1,2}[A-Z]\d{3}F(\d{2,4})[GI][MQLJ](\d{2,3})(?:-[A-Z])?$")
# EFM8: EFM8BB31F16A-D-4QFN24 -> flash 16 KB, package QFN24.
_SILABS_8 = re.compile(r"^EFM8[A-Z]{2}\d{1,2}F(\d{1,3})[A-Z]-[A-Z]-\d?(?:QFN|QSOP|SOIC|TSSOP)(\d{2})")

# Microchip AVR Dx/Ex: AVR<flash KB>D[ABDE]<pins>.
_AVR_DX = re.compile(r"^AVR(\d{2,3})(?:DA|DB|DD|DU|EA|EB)(\d{2})$")
# tinyAVR 0/1/2-series: ATtiny<flash KB><series><pin code>; pin code 2=8,
# 4=14, 6=20, 7=24 pins.
_ATTINY = re.compile(r"^ATtiny(\d{1,2})([012])([2467])$", re.I)
_ATTINY_PINS = {"2": 8, "4": 14, "6": 20, "7": 24}
# megaAVR 0-series: ATmega<flash KB>0<8|9>; the pin digit covers two packages
# (8 -> 28/32, 9 -> 40/48) so only flash is decoded.
_ATMEGA0 = re.compile(r"^ATmega(\d{1,2})0[89]$", re.I)
# PIC32MX/MZ: ...F<flash KB><pin letter>; PIC24FJ<flash KB>G...
_PIC32 = re.compile(r"^PIC32M[XZK]\d{3}F(\d{2,4})[A-Z]")
_PIC24FJ = re.compile(r"^PIC24FJ(\d{2,4})G[ABCU]\d{3}$")
_PIC32_PINS = {"B": 28, "C": 36, "D": 44, "H": 64, "L": 100}

# Renesas RA: R7FA<series><group><variant><flash code><...><temp><package>.
# Flash code (RA numbering figure): 3=16, 5=32, 7=64, 9=128, B=256, D=512,
# F=1024, H=2048 KB.
_RA = re.compile(r"^R7FA\d[A-Z]\d[A-Z0-9]([3579BDFH])\d[A-Z][A-Z]{2}$")
_RA_FLASH = {"3": 16, "5": 32, "7": 64, "9": 128, "B": 256, "D": 512, "F": 1024, "H": 2048}


def decode(part_number: str | None) -> dict[str, int]:
    """Decode what the vendor's printed ordering scheme states for a part."""
    if not part_number:
        return {}
    part = part_number.strip()
    out: dict[str, int] = {}

    if part.startswith("STM32") and not part.startswith(("STM32WB", "STM32MP", "STM32N", "STM32WL")):
        match = _ST_PART.match(part)
        if match:
            pins, flash = ST_PIN_LETTER.get(match.group(1)), ST_MEMORY_LETTER_KB.get(match.group(2))
            out["scheme"] = "stm32"
            if flash:
                out["flash_kb"] = flash
            if pins:
                # ST's pin letter is the package class, not the exact ball
                # count: WLCSP variants print 12 under "D" (14) and 99 under
                # "V" (100). Corroborates when equal; never contradicts.
                out["pin_count_nominal"] = pins
        return out

    if part.startswith("GD32"):
        match = _GD_PART.match(part)
        if match:
            pins, flash = ST_PIN_LETTER.get(match.group(1)), ST_MEMORY_LETTER_KB.get(match.group(2))
            out["scheme"] = "gd32"
            if flash:
                out["flash_kb"] = flash
            if pins:
                out["pin_count"] = pins
        return out

    match = _SILABS_32.match(part)
    if match:
        return {"scheme": "silabs_efm32_efr32", "flash_kb": int(match.group(1)), "pin_count": int(match.group(2))}
    match = _SILABS_8.match(part)
    if match:
        return {"scheme": "silabs_efm8", "flash_kb": int(match.group(1)), "pin_count": int(match.group(2))}

    match = _AVR_DX.match(part)
    if match:
        return {"scheme": "avr_dx", "flash_kb": int(match.group(1)), "pin_count": int(match.group(2))}
    match = _ATTINY.match(part)
    if match:
        return {"scheme": "tinyavr", "flash_kb": int(match.group(1)), "pin_count": _ATTINY_PINS[match.group(3)]}
    match = _ATMEGA0.match(part)
    if match:
        return {"scheme": "megaavr0", "flash_kb": int(match.group(1))}
    match = _PIC32.match(part)
    if match:
        out = {"scheme": "pic32", "flash_kb": int(match.group(1))}
        letter = part[match.end() - 1]
        if letter in _PIC32_PINS:
            out["pin_count"] = _PIC32_PINS[letter]
        return out
    match = _PIC24FJ.match(part)
    if match:
        return {"scheme": "pic24fj", "flash_kb": int(match.group(1))}

    match = _RA.match(part)
    if match:
        return {"scheme": "renesas_ra", "flash_kb": _RA_FLASH[match.group(1)]}

    return {}


def source_label(decoded: dict[str, int | str], key: str) -> str:
    """Provenance tag for a decoded value."""
    return f"ordering_code:{decoded.get('scheme', 'vendor')}:{key}"
