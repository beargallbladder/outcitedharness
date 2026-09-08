"""Deterministic reader for family device-comparison tables (knife lane v1).

Input: a document the census marked ``above_opn`` / ``family_matrix``. Output:
one family record in the CR conventions shape (``_meta`` receipts, typed
numeric leaves ``{typ|min|max, unit}``, ``part_number: null`` + ``family`` +
``part_numbers_covered``) with two layers:

  shared    attributes the table prints identically for every variant
            (merged cell or equal values) -- the family content layer;
  variants  one row per data column the table carries, bound to a concrete
            part number only when the document itself binds it (header token,
            header code list, or a printed device-summary join) -- the knife
            layer.

Every value keeps its verbatim string, page, row label and column. A value the
reader cannot type with certainty is emitted as ``unknown`` with the verbatim
retained; nothing is guessed. Column-to-part binding that cannot be made from
printed text is ``part_number: null`` with the header token kept, never a
minted part number.

The label grammar here is deliberately stricter than the census's
``attributes_in_label`` (which only asks "does this row address the
attribute?"). Here the label decides which knife the value lands in, so
"GPIOs" is ``gpio_count`` not ``pin_count``, "Tamper pins" is nothing,
"Maximum CPU frequency" is ``freq_mhz`` and not ``core``, and a row that
prints "SPI / I2S : 4/3" becomes ``spi_count=4`` and ``i2s_count=3`` because
the document itself paired them.

PyMuPDF tables only. Frontier/local models never run here; they are the
teacher for the ``unknown`` residue downstream.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from harness.electronics.family_census import (
    PART_TOKEN,
    attributes_in_label,
    classify_table,
    is_concrete_part_token,
    is_wildcard_token,
    packages_from_text,
    strip_title_rows,
)

MATRIX_SCHEMA = "harness.electronics-family-device-matrix.v1"
EXTRACTED_BY = "harness family_device_matrix v1 (deterministic PyMuPDF)"

# Attributes that get a typed numeric leaf and their default unit.
NUMERIC_ATTRIBUTES: dict[str, str] = {
    "code_flash_kb": "KB",
    "data_flash_kb": "KB",
    "eeprom_kb": "KB",
    "sram_kb": "KB",
    "freq_mhz": "MHz",
    "pin_count": "pins",
    "gpio_count": "count",
    "adc_count": "count",
    "adc_channels": "channels",
    "dac_count": "count",
    "dac_channels": "channels",
    "timers": "count",
    "timer_advanced": "count",
    "timer_general_purpose": "count",
    "timer_basic": "count",
    "timer_low_power": "count",
    "timer_watchdog": "count",
    "timer_systick": "count",
    "rtc": "count",
    "pwm_channels": "channels",
    "usart_count": "count",
    "uart_count": "count",
    "lpuart_count": "count",
    "spi_count": "count",
    "i2s_count": "count",
    "i2c_count": "count",
    "can_count": "count",
    "sdio_count": "count",
    "comparator_count": "count",
    "opamp_count": "count",
    "operating_voltage": "V",
    "temp_range": "degC",
    "standby_current_ua": "uA",
}
BOOLEAN_ATTRIBUTES = {"usb", "ethernet", "trustzone", "crypto", "lcd", "camera", "fsmc"}

# (attribute, must-match, must-not-match). Evaluated against the full label
# (carried group label + sub label); first match wins per token group.
_LABEL_RULES: tuple[tuple[str, re.Pattern[str], re.Pattern[str] | None], ...] = (
    ("data_flash_kb", re.compile(r"\bdata\s*(?:flash|area|memory)\b", re.I), None),
    ("eeprom_kb", re.compile(r"\beeprom\b", re.I), None),
    ("code_flash_kb", re.compile(r"\bflash\b|\bprogram\s+memory\b|\bcode\s+(?:area|memory)\b", re.I), re.compile(r"\bdata\b|\bexternal\b|\boption\b|\bsector\b|\bpage\b|\bbank|\bfs?mc\b|\bnand\b|\bnor\b|controller|\becc\b|protection|interface|\botfdec\b|\bxspi\b|\bqspi\b|\bospi\b", re.I)),
    ("sram_kb", re.compile(r"\bs?ram\b", re.I), re.compile(r"\bbackup\b|\bcache\b|\bparity\b\s*only|\bdma\b|\bfs?mc\b|controller|\bexternal\b|\bpsram\b|\bsdram\b|\becc\b|\bsram\d\b|\baxi\b|\bahb\b|\bd\d\s+domain\b|\bitcm\b|\bdtcm\b|\btcm\b|\bccm\b|\bretention\b|instruction|flexible", re.I)),
    ("freq_mhz", re.compile(r"\b(?:frequency|freq\.?|clock\s+speed|cpu\s+speed|speed)\b|\bmhz\b", re.I), re.compile(r"\badc\b|\bbus\b|\bexternal\b", re.I)),
    ("core", re.compile(r"\b(?:core|cpu|cortex)\b", re.I), re.compile(r"frequen|speed|mhz|\bram\b", re.I)),
    ("package", re.compile(r"\bpackages?\b", re.I), re.compile(r"\bpins?\b|tamper|wakeup|\blegacy\b|\bsmps\b|dedicated|thermal|\bfootprint\b", re.I)),
    ("gpio_count", re.compile(r"\bgpios?\b|\bi/os?\b|\bgeneral[\s-]purpose\s+i/?os?\b|\bio\s+pins?\b", re.I), re.compile(r"\bwakeup\b|\btamper\b|\bfast\b|\bnormal\b|\bfs?mc\b|tolerant|\b5\s*v\b|\btc\b|\btta?\b|\bft\b|\bhigh[\s-]sink\b|\bmultiplexed\b", re.I)),
    ("timer_advanced", re.compile(r"\btimers?\b.*\badvanced\b|\badvanced[\s-]control\b", re.I), None),
    ("timer_general_purpose", re.compile(r"\btimers?\b.*\bgeneral\b|\bgeneral\s*-?\s*purpose\b", re.I), re.compile(r"\bi/?os?\b|input|output|gpio", re.I)),
    ("timer_basic", re.compile(r"\btimers?\b.*\bbasic\b", re.I), None),
    ("timer_low_power", re.compile(r"\btimers?\b.*\blow[\s-]power\b|\blptim\b", re.I), None),
    ("timer_systick", re.compile(r"\bsystick\b", re.I), None),
    ("timer_watchdog", re.compile(r"\bwatchdog\b|\bwdg\b", re.I), None),
    ("rtc", re.compile(r"\brtc\b|\breal[\s-]time\s+clock\b", re.I), None),
    ("pwm_channels", re.compile(r"\bpwm\b", re.I), re.compile(r"advanced|general|basic|motor|low[\s-]power|except|complementary", re.I)),
    ("timers", re.compile(r"\btimers?\b", re.I), re.compile(r"advanced|general|basic|low|systick|watchdog|wdg|\blp\b|\b16-bit\b|\b32-bit\b|encoder|motor|pwm", re.I)),
    ("adc_channels", re.compile(r"\badcs?\b.*\bchannels?\b|\bchannels?\b.*\badcs?\b", re.I), None),
    ("adc_count", re.compile(r"\badcs?\b", re.I), re.compile(r"channel|resolution|speed|rate|bit\b", re.I)),
    ("dac_channels", re.compile(r"\bdacs?\b.*\bchannels?\b|\bchannels?\b.*\bdacs?\b", re.I), None),
    ("dac_count", re.compile(r"\bdacs?\b", re.I), re.compile(r"channel|resolution|bit\b", re.I)),
    ("comparator_count", re.compile(r"\bcomparators?\b|\bcomp\b", re.I), None),
    ("opamp_count", re.compile(r"\bop[\s-]?amps?\b|\boperational\s+amplifier", re.I), None),
    ("operating_voltage", re.compile(r"\b(?:operating|supply|power\s+supply)\s+voltage\b|\bvdd\b", re.I), re.compile(r"temperature|\busb\b|adc|analog|\bvbat\b|\bvref\b|\bvdda\b|\bvddio", re.I)),
    ("temp_range", re.compile(r"\btemperature", re.I), None),
    ("standby_current_ua", re.compile(r"\b(?:standby|stop|shutdown|deep\s*sleep|power[\s-]?down)\b.*?\b(?:current|consumption)\b", re.I), None),
    ("lpuart_count", re.compile(r"\blpuart\b|\blow[\s-]power\s+uart\b", re.I), None),
    ("usart_count", re.compile(r"\busart\b", re.I), None),
    ("uart_count", re.compile(r"\buart\b", re.I), None),
    ("i2s_count", re.compile(r"\bi2s\b", re.I), None),
    ("spi_count", re.compile(r"\bspi\b", re.I), re.compile(r"\bquad|\bocto|\bqspi\b|\bospi\b|\bxspi\b|\bhyper", re.I)),
    ("i2c_count", re.compile(r"\bi2c\b|\bi²c\b", re.I), None),
    ("can_count", re.compile(r"\bcan(?:\s*fd)?\b|\bfdcan\b|\bbxcan\b", re.I), None),
    ("sdio_count", re.compile(r"\bsdio\b|\bsdmmc\b|\bsd/mmc\b", re.I), None),
    ("usb", re.compile(r"\busb\b", re.I), None),
    ("ethernet", re.compile(r"\bethernet\b|\beth\b", re.I), None),
    ("trustzone", re.compile(r"\btrustzone\b", re.I), None),
    ("crypto", re.compile(r"\bcrypto|\baes\b|\bhash\b|\bpka\b", re.I), None),
    ("lcd", re.compile(r"\blcd\b|\bltdc\b", re.I), None),
    ("camera", re.compile(r"\bcamera\b|\bdcmi\b", re.I), None),
    ("fsmc", re.compile(r"\bfsmc\b|\bfmc\b", re.I), re.compile(r"\bnor\w*|\bnand\b|\bs?ram\b|\bpsram\b|\bsdram\b|multiplexed|\bmux\b|controller|\bi/o", re.I)),
)

_UNIT_IN_LABEL = (
    (re.compile(r"\b(?:kbytes?|kb|kbyte|kilobytes?)\b", re.I), "KB"),
    (re.compile(r"\b(?:mbytes?|mb|mbyte)\b", re.I), "MB"),
    (re.compile(r"\bmhz\b", re.I), "MHz"),
    (re.compile(r"\bkhz\b", re.I), "kHz"),
    (re.compile(r"[µu]a\b", re.I), "uA"),
    (re.compile(r"\bma\b", re.I), "mA"),
    (re.compile(r"°\s*c\b|\bdeg\s*c\b", re.I), "degC"),
    (re.compile(r"(?<![a-z])v\b", re.I), "V"),
)

_FOOTNOTE = re.compile(r"(?<=\S)\s*\(\d{1,2}\)$")
_INT = re.compile(r"^\d{1,6}$")
_NUMBER = re.compile(r"^\d{1,6}(?:\.\d+)?$")
_INT_WITH_BREAKDOWN = re.compile(r"^(\d{1,6})\s*\((?:\d+\s*[+x×]\s*)+\d+\)$")
_INT_WITH_NOTE = re.compile(r"^(\d{1,6})\s*(\([^()]*\)|[a-z][a-z /.-]*)$", re.I)
_INT_WITH_UNIT = re.compile(
    r"^(\d{1,6}(?:\.\d+)?)\s*(k\s*bytes?|kbytes?|kb|k|m\s*bytes?|mbytes?|mb|m|mhz|khz|[µu]a|ma|v|°c)$", re.I
)
_EXTRA_CLAUSE = re.compile(r"\s*\+\s*\d+\s*extra\b.*$", re.I)
_INT_TOKENS = re.compile(r"^\d{1,4}(?:\s*[/ ]\s*\d{1,4})+$")
_RANGE = re.compile(
    r"^([-+]?\d+(?:\.\d+)?)\s*(?:to|-|\.\.)\s*([-+]?\d+(?:\.\d+)?)\s*([a-zµ°]*)$",
    re.I,
)
# First printed range inside a longer cell ("Ambient: -40 to 85 °C / -40 to
# 105 °C", "1.8 V to 3.6 V (down to 1.65 V ...)").
_RANGE_ANY = re.compile(
    r"([-+]?\s?\d+(?:\.\d+)?)\s*(?:V|°\s*C)?\s*(?:to|-)\s*([-+]?\s?\d+(?:\.\d+)?)", re.I
)
# "5 (16-bit) 2 (32-bit)", "2 (32 bits) and 8 (16 bits)", "1 (16-bit) high frequency"
_BITWIDTH_GROUP = re.compile(r"(\d{1,3})\s*\(\s*\d{1,2}\s*-?\s*bits?\s*\)", re.I)
_YES = re.compile(r"^(yes|y|✓|x)$", re.I)
_NO = re.compile(r"^(no|n|n/a|na)$", re.I)
_EXPLICIT_NONE = re.compile(r"^(-{1,2}|—|–|none|0)$", re.I)
_YES_NO = re.compile(r"^(yes|no|y|n|-|—|–|n/a|na|✓|x)$", re.I)
_COMPOSITE_LABEL = re.compile(r"\s*(?:/|\[|\])\s*")
_MULTI_VALUE = re.compile(r"\bor\b", re.I)

_UNIT_WORDS = {
    "kbytes": "KB", "kbyte": "KB", "kb": "KB", "k": "KB", "k bytes": "KB", "k byte": "KB",
    "mbytes": "MB", "mbyte": "MB", "mb": "MB", "m": "MB", "m bytes": "MB", "m byte": "MB",
    "mhz": "MHz", "khz": "kHz", "ua": "uA", "µa": "uA",
    "ma": "mA", "v": "V", "°c": "degC",
}

# ST ordering-code memory letter -> code flash KB (public ST grammar, used
# only as a self-consistency check, never as a value source).
ST_FLASH_CODE_KB = {
    "4": 16, "6": 32, "8": 64, "B": 128, "C": 256, "D": 384, "E": 512,
    "F": 768, "G": 1024, "H": 1536, "I": 2048, "J": 4096,
}
# STM32 + product line (F446, G0B1, WB55, WBA23, WLE5) + pin letter + memory
# letter; anything after is package/temperature/option suffix.
_ST_PART = re.compile(
    r"^STM32(?:[A-Z]\d[A-Z0-9]\d|[A-Z]{2}\d{2}|[A-Z]{3}\d{1,2}|[A-Z]{3}\d)([A-Z])([0-9A-Z])(?:[A-Z0-9]*)$"
)
_HEADER_LIST_SPLIT = re.compile(r"\s*[,/]\s*")


def _norm(text: Any) -> str:
    return " ".join(str(text).split()) if text is not None else ""


def _strip_footnotes(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = _FOOTNOTE.sub("", text).strip()
    return text


def unit_from_label(label: str) -> str | None:
    for pattern, unit in _UNIT_IN_LABEL:
        if pattern.search(label):
            return unit
    return None


def label_attribute(label: str) -> str | None:
    """Single attribute for a (sub)label, or None when it addresses nothing we
    read. First matching rule wins."""
    for attribute, must, must_not in _LABEL_RULES:
        if must.search(label) and not (must_not and must_not.search(label)):
            return attribute
    return None


def split_composite_label(sub_label: str) -> list[str]:
    """'SPI / I2S' -> ['SPI', 'I2S']; 'SPI [I2S]' -> ['SPI', 'I2S'];
    'USART/ UART' -> ['USART', 'UART']. Anything else -> [sub_label]."""
    # Only a slash between two multi-character names is a pairing; "I/O",
    # "D/A", "A/D", "R/W" are single names.
    if not re.search(r"(?<=\w\w)\s*/\s*(?=\w\w)|\[", sub_label):
        return [sub_label]
    parts = [p for p in re.split(r"(?<=\w\w)\s*/\s*(?=\w\w)|\s*\[\s*|\s*\]\s*", sub_label) if p]
    return parts if 2 <= len(parts) <= 3 else [sub_label]


def split_composite_value(text: str, count: int) -> list[str] | None:
    """'4/3 (simplex)' -> ['4', '3 (simplex)'] ; '2 [1]' -> ['2', '1'];
    '4/1 FMP +' -> ['4', '1 FMP +']. None when the value does not carry
    exactly ``count`` components."""
    parts = [p.strip() for p in re.split(r"\s*/\s*|\s*\[\s*|\s*\]\s*", text) if p.strip()]
    return parts if len(parts) == count else None


def row_attributes(group_label: str, sub_label: str) -> list[tuple[str, int | None]]:
    """Attributes addressed by one table row.

    Returns [(attribute, component_index)] where component_index is None for
    a plain row and 0..n-1 for a composite row like 'SPI / I2S'.
    """
    if sub_label:
        components = split_composite_label(sub_label)
        if len(components) > 1:
            out: list[tuple[str, int | None]] = []
            for index, component in enumerate(components):
                attribute = label_attribute(f"{group_label} {component}")
                if attribute is None:
                    attribute = label_attribute(component)
                if attribute:
                    out.append((attribute, index))
            if out:
                return out
    full = " ".join(p for p in (group_label, sub_label) if p)
    attribute = label_attribute(full)
    if attribute is None and sub_label:
        attribute = label_attribute(sub_label)
    return [(attribute, None)] if attribute else []


def parse_value(verbatim: str, attribute: str, label_unit: str | None) -> dict[str, Any]:
    """Type one printed cell. Returns a leaf with ``status`` in
    {typed, boolean, verbatim, unknown}; the verbatim string is always kept."""
    raw = _norm(verbatim)
    text = _strip_footnotes(raw)
    leaf: dict[str, Any] = {"verbatim": raw}
    if not text:
        leaf["status"] = "unknown"
        leaf["reason"] = "empty_cell"
        return leaf

    if attribute in BOOLEAN_ATTRIBUTES:
        if _YES_NO.match(text):
            leaf["status"] = "boolean"
            leaf["value"] = text.lower() in {"yes", "y", "✓", "x"}
            return leaf
        lead = re.match(r"^(yes|no)\b[\s:(,-]*(.*)$", text, re.I)
        if lead:
            # "Yes (6-Endpoints)", "No (see note)": the flag plus a qualifier.
            leaf["status"] = "boolean"
            leaf["value"] = lead.group(1).lower() == "yes"
            if lead.group(2).strip(" )"):
                leaf["note"] = lead.group(2).strip(" )")
            return leaf
        if _NUMBER.match(text) or re.match(r"^\d+\s", text):
            # "1", "2 (FS + HS)": instances printed where a flag was expected.
            leaf["status"] = "boolean"
            leaf["value"] = float(text.split()[0]) > 0
            leaf["note"] = text
            return leaf
        leaf["status"] = "verbatim"
        return leaf

    if attribute == "package":
        packages = packages_from_text(text)
        leaf["status"] = "verbatim"
        if packages:
            leaf["packages"] = packages
        return leaf

    if attribute not in NUMERIC_ATTRIBUTES:
        leaf["status"] = "verbatim"
        return leaf

    unit = label_unit or NUMERIC_ATTRIBUTES[attribute]

    if attribute in {"operating_voltage", "temp_range"}:
        cleaned = text.replace("−", "-").replace("–", "-").replace("—", "-").replace("º", "°")
        cleaned = re.sub(r"(?<=\d)\s*(?:V|°\s*C)\b", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*(?:V|°\s*C)\s*$", "", cleaned, flags=re.I)
        cleaned = re.sub(r"(?<![\d.])-\s+(?=\d)", "-", cleaned)
        match = _RANGE.match(cleaned)
        if match is None and attribute == "temp_range":
            # Multi-grade cells ("-40 to 85 / -40 to 105 / -40 to 125"): the
            # document states a set of grades. min/max carry the first printed
            # grade; every grade is kept in ``grades``. A junction-temperature
            # clause is not an operating range.
            ambient = re.split(r"\bjunction\b", cleaned, flags=re.I)[0]
            grades = [(float(a.replace(" ", "").replace("+", "")), float(b.replace(" ", "").replace("+", ""))) for a, b in _RANGE_ANY.findall(ambient)]
            if grades and not re.match(r"^\s*junction", cleaned, re.I):
                leaf.update({"status": "typed", "min": grades[0][0], "max": grades[0][1], "unit": unit, "grades": [list(g) for g in grades], "note": "first of several printed grades" if len(grades) > 1 else "range within longer cell"})
                return leaf
        if match is None and attribute == "operating_voltage":
            first = _RANGE_ANY.search(cleaned)
            if first:
                leaf.update({"status": "typed", "min": float(first.group(1).replace("+", "")), "max": float(first.group(2).replace("+", "")), "unit": unit, "note": cleaned[first.end():].strip(" ()") or "range within longer cell"})
                return leaf
        if match:
            leaf.update(
                {
                    "status": "typed",
                    "min": float(match.group(1).replace("+", "")),
                    "max": float(match.group(2).replace("+", "")),
                    "unit": unit,
                }
            )
            return leaf
        leaf["status"] = "unknown"
        leaf["reason"] = "not_a_range"
        return leaf

    # Presence flags in a count row ("RTC: Yes", "SDIO: Yes") are kept as
    # booleans; ST's "-" is its explicit none.
    if _EXPLICIT_NONE.match(text):
        leaf.update({"status": "typed", "typ": 0, "unit": unit, "note": "explicit none"})
        return leaf
    if _YES.match(text):
        leaf.update({"status": "boolean", "value": True})
        return leaf
    if _NO.match(text):
        leaf.update({"status": "boolean", "value": False})
        return leaf
    if _MULTI_VALUE.search(text) and not _BITWIDTH_GROUP.search(text):
        leaf["status"] = "unknown"
        leaf["reason"] = "alternative_values"
        return leaf

    number: float | None = None
    if _NUMBER.match(text):
        number = float(text)
    elif unit == "count" and len(_BITWIDTH_GROUP.findall(text)) >= 1 and _bitwidth_only(text):
        groups = _BITWIDTH_GROUP.findall(text)
        number = float(sum(int(g) for g in groups))
        leaf["components"] = [m.group(0) for m in _BITWIDTH_GROUP.finditer(text)]
        leaf["note"] = "sum of bit-width groups"
    else:
        match = _INT_WITH_BREAKDOWN.match(text)
        if match:
            number = float(match.group(1))
            leaf["breakdown"] = text[len(match.group(1)):].strip()
        else:
            match = _INT_WITH_UNIT.match(text)
            if match:
                number = float(match.group(1))
                unit = _UNIT_WORDS.get(" ".join(match.group(2).lower().split()), unit)
            else:
                match = _INT_WITH_NOTE.match(text)
                if match:
                    number = float(match.group(1))
                    leaf["note"] = text[len(match.group(1)):].strip()
    if number is None:
        leaf["status"] = "unknown"
        leaf["reason"] = "unparsed_numeric"
        return leaf

    if attribute.endswith("_kb") and unit == "MB":
        number *= 1024
        unit = "KB"
    if number.is_integer():
        number = int(number)
    leaf.update({"status": "typed", "typ": number, "unit": unit})
    return leaf


def _bitwidth_only(text: str) -> bool:
    """True when the cell is only bit-width groups joined by and/+/,/space,
    optionally followed by a short qualifier ('high frequency')."""
    rest = _BITWIDTH_GROUP.sub("", text)
    rest = re.sub(r"\band\b|\+|,|/", " ", rest, flags=re.I).strip()
    return len(rest.split()) <= 3 and not re.search(r"\d", rest)


def _header_token(cell: str) -> str:
    """'STM32\\nF446MC' -> 'STM32F446MC'; keep printed casing."""
    return re.sub(r"\s+", "", cell)


_CODE = re.compile(r"[A-Z][0-9A-Z](?:xxN|xxP|xxQ)?")


def _split_stem_codes(cell: str, expected: int | None = None) -> tuple[str, list[str]] | None:
    """'STM32G050 _ F6 K6 K8 C6 C8' -> ('STM32G050', ['F6','K6','K8','C6','C8']).

    Also 'STM32C071__F8/_FB_F8xxN/...' (codes separated by / or _) and, when
    ``expected`` is given, a glued run 'G8GBK8KB' that chunks into exactly
    ``expected`` two-character codes.
    """
    text = re.sub(r"[_/]+", " ", _norm(cell)).strip()
    parts = text.split()
    if len(parts) < 2:
        return None
    stem = parts[0]
    if not PART_TOKEN.fullmatch(stem) and not is_wildcard_token(stem):
        return None
    rest = parts[1:]
    codes = [p for p in rest if _CODE.fullmatch(p)]
    if len(codes) >= 2 and len(codes) == len(rest):
        return stem, codes
    if expected and len(rest) >= 1:
        # Glued codes: split each blob into 2-char codes (with optional xxN).
        glued: list[str] = []
        for blob in rest:
            pos = 0
            while pos < len(blob):
                match = _CODE.match(blob, pos)
                if not match or match.end() == pos:
                    return None
                glued.append(match.group(0))
                pos = match.end()
        if len(glued) == expected:
            return stem, glued
    return None


def _wildcard_base(token: str) -> str:
    """'GD32F405xx' -> 'GD32F405'; 'STM32F101Tx' -> 'STM32F101T'; composite
    'STM32C011x4/x6' -> 'STM32C011'."""
    token = token.split("/")[0]
    return re.sub(r"x[A-Za-z0-9x]*$", "", token) if "x" in token else token


def _join_wildcard(token: str, device_summary: dict[str, list[str]]) -> list[str]:
    """Members of a header wildcard like STM32F101Tx from a printed device
    summary {STM32F101x8: [STM32F101C8, ...]} -> parts matching both."""
    pattern = re.compile("^" + re.escape(_wildcard_base(token)) + r"[0-9A-Z]$")
    members: list[str] = []
    for parts in device_summary.values():
        for part in parts:
            if pattern.match(part) and part not in members:
                members.append(part)
    return sorted(members)


def _members_from_parts_named(token: str, parts_named: list[str]) -> list[str]:
    """Concrete parts the document names elsewhere that instantiate a header
    wildcard (STM32F479Vx -> STM32F479VG, STM32F479VI), ordered by the ST
    memory letter when every member has one, else alphabetically. The
    ordering-code self-check later confirms or rejects the binding."""
    pattern = re.compile("^" + re.escape(_wildcard_base(token)) + r"[0-9A-Z]$")
    members = sorted({p for p in parts_named if pattern.match(p)})
    sizes = [st_flash_code_kb(p) for p in members]
    if members and all(s is not None for s in sizes):
        members.sort(key=lambda p: st_flash_code_kb(p) or 0)
    return members


_PART_PACKAGE_SUFFIX = re.compile(r"^([A-Z]{2,}[0-9][A-Z0-9]{3,}?)([A-Z])[xX]{2,4}$")


def bind_columns(
    grid: list[list[str]],
    label_cols: int,
    device_summary: dict[str, list[str]] | None,
    parts_named: list[str] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Bind each data column to a part number using only printed header text.

    Returns (number_of_header_rows, bindings). Each binding has
    ``part_number`` (or None), ``family_token``, ``binding`` in
    {header_token, header_codes, header_wildcard_codes, device_summary_join,
    unbound} and ``column_index``.
    """
    width = len(grid[0])
    data_cols = list(range(label_cols, width))
    row0 = grid[0]
    row1 = grid[1] if len(grid) > 1 else [""] * width

    # Case B: one header cell listing stem + codes.
    for cell in row0[label_cols:]:
        split = _split_stem_codes(cell, expected=len(data_cols))
        if split:
            stem, codes = split
            if len(codes) == len(data_cols):
                base = _wildcard_base(stem)
                out: list[dict[str, Any]] = []
                for col, code in zip(data_cols, codes):
                    if len(code) > 2:
                        # 'F8xxN' is an ordering-suffix variant, not a part
                        # number the document spells out; keep it verbatim.
                        out.append({"column_index": col, "part_number": None, "family_token": base + code, "binding": "unbound", "reason": "ordering_suffix_variant"})
                    else:
                        out.append({"column_index": col, "part_number": base + code, "family_token": stem, "binding": "header_codes"})
                return 1, out
            return 1, [
                {"column_index": col, "part_number": None, "family_token": stem, "binding": "unbound", "reason": "code_count_mismatch"}
                for col in data_cols
            ]

    filled0: list[str] = []
    carry = ""
    for col in range(width):
        cell = _norm(row0[col])
        if cell:
            carry = cell
        filled0.append(carry if col >= label_cols else cell)

    tokens = [_header_token(filled0[col]) for col in data_cols]
    codes1 = [_norm(row1[col]) for col in data_cols]
    codes1_ok = all(re.fullmatch(r"[A-Z][0-9A-Z]{1,3}", c or "") for c in codes1)
    second_header_is_labels = bool(row_attributes(_norm(row1[0]), _norm(row1[1]) if label_cols > 1 else ""))

    bindings: list[dict[str, Any]] = []
    if codes1_ok and not second_header_is_labels and len(set(codes1)) >= 2:
        # Case C: wildcard/stem over a code row (GD32F405xx / RE RG RK ...).
        for col, token, code in zip(data_cols, tokens, codes1):
            base = _wildcard_base(token) if token else ""
            if base:
                bindings.append({"column_index": col, "part_number": base + code, "family_token": token, "binding": "header_wildcard_codes"})
            else:
                bindings.append({"column_index": col, "part_number": None, "family_token": token or None, "binding": "unbound", "reason": "no_header_token"})
        return 2, bindings

    # Case A: concrete token per column; a token spanning several columns is
    # ambiguous unless a printed device summary resolves the members.
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    for col, token in zip(data_cols, tokens):
        if token and is_concrete_part_token(token) and counts[token] == 1:
            bindings.append({"column_index": col, "part_number": token, "family_token": token, "binding": "header_token"})
            continue
        packaged = _PART_PACKAGE_SUFFIX.match(token or "")
        if packaged and counts[token] == 1 and is_concrete_part_token(packaged.group(1)):
            # "STM32U3C5VIYxxxx": the concrete part followed by its package
            # letter and the ordering wildcards. One part, one package column.
            bindings.append(
                {"column_index": col, "part_number": packaged.group(1), "family_token": token, "binding": "header_token_package", "package_letter": packaged.group(2)}
            )
            continue
        if token and counts[token] == 1 and _HEADER_LIST_SPLIT.search(token):
            # Several parts named for one column ("STM32L552CE,STM32L552CC/
            # STM32L552CExxP"): the column binds to every concrete part the
            # header prints; ordering-suffix wildcards are kept verbatim.
            pieces = [p for p in _HEADER_LIST_SPLIT.split(token) if p]
            concrete = [p for p in pieces if is_concrete_part_token(p) and not re.search(r"xx", p)]
            if concrete:
                bindings.append(
                    {"column_index": col, "part_number": concrete[0], "part_numbers": concrete, "also_covers": [p for p in pieces if p not in concrete], "family_token": token, "binding": "header_token_list"}
                )
                continue
        if token and is_wildcard_token(token) and device_summary:
            members = _join_wildcard(token, device_summary)
            if len(members) == counts[token]:
                offset = [c for c, t in zip(data_cols, tokens) if t == token].index(col)
                bindings.append(
                    {"column_index": col, "part_number": members[offset], "family_token": token, "binding": "device_summary_join", "join_members": members}
                )
                continue
        if token and is_wildcard_token(token) and parts_named:
            members = _members_from_parts_named(token, parts_named)
            span = counts[token]
            if len(members) == span and span > 1:
                offset = [c for c, t in zip(data_cols, tokens) if t == token].index(col)
                bindings.append(
                    {"column_index": col, "part_number": members[offset], "family_token": token, "binding": "parts_named_join", "join_members": members}
                )
                continue
            if span == 1 and len(members) > 1:
                bindings.append(
                    {"column_index": col, "part_number": members[0], "part_numbers": members, "family_token": token, "binding": "parts_named_join", "members": [{"part_number": m, "token": m} for m in members]}
                )
                continue
        bindings.append(
            {"column_index": col, "part_number": None, "family_token": token or None, "binding": "unbound", "reason": "spanning_or_wildcard_header" if token else "empty_header"}
        )
    return 1, bindings


def read_device_summary(document: Any, pages: Iterable[int]) -> dict[str, list[str]]:
    """ST 'Table 1. Device summary': Reference -> Part number list."""
    summary: dict[str, list[str]] = {}
    for page_number in pages:
        try:
            tables = document[page_number - 1].find_tables().tables
        except Exception:  # noqa: BLE001
            continue
        for table in tables:
            try:
                rows = table.extract()
            except Exception:  # noqa: BLE001
                continue
            if not rows or len(rows[0]) < 2:
                continue
            header = " ".join(_norm(c) for c in rows[0]).lower()
            if "reference" not in header or "part" not in header:
                continue
            for row in rows[1:]:
                reference = _norm(row[0])
                if not reference:
                    continue
                parts = [
                    t for t in PART_TOKEN.findall(" ".join(_norm(c) for c in row[1:]))
                    if is_concrete_part_token(t)
                ]
                if parts:
                    bucket = summary.setdefault(reference, [])
                    for part in parts:
                        if part not in bucket:
                            bucket.append(part)
    return summary


def st_flash_code_kb(part_number: str | None) -> int | None:
    if not part_number or part_number.startswith(("STM32WB", "STM32MP", "STM32N")):
        # Wireless/MPU lines use a different memory-letter table; not checked.
        return None
    match = _ST_PART.match(part_number)
    if not match:
        return None
    return ST_FLASH_CODE_KB.get(match.group(2))


_GEO_CODE = re.compile(r"^_?([A-Z][0-9A-Z])/?$")
_GEO_SUFFIX = re.compile(r"^(?:xx[A-Z]|_?[A-Z][0-9A-Z]xx[A-Z])/?$")


def _column_ranges(table: Any, width: int) -> list[tuple[float, float]] | None:
    """x-ranges of the extracted grid's columns from the table's cell boxes."""
    xs: set[float] = set()
    right = 0.0
    for cell in getattr(table, "cells", None) or []:
        if cell:
            xs.add(round(cell[0], 1))
            right = max(right, cell[2])
    bounds = sorted(xs)
    if len(bounds) != width:
        return None
    return [(bounds[i], bounds[i + 1] if i + 1 < width else right) for i in range(width)]


def geometry_bindings(page: Any, table: Any, grid: list[list[str]], label_cols: int) -> list[dict[str, Any]] | None:
    """Bind columns from the header's word positions when the header text is
    a rotated code grid ("STM32G071_" over "_G8 _GB _K8 ..." with "xxN" pieces
    stacked under some codes, or "F8 / FB" stacked in one column).

    Only printed pieces are used: a code lands in the column whose x-range
    contains the word's centre; a suffix piece attaches to the code directly
    above it in the same column. Nothing is minted when a column shows no code.
    """
    width = len(grid[0])
    ranges = _column_ranges(table, width)
    if ranges is None:
        return None
    header_cells = [c for c in table.rows[0].cells if c]
    if not header_cells:
        return None
    x0 = min(c[0] for c in header_cells)
    y0 = min(c[1] for c in header_cells)
    x1 = max(c[2] for c in header_cells)
    y1 = max(c[3] for c in header_cells)
    words = [w for w in page.get_text("words") if w[0] >= x0 - 1 and w[2] <= x1 + 1 and w[1] >= y0 - 1 and w[3] <= y1 + 1]
    stem = None
    for w in words:
        token = w[4].strip("_")
        if re.search(r"\d", token) and (PART_TOKEN.fullmatch(token) or is_wildcard_token(token)):
            stem = token
            break
    if stem is None:
        return None
    base = _wildcard_base(stem)

    def column_of(word: Any) -> int | None:
        centre = (word[0] + word[2]) / 2
        for index, (left, right) in enumerate(ranges):
            if left <= centre < right:
                return index
        return None

    per_column: dict[int, list[tuple[float, str, str, bool]]] = {}
    for w in words:
        text = w[4]
        col = column_of(w)
        if col is None or col < label_cols:
            continue
        rotated = (w[3] - w[1]) > (w[2] - w[0])
        if _GEO_CODE.match(text):
            per_column.setdefault(col, []).append((w[1], "code", _GEO_CODE.match(text).group(1), rotated))
        elif _GEO_SUFFIX.match(text):
            per_column.setdefault(col, []).append((w[1], "suffix", text.strip("_/"), rotated))
    if not per_column:
        return None

    bindings: list[dict[str, Any]] = []
    for col in range(label_cols, width):
        raw = per_column.get(col, [])
        # Horizontal words stack top-to-bottom; rotated (vertical) words read
        # bottom-to-top. Either way the printed order is the order of the
        # slash-separated values in the column's cells.
        rotated_column = sum(1 for p in raw if p[3]) > len(raw) / 2
        pieces = [(y, kind, text) for y, kind, text, _ in sorted(raw, key=lambda p: p[0], reverse=rotated_column)]
        members: list[dict[str, Any]] = []
        for y, kind, text in pieces:
            if kind == "code":
                members.append({"part_number": base + text, "token": base + text})
            elif kind == "suffix":
                if len(text) > 3:  # full 'F8xxN' word
                    members.append({"part_number": None, "token": base + text})
                elif members and members[-1]["part_number"]:
                    # 'xxN' stacked under a code: the column is that ordering variant.
                    members[-1] = {"part_number": None, "token": members[-1]["token"] + text}
        if not members:
            bindings.append({"column_index": col, "part_number": None, "family_token": stem, "binding": "unbound", "reason": "no_code_in_column"})
            continue
        concrete = [m["part_number"] for m in members if m["part_number"]]
        binding = {
            "column_index": col,
            "part_number": concrete[0] if concrete else None,
            "family_token": stem,
            "binding": "header_geometry" if concrete else "unbound",
            "members": members,
        }
        if len(members) > 1:
            binding["part_numbers"] = concrete
        if not concrete:
            binding["reason"] = "ordering_suffix_variant"
            binding["family_token"] = members[0]["token"]
        bindings.append(binding)
    if not any(b["part_number"] for b in bindings):
        return None
    return bindings


def _read_parts_as_rows(rows: list[list[Any]], *, page: int) -> dict[str, Any] | None:
    """Selection-guide / product-list layout: one part per row, attributes
    across the header. Transposed into the same shape as parts_as_columns so
    the record builder does not care. A blank cell under a value in the same
    column is a merged cell (fill-down) and is flagged as such."""
    width = max(len(r) for r in rows)
    grid = [[_norm(c) for c in r] + [""] * (width - len(r)) for r in rows]
    header = grid[0]
    columns: list[tuple[int, list[tuple[str, int | None]], str | None, str]] = []
    for col in range(1, width):
        label = header[col]
        attributes = row_attributes(label, "")
        if attributes:
            columns.append((col, attributes, unit_from_label(label), label))
    if len(columns) < 2:
        return None
    bindings: list[dict[str, Any]] = []
    carried = ""
    body: list[list[str]] = []
    for row_index, row in enumerate(grid[1:], start=1):
        token = _header_token(row[0]) if row[0] else ""
        if token:
            carried = token
        part = carried
        if not part:
            continue
        if is_concrete_part_token(part):
            bindings.append({"column_index": row_index, "part_number": part, "family_token": part, "binding": "row_token"})
        elif is_wildcard_token(part):
            bindings.append({"column_index": row_index, "part_number": None, "family_token": part, "binding": "unbound", "reason": "wildcard_row"})
        else:
            continue
        body.append(row)
    if len(bindings) < 2:
        return None
    attribute_rows: list[dict[str, Any]] = []
    for col, attributes, unit, label in columns:
        values: list[tuple[str, bool]] = []
        carry = ""
        for row in body:
            cell = row[col]
            merged = False
            if cell:
                carry = cell
            else:
                merged = bool(carry)
            values.append((cell if cell else carry, merged))
        if any(v for v, _ in values):
            attribute_rows.append({"row_index": col, "label": label, "attributes": attributes, "label_unit": unit, "values": values})
    if len(attribute_rows) < 2:
        return None
    return {
        "page": page,
        "rows": len(rows),
        "columns": width,
        "label_columns": 1,
        "header_rows": 1,
        "orientation": "parts_as_rows",
        "bindings": bindings,
        "attribute_rows": attribute_rows,
        "census_attributes": attributes_in_label(" ".join(r["label"] for r in attribute_rows)),
    }


def read_matrix_table(
    rows: list[list[Any]],
    *,
    page: int,
    device_summary: dict[str, list[str]] | None,
    table: Any = None,
    pdf_page: Any = None,
    parts_named: list[str] | None = None,
) -> dict[str, Any] | None:
    rows = strip_title_rows(rows)
    description = classify_table(rows)
    if description is None:
        return None
    if description["orientation"] == "parts_as_rows":
        return _read_parts_as_rows(rows, page=page)
    width = max(len(r) for r in rows)
    grid = [[_norm(c) for c in r] + [""] * (width - len(r)) for r in rows]
    label_cols = min(2, width - 1)
    header_rows, bindings = bind_columns(grid, label_cols, device_summary, parts_named)
    if table is not None and pdf_page is not None and not any(b["part_number"] for b in bindings):
        geometry = geometry_bindings(pdf_page, table, grid, label_cols)
        if geometry:
            bindings = geometry
            header_rows = 1
    if not any(
        b["part_number"] or (b.get("family_token") and re.search(r"\d", b["family_token"]) and (PART_TOKEN.search(b["family_token"]) or is_wildcard_token(b["family_token"])))
        for b in bindings
    ):
        # No column header names a part or a series: not a device table
        # (low-power mode matrices and the like slip through the census).
        return None
    data_cols = [b["column_index"] for b in bindings]

    attribute_rows: list[dict[str, Any]] = []
    carried = ""
    for row_index in range(header_rows, len(grid)):
        row = grid[row_index]
        if row[0]:
            carried = row[0]
        sub = row[1] if label_cols > 1 else ""
        # Second label column is sometimes the start of data (2-col label
        # tables where col1 holds a value): treat as data when it has no label.
        attributes = row_attributes(carried, sub)
        if not attributes:
            continue
        label = " ".join(p for p in (carried, sub) if p)
        unit = unit_from_label(label)
        values: list[tuple[str, bool]] = []
        carry_value = ""
        for col in data_cols:
            cell = row[col]
            merged = False
            if cell:
                carry_value = cell
            else:
                merged = bool(carry_value)
            values.append((cell if cell else carry_value, merged))
        if not any(v for v, _ in values):
            continue
        values = _split_glued_runs(values)
        attribute_rows.append(
            {"row_index": row_index, "label": label, "attributes": attributes, "label_unit": unit, "values": values}
        )
    if len(attribute_rows) < 2:
        return None
    return {
        "page": page,
        "rows": len(rows),
        "columns": width,
        "label_columns": label_cols,
        "header_rows": header_rows,
        "bindings": bindings,
        "attribute_rows": attribute_rows,
        "census_attributes": attributes_in_label(" ".join(r["label"] for r in attribute_rows)),
    }


def _split_glued_runs(values: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
    """A cell like '2 2' or '1/1/1/1' that PyMuPDF glued across N columns is
    N per-column values when the token count equals the run length. Anything
    else is left as printed (and will surface as unknown)."""
    out = list(values)
    index = 0
    while index < len(out):
        text, merged = out[index]
        if merged:
            index += 1
            continue
        run_end = index + 1
        while run_end < len(out) and out[run_end][1]:
            run_end += 1
        run = run_end - index
        if run > 1 and _INT_TOKENS.match(_strip_footnotes(text)):
            tokens = re.split(r"\s*[/ ]\s*", _strip_footnotes(text))
            if len(tokens) == run:
                for offset, token in enumerate(tokens):
                    out[index + offset] = (token, False)
        index = run_end
    return out


def _leaf_receipt(page: int, label: str, column: int, merged: bool) -> dict[str, Any]:
    receipt = {"page": page, "row_label": label, "column_index": column}
    if merged:
        receipt["merged_cell"] = True
    return receipt


def _cell_leaves(verbatim: str, attributes: list[tuple[str, int | None]], label_unit: str | None) -> list[tuple[str, dict[str, Any]]]:
    """Leaves for one cell: plain rows give one leaf; composite rows split the
    value into components and give one leaf per component."""
    composite = [a for a in attributes if a[1] is not None]
    if not composite:
        return [(attribute, parse_value(verbatim, attribute, label_unit)) for attribute, _ in attributes]
    text = _strip_footnotes(_norm(verbatim))
    extra = _EXTRA_CLAUSE.search(text)
    if extra:
        text = text[: extra.start()].strip()
    components = split_composite_value(text, max(i for _, i in composite) + 1)
    out: list[tuple[str, dict[str, Any]]] = []
    repeated = {a for a in {a for a, _ in composite} if sum(1 for b, _ in composite if b == a) > 1}
    for attribute in sorted(repeated):
        # "FDCAN/TT-FDCAN: 1/1" names two kinds of one interface class;
        # the class count is the sum of the printed components.
        indices = [i for a, i in composite if a == attribute]
        if components is None:
            out.append((attribute, {"verbatim": _norm(verbatim), "status": "unknown", "reason": "composite_value_mismatch"}))
            continue
        leaves = [parse_value(components[i], attribute, label_unit) for i in indices]
        if attribute in NUMERIC_ATTRIBUTES and all(l["status"] == "typed" for l in leaves):
            leaf = {"verbatim": _norm(verbatim), "status": "typed", "typ": sum(l["typ"] for l in leaves), "unit": leaves[0].get("unit"), "note": "sum of components", "components": [components[i] for i in indices]}
        else:
            leaf = {"verbatim": _norm(verbatim), "status": "unknown", "reason": "composite_components_untyped"}
        out.append((attribute, leaf))
    composite = [(a, i) for a, i in composite if a not in repeated]
    for attribute, index in composite:
        if components is None:
            out.append((attribute, {"verbatim": _norm(verbatim), "status": "unknown", "reason": "composite_value_mismatch"}))
            continue
        leaf = parse_value(components[index], attribute, label_unit)
        leaf["verbatim"] = _norm(verbatim)
        leaf["component"] = components[index]
        if extra:
            leaf["note"] = extra.group(0).strip(" +")
        out.append((attribute, leaf))
    return out


def build_family_record(
    *,
    source_path: Path,
    document_sha256: str,
    census_row: dict[str, Any],
    tables: list[dict[str, Any]],
    device_summary: dict[str, list[str]],
) -> dict[str, Any]:
    series_tokens = list(census_row.get("series_tokens") or [])
    parts_named = [p["token"] for p in census_row.get("parts_named") or []]
    family = None
    if series_tokens:
        family = _wildcard_base(series_tokens[0]).rstrip("x")
    elif census_row.get("family_stems"):
        family = census_row["family_stems"][0]

    variants: dict[str, dict[str, Any]] = {}
    unbound_columns: list[dict[str, Any]] = []
    shared: dict[str, dict[str, Any]] = {}
    unknown: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    counts = {"cells_total": 0, "cells_typed": 0, "cells_verbatim": 0, "cells_unknown": 0}

    def _count(leaf: dict[str, Any]) -> None:
        counts["cells_total"] += 1
        status = leaf["status"]
        if status in ("typed", "boolean"):
            counts["cells_typed"] += 1
        elif status == "verbatim":
            counts["cells_verbatim"] += 1
        else:
            counts["cells_unknown"] += 1

    pending: list[dict[str, Any]] = []
    for table_index, table in enumerate(tables):
        page = table["page"]
        bindings = table["bindings"]
        for binding in bindings:
            if binding["part_number"] is None:
                unbound_columns.append(dict(binding, page=page))
        for arow in table["attribute_rows"]:
            label = arow["label"]
            by_attribute: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
            for binding, (verbatim, merged) in zip(bindings, arow["values"]):
                if not verbatim:
                    continue
                members = binding.get("members")
                if members and len(members) > 1 and not any(i is not None for _, i in arow["attributes"]):
                    # Column names several parts ("F8 / FB") and the cell may
                    # print one value per member ("64/128"); otherwise the one
                    # value applies to all members.
                    pieces = [p.strip() for p in re.split(r"\s*/\s*", _strip_footnotes(_norm(verbatim))) if p.strip()]
                    per_member = pieces if len(pieces) == len(members) else [verbatim] * len(members)
                    for member, piece in zip(members, per_member):
                        sub_binding = {"column_index": binding["column_index"], "part_number": member["part_number"], "family_token": member["token"], "binding": binding["binding"] if member["part_number"] else "unbound"}
                        for attribute, leaf in _cell_leaves(piece, arow["attributes"], arow["label_unit"]):
                            leaf["verbatim"] = _norm(verbatim)
                            if len(pieces) == len(members):
                                leaf["component"] = piece
                            leaf["receipt"] = _leaf_receipt(page, label, binding["column_index"], merged)
                            by_attribute.setdefault(attribute, []).append((sub_binding, leaf))
                    continue
                for attribute, leaf in _cell_leaves(verbatim, arow["attributes"], arow["label_unit"]):
                    leaf["receipt"] = _leaf_receipt(page, label, binding["column_index"], merged)
                    by_attribute.setdefault(attribute, []).append((binding, leaf))
            for attribute, cols in by_attribute.items():
                for binding, leaf in cols:
                    _count(leaf)
                    if leaf["status"] == "unknown":
                        unknown.append(
                            {"attribute": attribute, "part_number": binding["part_number"], "column_index": binding["column_index"], "page": page, "row_label": label, "verbatim": leaf["verbatim"], "reason": leaf.get("reason")}
                        )
                distinct = {leaf.get("component", leaf["verbatim"]) for _, leaf in cols}
                covered = {b["column_index"] for b, _ in cols}
                whole_table = len(distinct) == 1 and covered == {b["column_index"] for b in bindings}
                pending.append({"attribute": attribute, "table": table_index, "page": page, "label": label, "cols": cols, "whole_table": whole_table, "value": next(iter(distinct)) if whole_table else None})

    # Shared only when every table in the document prints the same single
    # value across all of its columns; a value common to one table but
    # different in another is per-variant, not family-wide.
    by_attr: dict[str, list[dict[str, Any]]] = {}
    for entry in pending:
        by_attr.setdefault(entry["attribute"], []).append(entry)

    def _table_parts(index: int) -> set[str]:
        out: set[str] = set()
        for b in tables[index]["bindings"]:
            for m in b.get("members") or [{"part_number": b["part_number"], "token": b.get("family_token")}]:
                out.add(m["part_number"] or f"column:{tables[index]['page']}:{b['column_index']}")
            for p in b.get("part_numbers") or []:
                out.add(p)
        return out

    all_parts = set().union(*(_table_parts(i) for i in range(len(tables)))) if tables else set()
    for attribute, entries in by_attr.items():
        covered_parts = set().union(*(_table_parts(e["table"]) for e in entries))
        if all(e["whole_table"] for e in entries) and len({e["value"] for e in entries}) == 1 and covered_parts == all_parts:
            leaf = dict(entries[0]["cols"][0][1])
            leaf["receipt"] = dict(leaf["receipt"], applies_to="all_variants")
            shared[attribute] = leaf
            continue
        for entry in entries:
            page = entry["page"]
            label = entry["label"]
            for binding, leaf in entry["cols"]:
                for part in binding.get("part_numbers") or [binding["part_number"]]:
                    key = part or f"column:{page}:{binding['column_index']}"
                    if part and binding.get("package_letter"):
                        key = f"{part}/{binding['package_letter']}"
                    variant = variants.setdefault(
                        key,
                        {"part_number": part, "family_token": binding["family_token"], "binding": binding["binding"], "column_index": binding["column_index"], "page": page, "attributes": {}},
                    )
                    if binding.get("package_letter"):
                        variant["package_letter"] = binding["package_letter"]
                    if binding.get("also_covers"):
                        variant["also_covers"] = binding["also_covers"]
                    existing = variant["attributes"].get(attribute)
                    if existing is not None:
                        if attribute in BOOLEAN_ATTRIBUTES and existing["status"] == "boolean" and leaf["status"] == "boolean":
                            # Several rows for one flag (USB OTG FS, USB OTG
                            # HS): present if any row says so; keep both
                            # verbatims and labels.
                            if leaf["value"] and not existing["value"]:
                                existing["value"] = True
                            existing["verbatim"] = f"{existing['verbatim']} | {leaf['verbatim']}"
                            existing.setdefault("rows", [existing["receipt"]["row_label"]]).append(label)
                            counts["cells_total"] -= 1
                            counts["cells_typed"] -= 1
                            continue
                        if leaf["status"] == "unknown":
                            # A second, unparseable row (junction temperature
                            # under "Operating temperatures") does not unseat a
                            # parsed one; it is already in the unknown list.
                            continue
                        if existing["status"] == "unknown" and existing.get("reason") != "repeated_row_disagrees":
                            variant["attributes"][attribute] = dict(leaf) if len(binding.get("part_numbers") or []) > 1 else leaf
                            continue
                        if attribute == "package" and existing["verbatim"] != leaf["verbatim"]:
                            # Packages printed on several rows: union.
                            existing["verbatim"] = f"{existing['verbatim']} | {leaf['verbatim']}"
                            seen = {(p["package_family"], p["pin_count"]) for p in existing.get("packages", [])}
                            for p in leaf.get("packages", []):
                                if (p["package_family"], p["pin_count"]) not in seen:
                                    existing.setdefault("packages", []).append(p)
                            counts["cells_total"] -= 1
                            counts["cells_verbatim"] -= 1
                            continue
                        if (
                            attribute.startswith("timer_")
                            and existing["status"] == "typed" and leaf["status"] == "typed"
                            and existing["receipt"]["row_label"] != label
                            and re.search(r"\bbits?\b", existing["receipt"]["row_label"] + " " + label, re.I)
                        ):
                            # "General purpose (32-bit)" + "General purpose (16-bit)":
                            # two rows of one timer class, printed by width. Sum.
                            existing["typ"] = existing["typ"] + leaf["typ"]
                            existing["verbatim"] = f"{existing['verbatim']} + {leaf['verbatim']}"
                            existing["note"] = "sum of bit-width rows"
                            existing.setdefault("rows", [existing["receipt"]["row_label"]]).append(label)
                            counts["cells_total"] -= 1
                            counts["cells_typed"] -= 1
                            continue
                        if existing["verbatim"] != leaf["verbatim"]:
                            # Two rows of the same table landed in one
                            # attribute with different values: the label
                            # grammar could not separate them, so neither
                            # value is trusted.
                            conflicts.append(
                                {"part_number": part, "attribute": attribute, "values": [existing["verbatim"], leaf["verbatim"]], "labels": [existing["receipt"]["row_label"], label], "pages": [existing["receipt"]["page"], page], "kind": "repeated_row_disagrees"}
                            )
                            if existing["status"] in ("typed", "boolean"):
                                counts["cells_typed"] -= 1
                                counts["cells_unknown"] += 1
                            elif existing["status"] == "verbatim":
                                counts["cells_verbatim"] -= 1
                                counts["cells_unknown"] += 1
                            for k in ("typ", "min", "max", "value"):
                                existing.pop(k, None)
                            existing["status"] = "unknown"
                            existing["reason"] = "repeated_row_disagrees"
                        continue
                    variant["attributes"][attribute] = dict(leaf) if len(binding.get("part_numbers") or []) > 1 else leaf

    # Derived: pin_count from the package cell when every package printed for
    # the variant has one pin count. Kept separate from gpio_count.
    def _derive_pins(attrs: dict[str, dict[str, Any]], receipt_extra: dict[str, Any]) -> None:
        package = attrs.get("package")
        if not package or "pin_count" in attrs:
            return
        pins = sorted({p["pin_count"] for p in package.get("packages", [])})
        if len(pins) == 1:
            attrs["pin_count"] = {"verbatim": package["verbatim"], "status": "typed", "typ": pins[0], "unit": "pins", "derived_from": "package", "receipt": dict(package["receipt"], **receipt_extra)}
        elif len(pins) > 1:
            attrs["pin_count"] = {"verbatim": package["verbatim"], "status": "unknown", "reason": "multiple_pin_counts", "receipt": dict(package["receipt"], **receipt_extra)}

    _derive_pins(shared, {"applies_to": "all_variants"})
    for variant in variants.values():
        _derive_pins(variant["attributes"], {})

    # Plausibility: channel/GPIO counts cannot exceed the package pin count.
    def _plausible(attrs: dict[str, dict[str, Any]], part: str | None) -> None:
        pins = (attrs.get("pin_count") or shared.get("pin_count") or {})
        if pins.get("status") != "typed":
            return
        for attribute in ("gpio_count", "adc_channels"):
            leaf = attrs.get(attribute)
            if leaf and leaf.get("status") == "typed" and leaf["typ"] > pins["typ"]:
                leaf["status"] = "unknown"
                leaf["reason"] = f"exceeds_pin_count_{pins['typ']}"
                leaf.pop("typ", None)
                unknown.append({"attribute": attribute, "part_number": part, "column_index": leaf["receipt"]["column_index"], "page": leaf["receipt"]["page"], "row_label": leaf["receipt"]["row_label"], "verbatim": leaf["verbatim"], "reason": leaf["reason"]})
                counts["cells_typed"] -= 1
                counts["cells_unknown"] += 1

    for variant in variants.values():
        _plausible(variant["attributes"], variant["part_number"])

    # Self-consistency: ST memory letter vs printed code flash.
    consistency: list[dict[str, Any]] = []
    for variant in variants.values():
        part = variant["part_number"]
        expected = st_flash_code_kb(part)
        leaf = variant["attributes"].get("code_flash_kb") or shared.get("code_flash_kb")
        if expected is None or leaf is None or leaf.get("status") != "typed":
            continue
        ok = leaf.get("typ") == expected
        consistency.append({"part_number": part, "check": "st_flash_code_vs_printed_code_flash", "expected_kb": expected, "printed_kb": leaf.get("typ"), "ok": ok, "binding": variant["binding"]})
        if not ok and variant["binding"] == "parts_named_join":
            # The join ordered members by memory letter; the printed flash
            # disagrees, so the binding was wrong. Revoke it, keep the column.
            unbound_columns.append({"column_index": variant["column_index"], "part_number": None, "family_token": variant["family_token"], "binding": "unbound", "reason": "parts_named_join_failed_flash_check", "page": variant["page"], "attempted": part})
            variant["part_number"] = None
            variant["binding"] = "unbound"
            variant["reason"] = "parts_named_join_failed_flash_check"
            continue
        if ok and variant["binding"] == "parts_named_join":
            variant["join_verified_by"] = "st_flash_code"
        if not ok:
            conflicts.append(
                {"part_number": part, "attribute": "code_flash_kb", "values": [leaf.get("verbatim"), f"ordering code implies {expected} KB"], "pages": [leaf["receipt"]["page"]], "kind": "ordering_code_mismatch"}
            )

    bound_parts = sorted({v["part_number"] for v in variants.values() if v["part_number"]})
    covered = sorted(set(parts_named) | set(bound_parts) | {p for ps in device_summary.values() for p in ps})
    attributes_addressed = sorted(set(shared) | {a for v in variants.values() for a in v["attributes"]})
    return {
        "schema": MATRIX_SCHEMA,
        "_meta": {
            "extracted_by": EXTRACTED_BY,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "document_sha256": document_sha256,
            "source_path": str(source_path),
            "datasheet_url": None,
            "revision": None,
            "device_table_pages": sorted({t["page"] for t in tables}),
            "models_used": [],
        },
        "overview": {
            "part_number": None,
            "family": family,
            "series_tokens": series_tokens,
            "part_numbers_covered": covered,
            "device_summary": device_summary,
            "vendor": census_row.get("vendor"),
        },
        "shared": shared,
        "variants": sorted(variants.values(), key=lambda v: (v["part_number"] is None, v["part_number"] or "", v["column_index"])),
        "unbound_columns": unbound_columns,
        "unknown": unknown,
        "conflicts": conflicts,
        "consistency": consistency,
        "bogey": {
            "parts_named": len(parts_named),
            "variant_columns": sum(len(t["bindings"]) for t in tables),
            "variants_bound": len(bound_parts),
            "attributes_addressed": attributes_addressed,
            **counts,
        },
    }


def extract_family_matrix(path: Path, census_row: dict[str, Any]) -> dict[str, Any]:
    import pymupdf

    document = pymupdf.open(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    census_pages = sorted({t["page"] for t in census_row.get("device_tables") or []})
    front = list(range(1, min(document.page_count, 3) + 1))
    device_summary = read_device_summary(document, front)

    tables: list[dict[str, Any]] = []
    for page_number in census_pages:
        try:
            found = document[page_number - 1].find_tables().tables
        except Exception:  # noqa: BLE001
            continue
        for table in found:
            try:
                rows = table.extract()
            except Exception:  # noqa: BLE001
                continue
            matrix = read_matrix_table(
                rows,
                page=page_number,
                device_summary=device_summary,
                table=table,
                pdf_page=document[page_number - 1],
                parts_named=[p["token"] for p in census_row.get("parts_named") or []],
            )
            if matrix and len(matrix["attribute_rows"]) >= 3:
                tables.append(matrix)
    record = build_family_record(
        source_path=path, document_sha256=sha, census_row=census_row, tables=tables, device_summary=device_summary
    )
    if sha != census_row.get("document_sha256"):
        record["_meta"]["census_sha256_mismatch"] = census_row.get("document_sha256")
    return record
