"""Ten-group Key Features grid rows from a family-document record.

CR's contract (`grid-target-ten-groups-20260908`): ten fixed groups at the
grain the document states, each a flat list of short strings the document
prints, every row with page receipts. Counts as an integer field next to the
label, qualifiers verbatim, multi-valued cells flagged `varies_by_part` and
never expanded, a fact may sit in two groups, register as printed.

Rows carry `tier`: "grid" for rows short enough to render on the page,
"below_grid" for the full-read detail (chapter feature sentences, chapter
locations) that feeds knives and the selector but does not render.

Absence is never emitted as a fact: `class_presence` lists asserted_present
classes only; `not_observed` is a separate list and means exactly that.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

SCHEMA = "harness.electronics-key-features-grid.v1"

GROUPS: tuple[tuple[str, str], ...] = (
    ("processing", "Processing"),
    ("memory", "Memory"),
    ("peripherals", "Peripherals"),
    ("connectivity", "Connectivity"),
    ("analogue", "Analogue"),
    ("timers_pwm_control", "Timers, PWM & Control"),
    ("power_clock_reset", "Power, Clock & Reset"),
    ("security_safety_identity", "Security, Safety & Identity"),
    ("io_package_environment", "I/O, Package & Environment"),
    ("graphics_vision_touch_hmi", "Graphics, Vision, Touch & HMI"),
)

# CR's mapping, versioned by CR. Chapter/feature class -> group(s). A class
# may sit in two groups (watchdog: timers + security; trustzone: processing +
# security; safety/crc: peripherals + security) per the RA4C1 page.
CLASS_GROUPS: dict[str, tuple[str, ...]] = {
    "cpu_core": ("processing",),
    "npu": ("processing",),  # flagged: CR undecided
    "flash": ("memory",),
    "sram": ("memory",),
    "eeprom": ("memory",),
    "memory_map": ("memory",),
    "dma": ("peripherals",),
    "interrupts": ("peripherals",),
    "debug": ("peripherals",),
    "safety": ("peripherals", "security_safety_identity"),
    "usart": ("connectivity",),
    "spi": ("connectivity",),
    "i2c": ("connectivity",),
    "i3c": ("connectivity",),
    "can": ("connectivity",),
    "lin": ("connectivity",),
    "usb": ("connectivity",),
    "ethernet": ("connectivity",),
    "sdmmc": ("connectivity",),
    "xspi": ("connectivity",),
    "fmc": ("connectivity",),
    "wireless": ("connectivity",),
    "adc": ("analogue",),
    "temp_sensor": ("analogue",),
    "dac": ("analogue",),
    "comparator": ("analogue",),
    "opamp": ("analogue",),
    "timer": ("timers_pwm_control",),
    "lptim": ("timers_pwm_control",),
    "hrtim": ("timers_pwm_control",),
    "rtc": ("timers_pwm_control",),
    "watchdog": ("timers_pwm_control", "security_safety_identity"),
    "power": ("power_clock_reset",),
    "reset_clock": ("power_clock_reset",),
    "crypto": ("security_safety_identity",),
    "trustzone": ("processing", "security_safety_identity"),
    "gpio": ("io_package_environment",),
    "display": ("graphics_vision_touch_hmi",),
    "dcmi": ("graphics_vision_touch_hmi",),
    "touch": ("graphics_vision_touch_hmi",),
    "audio": ("connectivity",),  # CR decision 2026-09-08: SAI/I2S are buses
}

# Instance class (from INSTANCE_GRAMMAR) -> (peripheral_class, display label).
INSTANCE_CLASS: dict[str, tuple[str, str]] = {
    "TIM": ("timer", "Timers (TIM)"),
    "LPTIM": ("lptim", "Low-power timers (LPTIM)"),
    "HRTIM": ("hrtim", "High-resolution timer (HRTIM)"),
    "TIMG": ("timer", "General-purpose timers (TIMG)"),
    "TIMA": ("timer", "Advanced timers (TIMA)"),
    "EPWM": ("timer", "ePWM modules"),
    "ECAP": ("timer", "eCAP modules"),
    "EQEP": ("timer", "eQEP modules"),
    "USART": ("usart", "USART"),
    "UART": ("usart", "UART"),
    "LPUART": ("usart", "Low-power UART (LPUART)"),
    "SCI": ("usart", "Serial Communications Interface (SCI)"),
    "SPI": ("spi", "SPI"),
    "I2S": ("spi", "I2S"),
    "SAI": ("audio", "Serial audio interface (SAI)"),
    "I2C": ("i2c", "I2C"),
    "I3C": ("i3c", "I3C"),
    "FDCAN": ("can", "FDCAN"),
    "CAN": ("can", "CAN"),
    "MCAN": ("can", "CAN-FD (MCAN)"),
    "DCAN": ("can", "DCAN"),
    "ADC": ("adc", "ADC"),
    "DAC": ("dac", "DAC"),
    "COMP": ("comparator", "Analog comparator (COMP)"),
    "OPAMP": ("opamp", "Operational amplifier (OPAMP)"),
    "DMA": ("dma", "DMA controllers"),
    "GPIO": ("gpio", "GPIO ports"),
    "USB": ("usb", "USB"),
    "ETH": ("ethernet", "Ethernet"),
    "SDMMC": ("sdmmc", "SDMMC"),
    "OCTOSPI": ("xspi", "Octo/Quad-SPI"),
    "FMC": ("fmc", "Flexible memory controller (FMC/FSMC)"),
    "RTC": ("rtc", "RTC"),
    "IWDG": ("watchdog", "Watchdogs"),
    "AES": ("crypto", "Cryptographic accelerators"),
    "LTDC": ("display", "Display / camera controllers"),
    "UCPD": ("usb", "USB Type-C / Power Delivery (UCPD)"),
    "TSC": ("touch", "Touch sensing controller (TSC)"),
    "RSPI": ("spi", "Serial Peripheral Interface (RSPI)"),
    "RIIC": ("i2c", "I2C bus interface (RIIC)"),
    "MTU": ("timer", "Multi-function timer pulse unit (MTU)"),
    "TPU": ("timer", "16-bit timer pulse unit (TPU)"),
    "CMT": ("timer", "Compare match timer (CMT)"),
    "TMR": ("timer", "8-bit timer (TMR)"),
    "GPT32": ("timer", "General PWM timer (GPT)"),
    "AGT": ("timer", "Asynchronous general-purpose timer (AGT)"),
    "S12AD": ("adc", "12-bit A/D converter (S12AD)"),
    "RSCAN": ("can", "CAN (RSCAN)"),
    "DMAC": ("dma", "DMA controller (DMAC)"),
    "DTC": ("dma", "Data transfer controller (DTC)"),
    "ELC": ("interrupts", "Event link controller (ELC)"),
    "MIBSPI": ("spi", "Multi-buffered SPI (MibSPI)"),
    "N2HET": ("timer", "High-end timer (N2HET)"),
    "RTI": ("timer", "Real-time interrupt module (RTI)"),
    "GIO": ("gpio", "GIO ports"),
    "MIBADC": ("adc", "Multi-buffered ADC (MibADC)"),
    "LIN": ("lin", "LIN"),
    "EMAC": ("ethernet", "Ethernet MAC (EMAC)"),
    # NXP Kinetis / LPC / MCX / i.MX RT.
    "TPM": ("timer", "Timer/PWM module (TPM)"),
    "FTM": ("timer", "FlexTimer module (FTM)"),
    "PIT": ("timer", "Periodic interrupt timer (PIT)"),
    "LPTMR": ("lptim", "Low-power timer (LPTMR)"),
    "LPIT": ("lptim", "Low-power periodic interrupt timer (LPIT)"),
    "CTIMER": ("timer", "Standard counter/timers (CTIMER)"),
    "SCTIMER": ("timer", "SCTimer/PWM"),
    "TSI": ("touch", "Touch sensing input (TSI)"),
    "CMP": ("comparator", "Analog comparator (CMP)"),
    "LPI2C": ("i2c", "Low-power I2C (LPI2C)"),
    "LPSPI": ("spi", "Low-power SPI (LPSPI)"),
    "FLEXCOMM": ("usart", "Flexcomm serial interfaces"),
    "FLEXCAN": ("can", "FlexCAN"),
    "ENET": ("ethernet", "Ethernet (ENET)"),
    "SDHC": ("sdmmc", "SD host controller (SDHC)"),
    "FLEXIO": ("usart", "FlexIO"),
    "FLEXSPI": ("xspi", "FlexSPI"),
    "LLWU": ("power", "Low-leakage wakeup unit (LLWU)"),
    "DMAMUX": ("dma", "DMA channel mux (DMAMUX)"),
    "EWM": ("watchdog", "External watchdog monitor (EWM)"),
    "CRC": ("safety", "CRC"),
    "LCDK": ("display", "LCD controller"),
    "MIPI": ("display", "MIPI DSI/CSI"),
    "LPADC": ("adc", "Low-power ADC (LPADC)"),
    "TIMER": ("timer", "Timer/Counter (TIMER)"),
    "WTIMER": ("timer", "Wide Timer/Counter (WTIMER)"),
    "LETIMER": ("lptim", "Low Energy Timer (LETIMER)"),
    "LEUART": ("usart", "Low Energy UART (LEUART)"),
    "PCNT": ("timer", "Pulse Counter (PCNT)"),
    "ACMP": ("comparator", "Analog Comparator (ACMP)"),
    "IDAC": ("dac", "Current DAC (IDAC)"),
    "VDAC": ("dac", "Voltage DAC (VDAC)"),
    "RTCC": ("rtc", "Real Time Counter and Calendar (RTCC)"),
    "CRYOTIMER": ("lptim", "Ultra Low Energy Timer (CRYOTIMER)"),
    "LDMA": ("dma", "Linked DMA (LDMA)"),
    "GPCRC": ("safety", "General Purpose CRC (GPCRC)"),
    "PRS": ("interrupts", "Peripheral Reflex System (PRS)"),
    "CSEN": ("touch", "Capacitive Sense (CSEN)"),
    "LESENSE": ("touch", "Low Energy Sensor Interface (LESENSE)"),
    "WDOG": ("watchdog", "Watchdog (WDOG)"),
    "PCA": ("timer", "Programmable Counter Array (PCA)"),
    "SMB": ("i2c", "SMBus (SMB)"),
}

_RANGE = re.compile(r"\d+(?:\.\d+)?\s*(?:-|–|to)\s*\d+(?:\.\d+)?")
# "48, 64, 72 and 100 leads" is a list; "100,000 cycles" is a thousands separator.
_COMMA_LIST = re.compile(r"\b\d{1,3}(?:,\s+\d{1,3})+(?:,?\s+(?:and|or)\s+\d{1,3})?\b|\b\d{1,3}\s+(?:and|or)\s+\d{1,3}\s+(?:leads|pins|packages)\b")
# "16 or 32 Kbytes", "8/16KB", "26/37/51 I/Os", "64/128/256 KB": one cell, several members.
_ALT_LIST = re.compile(r"\b\d{1,4}\s*(?:or|/)\s*\d{1,4}(?:\s*/\s*\d{1,4})*\s*-?\s*(?:KB|MB|Kbytes?|Mbytes?|Kbit|Mbit|I/Os?|pins?|leads?|MHz|channels?)\b", re.I)
_VARIES = re.compile(r"depending on|varies|device[- ]dependent|according to the (?:device|part|package)|see (?:the )?datasheet", re.I)
_SENTENCE_START = re.compile(r"^(?:the|this|these|it|for|refer|see|section|to|when|if|in|on|a|an|all|note|may|allows?|supports?|provides?|enables?|includes?|load|output|pin|gluelessly|devices?)\b|(?-i:^Can\s+[a-z])", re.I)
_CLASS_OVER_SECTION = {"flash", "sram", "eeprom", "memory_map", "debug", "dma", "package", "temp_sensor"}
_GRID_MAX_LEN = 72


def _norm_key(text: str | None) -> str:
    return re.sub(r"[^a-z]+", " ", (text or "").lower()).strip()


# Vendor section headings that name a group directly (Renesas/Microchip/NXP
# cover pages). Keys are normalised lowercase words.
_SECTION_GROUP_DIRECT: dict[str, str] = {
    "memory": "memory",
    "memories": "memory",
    "connectivity": "connectivity",
    "communication interfaces": "connectivity",
    "communications": "connectivity",
    "analog": "analogue",
    "analogue": "analogue",
    "advanced analog features": "analogue",
    "analog peripherals": "analogue",
    "timers": "timers_pwm_control",
    "timers output compare input capture": "timers_pwm_control",
    "high speed pwm": "timers_pwm_control",
    "safety": "security_safety_identity",
    "security": "security_safety_identity",
    "security and encryption": "security_safety_identity",
    "qualification and class b support": "security_safety_identity",
    "human machine interface hmi": "graphics_vision_touch_hmi",
    "human machine interface": "graphics_vision_touch_hmi",
    "graphics": "graphics_vision_touch_hmi",
    "power management": "power_clock_reset",
    "clock management": "power_clock_reset",
    "power": "power_clock_reset",
    "clocks": "power_clock_reset",
    "low power": "power_clock_reset",
    "operating conditions": "power_clock_reset",
    "operating temperature and packages": "io_package_environment",
    "packages": "io_package_environment",
    "input output": "io_package_environment",
    "general purpose i o ports": "io_package_environment",
    "system and power management": "power_clock_reset",
    "multiple clock sources": "power_clock_reset",
    "clock sources": "power_clock_reset",
    "reset and clock control": "power_clock_reset",
    "power supply": "power_clock_reset",
    "arm cortex m33 core": "processing",
    "arm cortex m4 core": "processing",
    "arm cortex m23 core": "processing",
    "arm cortex m85 core": "processing",
    "arm cortex m0 core": "processing",
    "cpu core": "processing",
    "event link": "peripherals",
    "direct memory access dma": "peripherals",
    "dma": "peripherals",
    "debugger development support": "peripherals",
    "core": "processing",
    "cpu": "processing",
}
_SECTION_WORDS: dict[str, str] = {}
# Vendor headings that hold several groups' worth of facts; the bullet decides.
_MIXED_SECTIONS = {"system and power management", "system", "peripherals", "other", "others", "miscellaneous", "additional features", "description"}
# Classes the RA4C1 page files in two groups on purpose.
_DUAL_HOME_CLASSES = {"watchdog", "safety", "trustzone", "crypto"}
_SIZING_UNITS = {"mbyte", "kbyte", "mb", "kb", "mbit", "kbit", "mhz", "khz", "ghz", "v", "channel", "i/o", "pin", "°c"}
_PROSE_GROUP: dict[str, tuple[str, str]] = {
    "core": ("processing", "cpu_core"),
    "max_frequency": ("processing", "cpu_core"),
    "supply_range": ("power_clock_reset", "power"),
    "temperature_range": ("io_package_environment", "gpio"),
    "packages": ("io_package_environment", "gpio"),
}


def varies_by_part(text: str) -> bool:
    return bool(_RANGE.search(text) or _COMMA_LIST.search(text) or _ALT_LIST.search(text) or _VARIES.search(text))


# Sizes that are not a memory capacity: programming/erase granularity, ECC
# word sizes, sector/page/block sizes, partitions, note text.
_NOT_CAPACITY = re.compile(r"^\s*note\b|^\s*\(|\bfirst\b|\barea of\b|\bunits? of\b|\bgranularity|\berase\b|\bprogram(?:ming)?\s*:|\bblank\s+check|\bpage\s+size|\bblock\s+size|\bsector\s+size|\bsectors?\s+of\b|\bword\b|\bper\s+(?:sector|page|block)|\bminimum\s+(?:erase|program)|\bphrase\b|\brecord\s+size|\bcache\s+line|\bline\s+size|\bboundar", re.I)

_STANDALONE_LEAF_CLASSES = {"temp_sensor", "dma", "rtc", "watchdog", "safety", "comparator", "dac", "adc", "touch", "display", "usb", "can", "ethernet", "crypto", "opamp"}
_MODE_LEAF = re.compile(r"^(?:simple|smart\s*card|manchester|asynchronous|synchronous|half[- ]duplex|full[- ]duplex|master|slave|mode|modes|supports?|support\s+for|peripherals?\s+supported)\b", re.I)
_SUPPLY_RANGE_TEXT = re.compile(r"\b\d\.\d{1,2}\s*V?\s*(?:to|–|-|~)\s*\d\.\d{1,2}\s*V\b")
_TEMP_RANGE_TEXT = re.compile(r"[-–−]\s*\d{1,3}\s*(?:°\s*C|ºC|℃)?\s*(?:to|~|–|-|…)\s*\+?\s*\d{1,3}\s*(?:°\s*C|ºC|℃)")

# Vendor section headings that carry a count or a long name; checked when the
# exact key is not in _SECTION_GROUP_DIRECT.
_SECTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # keys arrive through _norm_key: lower-case letters only ("6 timers" -> "timers").
    (re.compile(r"^timers?\b"), "timers_pwm_control"),
    (re.compile(r"^communication"), "connectivity"),
    (re.compile(r"^up to (?:fast )?i o ports?$|^i os?$|^i o ports?$|^i o$"), "io_package_environment"),
    (re.compile(r"^clock reset and supply management$|^clocks? and reset|^reset and clock|^supply and reset|^clock(?:ing)? and power"), "power_clock_reset"),
    (re.compile(r"^operating voltage$|^voltage range$|^supply voltage"), "power_clock_reset"),
    (re.compile(r"^debug(?: mode| and trace)?$|^debug"), "peripherals"),
    (re.compile(r"^(?:arm )?cortex m\d+\S* core$|^(?:arm|risc v) .*core$|^processor$|^cpu$|^core$"), "processing"),
    (re.compile(r"^analog(?:ue)?"), "analogue"),
    (re.compile(r"^memor(?:y|ies)"), "memory"),
    (re.compile(r"^security|^safety|^cryptograph"), "security_safety_identity"),
    (re.compile(r"^graphics|^display|^human machine|^hmi$|^touch"), "graphics_vision_touch_hmi"),
    (re.compile(r"^packages?|^operating temperature|^temperature range"), "io_package_environment"),
    # TI features-table sections.
    (re.compile(r"^performance$"), "processing"),
    (re.compile(r"^advanced motion control$|^motion control"), "timers_pwm_control"),
    (re.compile(r"^package information$|^packaging"), "io_package_environment"),
)


def _section_group(section_key: str) -> str | None:
    if section_key in _SECTION_GROUP_DIRECT:
        return _SECTION_GROUP_DIRECT[section_key]
    for pattern, group in _SECTION_PATTERNS:
        if pattern.search(section_key):
            return group
    return None


def _grid_eligible(text: str) -> bool:
    return len(text) <= _GRID_MAX_LEN and not _SENTENCE_START.match(text) and not text.rstrip().endswith((":", ",", ";")) and not text.endswith(".")


def _single_number(numbers: list[dict[str, Any]], text: str = "") -> tuple[Any, str | None]:
    typed = [n for n in numbers if n.get("unit")]
    if text and (_RANGE.search(text) or _ALT_LIST.search(text) or _COMMA_LIST.search(text)):
        return None, None  # "2.0 to 3.6 V", "16 or 32 Kbytes": a range or list, not a scalar
    if len(typed) == 1 and len(numbers) <= 2:
        return typed[0]["value"], typed[0]["unit"]
    # "8 KB data flash memory (100,000 P/E cycles)": the sized number leads and
    # the unitless numbers after it are detail. A unitless number before the
    # sized one ("1 to 20 MHz", "48/64/80 MHz") makes it a range or list: no scalar.
    if len(typed) == 1 and numbers and numbers[0] is typed[0]:
        return typed[0]["value"], typed[0]["unit"]
    return None, None


def build_grid(record: dict[str, Any], vendor_by_sha: dict[str, str] | None = None) -> dict[str, Any]:
    meta = record["_meta"]
    identity = record["identity"]
    artifact = meta["source_path"].rsplit("/", 1)[-1]
    scope = _scope_as_printed(identity)
    lines = identity["lines_covered"]
    grain = "series" if (lines.get("group_tokens") or lines["wildcards"] or lines["parts"]) else "family"
    vendor = normalize_vendor(meta.get("vendor"))
    vendor_basis = "document" if vendor else None
    if not vendor:
        vendor = normalize_vendor((vendor_by_sha or {}).get(meta["document_sha256"]))
        vendor_basis = "inventory" if vendor else None
    if not vendor:
        # A null vendor cannot be routed to a catalogue node (CR grid-v0
        # review). The document ID, scope and filename name the vendor in
        # nearly every case; the basis says so.
        vendor = infer_vendor(identity.get("document_id"), scope, artifact)
        vendor_basis = "inferred" if vendor else None
    base = {"vendor": vendor, "vendor_basis": vendor_basis, "grain": grain, "scope_as_printed": scope, "source_artifact": artifact, "document_sha256": meta["document_sha256"]}
    chapter_titles: dict[str, str] = {}
    for cls, entries in record.get("chapters", {}).get("peripheral_classes", {}).items():
        for entry in entries:
            if entry.get("level") == 1:
                chapter_titles.setdefault(cls, re.sub(r"^\s*\d{1,2}(?:\.\d+)*\s+", "", entry["title"]))
                break
    rows: list[dict[str, Any]] = []

    def emit(group: str, cls: str | None, label: str, *, pages: list[int], verbatim: str, tier: str, instances: int | None = None, value: Any = None, unit: str | None = None, qualifier: str | None = None, section: str | None = None, flags: dict | None = None) -> None:
        row = {
            **base,
            "group": group,
            "peripheral_class": cls,
            "label": label,
            "instances": instances,
            "value": value,
            "unit": unit,
            "qualifier_verbatim": qualifier,
            # A supply range is the envelope of every member (CR decision);
            # temperature and anything "depending on MPN" varies by part.
            "varies_by_part": False if (section == "supply_range" or _SUPPLY_RANGE_TEXT.search(label) or _SUPPLY_RANGE_TEXT.search(verbatim)) else varies_by_part(verbatim),
            "tier": tier,
            "source_pages": sorted(set(pages))[:12],
            "verbatim": verbatim[:300],
        }
        if section:
            row["section"] = section
        if flags:
            row.update(flags)
        rows.append(row)

    # 1. Instance counts: the display ("SCI x6"), count as integer.
    for icls, data in record.get("peripheral_instances", {}).items():
        strong = [i for i in data["instances"] if not i["weak"]]
        if not strong or icls not in INSTANCE_CLASS:
            continue
        pcls, group_label = INSTANCE_CLASS[icls]
        names = [i["instance"] for i in strong]
        # Label is the vendor's: the chapter title when the document has one for
        # this class, else the vendor's own instance stem. Our synthesized
        # wording stays in `group_label` as a grouping aid only.
        stem = _vendor_stem(names, icls)
        title = chapter_titles.get(pcls)
        # The chapter title is the label only when it names this stem (SPI's
        # chapter is not I2S's label; USART's chapter is not UART's).
        label = title if title and re.search(rf"(?<![A-Za-z]){re.escape(stem.split(' / ')[0])}(?![A-Za-z])", title) else stem
        for group in CLASS_GROUPS.get(pcls, ()):
            emit(group, pcls, label, pages=[i["first_page"] for i in strong], verbatim=", ".join(names), tier="grid", instances=len(strong), flags={"group_label": group_label, "instance_names": names, "weak_instance_names": [i["instance"] for i in data["instances"] if i["weak"]][:16]})

    # 2. Cover features: the vendor's own short list (family data sheets).
    #    The vendor's section heading ("■ Memory", "■ Connectivity") decides the
    #    group when it classifies; the bullet's own words otherwise.
    has_features_table = any(f.get("source") == "features_table" for f in record.get("features", []))
    for feature in record.get("features", []):
        text = feature["verbatim"]
        label = feature.get("label") or text
        classes = feature.get("classes") or []
        section = feature.get("vendor_section")
        section_class = feature.get("section_class") or (_SECTION_WORDS.get(_norm_key(section)) if section else None)
        value, unit = _single_number(feature.get("numbers", []), label)
        groups: set[str] = set()
        cls: str | None = classes[0] if classes else None
        section_key = _norm_key(section)
        section_group = _section_group(section_key)
        bullet_groups = {g for c in classes for g in CLASS_GROUPS.get(c, ())}
        table_row = feature.get("source") == "features_table"
        if section_group and section_key not in _MIXED_SECTIONS and not (table_row and cls in _CLASS_OVER_SECTION):
            # The vendor filed it under this heading; that is the group. A
            # bullet whose own class also belongs elsewhere keeps both homes.
            # In a features table the section rows are coarse ("Performance"
            # holds Flash, SRAM and EEPROM): a memory/debug class wins there.
            groups = {section_group}
            if cls in _DUAL_HOME_CLASSES:
                groups |= bullet_groups
        elif bullet_groups:
            groups = bullet_groups
        elif section_class:
            groups = set(CLASS_GROUPS.get(section_class, ()))
            cls = cls or section_class
        elif section_group:
            groups = {section_group}
        if not groups and _package_or_temp(text):
            groups = {"io_package_environment"}
        if not groups and _power_fact(text):
            groups = {"power_clock_reset"}
        # Sub-bullets are detail under their parent: below the grid — except a
        # short security-function line under a security engine ("Symmetric
        # algorithms: AES", "128-bit unique ID"), which the target page lists.
        security_leaf = "security_safety_identity" in groups and len(label) <= 48 and any(p.search(label) for _, p in _SECURITY_VOCAB)
        # A sub-bullet that names a peripheral in its own right ("Temperature
        # sensor" under the ADC, "7-channel DMA controller") is a grid line; a
        # mode of its parent ("Simple SPI" under SCI) is not.
        standalone_leaf = cls in _STANDALONE_LEAF_CLASSES and len(label) <= 48 and not _MODE_LEAF.match(label)
        tier = "grid" if (_grid_eligible(label) and (feature.get("level", 1) == 1 or security_leaf or standalone_leaf)) else "below_grid"
        # A document with a features table (TI) summarises the family there;
        # its long "Features" section is per-peripheral detail. From that
        # section only a counted or sized line is a grid fact.
        if has_features_table and feature.get("source") == "features_continuation":
            if not (feature.get("count") or (value is not None and unit)):
                tier = "below_grid"
        for group in sorted(groups):
            emit(group, cls, label, pages=[feature["receipt"]["page"]], verbatim=text, tier=tier, instances=feature.get("count"), value=value, unit=unit, qualifier=feature.get("qualifier_verbatim"), section=section or "features", flags={"parent": feature["parent"]} if feature.get("parent") else None)

    # 2b. Vendor section headings that are facts themselves ("■ Arm Cortex-M4
    #     Core with Floating Point Unit (FPU)") rather than group words.
    seen_sections: set[str] = set()
    for feature in record.get("features", []):
        section = feature.get("vendor_section")
        if not section or _norm_key(section) in _SECTION_GROUP_DIRECT or section in seen_sections:
            continue
        seen_sections.add(section)
        cls = feature.get("section_class")
        for group in CLASS_GROUPS.get(cls or "", ()):
            emit(group, cls, section, pages=[feature["receipt"]["page"]], verbatim=section, tier="grid" if _grid_eligible(section) else "below_grid", section="features_heading")

    # 3. Chapter "main features" bullets: family detail, below the grid except
    #    short capacity/speed facts with a sizing unit.
    for feature in record.get("chapter_features", []):
        text = feature["verbatim"]
        cls = feature.get("chapter_class")
        groups = CLASS_GROUPS.get(cls or "", ())
        if not groups:
            continue
        value, unit = _single_number(feature.get("numbers", []))
        sizing = unit is not None and unit.lower().rstrip("s") in _SIZING_UNITS
        # Only capacity/speed facts of the memory and core chapters render;
        # a peripheral chapter's bit rates are detail.
        tier = "grid" if (_grid_eligible(text) and len(text) <= 56 and sizing and cls in ("flash", "sram", "memory_map", "cpu_core") and not _NOT_CAPACITY.search(text)) else "below_grid"
        for group in groups:
            emit(group, cls, text, pages=[feature["receipt"]["page"]], verbatim=text, tier=tier, value=value, unit=unit, qualifier=feature.get("qualifier_verbatim"), section=feature.get("section"))

    # 4. Memory regions with a size (one row per distinct label).
    seen_memory: set[str] = set()
    for row in record.get("memory", []):
        text = row["verbatim"]
        if text.lower() in seen_memory:
            continue
        seen_memory.add(text.lower())
        # Sub-kilobyte numbers are programming/erase granularity, ECC word
        # sizes or note text, never a family memory capacity; footnotes and
        # "first N KB of" partitions are detail. Both stay below the grid.
        capacity = row["size_kb"] >= 1 and not _NOT_CAPACITY.search(text)
        emit("memory", "flash" if re.search(r"flash|program memory|otp", text, re.I) else "sram", text, pages=[row["receipt"]["page"]], verbatim=text, tier="grid" if (capacity and _grid_eligible(text)) else "below_grid", value=row["size_kb"], unit="KB", qualifier=row.get("qualifier_verbatim"), section="memory")

    # 5. Prose facts from the opening pages: core, max frequency, supply,
    #    temperature, packages.
    for fact in record.get("prose_facts", []):
        group, cls = _PROSE_GROUP[fact["kind"]]
        label = fact["verbatim"]
        if fact["kind"] == "packages":
            label = ", ".join(fact["value"])
        flags: dict[str, Any] = {}
        if fact["kind"] == "packages":
            flags["packages"] = fact["value"]
        if fact.get("context"):
            flags["context"] = fact["context"]
        emit(group, cls, label, pages=[fact["receipt"]["page"]], verbatim=fact["verbatim"], tier="grid" if len(label) <= _GRID_MAX_LEN else "below_grid", value=fact.get("value") if fact["kind"] != "packages" else None, unit=fact.get("unit"), qualifier=fact.get("qualifier_verbatim"), section=fact["kind"], flags=flags or None)

    # 5b. Package-qualified I/O counts. One row per (pin count, package); the
    #     I/O number is a value only under that package qualifier. CR: never a
    #     scalar, never expanded to members. `varies_by_part` is forced on.
    #     Two quantities, kept apart: a count the document states ("I/O pins:
    #     80") and a count of the port pins its pinout table lists for that
    #     package. They differ where a vendor counts remappable oscillator or
    #     debug pins as I/O; both are emitted so the merge can see it.
    for row in record.get("io_by_package", []):
        counted = row["pattern"] == "pin_table_count"
        label = f"{row['io_count']} {'port pins listed' if counted else 'I/O'} ({row['pin_count']}-pin {row['package']})"
        pages = row["receipt"].get("pages") or [row["receipt"]["page"]]
        emit("io_package_environment", "gpio", label, pages=pages, verbatim=row["verbatim"], tier="grid", value=row["io_count"], unit="I/O", section="io_by_package", flags={"package_qualifier": {"pin_count": row["pin_count"], "package": row["package"]}, "varies_by_part": True, "pattern": row["pattern"], "quantity_qualifier": "port_pins_listed" if counted else "stated"})

    # 6. Chapter presence: asserted present, with location. Never absence.
    presence = []
    for cls, entries in record.get("chapters", {}).get("peripheral_classes", {}).items():
        presence.append({"peripheral_class": cls, "presence": "asserted_present", "evidence": "chapter", "chapters": [{"title": e["title"], "page": e["page"]} for e in entries[:6]], "groups": list(CLASS_GROUPS.get(cls, ()))})
        for group in CLASS_GROUPS.get(cls, ()):
            top = entries[0]
            emit(group, cls, top["title"], pages=[e["page"] for e in entries[:6]], verbatim=top["title"], tier="below_grid", section="chapter")
    observed = {p["peripheral_class"] for p in presence} | {INSTANCE_CLASS[k][0] for k, v in record.get("peripheral_instances", {}).items() if k in INSTANCE_CLASS and v["count_asserted"]}
    not_observed = sorted(set(CLASS_GROUPS) - observed)

    rows = _dedupe(rows)
    for row in rows:
        row["quantity_qualifier"] = row.get("quantity_qualifier") or quantity_qualifier(row)
        facts = typed_facts(row)
        if facts:
            row["typed"] = facts
    sram_check = _sram_bank_check(rows)
    typed_by_group: Counter = Counter()
    typed_kinds: Counter = Counter()
    for row in rows:
        for fact in row.get("typed", []):
            typed_by_group[row["group"]] += 1
            typed_kinds[fact["kind"]] += 1
    grid_rows = [r for r in rows if r["tier"] == "grid"]
    populated = Counter(r["group"] for r in grid_rows)
    return {
        "schema": SCHEMA,
        "_meta": {**base, "document_id": identity.get("document_id"), "revision": identity.get("revision"), "title_verbatim": identity.get("title_verbatim"), "page_count": meta.get("page_count"), "extracted_at": meta.get("extracted_at"), "models_used": []},
        "groups": [{"key": k, "label": l, "grid_rows": populated.get(k, 0)} for k, l in GROUPS],
        "groups_populated": len(populated),
        "meets_six_of_ten": len(populated) >= 6,
        "rows": rows,
        "class_presence": presence,
        "not_observed": not_observed,
        "flags": {"npu_group_undecided": "npu" in observed, **sram_check},
        "quantity_qualifiers": dict(Counter(r["quantity_qualifier"] for r in grid_rows if r.get("value") is not None)),
        "typed_facts_by_group": dict(typed_by_group),
        "typed_fact_kinds": dict(typed_kinds),
    }


# Typed facts for the six groups where the family read is the only source.
# Each grammar reads a row's text (label + verbatim) and returns typed
# {kind, value, unit?, condition?} facts. Nothing inferred: every kind needs
# its anchor word in the text.
_MODE_WORDS = r"(run|sleep|low[- ]power\s+run|low[- ]power\s+sleep|lprun|lpsleep|stop\s*\d?|standby|shutdown|deep[- ]sleep|deep\s+power[- ]down|power[- ]down|vbat|backup|idle|snooze|hibernate|em\d|active|software\s+standby|halt|wait)"
_CURRENT = re.compile(rf"(\d+(?:\.\d+)?)\s*(nA|µA|uA|μA|mA)(?:/MHz)?\b", re.I)
_CURRENT_PER_MHZ = re.compile(r"(\d+(?:\.\d+)?)\s*(µA|uA|μA)\s*/\s*MHz", re.I)
_WAKEUP = re.compile(r"wake[- ]?up[^.\n]{0,40}?(\d+(?:\.\d+)?)\s*(µs|us|μs|ns|ms)|(\d+(?:\.\d+)?)\s*(µs|us|μs|ns|ms)[^.\n]{0,20}wake", re.I)
_BITS = re.compile(r"(\d{1,2})[- ]bit", re.I)
_CHANNELS = re.compile(r"(?:up\s+to\s+)?(\d{1,3})\s*(?:external\s+|input\s+|analog\s+)?(?:channels?|ch\b|inputs?)", re.I)
_SAMPLE_RATE = re.compile(r"(\d+(?:\.\d+)?)\s*(Msps|ksps|MSPS|kSPS|Ms/s|ks/s|MHz\s+sampling)", re.I)
_BITRATE = re.compile(r"(\d+(?:\.\d+)?)\s*(Mbit/s|Mbps|kbit/s|kbps|Gbit/s|Mbyte/s|MB/s)", re.I)
_SECURITY_VOCAB: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aes", re.compile(r"\bAES(?:[- ]?(?:128|192|256))?\b")),
    ("sha", re.compile(r"\bSHA(?:[- ]?(?:1|2|224|256|384|512))?\b")),
    ("trng", re.compile(r"\bTRNG\b|\btrue\s+random|\bRNG\b|random\s+number\s+generator", re.I)),
    ("ecc_crypto", re.compile(r"\bECC\b(?=[^.\n]{0,40}(?:crypto|asymmetric|curve|ECDSA|ECDH|public))|\bECDSA\b|\bECDH\b|\bPKA\b|elliptic", re.I)),
    ("rsa", re.compile(r"\bRSA\b")),
    ("trustzone", re.compile(r"TrustZone|\bTZ\b", re.I)),
    ("secure_boot", re.compile(r"secure\s+boot|root\s+of\s+trust|\bRoT\b", re.I)),
    ("tamper", re.compile(r"\btamper", re.I)),
    ("unique_id", re.compile(r"unique\s+(?:device\s+)?ID|\bUID\b|\bUUID\b", re.I)),
    ("readout_protection", re.compile(r"read-?out\s+protection|\bRDP\b|code\s+protection|flash\s+(?:area\s+)?protection|write\s+protection", re.I)),
    ("mpu", re.compile(r"\bMPU\b|memory\s+protection\s+unit", re.I)),
    ("memory_ecc", re.compile(r"\bECC\b(?![^.\n]{0,40}(?:crypto|asymmetric|curve|ECDSA|ECDH|public))|error\s+correct", re.I)),
    ("parity", re.compile(r"\bparity\b", re.I)),
    ("crc", re.compile(r"\bCRC(?:-?\d+)?\b")),
    ("watchdog", re.compile(r"watchdog|\bWDT\b|\bIWDG\b|\bWWDG\b|\bIWDT\b", re.I)),
    ("secure_debug", re.compile(r"debug\s+(?:access\s+level|protection|authentication)|secure\s+debug", re.I)),
    ("key_storage", re.compile(r"key\s+(?:storage|injection|wrap|management)|\bHUK\b|\bPUF\b", re.I)),
    ("functional_safety", re.compile(r"IEC\s*61508|ISO\s*26262|\bSIL\s?\d|\bASIL|class\s+B|UL\s*60730", re.I)),
    ("hardware_crypto_generic", re.compile(r"cryptograph|\bcrypto\b|\bRSIP\b|\bSCE\d?\b|\bHASH\b|\bCRYP\b|\bSAES\b", re.I)),
)
_USB_SPEED = re.compile(r"\b(full[- ]speed|high[- ]speed|low[- ]speed|super[- ]speed|\bFS\b|\bHS\b)", re.I)
_LCD_SEG = re.compile(r"(\d{1,2})\s*[×x]\s*(\d{1,3})\s*segment", re.I)
_TOUCH_CH = re.compile(r"(\d{1,3})[- ]channel\s+(?:capacitive\s+)?touch|touch[^.\n]{0,30}?(\d{1,3})\s*(?:channels?|electrodes?|inputs?)", re.I)
_RESOLUTION_PX = re.compile(r"(\d{3,4})\s*[×x]\s*(\d{3,4})", re.I)
_TIMER_WIDTH_COUNT = re.compile(r"(\d{1,2})[- ]bit\b[^.\n]{0,40}?(?:[×x]\s*(\d{1,2})|(\d{1,2})\s*[×x])|(\d{1,2})\s+(\d{1,2})[- ]bit", re.I)
_PWM_CH = re.compile(r"(\d{1,3})\s*(?:PWM\s+)?(?:channels?|outputs?)[^.\n]{0,15}PWM|PWM[^.\n]{0,30}?(\d{1,3})\s*(?:channels?|outputs?)", re.I)


def _n(text: str) -> float | int:
    value = float(text)
    return int(value) if value.is_integer() else value


# Processing: the core as printed, its clock, its performance and its options.
_CORE_NAME = re.compile(r"\b(Cortex[®™]?-?\s?[MRA]\d{1,2}\+?F?|C8051|CIP-51|8051|RISC-V|C28x|RXv[123]|RL78|MIPS32|MIPS\s+M\d{1,2}[Kk]|microAptiv|AVR|PIC18|PIC16|TriCore|Xtensa|ARM7TDMI|ARM9|ARM926)\b", re.I)
_CORE_MHZ = re.compile(r"(?:up\s+to\s+|@\s*|at\s+|of\s+)?(\d{1,4}(?:\.\d+)?)\s*-?\s*MHz", re.I)
_DMIPS = re.compile(r"(\d{2,4}(?:\.\d+)?)\s*DMIPS(?!/MHz)|(\d(?:\.\d+)?)\s*DMIPS/MHz", re.I)
_COREMARK = re.compile(r"(\d{2,4}(?:\.\d+)?)\s*CoreMark", re.I)
_FPU = re.compile(r"\b(?:FPU|floating[- ]point(?:\s+unit)?)\b", re.I)
_FPU_PRECISION = re.compile(r"\b(single|double)[- ]precision", re.I)
_CORE_WIDTH = re.compile(r"\b(8|16|32|64)[- ]bit\b", re.I)
_MULTI_CORE = re.compile(r"\b(dual|two|2|quad|four|4|triple|three|3)[- ]?cores?\b|\b(dual|quad)[- ]core\b", re.I)
_MEMORY_KIND = (
    ("eeprom", re.compile(r"\beeprom\b", re.I)),
    ("data_flash", re.compile(r"\bdata\s+flash\b", re.I)),
    ("flash", re.compile(r"\bflash\b|\bprogram\s+memory\b|\bcode\s+memory\b", re.I)),
    ("rom", re.compile(r"\bROM\b|\bboot\s+rom\b|\bmask\s+rom\b", re.I)),
    ("sram", re.compile(r"\bs?ram\b|\bTCM\b|\bCCM\b", re.I)),
)
_OSC = re.compile(
    r"\b(HSI48|HSI|HSE|LSI|LSE|MSI|CSI|HSI16|IRC48M|IRC8M|IRC40K|IRC32K|LOCO|HOCO|MOCO|SOSC|MOSC|FRO|IRC|LPOSC|OSC32K|XOSC32K|XOSC|DFLL48M|DFLL|OSC8M|OSC16M|OSC48M|HFRCO|LFRCO|AUXHFRCO|HFXO|LFXO|ULFRCO|PLL\d?|SYSOSC|LFOSC|BLPE|ROSC|SIRC|FIRC|SOSC|MCG)\b"
    r"|\b(internal|on-chip|external|low[- ]speed|high[- ]speed|low[- ]frequency|high[- ]frequency|low[- ]power)\s+(?:\w+\s+){0,2}?(?:RC\s+)?(?:oscillator|crystal|clock|reference\s+clock)\b",
    re.I,
)
_OSC_FREQ = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*(MHz|kHz|Hz)\b", re.I)
_LP_MODE = re.compile(r"\b(Sleep|Stop\s?\d?|Standby|Shutdown|Deep[- ]?Sleep|Deep\s+power[- ]down|Power[- ]down|Hibernate|Snooze|Idle|Halt|Wait|Backup|VBAT|Low[- ]power\s+run|Low[- ]power\s+sleep|LPRUN|LPSLEEP|EM[1-4]|VLLS\d?|LLS\d?|VLPS|VLPR|VLPW|Software\s+standby|Deep\s+software\s+standby)\b", re.I)
_PROTOCOL_FLAGS = (
    ("lin", re.compile(r"\bLIN\b", re.I)),
    ("irda", re.compile(r"\bIrDA\b", re.I)),
    ("smartcard", re.compile(r"\bsmart\s?card\b|ISO\s?7816", re.I)),
    ("modbus", re.compile(r"\bModbus\b", re.I)),
    ("ieee1588", re.compile(r"IEEE\s?1588|\bPTP\b", re.I)),
    ("rs485", re.compile(r"RS-?485", re.I)),
    ("i2c_fm_plus", re.compile(r"fast[- ]mode\s+plus|\bFm\+|1\s*Mbit/s\s*(?:\(fast\s+mode\s+plus\))?", re.I)),
    ("smbus", re.compile(r"\bSMBus\b", re.I)),
    ("pmbus", re.compile(r"\bPMBus\b", re.I)),
    ("i2s", re.compile(r"\bI2S\b", re.I)),
    ("spdif", re.compile(r"S/PDIF|\bSPDIF", re.I)),
    ("quad_spi", re.compile(r"quad[- ]?SPI|\bQSPI\b|\bQUADSPI\b", re.I)),
    ("octo_spi", re.compile(r"octo[- ]?SPI|\bOSPI\b|\bOCTOSPI\b|\bXSPI\b|hexadeca", re.I)),
    ("usb_type_c", re.compile(r"type[- ]C|\bUCPD\b|power\s+delivery", re.I)),
    ("hdmi_cec", re.compile(r"HDMI[- ]CEC|\bCEC\b", re.I)),
    ("sdio", re.compile(r"\bSDIO\b|\bSDMMC\b|\bSD/MMC|\bSDHC\b|\beMMC\b", re.I)),
    ("can_fd", re.compile(r"CAN[- ]?FD|FDCAN|flexible\s+data", re.I)),
    ("tt_can", re.compile(r"TT-?CAN|time[- ]triggered", re.I)),
)
_TIMER_CHANNELS = re.compile(r"(\d{1,2})(?:\s+or\s+(\d{1,2}))?[- ]channels?\b|(\d{1,2})(?:\s+or\s+(\d{1,2}))?\s+(?:compare/capture(?:/PWM)?|capture/compare(?:/PWM)?|input\s+capture|output\s+compare|IC/OC(?:/PWM)?|PWM)\s+channels?", re.I)
_TIMER_MHZ = re.compile(r"(?:up\s+to\s+|@\s*|at\s+)?(\d{1,4})\s*MHz", re.I)
_VREF = re.compile(r"(?:reference\s+voltage|voltage\s+reference|VREF(?:INT|BUF)?)[^.\n]{0,25}?\(?\s*(\d(?:\.\d+)?)\s*V\b|(\d(?:\.\d+)?)\s*V\s+(?:internal\s+)?(?:reference|VREF)", re.I)
_CONVERSION_TIME = re.compile(r"(\d+(?:\.\d+)?)\s*(µs|us|μs|ns)\s*(?:conversion|conv\.)|conversion(?:\s+time)?[^.\n]{0,20}?(\d+(?:\.\d+)?)\s*(µs|us|μs|ns)", re.I)


def _core_facts(text: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    core = _CORE_NAME.search(text)
    if core:
        printed = core.group(1)
        facts.append({"kind": "core_name", "value": re.sub(r"[®™]", "", printed), "value_normalized": re.sub(r"[®™\s]", "", printed).upper().replace("CORTEX", "Cortex").replace("CORTEX-", "Cortex-")})
    mhz = _CORE_MHZ.search(text)
    if mhz and not re.search(r"oscillator|crystal|\bRC\b|\bHSI\b|\bHSE\b|\bLSE\b|\bLSI\b|\bIRC\b|\bPLL\b|flash|SPI|USB|Ethernet|timer|ADC|bus|APB|AHB", text, re.I):
        value = _n(mhz.group(1))
        if 1 <= value <= 3000:
            facts.append({"kind": "core_max_mhz", "value": value, "qualifier_verbatim": "up to" if re.search(r"up\s+to|maximum|max\.?", text, re.I) else None})
    dmips = _DMIPS.search(text)
    if dmips:
        if dmips.group(1):
            facts.append({"kind": "dmips", "value": _n(dmips.group(1))})
        else:
            facts.append({"kind": "dmips_per_mhz", "value": _n(dmips.group(2))})
    coremark = _COREMARK.search(text)
    if coremark:
        facts.append({"kind": "coremark", "value": _n(coremark.group(1))})
    if _FPU.search(text):
        precision = _FPU_PRECISION.search(text)
        facts.append({"kind": "fpu", "value": True, "precision": precision.group(1).lower() if precision else None})
    if re.search(r"\bMPU\b|memory\s+protection\s+unit", text, re.I):
        facts.append({"kind": "mpu", "value": True})
    if re.search(r"\bDSP\b(?!\s*/)|DSP\s+instructions|digital\s+signal\s+processing", text, re.I):
        facts.append({"kind": "dsp_extension", "value": True})
    if re.search(r"TrustZone", text, re.I):
        facts.append({"kind": "trustzone", "value": True})
    multi = _MULTI_CORE.search(text)
    if multi:
        word = (multi.group(1) or multi.group(2)).lower()
        facts.append({"kind": "core_count", "value": {"dual": 2, "two": 2, "2": 2, "triple": 3, "three": 3, "3": 3, "quad": 4, "four": 4, "4": 4}[word]})
    width = _CORE_WIDTH.search(text)
    if width and core:
        facts.append({"kind": "core_width_bits", "value": int(width.group(1))})
    return facts


def _memory_facts(text: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    """A memory size typed in KB with its kind, bound and quantity qualifier.
    Multi-valued cells keep every member (never expanded to a scalar)."""
    unit = (row.get("unit") or "").lower()
    value = row.get("value")
    if value is None or unit not in ("kb", "mb", "gb", "bytes", "byte", "b", "kbytes", "kbyte", "mbytes", "mbyte", "kbit", "mbit"):
        return []
    scale = {"kb": 1, "kbytes": 1, "kbyte": 1, "mb": 1024, "mbytes": 1024, "mbyte": 1024, "gb": 1024 * 1024, "bytes": 1 / 1024, "byte": 1 / 1024, "b": 1 / 1024, "kbit": 1 / 8, "mbit": 128}[unit]
    members = value if isinstance(value, list) else [value]
    if not all(isinstance(m, (int, float)) for m in members):
        return []
    kind = next((name for name, pattern in _MEMORY_KIND if pattern.search(text)), None)
    if kind is None:
        return []
    sizes = [_n(str(round(m * scale, 4))) for m in members]
    return [{
        "kind": f"{kind}_kb",
        "value": sizes if isinstance(value, list) else sizes[0],
        "quantity_qualifier": row.get("quantity_qualifier"),
        "qualifier_verbatim": row.get("qualifier_verbatim"),
        "varies_by_part": bool(row.get("varies_by_part")),
    }]


def typed_facts(row: dict[str, Any]) -> list[dict[str, Any]]:
    text = f"{row.get('label') or ''} — {row.get('verbatim') or ''}"
    cls = row.get("peripheral_class") or ""
    group = row["group"]
    facts: list[dict[str, Any]] = []

    if group == "processing":
        facts.extend(_core_facts(text, row))

    elif group == "memory":
        facts.extend(_memory_facts(text, row))

    elif group == "analogue":
        bits = _BITS.search(text)
        if bits and cls in ("adc", "dac", "comparator", "opamp", ""):
            kind = "dac" if re.search(r"\bDAC|D/A", text, re.I) else "adc" if re.search(r"\bADC|A/D|S12AD|analog[- ]to[- ]digital|SAR|sigma", text, re.I) else cls or None
            if kind in ("adc", "dac"):
                # "3× 12-bit ADC": the count rides with the resolution, as for timers.
                facts.append({"kind": f"{kind}_resolution_bits", "value": int(bits.group(1)), "count": row.get("instances")})
        ch = _CHANNELS.search(text)
        if ch and re.search(r"\bADC|A/D|S12AD|analog", text, re.I) and not re.search(r"\bDAC|D/A", text, re.I):
            facts.append({"kind": "adc_channels", "value": int(ch.group(1)), "qualifier_verbatim": "Up to" if re.search(r"up\s+to", ch.group(0), re.I) else None})
        elif ch and re.search(r"\bDAC|D/A", text, re.I):
            facts.append({"kind": "dac_channels", "value": int(ch.group(1)), "qualifier_verbatim": "Up to" if re.search(r"up\s+to", ch.group(0), re.I) else None})
        rate = _SAMPLE_RATE.search(text)
        if rate:
            facts.append({"kind": "adc_sample_rate", "value": _n(rate.group(1)), "unit": rate.group(2)})
        conv = _CONVERSION_TIME.search(text)
        if conv and re.search(r"\bADC|A/D|analog", text, re.I):
            value, unit = (conv.group(1), conv.group(2)) if conv.group(1) else (conv.group(3), conv.group(4))
            facts.append({"kind": "adc_conversion_time", "value": _n(value), "unit": unit.replace("u", "µ").replace("μ", "µ")})
        vref = _VREF.search(text)
        if vref:
            facts.append({"kind": "internal_reference_v", "value": _n(vref.group(1) or vref.group(2))})
        if re.search(r"temp(?:erature)?\s+sensor", text, re.I):
            facts.append({"kind": "temperature_sensor", "value": True})
        if row.get("instances") and cls in ("adc", "dac", "comparator", "opamp"):
            facts.append({"kind": f"{cls}_instances", "value": row["instances"]})

    elif group == "timers_pwm_control":
        if row.get("instances") and cls in ("timer", "lptim", "hrtim", "rtc", "watchdog"):
            facts.append({"kind": f"{cls}_instances", "value": row["instances"]})
        bits = _BITS.search(text)
        if bits and cls in ("timer", "lptim", "hrtim"):
            facts.append({"kind": "timer_width_bits", "value": int(bits.group(1)), "count": row.get("instances")})
        pwm = _PWM_CH.search(text)
        if pwm:
            facts.append({"kind": "pwm_channels", "value": int(pwm.group(1) or pwm.group(2)), "qualifier_verbatim": "Up to" if re.search(r"up\s+to", text, re.I) else None})
        chans = _TIMER_CHANNELS.search(text)
        if chans and cls in ("timer", "lptim", "hrtim", ""):
            lo = int(chans.group(1) or chans.group(3))
            hi = chans.group(2) or chans.group(4)
            facts.append({"kind": "timer_channels", "value": [lo, int(hi)] if hi else lo, "per_instance": True})
        tmhz = _TIMER_MHZ.search(text)
        if tmhz and cls in ("timer", "lptim", "hrtim"):
            facts.append({"kind": "timer_clock_mhz", "value": int(tmhz.group(1)), "qualifier_verbatim": "up to" if re.search(r"up\s+to", text, re.I) else None})
        if cls == "rtc" or re.search(r"\bRTC\b|real[- ]time\s+clock", text, re.I):
            if re.search(r"calendar", text, re.I):
                facts.append({"kind": "rtc_calendar", "value": True})
            if re.search(r"sub-?second|tamper|alarm", text, re.I):
                facts.append({"kind": "rtc_features", "value": sorted({m.lower().replace("-", "") for m in re.findall(r"sub-?second|tamper|alarms?", text, re.I)})})
        if re.search(r"motor[- ]control|complementary\s+outputs?|dead[- ]time", text, re.I) and cls in ("timer", "hrtim", ""):
            facts.append({"kind": "motor_control_timer", "value": True})
        if re.search(r"quadrature|encoder\s+interface|\bQEI\b|\bQDEC\b", text, re.I):
            facts.append({"kind": "encoder_interface", "value": True})

    elif group == "power_clock_reset":
        for match in _OSC.finditer(text):
            name = match.group(1) or match.group(0)
            window = text[match.start(): match.end() + 40]
            before = text[max(0, match.start() - 30): match.start()]
            freq = _OSC_FREQ.search(window) or _OSC_FREQ.search(before)
            fact: dict[str, Any] = {"kind": "oscillator", "name": re.sub(r"\s+", " ", name.strip()), "internal": bool(re.search(r"internal|on-chip|\bRC\b|HSI|LSI|MSI|CSI|IRC|LOCO|HOCO|MOCO|FRO|SIRC|FIRC|HFRCO|LFRCO|ULFRCO|DFLL|OSC8M|OSC16M|OSC48M", name + " " + window[:20], re.I)) or None}
            if freq:
                fact["value"], fact["unit"] = _n(freq.group(1)), freq.group(2)
            facts.append(fact)
            if len([f for f in facts if f["kind"] == "oscillator"]) >= 6:
                break
        modes = sorted({re.sub(r"\s+", " ", m.group(1)).strip() for m in _LP_MODE.finditer(text)}, key=str.lower)  # as printed
        if modes and re.search(r"mode|power|energy|sleep|stop|standby", text, re.I):
            facts.append({"kind": "low_power_modes", "value": modes})
        if re.search(r"brown-?out|\bBOR\b|\bBOD\b", text, re.I):
            facts.append({"kind": "brown_out_reset", "value": True})
        if re.search(r"\bPVD\b|programmable\s+voltage\s+detector|\bLVD\b|low[- ]voltage\s+detect", text, re.I):
            facts.append({"kind": "voltage_detector", "value": True})
        if re.search(r"\bPOR\b|power[- ]on\s+reset", text, re.I):
            facts.append({"kind": "power_on_reset", "value": True})
        if re.search(r"\bVBAT\b|battery\s+backup|backup\s+domain", text, re.I):
            facts.append({"kind": "vbat_domain", "value": True})
        if re.search(r"\bSMPS\b|\bDC-?DC\b|step[- ]down\s+converter|buck\s+converter", text, re.I):
            facts.append({"kind": "integrated_smps", "value": True})
        if re.search(r"\bLDO\b|voltage\s+regulator", text, re.I):
            facts.append({"kind": "integrated_ldo", "value": True})
        per_mhz = _CURRENT_PER_MHZ.search(text)
        if per_mhz:
            facts.append({"kind": "active_current_per_mhz", "value": _n(per_mhz.group(1)), "unit": "µA/MHz"})
        for match in _CURRENT.finditer(text):
            if per_mhz and match.start() == per_mhz.start():
                continue
            window = text[max(0, match.start() - 70): match.end() + 70]
            mode = re.search(_MODE_WORDS, window, re.I)
            facts.append({"kind": "current_in_mode", "value": _n(match.group(1)), "unit": match.group(2).replace("u", "µ").replace("μ", "µ"), "mode_verbatim": mode.group(1) if mode else None, "condition_verbatim": window.strip()[:140]})
        wake = _WAKEUP.search(text)
        if wake:
            value, unit = (wake.group(1), wake.group(2)) if wake.group(1) else (wake.group(3), wake.group(4))
            facts.append({"kind": "wakeup_time", "value": _n(value), "unit": unit.replace("u", "µ").replace("μ", "µ"), "condition_verbatim": wake.group(0)[:120]})
        if row.get("section") == "supply_range" and isinstance(row.get("value"), list):
            facts.append({"kind": "supply_range_v", "value": row["value"], "quantity_qualifier": row.get("quantity_qualifier")})

    elif group == "security_safety_identity":
        for name, pattern in _SECURITY_VOCAB:
            match = pattern.search(text)
            if match:
                facts.append({"kind": "security_function", "value": name, "verbatim": match.group(0)})

    elif group == "connectivity":
        if row.get("instances") and cls:
            facts.append({"kind": f"{cls}_instances", "value": row["instances"]})
        rate = _BITRATE.search(text)
        if rate:
            facts.append({"kind": "bit_rate", "value": _n(rate.group(1)), "unit": rate.group(2), "peripheral_class": cls or None})
        if cls == "usb" or re.search(r"\bUSB\b", text):
            speed = _USB_SPEED.search(text)
            if speed:
                facts.append({"kind": "usb_speed", "value": speed.group(1).lower().replace(" ", "-")})
            if re.search(r"\bOTG\b|on-the-go", text, re.I):
                facts.append({"kind": "usb_otg", "value": True})
        if cls == "can" or re.search(r"\bCAN\b", text):
            if re.search(r"CAN[- ]?FD|FDCAN|flexible\s+data", text, re.I):
                facts.append({"kind": "can_fd", "value": True})
        if cls == "ethernet" or re.search(r"Ethernet|\bEMAC\b|\bETH\b", text):
            if re.search(r"10/100|100\s*Mb", text):
                facts.append({"kind": "ethernet_speed", "value": "10/100"})
            if re.search(r"gigabit|1000\s*Mb|\bGb\b", text, re.I):
                facts.append({"kind": "ethernet_speed", "value": "gigabit"})
        for flag, pattern in _PROTOCOL_FLAGS:
            if flag == "can_fd" and any(f["kind"] == "can_fd" for f in facts):
                continue
            if pattern.search(text):
                facts.append({"kind": "protocol_support", "value": flag, "peripheral_class": cls or None})

    elif group == "graphics_vision_touch_hmi":
        seg = _LCD_SEG.search(text)
        if seg:
            facts.append({"kind": "lcd_segments", "value": [int(seg.group(1)), int(seg.group(2))]})
        touch = _TOUCH_CH.search(text)
        if touch:
            facts.append({"kind": "touch_channels", "value": int(touch.group(1) or touch.group(2)), "qualifier_verbatim": "Up to" if re.search(r"up\s+to", text, re.I) else None})
        res = _RESOLUTION_PX.search(text)
        if res and re.search(r"resolution|display|LCD|LTDC|pixel", text, re.I):
            facts.append({"kind": "display_resolution", "value": [int(res.group(1)), int(res.group(2))]})
        if row.get("instances") and cls in ("display", "dcmi", "touch"):
            facts.append({"kind": f"{cls}_instances", "value": row["instances"]})

    return facts


# Which QUANTITY a numeric row is (CR reversal-extract-qualifiers-20260908).
# Sibling of qualifier_verbatim: that one says which bound ("Up to"), this one
# says which quantity (code flash vs data flash, Ta vs Tj). Anchored on the
# document's own words; a bare "flash" is left null, never called code flash.
_QQ_CODE_FLASH = re.compile(r"\bcode\s+flash|\bprogram\s+(?:flash|memory)|\bflash\s+program\s+memory|\bapplication\s+(?:flash|code)|\bmain\s+flash|\bembedded\s+flash\s+memory\s+\(.*code|\binstruction\s+flash|\bcode\s+memory|\bcode\s+storage|\bprogram\s+space", re.I)
_QQ_DATA_FLASH = re.compile(r"\bdata\s+flash|\bdata\s+memory\b(?!\s*\(sram)|\bemulated\s+eeprom|\bee\s+memory", re.I)
_QQ_EEPROM = re.compile(r"\beeprom\b", re.I)
# A bank has a designator the vendor gave it; special-purpose RAMs (backup,
# cache, message, DMA, USB, retention) are neither bank nor total.
_QQ_SRAM_BANK = re.compile(r"\bsram\s?\d\b|\bsram[A-D]\b|\bsram\s+bank|\bbank\s*\d|\bitcm|\bdtcm|\btcm\b|\bccm\b|\baxi\s+sram|\bahb\s+sram|\bmain\s+internal\s+sram|\bauxiliary\s+internal\s+sram|\bd\d\s+domain", re.I)
_QQ_SRAM_SPECIAL = re.compile(r"\bbackup\b|\bretention\b|\bcache\b|\bmessage\s+ram|\bdma\s+ram|\bdpsram|\busb\s+ram|\bstandby\s+sram|\bparity\s+ram|\becc\s+ram|\bflexram|\bocram|\bperipheral\s+ram|\bmailbox", re.I)
_QQ_AMBIENT = re.compile(r"\bTa\b|\bT\s*A\b|ambient", re.I)
_QQ_JUNCTION = re.compile(r"\bTj\b|\bT\s*J\b|junction", re.I)
_QQ_STORAGE = re.compile(r"\bstorage\b|\bTstg\b", re.I)
_QQ_ABS_MAX = re.compile(r"absolute\s+max|\babs\.?\s*max|\bmaximum\s+ratings?", re.I)
_QQ_RATED = re.compile(r"operating\s+(?:voltage|range|conditions?)|supply\s+voltage|power\s+supply|\bV(?:DD|CC)\b|recommended|\bsupply\b|\boperating\b|\boperation\b|\bvoltage\s+range\b", re.I)
_QQ_TYPICAL = re.compile(r"\btyp(?:ical|\.)?\b", re.I)
_QQ_MINIMUM = re.compile(r"\bmin(?:imum|\.)?\b", re.I)
_QQ_MAXIMUM = re.compile(r"\bmax(?:imum|\.)?\b|\bup\s+to\b", re.I)


_SIZE_IN_TEXT = re.compile(r"\d\s*-?\s*(?:KB|MB|Kbytes?|Mbytes?|bytes|Kbit|Mbit)\b", re.I)


def quantity_qualifier(row: dict[str, Any]) -> str | None:
    # Fields are joined with a separator so a label ending "SRAM" and a
    # verbatim starting "6 or 10" do not read as a bank designator "SRAM 6".
    # The vendor's section heading counts as wording ("Operating Voltage" over "1.62V – 3.63V").
    text = " ¦ ".join(str(row.get(k) or "") for k in ("label", "verbatim", "context", "section", "parent"))
    unit = (row.get("unit") or "").lower()
    cls = row.get("peripheral_class")
    section = row.get("section") or ""
    memory_row = unit in ("kb", "mb", "kbyte", "kbytes", "mbyte", "mbytes", "bytes", "kbit", "mbit") or cls in ("flash", "sram", "memory_map")
    # A multi-valued memory line ("16 or 32 Kbytes of Flash") has no scalar
    # value but is exactly the family fact that needs its qualifier.
    supply_text = _SUPPLY_RANGE_TEXT.search(str(row.get("label") or ""))
    temp_text = _TEMP_RANGE_TEXT.search(str(row.get("label") or ""))
    if row.get("value") is None and not (memory_row and row.get("group", "memory") == "memory" and _SIZE_IN_TEXT.search(text)) and not supply_text and not temp_text:
        return None
    if temp_text and row.get("value") is None and not supply_text:
        if _QQ_STORAGE.search(text):
            return None
        if _QQ_JUNCTION.search(text):
            return "junction"
        if _QQ_AMBIENT.search(text):
            return "ambient"
        return None
    if supply_text and row.get("value") is None:
        if _QQ_ABS_MAX.search(text):
            return "absolute_maximum"
        if _QQ_RATED.search(text) or re.search(r"\bsupply\b|\boperating\b|\bV(?:DD|CC)\b", text, re.I):
            return "rated"
        return None
    if memory_row:
        if _QQ_EEPROM.search(text) and re.search(r"emulat|high[- ]cycl|to use for data|use as|used as", text, re.I):
            return "data_flash"  # flash region used as EEPROM (ST "high cycling ... (EEPROM emulation)")
        if _QQ_EEPROM.search(text):
            # Its own kind (CR: eeprom_kb is a dictionary attribute); emulated EEPROM
            # is data flash. A prose row that also sizes flash or SRAM carries more
            # than one quantity, so no single qualifier is honest.
            if re.search(r"\bflash\b|\bs?ram\b", text, re.I):
                return None
            return "eeprom"
        if _QQ_DATA_FLASH.search(text):
            return "data_flash"
        if _QQ_CODE_FLASH.search(text):
            return "code_flash"
        if re.search(r"\bs?ram(?:\s?\d|[A-D]\b|\b)", text, re.I) and not re.search(r"\bflash\b", text, re.I):
            if _QQ_SRAM_BANK.search(text):
                return "sram_bank"
            if _QQ_SRAM_SPECIAL.search(text):
                return None
            return "total_sram"
        # Bare "Flash": the document did not say which. CR (renesas-correction-
        # qualifier-ack-20260908): the null is wanted, not a guess; ST is
        # disambiguated from its own parametric export on their side.
        return None
    if unit in ("°c", "ºc", "c") or section == "temperature_range":
        if _QQ_STORAGE.search(text):
            return None
        if _QQ_JUNCTION.search(text):
            return "junction"
        if _QQ_AMBIENT.search(text):
            return "ambient"
        return None
    if unit == "v" or section == "supply_range":
        if _QQ_ABS_MAX.search(text):
            return "absolute_maximum"
        if _QQ_RATED.search(text):
            return "rated"
        return None
    if unit in ("mhz", "khz", "ghz"):
        if row.get("qualifier_verbatim") and _QQ_MAXIMUM.search(row["qualifier_verbatim"]):
            return "maximum"
        if _QQ_TYPICAL.search(text):
            return "typical"
        if _QQ_MINIMUM.search(text):
            return "minimum"
        if _QQ_MAXIMUM.search(text):
            return "maximum"
        return None
    return None


def _sram_bank_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Banks must sum to the stated total; if they do not, the total row is
    refused (moved below the grid with a reason) rather than reconciled."""
    totals = [r for r in rows if r.get("quantity_qualifier") == "total_sram" and r["tier"] == "grid" and (r.get("unit") or "").lower() == "kb"]
    banks = [r for r in rows if r.get("quantity_qualifier") == "sram_bank" and r["tier"] == "grid" and (r.get("unit") or "").lower() == "kb" and isinstance(r.get("value"), (int, float))]
    if not totals or not banks:
        return {"sram_banks_checked": False}
    bank_sum = sum(float(r["value"]) for r in {r["label"]: r for r in banks}.values())
    result: dict[str, Any] = {"sram_banks_checked": True, "sram_bank_sum_kb": bank_sum, "sram_totals_kb": sorted({float(r["value"]) for r in totals})}
    matched = any(abs(float(t["value"]) - bank_sum) < 0.5 for t in totals)
    result["sram_banks_sum_to_total"] = matched
    if not matched:
        for t in totals:
            t["tier"] = "below_grid"
            t["refused"] = "sram_banks_do_not_sum_to_total"
    return result


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same fact printed on several pages is one row with all its pages.
    Key: group, tier, label normalised (case, ®/™, spacing, hyphen-vs-space).
    Then, per group, a bare prose number (e.g. "up to 48 MHz") is dropped
    when a vendor bullet in the same group already carries that value+unit."""
    merged: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        key = (row["group"], row["tier"], _label_key(row["label"]))
        if key in merged and merged[key].get("instances") is not None and row.get("instances") is not None and merged[key]["instances"] != row["instances"]:
            key = key + (row["instances"],)  # "16 digital comparators" and "8 digital comparators" are two statements
        if key in merged:
            keep = merged[key]
            keep["source_pages"] = sorted(set(keep["source_pages"]) | set(row["source_pages"]))[:12]
            if keep.get("instances") is None and row.get("instances") is not None:
                keep["instances"] = row["instances"]
            continue
        merged[key] = dict(row)
    out = list(merged.values())
    valued = {(r["group"], json_value(r["value"]), (r["unit"] or "").lower()) for r in out if r["value"] is not None and r.get("section") not in ("max_frequency", "supply_range", "temperature_range")}
    final = []
    for row in out:
        if row.get("section") in ("max_frequency", "supply_range", "temperature_range") and (row["group"], json_value(row["value"]), (row["unit"] or "").lower()) in valued:
            continue
        final.append(row)
    return _prefer_specific(final)


_STOP_TOKENS = {"module", "modules", "with", "and", "the", "of", "a", "an", "for", "x", "×", "interface", "interfaces", "controller", "controllers", "unit", "units", "core", "cpu", "bus", "up", "to"}
_SOURCE_RANK = {"features": 3, "features_heading": 3, "memory": 2, "chapter": 1}


def _tokens(label: str) -> frozenset[str]:
    return frozenset(t for t in re.split(r"[^a-z0-9.]+", _label_key(label)) if t and t not in _STOP_TOKENS)


def _prefer_specific(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Within one group and tier, when two rows of the same peripheral class
    say the same thing (one label's tokens inside the other's, or equal token
    sets), keep the more specific label, carry the count and pages, and park
    the discarded label in `also_printed`. Different classes never merge."""
    by_bucket: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        by_bucket.setdefault((row["group"], row["tier"], row.get("peripheral_class")), []).append(row)
    out: list[dict[str, Any]] = []
    for bucket in by_bucket.values():
        if len(bucket) == 1:
            out.extend(bucket)
            continue
        toks = [_tokens(r["label"]) for r in bucket]
        keep = [True] * len(bucket)
        for i, a in enumerate(bucket):
            for j, b in enumerate(bucket):
                if i == j or not keep[i] or not toks[i]:
                    continue
                # Two counts or two values that disagree are two facts
                # ("16-bit timers ×2" and "general purpose 16-bit timers plus
                # one PWM timer ×3"), never one.
                if a.get("instances") is not None and b.get("instances") is not None and a["instances"] != b["instances"]:
                    continue
                if a.get("value") is not None and b.get("value") is not None and json_value(a["value"]) != json_value(b["value"]):
                    continue
                if toks[i] < toks[j] or (toks[i] == toks[j] and _specificity(a) < _specificity(b)) or (toks[i] == toks[j] and _specificity(a) == _specificity(b) and i > j):
                    keep[i] = False
                    b.setdefault("also_printed", [])
                    if a["label"] not in b["also_printed"]:
                        b["also_printed"].append(a["label"])
                    b["source_pages"] = sorted(set(b["source_pages"]) | set(a["source_pages"]))[:12]
                    if b.get("instances") is None and a.get("instances") is not None:
                        b["instances"] = a["instances"]
                    if b.get("value") is None and a.get("value") is not None:
                        b["value"], b["unit"] = a["value"], a["unit"]
                    if not b.get("qualifier_verbatim") and a.get("qualifier_verbatim"):
                        b["qualifier_verbatim"] = a["qualifier_verbatim"]
                    break
        out.extend(r for r, k in zip(bucket, keep) if k)
    return out


def _specificity(row: dict[str, Any]) -> tuple[int, int]:
    return (_SOURCE_RANK.get(row.get("section") or "", 0), len(row["label"]))


def json_value(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True)


def _vendor_stem(names: list[str], icls: str) -> str:
    """"SPI1, SPI2, SPI6" -> "SPI"; "OTG_FS, OTG_HS" -> "OTG_FS / OTG_HS"; "GPIOA..GPIOK" -> "GPIO"."""
    stems = []
    for name in names:
        stem = re.sub(r"(?<=[A-Za-z])[0-9]{1,2}$|(?<=GPIO)[A-Z]$|(?<=GIO)[A-H]$|(?<=SCI)[A-D]$|(?<=ADC)[A-D]$|(?<=DAC)[A-D]$", "", name)
        if stem not in stems:
            stems.append(stem)
    return " / ".join(stems[:4]) if len(stems) <= 4 else icls


def _label_key(label: str) -> str:
    # "UARTs" and "UART", "I2Cs" and "I2C": one key. Only acronym plurals;
    # "timers" stays distinct from "timer" as printed.
    label = re.sub(r"\b([A-Z][A-Z0-9]{1,7})s\b", r"\1", label)
    text = re.sub(r"[®™©]", "", label.lower())
    text = re.sub(r"[\s\-–_]+", " ", text)
    return re.sub(r"[^a-z0-9 ./()+]", "", text).strip()


_NON_SCOPE = re.compile(r"^(?:\d{1,2}-?bit\s+)?(?:mcu|mcus|bit mcu|microcontrollers?|group|series|family|arm|based)$", re.I)


def _scope_as_printed(identity: dict[str, Any]) -> str:
    """Bind-to-catalogue token: the vendor's group token first ("RA4C1"), then
    wildcards, then concrete parts, then a family phrase with a designator,
    then the filename, then the title. Generic words are never a scope."""
    lines = identity["lines_covered"]
    for candidates in (lines.get("group_tokens", []), lines["wildcards"], lines["parts"]):
        good = [c for c in candidates if not _NON_SCOPE.match(c)]
        if good:
            return ", ".join(good[:8])
    for phrase in lines.get("family_phrases", []):
        if not _NON_SCOPE.match(phrase) and re.search(r"\d", phrase):
            return phrase
    if lines.get("filename_tokens"):
        return lines["filename_tokens"][0]
    title = identity.get("pdf_title") or identity.get("title_verbatim") or ""
    return "" if _NON_SCOPE.match(title.strip()) else title


VENDOR_DOMAIN: dict[str, str] = {
    "ti": "ti.com", "texas instruments": "ti.com", "st": "st.com", "stmicroelectronics": "st.com", "stm": "st.com",
    "microchip": "microchip.com", "atmel": "microchip.com", "renesas": "renesas.com", "nxp": "nxp.com", "freescale": "nxp.com",
    "gigadevice": "gigadevice.com", "gd": "gigadevice.com", "silabs": "silabs.com", "silicon labs": "silabs.com",
    "infineon": "infineon.com", "cypress": "infineon.com", "nuvoton": "nuvoton.com", "espressif": "espressif.com",
    "adi": "analog.com", "analog devices": "analog.com", "ambiq": "ambiq.com", "nordic": "nordicsemi.com",
}
_VENDOR_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:RM|UM|PM|DS|AN|ES)\d{4}$|^STM32|^STM8|^rm\d{4}|^(?:L\d{4}|L\d{4}[A-Z]?|ST1S\d+|STSPIN\w*|STM|VIPER|L6\d{3}|L7\d{3}|L5\d{3}|L9\d{3})\b", re.I), "st.com"),
    (re.compile(r"^(?:SLAU|SLAS|SPRU|SPRS|SPNU|SPNS|SPNZ|SLVS|SLLS|SPMU|SPMS|SPRUI|SPRUH|SPRUG|SPRUF|SLAA|SWRU|SWRS|TIDU)[A-Z0-9]+|^(?:TMS320|TMS570|MSP430|MSPM0|MSPM33|TM4C|AM2|AM6|RM4|RM5|CC\d{4}|Tiva|Concerto|F28M|Jacinto|PRU\b)", re.I), "ti.com"),
    (re.compile(r"^R01(?:UH|DS|AN|TU)\d{4}|^REN_|^(?:RA\d|RX\d|RL78|RZ|RH850|RE01|R7F|R5F|DA1\d{4})", re.I), "renesas.com"),
    (re.compile(r"^DS\d{8}[A-Z]?$|^DS\d{5}[A-Z]$|^(?:PIC|dsPIC|AT(?:SAM|mega|tiny|xmega)|SAM[A-Z]\d|SAM9|CEC1\d{2}|MEC1\d{2}|AVR)", re.I), "microchip.com"),
    (re.compile(r"^GD32", re.I), "gigadevice.com"),
    (re.compile(r"^(?:EFM32|EFR32|EFM8|C8051|SiM3|Si\d{4}|BGM|MGM|EZR32)", re.I), "silabs.com"),
    (re.compile(r"^(?:MK|MKL|MKV|MKE|MKW|LPC|MIMXRT|IMXRT|i\.?MX|MCX|S32|KL\d|K\d{2}P\d{2,3}M|KE\d|KV\d|KW\d|KBTLDR|DRM\d{3}|FT10RM|Kinetis|Freescale|QN9|JN5|RW61)", re.I), "nxp.com"),
    (re.compile(r"^(?:XMC|PSoC|PSOC|CY8C|CYW|TLE|AURIX|TC2|TC3|TC4|infineon|infns)", re.I), "infineon.com"),
    (re.compile(r"^nRF\d{4,5}", re.I), "nordicsemi.com"),
)


def normalize_vendor(vendor: str | None) -> str | None:
    if not vendor:
        return None
    key = vendor.strip().lower()
    if "." in key:
        return key
    return VENDOR_DOMAIN.get(key, key)


def infer_vendor(document_id: str | None, scope: str, artifact: str) -> str | None:
    for probe in (document_id or "", scope.split(",")[0].strip(), artifact.split(".")[0]):
        for pattern, domain in _VENDOR_HINTS:
            if probe and pattern.search(probe):
                return domain
    return None


def _package_or_temp(text: str) -> bool:
    return bool(re.search(r"\b(?:LQFP|QFN|BGA|CSP|TSSOP|SOIC|DIP|WLCSP|UFBGA|TFBGA|packages?|pins?|leads?|°C|ºC|I/Os?|GPIOs?)\b", text))


def _power_fact(text: str) -> bool:
    return bool(re.search(r"\b(?:V|mV|µA|μA|uA|mA|nA)\b|(?<=\d\s)(?:µA|μA)|/MHz|supply|standby|wake-?up|\bPLL\b|reset|LVD|BOR|POR|\bcurrent\b|\bEM[0-4]\b|energy mode|dc-?dc|\bLDO\b|regulator", text, re.I))
