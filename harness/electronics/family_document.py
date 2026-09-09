"""Family-level document reader: the common characteristics of a line.

A reference manual, technical reference manual, hardware/user manual or
family data sheet describes every device of a family at once. This module
reads such a document in full and reports what it says about the family, at
the grain the cover states, with receipts. It does not resolve anything to
an orderable part; that is the knife owner's decision.

Four deterministic sources, all PyMuPDF, no model:

identity      cover page: document id (RM0481, SLAU846, DS70616), title, the
              lines it covers as printed, revision.
chapters      the PDF bookmark outline (fallback: printed table of contents)
              normalised to a peripheral vocabulary: which peripheral classes
              the family has, and where each chapter is.
instances     every peripheral instance name the document uses (TIM1..TIM17,
              SPI1..SPI6, USART1, LPUART1, FDCAN2, ADC3, GPIOK, DMA2,
              USB_OTG_HS), with the number of pages that name it. A name used
              on many pages is an instance the family has; a name on one page
              is quoted, not asserted, and stays in the record as weak.
features      the bullet features list on the first pages, each bullet kept
              verbatim with the numbers it carries typed and its qualifier
              ("Up to", "max") kept as printed.
memory        "Memory map"/"Flash memory organisation" style rows with a size:
              region name, size, verbatim.

Everything numeric carries `qualifier_verbatim` (or null) and a receipt
{page, verbatim}. Nothing is expanded to members and nothing is withheld.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf

from harness.electronics.family_census import (
    is_concrete_part_token,
    is_wildcard_token,
    parts_from_text,
    wildcards_from_text,
)

SCHEMA = "harness.electronics-family-document.v1"

# Document ids by vendor house style.
_DOC_ID = re.compile(
    r"\b(RM\d{4}|UM\d{4}|PM\d{4}|DS\d{4,5}|AN\d{4}|ES\d{4}|"          # ST
    r"SLAU\d{3}[A-Z]?|SLAS\d{3}[A-Z]?|SPRU[A-Z0-9]{3,4}|SPRS[A-Z0-9]{3,4}|SPNU\d{3}[A-Z]?|SPNS\d{3}[A-Z]?|SLVS[A-Z0-9]{3,4}|SLLS[A-Z0-9]{3,4}|"  # TI
    r"DS\d{8}[A-Z]?|DS\d{5}[A-Z]|"                                     # Microchip
    r"R01UH\d{4}[A-Z]{2}\d{4}|R01DS\d{4}[A-Z]{2}\d{4}|R01AN\d{4}[A-Z]{2}\d{4}|"  # Renesas
    r"[A-Z]{2,6}\d{2,4}RM|[A-Z]{2,6}\d{2,4}P\d{2,3}M\d{2,3}[A-Z]{2}\d{1,2}RM)\b"  # NXP KL43P64M48SF6RM
)
_REVISION = re.compile(r"\b(?:Rev(?:ision)?\.?\s*([0-9]+(?:\.[0-9]+)?[A-Z]?)|Version\s*([0-9]+(?:\.[0-9]+)?))\b", re.I)

# Peripheral vocabulary: canonical class -> chapter-title regex. A title is
# tried against CHAPTER_EXCLUDE[class] first; a hit there blocks the class.
CHAPTER_EXCLUDE: dict[str, re.Pattern[str]] = {
    "cpu_core": re.compile(r"\btimers?\b|arbitr|\bcpu (?:self-test|compare)|\bfilter\b", re.I),
    "comparator": re.compile(r"\bclock\b|\bbus\b|\bdcc\b|\bwindow\b", re.I),
    "trustzone": re.compile(r"\btrip", re.I),
    "sram": re.compile(r"\bmessage ram\b|\bprotection\b|\bmemory ram\b", re.I),
    "spi": re.compile(r"\bexport\b|\bclb\b", re.I),
    "timer": re.compile(r"\bwatchdog\b|\brtc\b|\breal-time clock", re.I),
    "gpio": re.compile(r"\bsysconfig\b|\btool", re.I),
    "flash": re.compile(r"\bprotection\b|\berase/program", re.I),
    "usb": re.compile(r"\bsignals\b", re.I),
}
CHAPTER_VOCAB: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("memory_map", re.compile(r"\bmemory (?:map|organi[sz]ation)\b|\bmemory and bus", re.I)),
    ("flash", re.compile(r"\b(?:embedded )?flash\b|\bnvm\b|\bprogram memory\b", re.I)),
    ("sram", re.compile(r"\bs?ram\b(?!p)", re.I)),
    ("ethernet", re.compile(r"\bethernet|\bemac\b|\beth\b", re.I)),
    ("usb", re.compile(r"\busb\b|\botg\b|\bucpd|\busb type-c", re.I)),
    ("sdmmc", re.compile(r"\bsdmmc|\bsdio\b|\bsd/mmc|\bemmc|\bmmc\b", re.I)),
    ("xspi", re.compile(r"\bquad-?spi|\bqspi|\bocto-?spi|\bospi|\bxspi|\bhexadeca|\bhyperbus", re.I)),
    ("fmc", re.compile(r"\bfmc\b|\bfsmc|\bflexible (?:static )?memory controller|\bemif\b|\bexternal memory interface|\bebi\b", re.I)),
    ("can", re.compile(r"\bcan\b|\bfdcan|\bbxcan|\bmcan|\bdcan|\becan|\bcontroller area network", re.I)),
    ("lin", re.compile(r"\blin\b", re.I)),
    ("i3c", re.compile(r"\bi3c\b", re.I)),
    ("i2c", re.compile(r"\bi2c|\bi²c|\binter-integrated", re.I)),
    ("usart", re.compile(r"\busart|\buart|\blpuart|\bsci\b|\bserial communication interface", re.I)),
    ("spi", re.compile(r"\bspi\b|\bserial peripheral interface|\bmibspi|\bssi\b|\bi2s\b", re.I)),
    ("display", re.compile(r"\bltdc|\blcd\b|\bdsi\b|\bdisplay|\bgfxmmu|\bdma2d|\bchrom-?art|\bgpu\b|\bneochrom", re.I)),
    ("dcmi", re.compile(r"\bdcmi|\bcamera|\bpssi\b|\bcsi\b", re.I)),
    ("audio", re.compile(r"\bsai\b|\bspdif|\bdfsdm|\bmdf\b|\badf\b|\bpdm\b", re.I)),
    ("touch", re.compile(r"\btsc\b|\btouch sensing|\bcaptivate", re.I)),
    ("hrtim", re.compile(r"\bhrtim|\bhigh-resolution timer", re.I)),
    ("lptim", re.compile(r"\blow[- ]power timer|\blptim", re.I)),
    ("rtc", re.compile(r"\breal[- ]time clock|\brtc\b", re.I)),
    ("watchdog", re.compile(r"\bwatchdog|\biwdg|\bwwdg|\bwdt\b", re.I)),
    ("adc", re.compile(r"\badc|analog[- ]to[- ]digital", re.I)),
    ("dac", re.compile(r"\bdac|digital[- ]to[- ]analog", re.I)),
    ("opamp", re.compile(r"\bop-?amp|\boperational amplifier|\bpga\b", re.I)),
    ("comparator", re.compile(r"\bcomp(?:arator)?s?\b", re.I)),
    ("crypto", re.compile(r"\baes\b|\bcryp\b|\bhash\b|\bpka\b|\bsaes\b|\brng\b|\btrng|\bcryptograph|\bsecurity\b|\bsecure\b|\bhsm\b|\bmathacl", re.I)),
    ("trustzone", re.compile(r"\btrustzone|\bgtzc\b|\btz\b", re.I)),
    ("npu", re.compile(r"\bnpu\b|\bneural|\bai accelerator|\btinyengine", re.I)),
    ("wireless", re.compile(r"\bradio\b|\bbluetooth|\bble\b|\b802\.15\.4|\bzigbee|\bthread\b|\bwi-?fi|\brf subsystem|\bsub-ghz", re.I)),
    ("safety", re.compile(r"\becc\b|\bcrc\b|\bpbist|\bstc\b|\besm\b|\berror signaling|\bccm-r4|\bself-test|\bfunctional safety", re.I)),
    ("debug", re.compile(r"\bdebug|\bdbg\b|\bswd\b|\bjtag|\btrace|\betm\b|\bcoresight", re.I)),
    ("power", re.compile(r"\bpower (?:control|management|supply)|\bpwr\b|\bpmcu\b|\bpower-saving|low[- ]power modes", re.I)),
    ("reset_clock", re.compile(r"\breset and clock|\brcc\b|\bclock (?:system|tree|configuration|control)|\boscillator|\bsysctl\b|\bclock module", re.I)),
    ("gpio", re.compile(r"\bgpio|\bgeneral[- ]purpose i/?os?|\bi/o ports?|\bport (?:control|i/o)|\biomux|\bpinmux|\bi/o multiplexing", re.I)),
    ("dma", re.compile(r"\bdma\b|\bdirect memory access|\bgpdma|\bmdma|\bbdma|\bdmamux|\bedma", re.I)),
    ("timer", re.compile(r"\btimers?\b|\btim\d|\btimg|\btima|\bgptm|\bepwm|\becap|\beqep|\brti\b|\bhet\b|\bn2het|\boutput compare|\binput capture", re.I)),
    ("interrupts", re.compile(r"\binterrupt|\bnvic\b|\bexti\b|\bvim\b|\bevents?\b", re.I)),
    ("cpu_core", re.compile(r"\b(?:cortex|cpu|processor core|core architecture|arm)\b", re.I)),
)

# Instance grammar: canonical class -> regex whose group(1) is the instance
# designator. Word-bounded, case-sensitive to avoid prose ("can", "spi").
INSTANCE_GRAMMAR: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("TIM", re.compile(r"\bTIM(\d{1,2})\b")),
    ("LPTIM", re.compile(r"\bLPTIM(\d)\b")),
    ("HRTIM", re.compile(r"\bHRTIM(\d)\b")),
    ("TIMG", re.compile(r"\bTIMG(\d{1,2})\b")),
    ("TIMA", re.compile(r"\bTIMA(\d)\b")),
    ("EPWM", re.compile(r"\bEPWM(\d{1,2})\b")),
    ("ECAP", re.compile(r"\bECAP(\d{1,2})\b")),
    ("EQEP", re.compile(r"\bEQEP(\d)\b")),
    ("USART", re.compile(r"\bUSART(\d)\b")),
    ("UART", re.compile(r"\bUART(\d)\b")),
    ("LPUART", re.compile(r"\bLPUART(\d)\b")),
    ("SCI", re.compile(r"\bSCI([A-D]|\d{1,2})\b")),
    # Renesas RX / RA / RL78 house names.
    ("RSPI", re.compile(r"\bRSPI(\d)\b")),
    ("RIIC", re.compile(r"\bRIIC(\d)\b")),
    ("MTU", re.compile(r"\bMTU(\d{1,2})\b")),
    ("TPU", re.compile(r"\bTPU(\d)\b")),
    ("CMT", re.compile(r"\bCMT(\d)\b")),
    ("TMR", re.compile(r"\bTMR(\d)\b")),
    ("GPT32", re.compile(r"\bGPT(?:32|16|32E|32EH)(\d)\b")),
    ("AGT", re.compile(r"\bAGT(\d)\b")),
    ("S12AD", re.compile(r"\bS12AD([A-Z]?\d?)\b")),
    ("RSCAN", re.compile(r"\bRSCAN(\d?)\b")),
    ("DMAC", re.compile(r"\bDMAC(\d)\b")),
    ("DTC", re.compile(r"\b(DTC)\b")),
    ("ELC", re.compile(r"\b(ELC)\b")),
    # TI Hercules house names.
    ("MIBSPI", re.compile(r"\bMib(?:SPI|SPIP)(\d)\b")),
    ("N2HET", re.compile(r"\bN2HET(\d)\b")),
    ("RTI", re.compile(r"\b(RTI)\b")),
    ("GIO", re.compile(r"\bGIO([A-H])\b")),
    ("MIBADC", re.compile(r"\bMibADC(\d)\b")),
    ("LIN", re.compile(r"\bLIN(\d)\b")),
    ("EMAC", re.compile(r"\b(EMAC)\b")),
    ("SPI", re.compile(r"\bSPI(\d)\b")),
    ("I2S", re.compile(r"\bI2S(\d)\b")),
    ("SAI", re.compile(r"\bSAI(\d)\b")),
    ("I2C", re.compile(r"\bI2C(\d)\b")),
    ("I3C", re.compile(r"\bI3C(\d)\b")),
    ("FDCAN", re.compile(r"\bFDCAN(\d)\b")),
    ("CAN", re.compile(r"\b(?:bx)?CAN(\d)\b")),
    ("MCAN", re.compile(r"\bMCAN(\d)\b")),
    ("DCAN", re.compile(r"\bDCAN(\d)\b")),
    ("ADC", re.compile(r"\bADC(\d|[A-D])\b")),
    ("DAC", re.compile(r"\bDAC(\d|[A-D])\b")),
    ("COMP", re.compile(r"\bCOMP(\d)\b")),
    ("OPAMP", re.compile(r"\bOPAMP(\d)\b")),
    ("DMA", re.compile(r"\b(?:GP|B|M)?DMA(\d)\b")),
    ("GPIO", re.compile(r"\bGPIO([A-Z])\b")),
    ("USB", re.compile(r"\b(USB_OTG_FS|USB_OTG_HS|USB_DRD_FS|OTG_FS|OTG_HS|USBFS|USBHS)\b")),
    ("ETH", re.compile(r"\bETH(\d)\b")),
    ("SDMMC", re.compile(r"\bSDMMC(\d)\b")),
    ("OCTOSPI", re.compile(r"\b(?:OCTO|X|HSPI|QUAD)SPI(\d)\b")),
    ("FMC", re.compile(r"\b(FMC|FSMC)\b")),
    ("RTC", re.compile(r"\b(RTC)\b")),
    ("IWDG", re.compile(r"\b(IWDG|WWDG|WDT\d?)\b")),
    ("AES", re.compile(r"\b(AES|SAES|HASH|PKA|RNG|CRYP)\b")),
    ("LTDC", re.compile(r"\b(LTDC|DSI|DMA2D|GFXMMU|DCMI|PSSI)\b")),
    ("UCPD", re.compile(r"\bUCPD(\d)\b")),
    ("TSC", re.compile(r"\b(TSC)\b")),
)

_BULLET = re.compile(r"^\s*(?:[•·▪◦■●\-–]|)\s*(.+)$")
_QUALIFIER = re.compile(r"\b(up\s+to|maximum|max\.?|minimum|min\.?|typical|typ\.?|at\s+least|as\s+low\s+as|down\s+to)\b", re.I)
_NUM_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(Mbytes?|Kbytes?|MB|KB|Kbit|Mbit|MHz|kHz|GHz|bits?|bytes?|channels?|V|mA|µA|uA|nA|°C|ms|µs|us|ns|I/Os?|pins?|x|×)?\b",
    re.I,
)
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(Mbytes?|Kbytes?|MB|KB|Kbit|Mbit|bytes?)\b", re.I)
_MEMORY_ROW = re.compile(r"^(?P<name>[A-Za-z][\w\s/().-]{1,40}?)\s+(?P<size>\d+(?:\.\d+)?\s*(?:Mbytes?|Kbytes?|MB|KB|Kbit|Mbit|bytes?))\b", re.I)


def _norm(text: str) -> str:
    return " ".join(text.split())


def read_identity(document: Any, page_texts: dict[int, str]) -> dict[str, Any]:
    """Cover-page facts: id, title, lines covered as printed, revision."""
    cover = page_texts.get(1) or ""
    second = page_texts.get(2) or ""
    doc_id = None
    for text in (cover, second):
        match = _DOC_ID.search(text)
        if match:
            doc_id = match.group(1)
            break
    revision = None
    match = _REVISION.search(cover) or _REVISION.search(second)
    if match:
        revision = match.group(1) or match.group(2)
    lines = cover.split("\n")
    title_lines = [_norm(l) for l in lines[:40] if _norm(l) and len(_norm(l)) > 3]
    # Lines covered: wildcards and concrete parts printed on the cover.
    covered_tokens = wildcards_from_text(cover)
    covered_parts = [p for p in parts_from_text(cover, None) if p != doc_id and not _DOC_ID.fullmatch(p)]
    # Microchip-style masks: dsPIC33EPXXX(GP/MC/MU)806/810/814, PIC24EPXXX(GP/GU)810/814.
    covered_tokens += [m for m in re.findall(r"\b(?:ds)?PIC\d{2}[A-Z]{1,3}X{2,3}(?:\([A-Z/]+\))?[0-9/]+", cover) if m not in covered_tokens]
    # Family words like "STM32H5 series", "MSPM0 G-Series", "RA2E1 Group", "TMS320F28P65x Real-Time Microcontrollers".
    family_phrases = [
        p for p in re.findall(
            r"\b([A-Z][A-Za-z0-9]{1,12}(?:[ -][A-Za-z0-9]{1,10}){0,2}[ -](?:series|family|families|group|line|MCUs?|microcontrollers))\b",
            cover,
            re.I,
        )
        if not _generic_scope(p)
    ]
    # "RA4C1 Group", "RX231 Group", "RL78/G13", "MSPM0 G-Series": the vendor's
    # own family token, the thing a document is bound to.
    group_tokens: list[str] = []
    for match in re.finditer(r"\b((?:[A-Z]{1,5}\d[A-Z0-9]{1,8})|RL78/[A-Z]\d{1,2}[A-Z]?)\s+(?:Group|Series|Family|Line)\b", cover):
        token = match.group(1)
        if token not in group_tokens and not _DOC_ID.fullmatch(token):
            group_tokens.append(token)
    # Filename as last resort: ra_ra4c1.pdf -> RA4C1, RA4W1_Group_Datasheet.pdf -> RA4W1.
    stem = Path(getattr(document, "name", "") or "").stem
    filename_tokens = [t.upper() for t in re.findall(r"(?i)\b([a-z]{1,5}\d[a-z0-9]{1,8})\b", stem.replace("_", " ").replace("-", " ")) if not _DOC_ID.fullmatch(t.upper())]
    metadata = document.metadata or {}
    return {
        "document_id": doc_id,
        "title_verbatim": " ".join(title_lines[:6])[:300],
        "pdf_title": metadata.get("title") or None,
        "revision": revision,
        "lines_covered": {
            "wildcards": covered_tokens[:40],
            "parts": covered_parts[:80],
            "group_tokens": group_tokens[:8],
            "family_phrases": sorted(set(_norm(p) for p in family_phrases))[:12],
            "filename_tokens": filename_tokens[:4],
        },
        "receipt": {"page": 1},
    }


_GENERIC_SCOPE_WORDS = {"bit", "mcu", "mcus", "microcontroller", "microcontrollers", "group", "series", "family", "families", "line", "arm", "based", "advanced", "renesas", "the", "of", "and", "32", "16", "8", "risc", "flash", "real", "time", "dual", "core", "wireless", "soc", "socs"}


def _generic_scope(phrase: str) -> bool:
    """"32-Bit MCU", "Renesas RA Family", "family of microcontrollers": not a scope."""
    tokens = [t for t in re.split(r"[\s\-/]+", phrase.lower()) if t]
    return all(t in _GENERIC_SCOPE_WORDS for t in tokens) or not any(re.search(r"\d", t) for t in tokens)


def read_chapters(document: Any, page_texts: dict[int, str]) -> dict[str, Any]:
    """Chapter list from bookmarks, normalised to the peripheral vocabulary."""
    toc = document.get_toc() or []
    top = [(lvl, _norm(title), page) for lvl, title, page in toc if lvl <= 2 and title and page > 0]
    source = "bookmarks"
    if len(top) < 5:
        # Printed table of contents fallback: "3 CPU .......... 430" lines.
        source = "printed_toc"
        top = []
        for pno in range(1, min(document.page_count, 30) + 1):
            for line in (page_texts.get(pno) or "").split("\n"):
                match = re.match(r"^\s*(\d{1,2}(?:\.\d{1,2})?)\s+([A-Z][^.]{3,80}?)\s*\.{3,}\s*(\d{1,4})\s*$", line)
                if match:
                    level = 1 if "." not in match.group(1) else 2
                    top.append((level, _norm(match.group(2)), int(match.group(3))))
    classes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for level, title, page in top:
        name = classify_chapter_title(title)
        if name:
            classes[name].append({"title": title, "page": page, "level": level})
    return {
        "source": source,
        "entries": len(top),
        "chapters": [{"level": l, "title": t, "page": p} for l, t, p in top if l == 1][:120],
        "all_entries": [{"level": l, "title": t, "page": p} for l, t, p in top],
        "peripheral_classes": {k: v[:12] for k, v in sorted(classes.items())},
    }


def classify_chapter_title(title: str) -> str | None:
    for name, pattern in CHAPTER_VOCAB:
        if pattern.search(title):
            exclude = CHAPTER_EXCLUDE.get(name)
            if exclude and exclude.search(title):
                continue
            return name
    return None


_FEATURES_HEADING = re.compile(
    r"^\s*(?:\d{1,2}(?:\.\d{1,2}){0,2}\s+)?(?P<subject>[A-Za-z][A-Za-z0-9/™®+\- ]{0,60}?)\s*(?:main\s+|key\s+|general\s+)?features\s*$",
    re.I | re.M,
)


def read_chapter_features(page_texts: dict[int, str], chapters: dict[str, Any], page_count: int) -> list[dict[str, Any]]:
    """Bullet lists under any "<subject> [main] features" heading anywhere in
    the document, tagged with the enclosing chapter. This is where a reference
    manual states what each peripheral does for the whole family."""
    entries = sorted(chapters.get("all_entries", []), key=lambda e: e["page"])
    top_level = [e for e in entries if e["level"] == 1]

    def enclosing(pno: int) -> dict[str, Any] | None:
        best = None
        for entry in top_level:
            if entry["page"] <= pno:
                best = entry
            else:
                break
        return best

    out: list[dict[str, Any]] = []
    seen_pages: set[int] = set()
    for pno in sorted(page_texts):
        text = page_texts[pno]
        heading = _FEATURES_HEADING.search(text)
        if not heading or pno in seen_pages:
            continue
        subject = _norm(heading.group("subject") or "")
        if len(subject) > 48 or re.search(r"\b(?:refer|see|for|the|this|these|following|with|of)\b", subject, re.I):
            continue  # a sentence ending in "features", not a heading
        chapter = enclosing(pno)
        chapter_class = classify_chapter_title(chapter["title"]) if chapter else None
        after = text[heading.end():]
        nxt = page_texts.get(pno + 1) or ""
        bullets = _bullet_items(after) + _bullet_items(nxt[:4000])
        if not bullets:
            continue
        seen_pages.update({pno, pno + 1})
        for item in bullets[:60]:
            row = _feature(item["text"], pno, level=item.get("level"), parent=item.get("parent"))
            row["section"] = subject or None
            row["chapter"] = chapter["title"] if chapter else None
            row["chapter_class"] = chapter_class or (classify_chapter_title(subject) if subject else None)
            out.append(row)
    return out[:3000]


# Bullet glyphs vendors use, including the private-use-area glyphs Wingdings
# and Symbol fonts leave in extracted text (Renesas \uf0b7, Microchip \uf0a7).
_GLYPH = r"[•·▪◦■●♦◆►▶➢➤✓\u2022\u2023\u25aa\u25ab\u25e6\u2043\u2219\uf000-\uf0ff]"
_GLYPH_LINE = re.compile(rf"^\s*({_GLYPH})\s*(.*)$")
_DASH_LINE = re.compile(r"^\s*[–\-]\s+(\S.*)$")
_COUNT_SUFFIX = re.compile(r"\s*(?:[×x]\s*(\d{1,2})|(\d{1,2})\s*[×x]|\((\d{1,2})\s*(?:channels?|units?|modules?|ch)\))\s*(\(.*\))?\s*$", re.I)


def _bullets(text: str) -> list[str]:
    """Bullet texts (flat) from a block; kept for callers that only need text."""
    return [item["text"] for item in _bullet_items(text)]


def _bullet_items(text: str) -> list[dict[str, Any]]:
    """Structured bullets from a text block: {text, section, level}.

    Marker hierarchy is learned per block: when two distinct glyphs appear,
    the one seen first is the section header level (Renesas "■ Memory" over
    "\uf0b7 512-KB code flash"); a dash line is a sub-bullet. A glyph alone on
    its line takes the following line as its text. Parsing stops at the next
    numbered heading."""
    lines = [l.rstrip() for l in text.split("\n")]
    glyph_order: list[str] = []
    for line in lines:
        match = _GLYPH_LINE.match(line)
        if match and match.group(1) not in glyph_order:
            glyph_order.append(match.group(1))
    header_glyph = glyph_order[0] if len(glyph_order) >= 2 else None

    items: list[dict[str, Any]] = []
    section: str | None = None
    current: dict[str, Any] | None = None
    pending_glyph: str | None = None

    def close() -> None:
        nonlocal current
        if current and current["text"].strip():
            current["text"] = _norm(current["text"])
            items.append(current)
        current = None

    for line in lines:
        if not line.strip():
            continue
        if re.match(r"^\s*\d{1,2}(?:\.\d{1,2}){1,2}\s+[A-Z]", line) and (current is not None or items):
            break
        match = _GLYPH_LINE.match(line)
        if match:
            glyph, rest = match.group(1), match.group(2).strip()
            if glyph == header_glyph:
                close()
                section = rest or None
                pending_glyph = "header" if not rest else None
                continue
            close()
            current = {"text": rest, "section": section, "level": 1}
            pending_glyph = None if rest else "bullet"
            continue
        dash = _DASH_LINE.match(line)
        if dash and (current is not None or items):
            parent = current["text"] if current else (items[-1]["text"] if items else None)
            close()
            current = {"text": dash.group(1).strip(), "section": section, "level": 2, "parent": _norm(parent) if parent else None}
            continue
        if pending_glyph == "header":
            section = _norm(line)
            pending_glyph = None
            continue
        if current is not None and len(line) < 90 and not line.strip().isupper():
            current["text"] = f"{current['text']} {line.strip()}"
    close()
    return items


def split_count(text: str) -> tuple[str, int | None]:
    """"Serial Communications Interface (SCI) × 4" -> ("Serial Communications Interface (SCI)", 4)."""
    match = _COUNT_SUFFIX.search(text)
    if not match:
        return text, None
    count = int(match.group(1) or match.group(2) or match.group(3))
    label = text[: match.start()].rstrip()
    if match.group(4):
        label = f"{label} {match.group(4)}"
    return label, count


def read_instances(page_texts: dict[int, str], weak_pages: int = 2) -> dict[str, Any]:
    """Every peripheral instance name the document uses, with page support."""
    support: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for pno, text in page_texts.items():
        for cls, pattern in INSTANCE_GRAMMAR:
            for match in pattern.finditer(text):
                support[cls][match.group(1)].add(pno)
    out: dict[str, Any] = {}
    for cls in sorted(support):
        instances = []
        for name, pages in support[cls].items():
            bare = cls in ("USB", "FMC", "RTC", "IWDG", "AES", "LTDC", "TSC", "DTC", "ELC", "RTI", "EMAC")
            instances.append({"instance": name if (bare or name.startswith(cls)) else f"{cls}{name}", "pages": len(pages), "first_page": min(pages), "weak": len(pages) < weak_pages})
        instances.sort(key=lambda i: (i["weak"], _natural(i["instance"])))
        strong = [i for i in instances if not i["weak"]]
        out[cls] = {"count_asserted": len(strong), "instances": instances[:64]}
    return out


def _natural(text: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", text))


def _typed_numbers(text: str) -> list[dict[str, Any]]:
    numbers = []
    for match in _NUM_UNIT.finditer(text):
        value = float(match.group(1))
        unit = (match.group(2) or "").strip() or None
        if unit is None and not re.search(r"\d", text.replace(match.group(1), "", 1)):
            pass
        numbers.append({"value": int(value) if value.is_integer() else value, "unit": unit})
    return numbers[:8]


def read_features(page_texts: dict[int, str], max_pages: int = 6) -> list[dict[str, Any]]:
    """Bullet features on the first pages, verbatim, numbers typed, qualifier as printed."""
    out: list[dict[str, Any]] = []
    for pno in range(1, min(max_pages, max(page_texts) if page_texts else 0) + 1):
        text = page_texts.get(pno) or ""
        if not re.search(r"\bfeatures\b", text, re.I) and pno > 2:
            continue
        # A "Features" heading on the page starts the list; otherwise the whole page.
        heading = re.search(r"^\s*(?:key\s+|device\s+|product\s+)?features\s*$", text, re.I | re.M)
        block = text[heading.end():] if heading else text
        for item in _bullet_items(block):
            out.append(_feature(item["text"], pno, section=item.get("section"), level=item.get("level"), parent=item.get("parent")))
    return out[:400]


def _feature(text: str, page: int, *, section: str | None = None, level: int | None = None, parent: str | None = None) -> dict[str, Any]:
    text = _norm(text)
    qualifier = _QUALIFIER.search(text)
    label, count = split_count(text)
    subject = f"{section} {label}" if section else label
    row = {
        "verbatim": text[:300],
        "label": label[:200],
        "count": count,
        "numbers": _typed_numbers(label),
        "qualifier_verbatim": qualifier.group(1) if qualifier else None,
        "classes": [name for name, pattern in CHAPTER_VOCAB if pattern.search(label) and not (CHAPTER_EXCLUDE.get(name) and CHAPTER_EXCLUDE[name].search(label))][:4],
        "receipt": {"page": page},
    }
    if section is not None:
        row["vendor_section"] = section
        row["section_class"] = classify_chapter_title(section)
    if level is not None:
        row["level"] = level
    if parent is not None:
        row["parent"] = parent[:200]
    return row


_CORE = re.compile(
    r"\b(?:Arm|ARM)\W{0,3}\s*(Cortex\W{0,3}[- ]?[MRA]\d{1,2}\+?(?:F|D)?(?:\s+(?:with|and)\s+(?:FPU|DSP|MPU)(?:\s+and\s+(?:FPU|DSP|MPU))?)?)"
    r"|\b(Cortex\W{0,3}[- ]?[MRA]\d{1,2}\+?(?:F|D)?)"
    r"|\b(RISC-V(?:\s+[A-Z0-9]{2,8})?|C28x|TMS320C28x|RXv[1-3] core|RL78 CPU core|CIP-51|Xtensa(?:\s+LX\d)?|AVR®?(?:\s+CPU)?|dsPIC33E|PIC24E|MIPS32®? [A-Za-z0-9]+|e200z\d|8051|Tricore™?|PowerPC)\b"
)
_MAX_FREQ = re.compile(r"\b(?:(up\s+to|max(?:imum)?(?:\s+of)?|maximum\s+frequency\s+of)\s*(\d{1,3}(?:\.\d)?)\s*MHz|(\d{1,3}(?:\.\d)?)\s*MHz\s+(max(?:imum)?(?:\s+(?:operating\s+)?frequency)?))", re.I)
_SUPPLY = re.compile(r"\b(\d\.\d{1,2})\s*V?\s*(?:to|–|-|~)\s*(\d\.\d{1,2})\s*V\b")
_TEMP = re.compile(r"[-–−]\s*40\s*°?\s*C?\s*(?:to|–|-|~|\.\.)\s*\+?\s*(85|105|125|150)\s*°\s*C")
_PACKAGE = re.compile(r"\b((?:LQFP|UFQFPN|UFBGA|TFBGA|WLCSP|LFBGA|QFN|VQFN|HWQFN|TQFP|VFQFPN|EWLCSP|LGA|TSSOP|SOIC|SSOP|SO|DIP|PDIP|SOT|VSSOP|HVQFN|HTQFP|nFBGA|PLCC|CSP|HLQFP|WQFN|DFN|UDFN|uDFN|LQFN|QFP|BGA)\s?-?\d{1,3}[A-Z]?)\b")


def read_prose_facts(page_texts: dict[int, str], max_pages: int = 160) -> list[dict[str, Any]]:
    """Family facts a document states in prose on its opening pages: core,
    maximum frequency, supply range, temperature range, packages. Verbatim
    match kept; nothing inferred."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, verbatim: str, page: int, value: Any = None, unit: str | None = None, qualifier: str | None = None, key_text: str | None = None) -> None:
        key = (kind, re.sub(r"[^a-z0-9.]+", "", (key_text or verbatim).lower()))
        if key in seen:
            return
        seen.add(key)
        out.append({"kind": kind, "verbatim": _norm(verbatim), "value": value, "unit": unit, "qualifier_verbatim": qualifier, "receipt": {"page": page}})

    supply_candidates: list[tuple[float, float, str, int]] = []
    for pno in range(1, min(max_pages, max(page_texts) if page_texts else 0) + 1):
        text = page_texts.get(pno) or ""
        for match in _CORE.finditer(text):
            core = match.group(1) or match.group(2) or match.group(3)
            if core:
                # One row per core designator (M4, M33, C28x); spelling variants collapse.
                stem = re.sub(r"[^a-z0-9]+", "", re.sub(r"\b(?:arm|with|fpu|dsp|mpu|and|core)\b", "", core.lower()))
                add("core", match.group(0), pno, key_text=stem)
        for line in text.split("\n"):
            # A CPU/system clock statement, not a peripheral's bit rate.
            if not re.search(r"frequency|\bcpu\b|\bcore\b|\bsystem clock|\bsysclk|\bhclk|running|operat|performance|\bmips\b|\bdmips\b", line, re.I):
                continue
            if re.search(r"\bi2c|\bspi\b|\busb|\badc|\bdac|\buart|\busart|\bsdio|\bsdmmc|\bdma\b|\btimer|\bcan\b|\bethernet|\bbus\b|\bpll\b|\bhse\b|\bhsi\b|\blse\b|\blsi\b|oscillator|\bxspi|\bqspi|\bfmc\b|\bsampling", line, re.I):
                continue
            for match in _MAX_FREQ.finditer(line):
                value = match.group(2) or match.group(3)
                qualifier = match.group(1) or match.group(4)
                add("max_frequency", match.group(0), pno, float(value) if "." in value else int(value), "MHz", qualifier, key_text=value)
        for line in text.split("\n"):
            if not re.search(r"operating voltage|supply voltage|power supply|\bV(?:DD|CC)\b|operating range|voltage range", line, re.I):
                continue
            for match in _SUPPLY.finditer(line):
                lo, hi = float(match.group(1)), float(match.group(2))
                if 0.9 <= lo < hi <= 6.0:
                    supply_candidates.append((lo, hi, match.group(0), pno))
        for match in _TEMP.finditer(text):
            add("temperature_range", match.group(0), pno, [-40, int(match.group(1))], "°C", key_text=match.group(1))
        packages = sorted({m.group(1).replace(" ", "") for m in _PACKAGE.finditer(text)})
        if packages and pno <= 12:
            add("packages", ", ".join(packages), pno, packages, None)
    if supply_candidates:
        # The widest range stated with a supply word is the family's operating range.
        lo, hi, verbatim, pno = max(supply_candidates, key=lambda c: (c[1] - c[0], -c[3]))
        add("supply_range", verbatim, pno, [lo, hi], "V")
    return out[:80]


def read_memory(document: Any, page_texts: dict[int, str], chapters: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows naming a memory region with a size, from memory-map/flash/SRAM chapter pages and the cover."""
    pages: list[int] = [1, 2, 3]
    for cls in ("memory_map", "flash", "sram"):
        for entry in chapters.get("peripheral_classes", {}).get(cls, [])[:3]:
            pages.extend(range(entry["page"], min(entry["page"] + 6, document.page_count + 1)))
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for pno in sorted(set(p for p in pages if 1 <= p <= document.page_count)):
        for line in (page_texts.get(pno) or "").split("\n"):
            line = _norm(line)
            if not _SIZE.search(line) or not re.search(r"flash|sram|ram\b|rom\b|eeprom|otp|backup|itcm|dtcm|tcm|ccm|program memory|data memory", line, re.I):
                continue
            if re.search(r"for example|if a device|for instance|e\.g\.|assume|^(?:table|figure|fig\.)\s*\d", line, re.I):
                continue
            size = _SIZE.search(line)
            value, unit = float(size.group(1)), size.group(2).lower()
            kb = value * 1024 if unit.startswith("m") else (value / 1024 if unit.startswith("byte") else value / 8 if "bit" in unit else value)
            key = (line[:60], size.group(0))
            if key in seen:
                continue
            seen.add(key)
            qualifier = _QUALIFIER.search(line)
            out.append({"verbatim": line[:200], "size_kb": round(kb, 3), "qualifier_verbatim": qualifier.group(1) if qualifier else None, "receipt": {"page": pno}})
    return out[:120]


def read_family_document(path: Path, *, vendor: str | None = None, max_text_pages: int | None = None) -> dict[str, Any]:
    document = pymupdf.open(str(path))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    limit = document.page_count if max_text_pages is None else min(document.page_count, max_text_pages)
    page_texts = {pno + 1: document[pno].get_text() for pno in range(limit)}
    identity = read_identity(document, page_texts)
    chapters = read_chapters(document, page_texts)
    instances = read_instances(page_texts)
    features = read_features(page_texts)
    chapter_features = read_chapter_features(page_texts, chapters, document.page_count)
    memory = read_memory(document, page_texts, chapters)
    prose_facts = read_prose_facts(page_texts)
    chapters.pop("all_entries", None)
    return {
        "schema": SCHEMA,
        "_meta": {
            "extracted_by": "harness.electronics.family_document",
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "document_sha256": sha,
            "source_path": str(path),
            "page_count": document.page_count,
            "pages_read": limit,
            "vendor": vendor,
            "models_used": [],
        },
        "identity": identity,
        "chapters": chapters,
        "peripheral_instances": instances,
        "features": features,
        "chapter_features": chapter_features,
        "memory": memory,
        "prose_facts": prose_facts,
        "bogey": {
            "chapter_entries": chapters["entries"],
            "peripheral_classes": len(chapters["peripheral_classes"]),
            "instance_classes": len(instances),
            "instances_asserted": sum(v["count_asserted"] for v in instances.values()),
            "features": len(features),
            "features_qualified": sum(1 for f in features if f["qualifier_verbatim"]),
            "chapter_features": len(chapter_features),
            "chapter_feature_sections": len({(f["receipt"]["page"], f["section"]) for f in chapter_features}),
            "memory_rows": len(memory),
        },
    }
