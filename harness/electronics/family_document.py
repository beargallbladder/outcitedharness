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
    "cpu_core": re.compile(r"\btimers?\b|arbitr|\bcpu (?:self-test|compare)|\bfilter\b|\bprimecell\b|dma\b", re.I),
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
    ("eeprom", re.compile(r"\beeprom\b(?!\s+emulation)", re.I)),
    ("sram", re.compile(r"\bs?ram\b(?!p)", re.I)),
    ("ethernet", re.compile(r"\bethernet|\bemac\b|\beth\b", re.I)),
    ("usb", re.compile(r"\busb\b|\botg\b|\bucpd|\busb type-c", re.I)),
    ("sdmmc", re.compile(r"\bsdmmc|\bsdio\b|\bsd/mmc|\bemmc|\bmmc\b", re.I)),
    ("xspi", re.compile(r"\bquad-?spi|\bqspi|\bocto-?spi|\bospi|\bxspi|\bhexadeca|\bhyperbus", re.I)),
    ("fmc", re.compile(r"\bfmc\b|\bfsmc|\bflexible (?:static )?memory controller|\bemif\b|\bexternal memory interface|\bebi\b", re.I)),
    ("can", re.compile(r"\bCAN\b|(?i:\bcan\s*(?:2\.0|fd|bus|controller|module|interface)\b|\bfdcan|\bbxcan|\bmcan|\bdcan|\becan|\bcontroller area network|\bflexcan|\btwai\b)")),
    ("lin", re.compile(r"\blin\b", re.I)),
    ("i3c", re.compile(r"\bi3c\b", re.I)),
    ("i2c", re.compile(r"\bi2c|\bi²c|\binter-integrated", re.I)),
    ("usart", re.compile(r"\busart|\buart|\blpuart|\bsci\b|\bserial communication interface", re.I)),
    ("spi", re.compile(r"\bspi\b|\bserial peripheral interface|\bmibspi|\bssi\b|\bi2s\b", re.I)),
    ("display", re.compile(r"\bltdc|\blcd\b|\bdsi\b|\bdisplay|\bgfxmmu|\bdma2d|\bchrom-?art|\bgpu\b|\bneochrom", re.I)),
    ("dcmi", re.compile(r"\bdcmi|\bcamera|\bpssi\b|\bcsi\b", re.I)),
    ("audio", re.compile(r"\bsai\b|\bspdif|\bdfsdm|\bmdf\b|\badf\b|\bpdm\b", re.I)),
    ("touch", re.compile(r"\btsc\b|\btouch sensing|\bcaptivate|\bctsu\b|\bptc\b|peripheral touch controller|capacitive touch|\btouch\s+(?:controller|sensor|key|button)", re.I)),
    ("hrtim", re.compile(r"\bhrtim|\bhigh-resolution timer", re.I)),
    ("lptim", re.compile(r"\blow[- ]power timer|\blptim", re.I)),
    ("rtc", re.compile(r"\breal[- ]time clock|\brtc\b", re.I)),
    ("watchdog", re.compile(r"\bwatchdog|\biwdg|\bwwdg|\bwdt\b", re.I)),
    ("adc", re.compile(r"\badc|analog[- ]to[- ]digital|\bA/D\b|\bS12AD|\bsigma[- ]delta\s+(?:adc|converter)|\bSDADC", re.I)),
    ("dac", re.compile(r"\bdac|digital[- ]to[- ]analog|\bD/A\b", re.I)),
    ("temp_sensor", re.compile(r"temperature\s+sensor|\bTSN\b|\bTEMPSENSOR\b", re.I)),
    ("opamp", re.compile(r"\bop-?amp|\boperational amplifier|\bpga\b", re.I)),
    ("comparator", re.compile(r"\bcomp(?:arator)?s?\b", re.I)),
    ("crypto", re.compile(r"\baes\b|\bcryp\b|\bcrypto\b|\bhash\b|\bpka\b|\bsaes\b|\brng\b|\btrng|\bcryptograph|\bsecurity\b|\bsecure\b|\bhsm\b|\bmathacl|\bsha-?[123]\b|\bsha-?256\b|advanced encryption standard|elliptic curve|public key|\becdsa\b|\becdh\b", re.I)),
    ("trustzone", re.compile(r"\btrustzone|\bgtzc\b|\btz\b", re.I)),
    ("npu", re.compile(r"\bnpu\b|\bneural|\bai accelerator|\btinyengine", re.I)),
    ("wireless", re.compile(r"\bradio\b|\bbluetooth|\bble\b|\b802\.15\.4|\bzigbee|\bthread\b|\bwi-?fi|\brf subsystem|\bsub-ghz", re.I)),
    ("safety", re.compile(r"\becc\b|\bcrc\b|cyclic redundancy|\bpbist|\bstc\b|\besm\b|\berror signaling|\bccm-r4|\bself-test|\bfunctional safety", re.I)),
    ("debug", re.compile(r"\bdebug|\bdbg\b|\bswd\b|\bjtag|\btrace|\betm\b|\bcoresight", re.I)),
    ("power", re.compile(r"\bpower (?:control|management|supply)|\bpwr\b|\bpmcu\b|\bpower-saving|low[- ]power modes|\bhibernat|\bbattery[- ]backed|\bdc-?dc\b|\bldo\b|voltage regulator|brown-?out|\bbod\b|energy management|\bemu\b|\benergy modes?\b|\bEM[0-4]\b", re.I)),
    ("reset_clock", re.compile(r"\breset and clock|\brcc\b|\bclock (?:system|tree|configuration|control)|\boscillator|\bsysctl\b|\bclock module", re.I)),
    ("gpio", re.compile(r"\bgpio|\bgeneral[- ]purpose i/?os?|\bi/o ports?|\bport (?:control|i/o)|\biomux|\bpinmux|\bi/o multiplexing", re.I)),
    ("dma", re.compile(r"\bdma\b|[µμu]dma\b|\bdirect memory access|\bgpdma|\bmdma|\bbdma|\bdmamux|\bedma", re.I)),
    ("timer", re.compile(r"\btimers?\b|\btim\d|\btimg|\btima|\bgptm|\bepwm|\becap|\beqep|\brti\b|\bhet\b|\bn2het|\boutput compare|\binput capture|\bpwm\b|\bqei\b|\bquadrature encoder|\bcryotimer|\bletimer|\bwtimer|\bpulse counter|\bpcnt\b|\bpca\b|programmable counter array", re.I)),
    ("interrupts", re.compile(r"\binterrupt|\bnvic\b|\bexti\b|\bvim\b|\bevents?\b|peripheral reflex system|\bprs\b", re.I)),
    ("cpu_core", re.compile(r"\b(?:cortex|cpu|processor core|core architecture|arm)\b|memory protection unit|\bmpu\b|floating[- ]point unit|\bfpu\b|\bc?8051\b|\bcip-51\b|\brisc-v\b|\bxtensa\b|\bc28x\b|\brl78\b|\brxv\d\b|\bavr\b(?:\s+(?:core|cpu))|\bmips\b", re.I)),
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
    ("USB", re.compile(r"\b(USB_OTG_FS|USB_OTG_HS|USB_DRD_FS|OTG_FS|OTG_HS|USBFS|USBHS|USB0|USB1|USBOTG|USBFSH|USBHSD|USBHSH)\b")),
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
    # NXP Kinetis / LPC / MCX / i.MX RT house names. `USB0`, `ADC0`, `I2C0`,
    # `SPI0`, `UART0` also count from zero; the generic rules above take the
    # digit, so only the Kinetis-only stems are listed here.
    ("TPM", re.compile(r"\bTPM(\d)\b")),
    ("FTM", re.compile(r"\bFTM(\d)\b")),
    ("PIT", re.compile(r"\b(PIT)\b")),
    ("LPTMR", re.compile(r"\bLPTMR(\d)\b")),
    ("LPIT", re.compile(r"\bLPIT(\d)\b")),
    ("CTIMER", re.compile(r"\bCTIMER(\d)\b")),
    ("SCTIMER", re.compile(r"\b(SCT\d?|SCTimer)\b")),
    ("TSI", re.compile(r"\bTSI(\d)\b")),
    ("CMP", re.compile(r"\bCMP(\d)\b")),
    ("LPI2C", re.compile(r"\bLPI2C(\d)\b")),
    ("LPSPI", re.compile(r"\bLPSPI(\d)\b")),
    ("FLEXCOMM", re.compile(r"\b(?:FLEXCOMM|Flexcomm)\s?(\d{1,2})\b")),
    ("FLEXCAN", re.compile(r"\bFlexCAN(\d)\b")),
    ("ENET", re.compile(r"\b(ENET\d?)\b")),
    ("SDHC", re.compile(r"\b(SDHC|uSDHC\d?|USDHC\d?)\b")),
    ("FLEXIO", re.compile(r"\b(?:FLEXIO|FlexIO)(\d)\b")),
    ("FLEXSPI", re.compile(r"\b(?:FLEXSPI|FlexSPI)(\d)\b")),
    # Silicon Labs EFM32/EFR32 house names (TIMER0, WTIMER0, LETIMER0, LEUART0,
    # PCNT0, ACMP0, IDAC0, RTCC, CRYOTIMER, LDMA, GPCRC, PRS, CRYPTO, VDAC0, CSEN).
    ("TIMER", re.compile(r"\bTIMER(\d)\b")),
    ("WTIMER", re.compile(r"\bWTIMER(\d)\b")),
    ("LETIMER", re.compile(r"\bLETIMER(\d)\b")),
    ("LEUART", re.compile(r"\bLEUART(\d)\b")),
    ("PCNT", re.compile(r"\bPCNT(\d)\b")),
    ("ACMP", re.compile(r"\bACMP(\d)\b")),
    ("IDAC", re.compile(r"\bIDAC(\d)\b")),
    ("VDAC", re.compile(r"\bVDAC(\d)\b")),
    ("RTCC", re.compile(r"\b(RTCC)\b")),
    ("CRYOTIMER", re.compile(r"\b(CRYOTIMER)\b")),
    ("LDMA", re.compile(r"\b(LDMA)\b")),
    ("GPCRC", re.compile(r"\b(GPCRC)\b")),
    ("PRS", re.compile(r"\b(PRS)\b")),
    ("CSEN", re.compile(r"\b(CSEN)\b")),
    ("LESENSE", re.compile(r"\b(LESENSE)\b")),
    ("WDOG", re.compile(r"\bWDOG(\d)\b")),
    ("PCA", re.compile(r"\bPCA(\d)\b")),
    ("SMB", re.compile(r"\bSMB(\d)\b")),
    ("LLWU", re.compile(r"\b(LLWU)\b")),
    ("DMAMUX", re.compile(r"\bDMAMUX(\d?)\b")),
    ("EWM", re.compile(r"\b(EWM)\b")),
    ("CRC", re.compile(r"\b(CRC)\b")),
    ("LCDK", re.compile(r"\b(LCDC|eLCDIF|LCDIF|SLCD)\b")),
    ("MIPI", re.compile(r"\b(MIPI_DSI|MIPI_CSI)\b")),
    ("LPADC", re.compile(r"\bLPADC(\d)\b")),
)

_BULLET = re.compile(r"^\s*(?:[•·▪◦■●\-–]|)\s*(.+)$")
_QUALIFIER = re.compile(r"\b(up\s+to|maximum|max\.?|minimum|min\.?|typical|typ\.?|at\s+least|as\s+low\s+as|down\s+to)\b", re.I)
_NUM_UNIT = re.compile(
    r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)[\s-]*(Mbytes?|Kbytes?|MB|KB|Kbit|Mbit|MHz|kHz|GHz|bits?|bytes?|channels?|V|mA|µA|uA|nA|°C|ms|µs|us|ns|I/Os?|pins?|x|×)?\b",
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
    r"^\s*(?:\d{1,2}(?:\.\d{1,2}){0,2}\s+)?(?P<subject>[A-Za-z][A-Za-z0-9/™®+\- ]{0,60}?)\s*(?:main\s+|key\s+|general\s+)?features(?:\s+overview)?\s*$",
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
_DASH_LINE = re.compile(r"^\s*[–\-]\s*(\S.*)?$")
_COUNT_SUFFIX = re.compile(r"\s*(?:[×x]\s*(\d{1,2})|(\d{1,2})\s*[×x]|\((\d{1,2})\s*(?:channels?|units?|modules?|ch)\))\s*(\(.*\))?\s*$", re.I)
_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "sixteen": 16}
# "2 x 12-bit A/D converters", "2 × USARTs", "Two 16-bit timers", "1 x I2C interface".
_COUNT_PREFIX = re.compile(
    rf"^(?:up\s+to\s+)?(?:(\d{{1,2}})\s*(?:×\s*|x\s+)|({'|'.join(_COUNT_WORDS)})\s+(?=[A-Za-z0-9])"
    rf"|(\d{{1,2}})\s+(?!(?:or|to|and|x|bits?|Kbytes?|KB|MB|Mbytes?|MHz|kHz|GHz|V|mA|µA|µs|ns|ms|regions?|wait|external|internal|independent|channels?|pins?|wire|lanes?|ports?|modes?|levels?|priorit\w+|cycles?|entries|vectors?|slots?|bytes?|words?|segments?|commons?)\b)(?=[A-Za-z]))",
    re.I,
)
# Where a long feature line stops being the fact and starts describing it.
_LABEL_CUT = re.compile(r",\s+(?:each|all|with|including|supporting|configurable|capable|featuring|providing|up to|for)\b|\s+(?:with|supporting|including|featuring|capable of|configurable as|for periodic|for use|operates?|running)\s+(?=\S)", re.I)
_NUMBERED_HEADING = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2}){1,2}\s+[A-Z][A-Za-z]+(?:[ \-/][A-Za-z()]+){0,7}\s*$")
# Bullets that are a vendor section heading rather than a fact.
_SECTION_HEADING = re.compile(
    r"^(?:processor|core|cpu|cpu\s+core|memor(?:y|ies)|system|low[- ]power(?:\s+modes?)?|power(?:\s+management)?|peripherals?|i/os?|packages?|"
    r"operating\s+(?:voltage|conditions)|debug(?:\s+mode)?|connectivity|communications?(?:\s+interfaces?)?|analog(?:ue)?(?:\s+peripherals)?|timers?(?:/counters?)?|counters?/timers?(?:\s+and\s+pwm)?|security(?:\s+and\s+\w+)?|safety|"
    r"clocks?(?:,\s*reset\s+and\s+supply\s+management)?|clock\s+management|reset\s+and\s+clock\s+control|dma|human\s+machine\s+interface(?:\s+\(hmi\))?|graphics|"
    r"system\s+and\s+power\s+management|multiple\s+clock\s+sources|general[- ]purpose\s+i/os?|input/output|features|key\s+features|main\s+features|other\s+features|"
    r"advanced\s+analog\s+features|up\s+to\s+\d+\s+(?:fast\s+)?i/o\s+ports|\d+\s+timers|\d+\s+communication\s+interfaces)\s*:?$",
    re.I,
)


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
    # A font-substituted bullet renders as one letter alone on its line
    # (Atmel "z" from Wingdings). If a single character stands alone on six
    # or more lines of the block, it is this block's bullet glyph.
    lone = Counter(l.strip() for l in lines if len(l.strip()) == 1 and not l.strip().isdigit() and not re.match(r"[–\-]", l.strip()))
    extra_glyphs = {ch for ch, n in lone.items() if n >= 6}
    glyph_line = re.compile(rf"^\s*({_GLYPH}|{'|'.join(re.escape(c) for c in extra_glyphs)})\s*(.*)$") if extra_glyphs else _GLYPH_LINE

    glyph_order: list[str] = []
    for line in lines:
        match = glyph_line.match(line)
        if match and match.group(1) not in glyph_order:
            glyph_order.append(match.group(1))
    header_glyph = glyph_order[0] if len(glyph_order) >= 2 else None
    has_dashes = sum(1 for l in lines if _DASH_LINE.match(l)) >= 3

    items: list[dict[str, Any]] = []
    section: str | None = None
    section_from_bullet = False  # section named by a heading-like bullet, not a header glyph
    current: dict[str, Any] | None = None
    pending_glyph: str | None = None
    last_bullet: dict[str, Any] | None = None  # most recent glyph bullet; dashes hang off it

    def close() -> None:
        nonlocal current, section, section_from_bullet, last_bullet
        if current and current["text"].strip():
            current["text"] = _norm(current["text"])
            is_glyph_bullet = current["level"] == 1 and "parent" not in current and current.get("glyph")
            if is_glyph_bullet:
                last_bullet = current
                # In a glyph-over-dash layout (ST "•" headings over "–" facts)
                # a heading bullet's scope ends at the next glyph bullet. In an
                # all-glyph layout (Atmel "z" at every level) it persists.
                if section_from_bullet and has_dashes:
                    section = None
                    section_from_bullet = False
                    current["section"] = None
            current.pop("glyph", None)
            # A level-1 bullet that is a vendor section heading ("Memories",
            # "Low power", "6 timers") names the section for what follows; it
            # is kept as a fact only when it carries a number.
            if current["level"] == 1 and _SECTION_HEADING.match(current["text"]):
                section = current["text"].rstrip(":")
                section_from_bullet = True
                current["section"] = section
                if re.search(r"\d", current["text"]):
                    items.append(current)
            else:
                items.append(current)
        current = None

    for line in lines:
        if not line.strip():
            continue
        # A numbered section heading ("2.1 Device overview") ends the list; a
        # figure like "1.25 DMIPS/MHz (Dhrystone 2.1)" does not.
        if _NUMBERED_HEADING.match(line) and (current is not None or items):
            break
        match = glyph_line.match(line)
        if match:
            glyph, rest = match.group(1), match.group(2).strip()
            if glyph == header_glyph:
                close()
                section = rest or None
                pending_glyph = "header" if not rest else None
                continue
            close()
            current = {"text": rest, "section": section, "level": 1, "glyph": True}
            pending_glyph = None if rest else "bullet"
            continue
        dash = _DASH_LINE.match(line)
        if dash and (current is not None or items or pending_glyph == "dash"):
            close()
            parent = last_bullet["text"] if last_bullet else None
            # Under a section-heading bullet (ST "• Memories" over "– 16 or 32
            # Kbytes of Flash") the dashes are the facts themselves: level 1.
            heading_parent = parent is not None and _SECTION_HEADING.match(_norm(parent)) is not None
            level = 1 if heading_parent else 2
            current = {"text": (dash.group(1) or "").strip(), "section": section, "level": level}
            if not heading_parent and parent:
                current["parent"] = _norm(parent)
            pending_glyph = None if dash.group(1) else "dash"
            continue
        if pending_glyph == "header":
            section = _norm(line)
            pending_glyph = None
            continue
        if current is not None and pending_glyph in ("bullet", "dash") and not current["text"]:
            # The glyph stood alone; this line is the bullet's text, whatever
            # its case ("DMA") or length.
            current["text"] = line.strip()
            pending_glyph = None
            continue
        if current is not None and len(line) < 120 and not line.strip().isupper():
            current["text"] = f"{current['text']} {line.strip()}"
            pending_glyph = None
    close()
    return items


def split_count(text: str) -> tuple[str, int | None]:
    """"Serial Communications Interface (SCI) × 4" -> ("Serial Communications Interface (SCI)", 4)."""
    match = _COUNT_SUFFIX.search(text)
    if not match:
        prefix = _COUNT_PREFIX.match(text)
        if prefix:
            if prefix.group(1) or prefix.group(3):
                count = int(prefix.group(1) or prefix.group(3))
            else:
                count = _COUNT_WORDS[prefix.group(2).lower()]
            return text[prefix.end():].strip(), count
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
            bare = cls in ("USB", "FMC", "RTC", "IWDG", "AES", "LTDC", "TSC", "DTC", "ELC", "RTI", "EMAC", "PIT", "SCTIMER", "ENET", "SDHC", "LLWU", "EWM", "CRC", "LCDK", "MIPI")
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
        value = float(match.group(1).replace(",", ""))
        unit = (match.group(2) or "").strip() or None
        numbers.append({"value": int(value) if value.is_integer() else value, "unit": unit})
    return numbers[:8]


def read_features(page_texts: dict[int, str], max_pages: int = 6) -> list[dict[str, Any]]:
    """Bullet features on the first pages, verbatim, numbers typed, qualifier as printed."""
    out: list[dict[str, Any]] = []
    if not page_texts:
        return out
    last = max(page_texts)
    # "Features", "Key Features", or "<Part> Microcontroller Features" (TI).
    heading_re = re.compile(
        r"^\s*(?:\d{1,2}(?:\.\d{1,2}){0,2}\s*\n?\s*)?(?:[A-Za-z][A-Za-z0-9/™®+\- ]{0,40}?\s+)?(?:key\s+|device\s+|product\s+|microcontroller\s+)?features(?:\s+overview)?\s*$"
        r"|^\s*(?:\d{1,2}(?:\.\d{1,2}){0,2}\s*\n?\s*)?[A-Za-z0-9][A-Za-z0-9/+\- ]{0,30}?\s+(?:sub-?family|family|series)\s+introduction\s*$",
        re.I | re.M,
    )

    def glyph_lines(text: str) -> int:
        return sum(1 for line in text.splitlines() if _GLYPH_LINE.match(line) or _DASH_LINE.match(line))

    # Pages to read: the opening pages, plus the "Features" page wherever the
    # front matter puts it (reference manuals put it after a 40-page TOC), plus
    # the pages that continue its bullet list.
    pages: list[int] = list(range(1, min(max_pages, last) + 1))
    heading_page: int | None = None
    for pno in range(max_pages + 1, min(FEATURES_SEARCH_PAGES, last) + 1):
        text = page_texts.get(pno) or ""
        if heading_re.search(text) and glyph_lines(text) >= 8:
            pages.append(pno)
            heading_page = pno
            nxt = pno + 1
            # Continuation pages: still dense with bullets, not a new chapter,
            # and not a features table (read_features_table owns that page).
            while nxt <= last and glyph_lines(page_texts.get(nxt) or "") >= 8 and not re.search(r"^\s*1\.\s+Overview\b|^\s*Table\s+1\.\d", page_texts.get(nxt) or "", re.M) and not _FEATURES_TABLE_CAPTION.search(page_texts.get(nxt) or ""):
                pages.append(nxt)
                nxt += 1
            break
    for pno in pages:
        text = page_texts.get(pno) or ""
        if not re.search(r"\bfeatures\b", text, re.I) and pno > 2 and pno <= max_pages:
            continue
        # A "Features" heading on the page starts the list; otherwise the whole page.
        heading = heading_re.search(text)
        block = text[heading.end():] if heading else text
        # A cover that lists applications ("Example applications:", "...
        # applications include the following:") mixes those bullets into the
        # feature column in text order; an application names a market, not a block.
        applications_page = _APPLICATIONS_LEAD.search(text) is not None
        for item in _bullet_items(block):
            if _TOC_LINE.match(item["text"]):
                continue
            if applications_page and _is_application(item["text"]):
                continue
            row = _feature(item["text"], pno, section=item.get("section"), level=item.get("level"), parent=item.get("parent"))
            # The page the Features heading is on carries the family summary;
            # its continuation pages are per-peripheral detail.
            row["source"] = "features_heading_page" if pno == heading_page or pno <= max_pages else "features_continuation"
            out.append(row)
    return out[:400]


FEATURES_SEARCH_PAGES = 120

_APPLICATIONS_LEAD = re.compile(r"\b(?:example\s+applications|applications?\s+include(?:\s+the\s+following)?|target\s+applications|typical\s+applications)\s*:?\s*$", re.I | re.M)
_APPLICATION_WORDS = re.compile(
    r"\b(?:automation|consumer|medical|lighting|health|fitness|accessor(?:y|ies)|iot|smart\s+\w+|metering|meters?|appliances?|automotive|wearables?|e-?bikes?|drones?|toys|gaming|building|industrial|white\s+goods|hvac|point\s+of\s+sale|expander|motor\s+control|sensor\s+controllers?|sensors|equipment|devices|systems|electronics|home|security|controls?|monitoring|thermostats?|locks?|remotes?|hubs?|gateways?|trackers?|infrastructure|energy\s+harvesting|inverters?|chargers?|robotics?|telecom|networking|communications?\s+equipment)\b",
    re.I,
)
_TOC_LINE = re.compile(r"^\d{1,4}\s+\d{1,2}(?:\.\d{1,2}){1,3}\s+\S|.*\.\s?\.\s?\.\s?\.|.*\s\.\s*$")


def _is_application(text: str) -> bool:
    """An application bullet names a market and carries no number, no unit and
    no peripheral: "Motor control", "Home automation and security"."""
    if re.search(r"\d", text) or len(text) > 48:
        return False
    if not _APPLICATION_WORDS.search(text):
        return False
    remainder = _APPLICATION_WORDS.sub(" ", text)
    return not any(pattern.search(_singular_acronyms(remainder)) for name, pattern in CHAPTER_VOCAB if name not in ("power", "gpio", "cpu_core"))


# A features *table* (TI Tiva/Hercules "Table 1-1. <Part> Microcontroller
# Features": Feature | Description, with one-cell section rows "Communication
# Interfaces", "Analog Support"). The description cell is the fact as printed;
# the feature cell names the peripheral ("Operating Range (Ambient)").
_FEATURES_TABLE_CAPTION = re.compile(r"^\s*Table\s+\d{1,2}[-.]\d{1,2}\.?\s+.{0,60}\b(?:Features|Module functional categories|functional categories)\s*$", re.I | re.M)
_FEATURES_TABLE_HEADER = re.compile(r"^(?:feature|module\s+category|category|module|function|block)s?$", re.I)


def read_features_table(document: Any, page_texts: dict[int, str], max_pages: int = 80) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pno in range(1, min(max_pages, document.page_count) + 1):
        text = page_texts.get(pno) or ""
        if not _FEATURES_TABLE_CAPTION.search(text):
            continue
        try:
            tables = document[pno - 1].find_tables().tables
        except Exception:
            continue
        for table in tables:
            raw_rows = [[(c or "") for c in row] for row in table.extract()]
            rows = [[_norm(c) for c in row] for row in raw_rows]
            if not rows or len(rows[0]) != 2:
                continue
            # The caption may be extracted as a first spanning row.
            start = 1 if _FEATURES_TABLE_CAPTION.match(rows[0][0]) and not rows[0][1] else 0
            if start >= len(rows):
                continue
            header = [c.lower() for c in rows[start]]
            if not (_FEATURES_TABLE_HEADER.match(header[0]) and "description" in header[1]):
                continue
            section: str | None = None
            for (feature_name, description), (_, raw_description) in zip(rows[start + 1:], raw_rows[start + 1:]):
                if feature_name and not description:
                    section = feature_name
                    continue
                if not description or len(description) < 3:
                    continue
                # A category row whose description is a bullet list (Kinetis
                # "Module functional categories"): each bullet is a fact, the
                # category is its section.
                if _GLYPH_LINE.match(raw_description.strip().splitlines()[0]):
                    for item in _bullet_items(raw_description):
                        if item.get("level", 1) != 1 or item["text"].rstrip().endswith(":"):
                            continue
                        row = _feature(item["text"], pno, section=feature_name, level=1)
                        row["source"] = "features_table"
                        out.append(row)
                    continue
                # Parenthetical ranges are the fact; the frame is the section.
                # "Industrial (-40°C to 85°C) temperature range" stays as printed.
                for part in re.split(r"\s*;\s*|\.\s+(?=[A-Z])|(?<=\brange)\s+(?=[A-Z])|,\s+(?=each\s+with\b|for\s+a\s+total\b|with\s+four\b)", description):
                    part = part.strip(" .")
                    if len(part) < 3 or re.match(r"^(?:each|for a total|with)\b", part, re.I):
                        continue  # the tail of a split is detail of its head
                    row = _feature(part, pno, section=section, level=1, parent=feature_name or None)
                    row["source"] = "features_table"
                    out.append(row)
        if out:
            break
    return out[:120]


# The "General description" paragraph: the family facts as one prose sentence
# list ("provides up to 3072 KB on-chip Flash memory and 128 KB SRAM ... up to
# three 12-bit ADCs, two 12-bit DACs, ... a SDIO, and an USBFS"). GigaDevice,
# ST "Description", Microchip and NXP introductions all use it.
_DESC_HEADING = re.compile(r"^\s*(?:\d{1,2}\.?\s+)?(?:general\s+description|description|introduction|overview|product\s+overview|device\s+overview)\s*$", re.I | re.M)
_DESC_LEAD = re.compile(
    r"^(?:(?:it|they|which|the\s+(?:devices?|series|family|mcus?|products?)|these\s+devices|this\s+(?:device|series|family)|all\s+devices|[A-Z][A-Za-z0-9/\-]+(?:\s+\S+){0,6}?\s+(?:devices?|series|family|mcus?|microcontrollers?))\s+)?"
    r"(?:also\s+|further\s+)?(?:provides?|incorporates?|offers?|features?|includes?|integrates?|has|have|contains?|supports?|comes?\s+with|is\s+equipped\s+with|embeds?|operates?\s+from|is\s+available\s+in|are\s+available\s+in|available\s+in|delivers?|combines?|operating\s+at)\s+(?:a\s+|an\s+|the\s+)?",
    re.I,
)
_DESC_STRIP = re.compile(r"^(?:as\s+well\s+as|and|also|with|plus|along\s+with|together\s+with|standard\s+and\s+advanced\s+communication\s+interfaces:|communication\s+interfaces:|peripherals:|including|namely|in\s+addition\s+to|operates?\s+from|from)\s+|^(?:a|an|the)\s+", re.I)
_DESC_NOUN = re.compile(r"\b(?!(?:up|to|or|and|of|from|at|KB|MB|Kbytes?|Mbytes?|Kbit|Mbit|MHz|kHz|GHz|bit|bits|V|mA|µA|ms|µs|ns|MSps|Msps|ksps)\b)[A-Za-z][A-Za-z0-9/\-]{2,}\b")
_DESC_FACT = re.compile(
    rf"\b(?:\d+(?:\.\d+)?|{'|'.join(_COUNT_WORDS)})\b.*(?:\b(?:KB|MB|Kbytes?|Mbytes?|Kbit|Mbit|MHz|kHz|V|bit|channels?|pins?|I/Os?|timers?|ADCs?|DACs?|SPIs?|I2Cs?|I2Ss?|USARTs?|UARTs?|CANs?|USB\w*|SDIO|comparators?|op-?amps?|DMA|GPIOs?|packages?|leads?|cores?)\b|°C)"
    r"|\b(?:SDIO|USB\w*|CAN\w*|Ethernet|RTC|CRC|LCD|TSC|RNG|AES|SHA|TrustZone|FPU|MPU|DSP|DMA|QSPI|OSPI|FSMC|EXMC|SDMMC|I3C|LIN)\b",
    re.I,
)


def read_description_facts(page_texts: dict[int, str], max_pages: int = 14) -> list[dict[str, Any]]:
    """Clauses of the description paragraph that state a counted, sized or
    named family fact, verbatim; leading verb phrases dropped so the clause
    reads as a feature line ("up to three 12-bit 2.6M MSPS ADCs")."""
    out: list[dict[str, Any]] = []
    for pno in range(1, min(max_pages, max(page_texts) if page_texts else 0) + 1):
        text = page_texts.get(pno) or ""
        heading = _DESC_HEADING.search(text)
        if not heading:
            continue
        block = text[heading.end():]
        nxt = re.search(r"^\s*\d{1,2}(?:\.\d{1,2})?\.?\s+[A-Z][a-z]+(?:\s+\w+){0,5}\s*$", block, re.M)
        if nxt and nxt.start() > 200:
            block = block[: nxt.start()]
        paragraph = _norm(block)
        if len(paragraph) < 200:
            continue
        for sentence in re.split(r"(?<=[A-Za-z0-9)%])\.\s+(?=[A-Z])", paragraph):
            if not re.search(r"\b(?:up\s+to|provides?|incorporates?|offers?|features?|includes?|integrates?|operates?|available|embeds?|contains?|supports?)\b", sentence, re.I):
                continue
            # A colon introduces a list; the part before it is framing.
            sentence = re.sub(r"^[^:]{0,120}?\binterfaces?:\s*", "", sentence)
            splitter = r",\s+(?:and\s+|as\s+well\s+as\s+)?|\s+and\s+(?=(?:up\s+to\s+)?(?:\d|(?:a|an|available|" + "|".join(_COUNT_WORDS) + r")\b))|;\s+|\s+(?=(?:operating|running)\s+at\b)|\s+with\s+(?=(?:up\s+to\s+)?\d)"
            for clause in re.split(splitter, sentence):
                clause = clause.strip(" .")
                clause = _DESC_LEAD.sub("", clause)
                for _ in range(2):
                    clause = _DESC_STRIP.sub("", clause)
                clause = re.sub(r"\s+(?:to\s+obtain|which|that|in\s+terms\s+of|for\s+enhanced|providing|allowing|enabling|with\s+flash\s+access|with\s+zero)\b.*$", "", clause, flags=re.I)
                clause = re.sub(r"^(?:operating|running)\s+at\s+", "", clause, flags=re.I)
                # A parenthetical list split by the comma rule leaves halves:
                # "embedded memories (12 Kbytes of SRAM" / "32 Kbytes of Flash)".
                if "(" in clause and ")" not in clause:
                    clause = clause[clause.index("(") + 1:]
                elif ")" in clause and "(" not in clause:
                    clause = clause[: clause.index(")")]
                # A verb in the middle starts a new statement; keep the head.
                clause = re.sub(r"\s+(?:features?|provides?|includes?|incorporates?|offers?|supports?|embeds?|integrates?|has|have)\s+(?:a|an|the|up\s+to)\b.*$", "", clause, flags=re.I)
                # "... and Reset pin enabled": an "and" tail that states no fact.
                head, sep, tail = clause.partition(" and ")
                if sep and not _DESC_FACT.search(tail):
                    clause = head
                clause = clause.strip(" .")
                if "•" in clause or "\uf0b7" in clause or re.search(r"\s(?:is|are|was|were|belongs?)\s", clause):
                    continue
                if re.search(r"\b(?:most\s+powerful|needed\s+for|delivering|enabled)\b|@", clause, re.I) or not _DESC_NOUN.search(clause):
                    continue  # a bare "32 Kbytes" or a test condition is not a family fact
                if len(clause) < 4 or len(clause) > 90 or not _DESC_FACT.search(clause):
                    continue
                if re.search(r"\b(?:suitable|applications?|such\s+as|areas?|markets?|ratio|efficien|ideal|designed|targets?|where|each|selected|configured|optimized|ranging)\b", clause, re.I):
                    continue
                if re.search(r"\bcan\s+(?:be|also|operate|run|support|perform|act|handle|reach|wake|use|form)\b", clause, re.I):
                    continue  # the verb, not the bus
                row = _feature(clause, pno, section="Description", level=1)
                row["source"] = "description"
                out.append(row)
        if out:
            break
    return out[:60]


def _singular_acronyms(text: str) -> str:
    """"three SPIs, two I2Cs, two CANs" -> "SPI", "I2C", "CAN" for classification only."""
    return re.sub(r"\b([A-Z][A-Z0-9]{1,7})s\b", r"\1", text)


def _feature(text: str, page: int, *, section: str | None = None, level: int | None = None, parent: str | None = None) -> dict[str, Any]:
    text = _norm(text)
    qualifier = _QUALIFIER.search(text)
    # The fact is the head of the line; ", each with up to 4 IC/OC/PWM ..." is
    # its description. The full line stays in verbatim.
    head = text
    cut = _LABEL_CUT.search(text)
    # "C8051 core with 25 MHz maximum operating frequency": the tail after
    # "with" is a sized fact of its own, so the line stays whole.
    sized_tail = cut and re.match(r"\s*,?\s*(?:with|supporting|including|featuring|up to|for)\s+(?:up\s+to\s+)?\d+(?:\.\d+)?\s*-?\s*(?:MHz|kHz|GHz|KB|MB|Kbytes?|Mbytes?|V)\b", text[cut.start():], re.I)
    if cut and cut.start() >= 8 and re.search(r"[A-Za-z]", text[: cut.start()]) and not sized_tail:
        head = text[: cut.start()].rstrip(" ,:;")
    if head.count("(") > head.count(")") and "(" in head:
        head = head[: head.rindex("(")].rstrip(" ,:;")  # a parenthetical the cut split open
    label, count = split_count(head)
    subject = f"{section} {label}" if section else label
    row = {
        "verbatim": text[:300],
        "label": label[:200],
        "count": count,
        "numbers": _typed_numbers(label),
        "qualifier_verbatim": qualifier.group(1) if qualifier else None,
        "classes": [name for name, pattern in CHAPTER_VOCAB if pattern.search(_singular_acronyms(label)) and not (CHAPTER_EXCLUDE.get(name) and CHAPTER_EXCLUDE[name].search(label))][:4],
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
_PACKAGE = re.compile(r"\b((?:LFQFP|TFLGA|PLQP|PTLG|PWQN|PLBG|LQFP|UFQFPN|UFBGA|TFBGA|WLCSP|LFBGA|QFN|VQFN|HWQFN|TQFP|VFQFPN|EWLCSP|LGA|TSSOP|SOIC|SSOP|SO|DIP|PDIP|SOT|VSSOP|HVQFN|HTQFP|nFBGA|PLCC|CSP|HLQFP|WQFN|DFN|UDFN|uDFN|LQFN|QFP|BGA)\s?-?\d{1,3}[A-Z]?)\b")


def read_prose_facts(page_texts: dict[int, str], max_pages: int = 160) -> list[dict[str, Any]]:
    """Family facts a document states in prose on its opening pages: core,
    maximum frequency, supply range, temperature range, packages. Verbatim
    match kept; nothing inferred."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, verbatim: str, page: int, value: Any = None, unit: str | None = None, qualifier: str | None = None, key_text: str | None = None, context: str | None = None) -> None:
        key = (kind, re.sub(r"[^a-z0-9.]+", "", (key_text or verbatim).lower()))
        if key in seen:
            return
        seen.add(key)
        row = {"kind": kind, "verbatim": _norm(verbatim), "value": value, "unit": unit, "qualifier_verbatim": qualifier, "receipt": {"page": page}}
        if context:
            row["context"] = _norm(context)[:240]
        out.append(row)

    def line_of(text: str, pos: int) -> str:
        start = text.rfind("\n", 0, pos) + 1
        end = text.find("\n", pos)
        return text[start: end if end != -1 else len(text)]

    supply_candidates: list[tuple[float, float, str, int, str]] = []
    # Cores: one row per designator (M4, M33, C28x), spelling variants
    # collapsed. A core the document names once in a comparison or migration
    # table is not this family's core: a row needs the opening pages or
    # repeated mention (page support >= 3, or >= 10% of the core mentions).
    core_pages: dict[str, set[int]] = defaultdict(set)
    core_first: dict[str, tuple[int, str]] = {}
    for pno in range(1, min(max_pages, max(page_texts) if page_texts else 0) + 1):
        text = page_texts.get(pno) or ""
        for match in _CORE.finditer(text):
            core = match.group(1) or match.group(2) or match.group(3)
            if core:
                stem = re.sub(r"[^a-z0-9]+", "", re.sub(r"\b(?:arm|with|fpu|dsp|mpu|and|core)\b", "", core.lower()))
                core_pages[stem].add(pno)
                core_first.setdefault(stem, (pno, match.group(0)))
    total_core_pages = sum(len(p) for p in core_pages.values()) or 1
    for stem, pages in core_pages.items():
        pno, verbatim = core_first[stem]
        if min(pages) <= 3 or (len(pages) >= 3 and len(pages) / total_core_pages >= 0.10):
            add("core", verbatim, pno, key_text=stem, context=f"pages={len(pages)}")
    for pno in range(1, min(max_pages, max(page_texts) if page_texts else 0) + 1):
        text = page_texts.get(pno) or ""
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
                    supply_candidates.append((lo, hi, match.group(0), pno, line))
        for match in _TEMP.finditer(text):
            context = line_of(text, match.start())
            # Ta / Tj distinguish two attributes; keep them as separate rows.
            axis = "j" if re.search(r"\bTj\b|junction", context, re.I) else "a" if re.search(r"\bTa\b|ambient", context, re.I) else "-"
            add("temperature_range", match.group(0), pno, [-40, int(match.group(1))], "°C", key_text=f"{match.group(1)}{axis}", context=context)
        packages = sorted({m.group(1).replace(" ", "") for m in _PACKAGE.finditer(text)})
        if packages and pno <= 12:
            add("packages", ", ".join(packages), pno, packages, None)
    if supply_candidates:
        # Rated (operating) and absolute-maximum are two attributes: the widest
        # range of each kind, with its line as context.
        rated = [c for c in supply_candidates if not re.search(r"absolute|abs\.?\s*max|maximum ratings?", c[4], re.I)]
        absolute = [c for c in supply_candidates if c not in rated]
        for group in (rated, absolute):
            if group:
                # The document's own statement of the supply ("operates from a
                # 2.6 to 3.6 V power supply", "operating voltage") outranks a
                # wider range found on a later page (an analog domain, a pin).
                stated = [c for c in group if re.search(r"operat|power\s+supply|supply\s+voltage|\bV(?:DD|CC)\b\s*(?:=|range|from)", c[4], re.I)]
                pool = stated or group
                lo, hi, verbatim, pno, line = max(pool, key=lambda c: (-c[3] if stated else 0, c[1] - c[0], -c[3]))
                add("supply_range", verbatim, pno, [lo, hi], "V", key_text=f"{lo}-{hi}", context=line)
    return out[:80]


_PKG_WORD = r"(?:LFQFP|TFLGA|PLQP|PTLG|PWQN|PLBG|PVQN|LQFP|UFQFPN|UFBGA|TFBGA|WLCSP|LFBGA|QFN|VQFN|HWQFN|TQFP|VFQFPN|EWLCSP|LGA|TSSOP|SOIC|SSOP|DIP|PDIP|VSSOP|HVQFN|HTQFP|nFBGA|PLCC|CSP|HLQFP|WQFN|DFN|UDFN|LQFN|QFP|BGA|WFLGA|WLBGA|VFBGA)"
# Renesas RA/RX house style: "I/O ports for the 100-pin LQFP" ... "I/O pins: 80".
_IO_FOR_PACKAGE = re.compile(
    rf"I/O\s+ports?\s+for\s+the\s+(?P<pins>\d{{2,3}})-pin\s+(?P<pkg>{_PKG_WORD})[^\n]*\n(?:[^\n]*\n){{0,2}}?[–\-\s]*I/O\s+pins?\s*[:：]\s*(?P<io>\d{{1,3}})",
    re.I,
)
# Generic proximity: "<N>-pin <PKG>" and "I/O (pins|ports)[:] <M>" within a short window, either order.
_IO_NEAR_PACKAGE = re.compile(
    rf"(?P<pins>\d{{2,3}})-(?:pin|lead)\s+(?P<pkg>{_PKG_WORD})\b(?P<mid>[^\n]{{0,60}}\n?[^\n]{{0,80}}?)\b(?:I/Os?|GPIOs?|I/O\s+(?:pins|ports)|general[- ]purpose\s+I/Os?)\s*[:：]?\s*(?P<io>\d{{1,3}})\b",
    re.I,
)


def read_io_by_package(page_texts: dict[int, str]) -> list[dict[str, Any]]:
    """Package-qualified I/O counts: the document's own "N-pin PACKAGE -> M
    I/O" statements. Each row is one (pin count, package) with its I/O count,
    verbatim and page. Never a scalar for the family."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int]] = set()
    for pno, text in page_texts.items():
        for pattern, kind in ((_IO_FOR_PACKAGE, "io_ports_for_package"), (_IO_NEAR_PACKAGE, "io_near_package")):
            for match in pattern.finditer(text):
                pins, io = int(match.group("pins")), int(match.group("io"))
                pkg = match.group("pkg").upper()
                if io > pins or io < 4:
                    continue  # plausibility: I/O never exceeds pins
                if kind == "io_near_package" and re.search(r"\d", match.group("mid") or "") and len(match.group("mid") or "") > 40:
                    continue  # too much text between; likely unrelated numbers
                key = (pins, pkg, io)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"pin_count": pins, "package": pkg, "io_count": io, "pattern": kind, "verbatim": _norm(match.group(0))[:200], "receipt": {"page": pno}})
    return out[:40]


# Port-pin names across vendors: PA0 (ST/GD/AVR/Tiva/EFM32), PTA0 (Kinetis),
# PA00 (SAM), RA0 (PIC), P1.0 (MSP430/EFM8/XMC), P000 (Renesas RA/RX),
# PIO0_0 / P0_0 (LPC), GPIO0 (C2000).
_PORT_PIN = re.compile(r"\b(?:P[A-K]\d{1,2}|PT[A-E]\d{1,2}|P[A-C]\d{2}|R[A-G]\d|P\d\.\d{1,2}|P\d{3}|PIO\d_\d{1,2}|P\d_\d{1,2}|GPIO\d{1,3})\b")
_PKG_CELL = re.compile(
    rf"(?P<pkg1>{_PKG_WORD})\s?-?\s?(?P<pins1>\d{{2,3}})\b|\b(?P<pins2>\d{{2,3}})\s?-?\s?(?:pin|lead)s?\s+(?P<pkg2>{_PKG_WORD})\b|\b(?P<pins3>\d{{2,3}})\s*(?P<pkg3>{_PKG_WORD})\b",
    re.I,
)
_PIN_TABLE_CAPTION_BARE = re.compile(r"signals?\s+by\s+pin\s+number|\bpin\s+definitions?\b|\bpinout\s+(?:table|description)|\bpin\s+assignments?\s+(?:table|for)\b", re.I)
# A pin-name cell is the port pin, possibly with its alternate names
# ("PC13-TAMPER-RTC", "PA0/WKUP"); a mux cell "PF3 (7)/PB5 (7)" is not.
_PIN_NAME_CELL = re.compile(rf"^\s*{_PORT_PIN.pattern}(?:\(\d\))?(?:\s*[-/_]\s*[A-Za-z0-9_+.\-]+(?:\s?(?:IN|OUT))?(?:\(\d\))?)*\s*_?\s*$")
_PIN_TABLE_CAPTION = re.compile(
    rf"(?:(?P<pkg>{_PKG_WORD})\s?-?\s?(?P<pins>\d{{2,3}})|(?P<pins_b>\d{{2,3}})\s?-?\s?(?:pin|lead)\s+(?P<pkg_b>{_PKG_WORD}))\s+(?:pin\s+(?:definitions?|assignments?|descriptions?|out|functions?)|pinout|signals?\s+by\s+pin)",
    re.I,
)


_PINOUT_PAGE = re.compile(r"pin\s*(?:out|definitions?|assignments?|descriptions?|functions?|list|map)|signals?\s+by\s+pin|pin\s+multiplexing|\bmultiplexing\b|pin\s+configuration", re.I)


def _pin_name_in_cell(cell: str) -> str | None:
    """The port pin a pin-name cell names: a strict pin-name cell, or a
    "Default: OSCIN / Remap: PD0" cell naming exactly one port pin. A mux list
    "PF3 (7)/PB5 (7)" names several and is not a pin-name cell."""
    joined = cell.replace("\n", "")
    if _PIN_NAME_CELL.match(joined):
        return _PORT_PIN.search(joined).group(0)
    names = set(_PORT_PIN.findall(joined))
    if len(names) == 1 and not re.search(r"\(\d\)\s*/", joined) and len(joined) <= 40:
        return next(iter(names))
    return None


def _package_in_cell(cell: str) -> tuple[str, int] | None:
    m = _PKG_CELL.search(cell.replace("\n", " "))
    if not m:
        return None
    for pkg, pins in (("pkg1", "pins1"), ("pkg2", "pins2"), ("pkg3", "pins3")):
        if m.group(pkg):
            return m.group(pkg).upper(), int(m.group(pins))
    return None


def read_io_from_pin_tables(document: Any, page_texts: dict[int, str]) -> list[dict[str, Any]]:
    """I/O per package counted from the pinout table itself: the distinct
    port-pin names that have a pin number in a package's column (ST/Renesas/
    SAM shape: one column per package) or in a table captioned with the
    package (GD32/TI shape: one table per package). The count is what the
    document lists, not a statement it makes; pattern says so."""
    per_package: dict[tuple[str, int], dict[str, Any]] = {}
    layout: tuple[int, dict[int, tuple[str, int]], int] | None = None  # (page, pkg columns, pin column)
    caption_layout: tuple[int, tuple[str, int], int, int] | None = None  # (page, package, width, pin column)
    # A document that names one package throughout (TI Tiva "64LQFP") owns
    # an uncaptioned pinout table.
    named = Counter(p for text in page_texts.values() for m in _PKG_CELL.finditer(text) if (p := _package_in_cell(m.group(0))))
    ranked = named.most_common(2)
    single_package = ranked[0][0] if ranked and ranked[0][1] >= 5 and (len(ranked) == 1 or ranked[0][1] >= 5 * ranked[1][1]) else None
    for pno in range(1, document.page_count + 1):
        text = page_texts.get(pno) or ""
        if len(_PORT_PIN.findall(text)) < 5:
            layout = caption_layout = None
            continue
        if not _PINOUT_PAGE.search(text) and not (layout and layout[0] == pno - 1) and not (caption_layout and caption_layout[0] == pno - 1):
            continue
        try:
            tables = document[pno - 1].find_tables().tables
        except Exception:
            continue
        caption = _PIN_TABLE_CAPTION.search(text)
        caption_pkg = None
        if caption:
            caption_pkg = ((caption.group("pkg") or caption.group("pkg_b")).upper(), int(caption.group("pins") or caption.group("pins_b")))
        elif single_package and _PIN_TABLE_CAPTION_BARE.search(text):
            caption_pkg = single_package
        for table in tables:
            rows = [[(c or "").strip() for c in row] for row in table.extract()]
            if len(rows) < 6:
                continue
            width = len(rows[0])
            # Header: the first two rows, looking for package cells.
            pkg_cols: dict[int, tuple[str, int]] = {}
            header_rows = 0
            for hi in range(min(2, len(rows))):
                found = {i: p for i, c in enumerate(rows[hi]) if (p := _package_in_cell(c))}
                if found:
                    pkg_cols = found
                    header_rows = hi + 1
                    break
            # Pin-name column: the one with the most port-pin cells.
            counts = [sum(1 for r in rows[header_rows:] if i < len(r) and _PIN_NAME_CELL.match(r[i].replace("\n", ""))) for i in range(width)]
            pin_col = max(range(width), key=lambda i: counts[i]) if width else 0
            if counts[pin_col] < 5:
                continue
            if not pkg_cols and layout and layout[0] == pno - 1 and layout[2] == pin_col and width >= max(layout[1]) + 1:
                pkg_cols = layout[1]  # continuation page of the same table
            if pkg_cols:
                layout = (pno, pkg_cols, pin_col)
                for col, (pkg, pins) in pkg_cols.items():
                    entry = per_package.setdefault((pkg, pins), {"pins": set(), "pages": set()})
                    for r in rows[header_rows:]:
                        if col >= len(r) or pin_col >= len(r):
                            continue
                        name = _pin_name_in_cell(r[pin_col])
                        cell = r[col].strip()
                        if name and cell and cell not in ("-", "—", "–", "N/A", "NC"):
                            entry["pins"].add(name)
                            entry["pages"].add(pno)
            else:
                pkg_key = caption_pkg
                if pkg_key is None and caption_layout and caption_layout[0] == pno - 1:
                    pkg_key = caption_layout[1]  # the captioned table continues on this page
                if pkg_key is None:
                    continue
                caption_layout = (pno, pkg_key, width, pin_col)
                entry = per_package.setdefault(pkg_key, {"pins": set(), "pages": set()})
                for r in rows[header_rows:]:
                    name = _pin_name_in_cell(r[pin_col]) if pin_col < len(r) else None
                    if name:
                        entry["pins"].add(name)
                        entry["pages"].add(pno)
    out: list[dict[str, Any]] = []
    for (pkg, pins), entry in sorted(per_package.items(), key=lambda kv: kv[0][1]):
        io = len(entry["pins"])
        if io < 6 or io >= pins - 1:
            continue  # a package always spends pins on supply; equality means the table was not a pinout
        pages = sorted(entry["pages"])
        out.append({"pin_count": pins, "package": pkg, "io_count": io, "pattern": "pin_table_count", "verbatim": f"{io} port pins listed with a pin number for {pkg}{pins} in the pinout table", "receipt": {"page": pages[0], "pages": pages[:8]}})
    return out[:24]


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
    features = read_features(page_texts) + read_features_table(document, page_texts) + read_description_facts(page_texts)
    chapter_features = read_chapter_features(page_texts, chapters, document.page_count)
    memory = read_memory(document, page_texts, chapters)
    prose_facts = read_prose_facts(page_texts)
    io_by_package = read_io_by_package(page_texts)
    for row in read_io_from_pin_tables(document, page_texts):
        # A counted pinout complements a stated count; when both exist for
        # the same package, keep both so a disagreement is visible downstream.
        io_by_package.append(row)
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
        "io_by_package": io_by_package,
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
            "io_by_package_rows": len(io_by_package),
        },
    }
