"""Read a document's cover-page scope statement: what does it govern?

CR's GD32F403 near-miss showed filename binding lies: GD32F4xx_User_Manual
names six series on its cover and F403 is not one of them. The authoritative
grain signal is the printed scope statement, so this reader returns it
verbatim, fail-closed.

Per document (JSONL row, schema harness.electronics-document-scope.v1):

  governs_series   : series/part tokens printed inside the scope statement
  scope_statement  : the verbatim printed sentence ("For GD32F405xx, ...")
  scope_page       : 1-based page carrying the statement
  document_class   : reference_manual|user_manual|datasheet|application_note|
                     errata|unknown  (from printed front-matter words only)
  title_verbatim   : best-effort verbatim title line(s) from the cover
  page_count       : PDF page count
  scope_absent     : true when no explicit scope statement is printed --
                     a first-class answer, never a guess from the filename

Deterministic PyMuPDF text only. Nothing is inferred: if the cover does not
print a scope sentence, scope_absent is the answer.

v2 additions (CR 2026-09-06 mails):

  status           : read_ok | extraction_failed. CR's GD32F4xx manual read
                     as zero hits for even "GD32" under a naive extractor --
                     an instrument artifact indistinguishable from "the
                     vendor states no scope". Those verdicts mean opposite
                     things, so unreadable pages are extraction_failed and
                     NEVER scope_absent.
  control_token    : caller-supplied token that MUST appear in the front
                     matter (the vendor's own family stem, e.g. STM32).
                     Missing token => extraction_failed:control_token_absent.
  document_subject : device_family | cpu_core | peripheral | unknown.
                     A Cortex programming manual is family-grain for no
                     family: it describes the core, not the device. 119 of
                     CR's gold pages rested on exactly that confusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pymupdf

FRONT_PAGES = 5

# A scope sentence opens with "For ", "Applicable to", "This manual/document
# ... applies to", etc., and must contain at least one part-series token.
# Cover statements wrap across lines ("... GD32F427xx,\nGD32F450xx and
# GD32F470xx"), so matching runs over whitespace-normalized page text and the
# statement ends at sentence punctuation or a token that stops reading like a
# series list.
_TOKEN = r"[A-Z]{2,10}\d[A-Za-z0-9]*(?:x{1,3})?"
# ST writes composite tokens ("STM32F405xx/07xx" meaning 405xx and 407xx);
# they are preserved exactly as printed, never expanded -- expansion is the
# consumer's judgement call, ours is verbatim capture.
_TOKEN_COMPOSITE = _TOKEN + r"(?:/[0-9A-Za-z]{1,8})?"
# "For GD32F405xx, GD32F407xx ... and GD32F470xx" -- self-delimiting: the
# match ends where the comma/and-joined token list ends, so cover pages
# without sentence punctuation still parse.
SCOPE_FOR_LIST = re.compile(
    rf"\bFor\s+{_TOKEN}(?:\s*(?:,|and|&)\s*{_TOKEN})*",
)
# ST reference-manual introductions: "It provides complete information on
# how to use the STM32F405xx/07xx, STM32F415xx/17xx, STM32F42xxx and
# STM32F43xxx microcontroller memory and peripherals."
SCOPE_USE_LIST = re.compile(
    rf"how\s+to\s+use\s+the\s+{_TOKEN_COMPOSITE}"
    rf"(?:\s*(?:,|and|&)\s*{_TOKEN_COMPOSITE})*",
)
# Inverted ST phrasings: "...how to use the memory and peripherals of
# STM32F446xx microcontrollers" (RM0390) and "...complements the datasheets
# of the STM32G0x1 microcontrollers" (RM0444).
SCOPE_OF_LIST = re.compile(
    rf"(?:peripherals|datasheets?)\s+of\s+(?:the\s+)?{_TOKEN_COMPOSITE}"
    rf"(?:\s*(?:,|and|&)\s*{_TOKEN_COMPOSITE})*",
)
SCOPE_PROSE = re.compile(
    r"\b(Applicable\s+(?:products?|to)[:\s].{4,300}?"
    r"|This\s+(?:user\s+manual|reference\s+manual|document|datasheet)\s+"
    r"(?:applies\s+to|covers|describes).{4,300}?)(?:[.]|$)",
    re.IGNORECASE,
)
# Vendor series tokens: an uppercase alphanumeric stem with digits, then a
# tail of uppercase/digits with lowercase `x` wildcards allowed ANYWHERE, not
# just trailing -- ST writes mid-token wildcards (STM32G0x1, STM32F411xC).
# Optional ST-style composite tail preserved verbatim (STM32F405xx/07xx).
SERIES_TOKEN = re.compile(
    r"\b([A-Z]{2,10}\d{1,4}[A-Zx0-9]*(?:/[0-9A-Za-z]{1,8})?)\b"
)

CLASS_WORDS = (
    ("reference manual", "reference_manual"),
    ("user manual", "user_manual"),
    ("programming manual", "programming_manual"),
    ("application note", "application_note"),
    ("errata", "errata"),
    ("datasheet", "datasheet"),
    ("data sheet", "datasheet"),
)

# Subject classification, printed front-matter words only.
# cpu_core: the document scopes itself to a processor core ("STM32 Cortex-M4
# MCUs and MPUs programming manual") rather than a device family. peripheral:
# it scopes to one peripheral block (dsPIC33/PIC24 FRM "I2C"). Signals are
# deliberately narrow; anything not clearly core- or peripheral-scoped that
# names device series stays device_family, else unknown.
_CORE_SIGNAL = re.compile(
    r"\b(cortex[\s-]?m\d+\w*|arm\S{0,8}\s+core|instruction\s+set|"
    r"core\s+programming)\b",
    re.IGNORECASE,
)
_PERIPHERAL_NAMES = re.compile(
    r"\b(I2C|SPI|UART|USART|USB|CAN|ADC|DAC|DMA|RTC|TIMER|PWM|ETHERNET|"
    r"FLASH\s+PROGRAM)\b",
)
_PERIPHERAL_DOC = re.compile(
    r"(framework\s+reference\s+manual|FRM|section\s+of\s+the)",
    re.IGNORECASE,
)


def _document_subject(
    title: str | None,
    front: str,
    document_class: str,
    has_series: bool,
) -> str:
    header = f"{title or ''}\n{front[:2000]}"
    if document_class == "programming_manual" and _CORE_SIGNAL.search(header):
        return "cpu_core"
    if _CORE_SIGNAL.search(title or "") and "programming" in header.lower():
        return "cpu_core"
    if title and _PERIPHERAL_NAMES.search(title) and (
        _PERIPHERAL_DOC.search(header)
        or document_class == "reference_manual"
    ) and not has_series:
        return "peripheral"
    if has_series:
        return "device_family"
    if document_class in {"reference_manual", "user_manual", "datasheet"}:
        return "device_family"
    return "unknown"


def read_scope(path: Path, control_token: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "schema": "harness.electronics-document-scope.v2",
        "source_path": str(path),
        "document_sha256": None,
        "page_count": None,
        "status": "read_ok",
        "extraction_failed_reason": None,
        "control_token": control_token,
        "control_token_found": None,
        "governs_series": [],
        "scope_statement": None,
        "scope_page": None,
        "document_class": "unknown",
        "document_subject": "unknown",
        "title_verbatim": None,
        "scope_absent": True,
        "scope_absent_reason": None,
    }
    try:
        document = pymupdf.open(path)
        row["page_count"] = int(document.page_count)
        row["document_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception as error:  # noqa: BLE001 - unreadable file is a verdict
        row["status"] = "extraction_failed"
        row["extraction_failed_reason"] = f"pdf_open_failed: {error}"
        row["scope_absent"] = None
        return row
    page_count = int(document.page_count)

    front_text: list[str] = []
    for index in range(min(page_count, FRONT_PAGES)):
        front_text.append(document[index].get_text())
    joined = "\n".join(front_text)
    if len(joined.strip()) < 40:
        # An unreadable page is an instrument failure, never a scope verdict:
        # CR's CID-encoded GD32F4xx manual read as zero text and would have
        # bound blind as "vendor states no scope".
        row["status"] = "extraction_failed"
        row["extraction_failed_reason"] = "no_text_layer_on_front_matter"
        row["scope_absent"] = None
        return row

    if control_token:
        found = control_token.upper() in joined.upper()
        row["control_token_found"] = found
        if not found:
            row["status"] = "extraction_failed"
            row["extraction_failed_reason"] = "control_token_absent"
            row["scope_absent"] = None
            return row

    # Classify by the EARLIEST class phrase printed on the cover: documents
    # self-declare their class in the header, while cross-references to other
    # classes ("Reference manuals of STM32F3 Series...") appear further down.
    # PM0214 is the canonical trap.
    cover_lowered = front_text[0].lower()
    lowered = joined.lower()
    for source in (cover_lowered, lowered):
        positions = [
            (source.find(needle), label)
            for needle, label in CLASS_WORDS
            if needle in source
        ]
        if positions:
            row["document_class"] = min(positions)[1]
            break

    # Title: first cover lines that are neither vendor boilerplate nor empty.
    cover_lines = [line.strip() for line in front_text[0].splitlines()]
    title_lines = [
        line
        for line in cover_lines
        if line
        and not re.fullmatch(r"[\s\d.()-]+", line)
        and "inc" not in line.lower()
        and "revision" not in line.lower()
    ]
    if title_lines:
        row["title_verbatim"] = " ".join(title_lines[:2])

    for page_index in range(min(page_count, FRONT_PAGES)):
        normalized = " ".join(front_text[page_index].split())
        candidates = (
            [match.group(0) for match in SCOPE_FOR_LIST.finditer(normalized)]
            + [
                match.group(0)
                for match in SCOPE_USE_LIST.finditer(normalized)
            ]
            + [
                match.group(0)
                for match in SCOPE_OF_LIST.finditer(normalized)
            ]
            + [match.group(1) for match in SCOPE_PROSE.finditer(normalized)]
        )
        for statement in candidates:
            statement = " ".join(statement.split())
            tokens = []
            for token in SERIES_TOKEN.findall(statement):
                if token not in tokens:
                    tokens.append(token)
            if not tokens:
                continue
            row["governs_series"] = tokens
            row["scope_statement"] = statement
            row["scope_page"] = page_index + 1
            row["scope_absent"] = False
            break
        if not row["scope_absent"]:
            break

    if row["scope_absent"]:
        row["scope_absent_reason"] = "no_scope_statement_printed"
    row["document_subject"] = _document_subject(
        row.get("title_verbatim"),  # type: ignore[arg-type]
        joined,
        str(row["document_class"]),
        bool(row["governs_series"]),
    )
    return row
