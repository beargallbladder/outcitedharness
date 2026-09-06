#!/usr/bin/env python3
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

Deterministic PyMuPDF text only; documents whose covers are scans come back
scope_absent with reason no_text_layer_on_front_matter so a vision pass can
be scoped separately. Nothing is inferred: if the cover does not print a
scope sentence, scope_absent is the answer.
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
# "For GD32F405xx, GD32F407xx ... and GD32F470xx" -- self-delimiting: the
# match ends where the comma/and-joined token list ends, so cover pages
# without sentence punctuation still parse.
SCOPE_FOR_LIST = re.compile(
    rf"\bFor\s+{_TOKEN}(?:\s*(?:,|and|&)\s*{_TOKEN})*",
)
SCOPE_PROSE = re.compile(
    r"\b(Applicable\s+(?:products?|to)[:\s].{4,300}?"
    r"|This\s+(?:user\s+manual|reference\s+manual|document|datasheet)\s+"
    r"(?:applies\s+to|covers|describes).{4,300}?)(?:[.]|$)",
    re.IGNORECASE,
)
# Vendor series tokens: an uppercase alphanumeric stem with digits, optional
# lowercase wildcard tail (GD32F405xx, STM32F42xxx, S32K39, EFR32FG23).
SERIES_TOKEN = re.compile(r"\b([A-Z]{2,10}\d{1,4}[A-Z0-9]*(?:x{1,3})?)\b")

CLASS_WORDS = (
    ("reference manual", "reference_manual"),
    ("user manual", "user_manual"),
    ("application note", "application_note"),
    ("errata", "errata"),
    ("datasheet", "datasheet"),
    ("data sheet", "datasheet"),
)


def read_scope(path: Path) -> dict[str, object]:
    document = pymupdf.open(path)
    page_count = int(document.page_count)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row: dict[str, object] = {
        "schema": "harness.electronics-document-scope.v1",
        "source_path": str(path),
        "document_sha256": digest,
        "page_count": page_count,
        "governs_series": [],
        "scope_statement": None,
        "scope_page": None,
        "document_class": "unknown",
        "title_verbatim": None,
        "scope_absent": True,
        "scope_absent_reason": None,
    }

    front_text: list[str] = []
    for index in range(min(page_count, FRONT_PAGES)):
        front_text.append(document[index].get_text())
    joined = "\n".join(front_text)
    if len(joined.strip()) < 40:
        row["scope_absent_reason"] = "no_text_layer_on_front_matter"
        return row

    lowered = joined.lower()
    for needle, label in CLASS_WORDS:
        if needle in lowered:
            row["document_class"] = label
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
        candidates = [
            match.group(0) for match in SCOPE_FOR_LIST.finditer(normalized)
        ] + [match.group(1) for match in SCOPE_PROSE.finditer(normalized)]
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
            return row

    row["scope_absent_reason"] = "no_scope_statement_printed"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [read_scope(path) for path in args.pdfs]
    if args.output:
        if args.output.exists():
            raise SystemExit(f"output already exists: {args.output}")
        with args.output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    for row in rows:
        print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
