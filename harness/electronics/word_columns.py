"""Deterministic word-level package-column extraction for digitally-born PDFs.

The page's own text spans are the primary source. Identifier words are
assigned to package columns by x-overlap with the printed column headers
and matched to signal names within the same row band, so every emitted
row claim carries coordinate provenance back to the source document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


PACKAGE_HEADER = re.compile(
    r"^(?:(?:LQFP|QFN|TQFN|UFBGA|TFBGA|WLCSP|UFQFPN|VFBGA|LGA|BGA|CSP|QFP"
    r"|TSSOP|SSOP|MSOP|SOP|SOIC|DIP)\d{2,4}(?:\+\d{1,3})?"
    r"|\d{2,4}-(?:LQFP|QFN|TQFN|UFBGA|TFBGA|WLCSP|UFQFPN|VFBGA|LGA|BGA"
    r"|CSP|QFP|TSSOP|SSOP|MSOP|SOP|SOIC|TQFP))$"
)
PIN_COLUMN_HEADER = re.compile(
    r"^(?:pins?|pin\s*\(s\)|pin\s*no\.?|pin\s*number|pin#|ball|ball\s*no\.?|no\.?)$",
    re.IGNORECASE,
)
BARE_PIN_HEADER = re.compile(r"^pins?$", re.IGNORECASE)
NAME_COLUMN_HEADER = re.compile(r"^(?:pin/ball|pin)?\s*name$", re.IGNORECASE)
IDENTIFIER = re.compile(r"^\d{1,4}$|^[A-Z]{1,2}\d{1,3}$")
SIGNAL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_/.#&-]{0,23}$")

COLUMN_LEFT_MARGIN = 5.0
COLUMN_RIGHT_MARGIN = 12.0
NAME_ANCHOR_MARGIN = 32.0
ROW_BAND_TOLERANCE = 9.0
HEADER_BAND_TOLERANCE = 30.0
PACKAGE_HEADER_BAND_TOLERANCE = 20.0
HEADER_ANCHOR_RELEVANCE = 60.0
NAME_JOIN_GAP = 10.0
MINIMUM_COLUMN_ROWS = 2

Word = tuple[float, float, float, float, str]


@dataclass(frozen=True)
class WordSpan:
    text: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class PinRowClaim:
    pin_no: str
    name: str
    identifier_span: WordSpan
    name_span: WordSpan


@dataclass(frozen=True)
class PackageColumn:
    header: str
    column_key: str
    header_span: WordSpan
    rows: tuple[PinRowClaim, ...]


def package_pin_bogey(package_token: str) -> tuple[int, int] | None:
    match = re.fullmatch(
        r"[A-Z]+(\d{2,4})(?:\+(\d{1,3}))?", package_token or ""
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _center(word: Word, axis: int) -> float:
    return (word[axis] + word[axis + 2]) / 2


def _window(header: Word) -> tuple[float, float]:
    return (
        header[0] - COLUMN_LEFT_MARGIN,
        header[2] + COLUMN_RIGHT_MARGIN,
    )


def _in(value: float, window: tuple[float, float]) -> bool:
    return window[0] <= value <= window[1]


def _runs(line: list[Word]) -> list[list[Word]]:
    runs: list[list[Word]] = []
    for word in sorted(line, key=lambda w: w[0]):
        if runs and word[0] - runs[-1][-1][2] <= NAME_JOIN_GAP:
            runs[-1].append(word)
        else:
            runs.append([word])
    return runs


def _voronoi_band(
    identifiers: list[Word],
    index: int,
) -> tuple[float, float]:
    y = _center(identifiers[index], 1)
    lower = (
        (_center(identifiers[index - 1], 1) + y) / 2
        if index > 0
        else y - ROW_BAND_TOLERANCE
    )
    upper = (
        (y + _center(identifiers[index + 1], 1)) / 2
        if index + 1 < len(identifiers)
        else y + ROW_BAND_TOLERANCE
    )
    return lower, upper


def _rows_for_anchor(
    identifier_words: list[Word],
    body: list[Word],
    anchor: Word,
    identifier_windows: list[tuple[float, float]],
    header_words: list[Word] | None = None,
) -> dict[str, PinRowClaim]:
    name_window = (
        anchor[0] - NAME_ANCHOR_MARGIN,
        anchor[2] + NAME_ANCHOR_MARGIN,
    )
    if header_words is not None:
        identifier_windows = [
            window
            for window, header in zip(identifier_windows, header_words)
            if abs(header[1] - anchor[1]) <= HEADER_ANCHOR_RELEVANCE
        ]
    anchor_x = _center(anchor, 0)
    identifiers = sorted(identifier_words, key=lambda w: _center(w, 1))
    rows: dict[str, PinRowClaim] = {}
    for index, identifier in enumerate(identifiers):
        y = _center(identifier, 1)
        band_low, band_high = _voronoi_band(identifiers, index)
        candidates = sorted(
            [
                word
                for word in body
                if _in(_center(word, 0), name_window)
                and not any(
                    _in(_center(word, 0), window)
                    for window in identifier_windows
                )
                and (
                    band_low <= _center(word, 1) < band_high
                    or abs(_center(word, 1) - y) <= ROW_BAND_TOLERANCE
                )
                and SIGNAL_NAME.match(word[4])
                and word[4] != identifier[4]
            ],
            key=lambda w: (_center(w, 1), w[0]),
        )
        if not candidates:
            continue
        lines: list[list[Word]] = []
        for word in candidates:
            if lines and (
                abs(_center(word, 1) - _center(lines[-1][0], 1)) <= 4.0
            ):
                lines[-1].append(word)
            else:
                lines.append([word])
        primary_line = min(
            lines, key=lambda line: abs(_center(line[0], 1) - y)
        )
        primary_run = min(
            _runs(primary_line),
            key=lambda run: abs(
                (_center(run[0], 0) + _center(run[-1], 0)) / 2 - anchor_x
            ),
        )
        parts: list[Word] = []
        for line in lines:
            line_runs = _runs(line)
            if line is primary_line:
                parts.extend(primary_run)
            elif len(line_runs) == 1:
                parts.extend(line_runs[0])
        ordered = sorted(parts, key=lambda w: (_center(w, 1), w[0]))
        text = "".join(word[4] for word in ordered)
        x0 = min(word[0] for word in ordered)
        y0 = min(word[1] for word in ordered)
        x1 = max(word[2] for word in ordered)
        y1 = max(word[3] for word in ordered)
        rows.setdefault(
            identifier[4],
            PinRowClaim(
                pin_no=identifier[4],
                name=text,
                identifier_span=WordSpan(
                    identifier[4],
                    (identifier[0], identifier[1], identifier[2], identifier[3]),
                ),
                name_span=WordSpan(text, (x0, y0, x1, y1)),
            ),
        )
    return rows


def _header_like(word: Word) -> bool:
    return bool(
        PACKAGE_HEADER.match(word[4])
        or PIN_COLUMN_HEADER.match(word[4])
        or NAME_COLUMN_HEADER.match(word[4])
    )


def _in_header_band(
    header: Word,
    words: list[Word],
    tolerance: float = HEADER_BAND_TOLERANCE,
) -> bool:
    return any(
        word is not header
        and _header_like(word)
        and abs(word[1] - header[1]) <= tolerance
        for word in words
    )


def _load_words(source: Path, page_1based: int) -> list[Word] | None:
    import pymupdf

    if page_1based < 1:
        return None
    document = pymupdf.open(source)
    if page_1based > document.page_count:
        document.close()
        return None
    page = document[page_1based - 1]
    raw = page.get_text("words")
    rotation = page.rotation % 360
    if rotation:
        mediabox = page.mediabox
        width = mediabox.x1 - mediabox.x0
        height = mediabox.y1 - mediabox.y0
        words: list[Word] = []
        for w in raw:
            x0, y0, x1, y1, text = (
                float(w[0]),
                float(w[1]),
                float(w[2]),
                float(w[3]),
                str(w[4]),
            )
            if rotation == 90:
                words.append((height - y1, x0, height - y0, x1, text))
            elif rotation == 180:
                words.append((width - x1, height - y1, width - x0, height - y0, text))
            elif rotation == 270:
                words.append((y0, width - x1, y1, width - x0, text))
            else:
                words.append((x0, y0, x1, y1, text))
    else:
        words = [
            (float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4]))
            for w in raw
        ]
    document.close()
    return words


def _pipeline(
    words: list[Word],
    *,
    body_words: list[Word] | None = None,
) -> dict[str, PackageColumn]:
    header_candidates = [
        w
        for w in words
        if (
            PACKAGE_HEADER.match(w[4]) or PIN_COLUMN_HEADER.match(w[4])
        )
        and _in_header_band(
            w,
            words,
            (
                PACKAGE_HEADER_BAND_TOLERANCE
                if PACKAGE_HEADER.match(w[4])
                else HEADER_BAND_TOLERANCE
            ),
        )
    ]
    name_anchors = [
        w
        for w in words
        if NAME_COLUMN_HEADER.match(w[4]) and _in_header_band(w, words)
    ]
    if not header_candidates or not name_anchors:
        return {}
    header_words: list[Word] = []
    for header in header_candidates:
        window = _window(header)
        beneath = [
            w
            for w in words
            if w[1] > header[3]
            and IDENTIFIER.match(w[4])
            and _in(_center(w, 0), window)
        ]
        if len(beneath) < MINIMUM_COLUMN_ROWS:
            continue
        if BARE_PIN_HEADER.match(header[4]):
            pure_numbers = [
                w for w in beneath if re.fullmatch(r"\d{1,4}", w[4])
            ]
            if len(pure_numbers) < MINIMUM_COLUMN_ROWS:
                continue
        header_words.append(header)
    if not header_words:
        return {}
    if body_words is None:
        header_bottom = min(w[3] for w in header_words)
        body = [
            w
            for w in words
            if w[1] > header_bottom
            and w[4].strip() not in {"-", "—", "–", ""}
        ]
    else:
        body = [
            w
            for w in body_words
            if w[4].strip() not in {"-", "—", "–", ""}
        ]
    identifier_windows = [_window(h) for h in header_words]

    columns: dict[str, PackageColumn] = {}
    for header in header_words:
        window = _window(header)
        identifier_words = [
            w
            for w in body
            if IDENTIFIER.match(w[4]) and _in(_center(w, 0), window)
        ]
        if not identifier_words:
            continue
        best_rows: dict[str, PinRowClaim] = {}
        best_anchor: Word | None = None
        for anchor in name_anchors:
            rows = _rows_for_anchor(
                identifier_words,
                body,
                anchor,
                identifier_windows,
                header_words=header_words,
            )
            anchor_distance = abs(
                _center(anchor, 0) - _center(header, 0)
            )
            best_distance = (
                abs(_center(best_anchor, 0) - _center(header, 0))
                if best_anchor is not None
                else None
            )
            if len(rows) > len(best_rows) or (
                len(rows) == len(best_rows)
                and rows
                and (best_distance is None or anchor_distance < best_distance)
            ):
                best_rows = rows
                best_anchor = anchor
        ordered = tuple(
            sorted(best_rows.values(), key=lambda c: c.identifier_span.bbox[1])
        )
        if len(ordered) >= MINIMUM_COLUMN_ROWS:
            key = f"{header[4]}@{round(header[0])}"
            columns[key] = PackageColumn(
                header=header[4],
                column_key=key,
                header_span=WordSpan(
                    header[4], (header[0], header[1], header[2], header[3])
                ),
                rows=ordered,
            )
    return columns


CONTINUATION_LOOKBACK = 8


def extract_pin_columns(
    source: Path,
    page_1based: int,
) -> dict[str, PackageColumn]:
    """Extract package pin columns from one page of a digitally-born PDF.

    Continuation pages of multi-page pin tables carry no header band; when
    a page yields no columns, the headers are re-validated on preceding
    pages and applied to this page's rows.
    """

    words = _load_words(source, page_1based)
    if words is None:
        return {}
    columns = _pipeline(words)
    if columns:
        return columns
    for lookback in range(1, CONTINUATION_LOOKBACK + 1):
        previous = _load_words(source, page_1based - lookback)
        if previous is None:
            break
        if _pipeline(previous):
            return _pipeline(previous, body_words=words)
    return {}


def select_package_columns(
    columns: dict[str, PackageColumn],
    requested_package: str,
) -> list[PackageColumn]:
    """Pick the columns serving a requested package.

    Exact normalized header text wins (covers digits-first table titles
    like 64-TQFP), then the locator's package identity matcher, then all
    generic pin columns merged (two-group tables split odd/even rows
    across side-by-side column groups).
    """

    from harness.electronics.locator import match_package_column

    requested_key = re.sub(r"[^A-Z0-9]", "", str(requested_package).upper())
    if not requested_key:
        return []
    exact = [
        column
        for column in columns.values()
        if re.sub(r"[^A-Z0-9]", "", column.header.upper()) == requested_key
    ]
    if len(exact) == 1:
        return exact
    matched = match_package_column(
        requested_package, [column.header for column in columns.values()]
    )
    if matched is not None:
        candidates = [
            column
            for column in columns.values()
            if column.header == matched
        ]
        if len(candidates) == 1:
            return candidates
    return [
        column
        for column in columns.values()
        if not re.search(r"[A-Z]+\d", column.header)
    ]


def capability_result(column: PackageColumn) -> dict[str, object]:
    return {
        "package": column.header,
        "pins": [
            {"pin_no": row.pin_no, "name": row.name} for row in column.rows
        ],
    }
