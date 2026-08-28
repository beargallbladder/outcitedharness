from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass

from harness.gci.models import ImportRecord, SemanticSlice, SymbolRecord
from harness.task.code_index import chunk_source


MAX_SLICE_CHARS = 450
SLICE_OVERLAP_CHARS = 90

_DECLARATION = re.compile(
    r"^\s*(?:(?:export|async|pub)\s+)*(class|def|function|interface|type|enum|struct|trait|fn)\s+"
    r"([A-Za-z_$][A-Za-z0-9_$]*)",
    re.MULTILINE,
)
_CONST_DECLARATION = re.compile(
    r"^\s*(?:(?:export|pub)\s+)?(?:const|let|var|static)\s+"
    r"([A-Za-z_$][A-Za-z0-9_$]*)",
    re.MULTILINE,
)
_IMPORT = re.compile(
    r"^\s*(?:from\s+([A-Za-z0-9_.$/:-]+)\s+import|import\s+(?:[^\"']+\s+from\s+)?[\"']?([^\"';\s]+))",
    re.MULTILINE,
)


@dataclass(frozen=True)
class _Unit:
    start_line: int
    end_line: int
    text: str


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def extract_symbols(path: str, content: str, language: str) -> list[SymbolRecord]:
    if language == "python":
        try:
            tree = ast.parse(content)
        except SyntaxError:
            tree = None
        if tree is not None:
            rows: list[SymbolRecord] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    rows.append(SymbolRecord(path=path, name=node.name, kind=kind, line=node.lineno))
            return sorted(rows, key=lambda row: (row.line, row.name))
    rows = [
        SymbolRecord(
            path=path,
            name=match.group(2),
            kind=match.group(1),
            line=_line_for_offset(content, match.start()),
        )
        for match in _DECLARATION.finditer(content)
    ]
    rows.extend(
        SymbolRecord(
            path=path,
            name=match.group(1),
            kind="constant",
            line=_line_for_offset(content, match.start()),
        )
        for match in _CONST_DECLARATION.finditer(content)
    )
    return sorted(rows, key=lambda row: (row.line, row.name))


def extract_imports(path: str, content: str, language: str) -> list[ImportRecord]:
    if language == "python":
        try:
            tree = ast.parse(content)
        except SyntaxError:
            tree = None
        if tree is not None:
            rows: list[ImportRecord] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    rows.extend(
                        ImportRecord(path=path, module=name.name, line=node.lineno)
                        for name in node.names
                    )
                elif isinstance(node, ast.ImportFrom):
                    rows.append(
                        ImportRecord(path=path, module=node.module or "." * node.level, line=node.lineno)
                    )
            return sorted(rows, key=lambda row: (row.line, row.module))
    rows = []
    for match in _IMPORT.finditer(content):
        module = match.group(1) or match.group(2)
        if module:
            rows.append(
                ImportRecord(
                    path=path,
                    module=module,
                    line=_line_for_offset(content, match.start()),
                )
            )
    return rows


def _structural_units(content: str) -> list[_Unit]:
    return [_Unit(start, end, body) for start, end, body in chunk_source(content)]


def _symbol_for_line(symbols: list[SymbolRecord], line: int) -> SymbolRecord | None:
    candidate = None
    for symbol in symbols:
        if symbol.line > line:
            break
        candidate = symbol
    return candidate


def semantic_slices(
    path: str,
    content: str,
    language: str,
    *,
    max_chars: int = MAX_SLICE_CHARS,
    overlap_chars: int = SLICE_OVERLAP_CHARS,
) -> list[SemanticSlice]:
    if max_chars < 80 or max_chars > MAX_SLICE_CHARS:
        raise ValueError(f"max_chars must be between 80 and {MAX_SLICE_CHARS}")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be nonnegative and smaller than max_chars")
    symbols = extract_symbols(path, content, language)
    units = _structural_units(content)
    if not units and content.strip():
        units = [_Unit(1, max(1, len(content.splitlines())), content)]
    out: list[SemanticSlice] = []
    step = max_chars - overlap_chars
    for unit in units:
        cursor = 0
        while cursor < len(unit.text):
            body = unit.text[cursor : cursor + max_chars]
            if not body.strip():
                break
            local_start = unit.text.count("\n", 0, cursor)
            local_end = local_start + body.count("\n")
            start_line = unit.start_line + local_start
            end_line = min(unit.end_line, unit.start_line + local_end)
            symbol = _symbol_for_line(symbols, start_line)
            out.append(
                SemanticSlice(
                    path=path,
                    symbol=symbol.name if symbol else None,
                    symbol_type=symbol.kind if symbol else None,
                    start_line=start_line,
                    end_line=max(start_line, end_line),
                    text=body,
                    text_hash=_digest(body),
                )
            )
            if cursor + max_chars >= len(unit.text):
                break
            cursor += step
    return out
