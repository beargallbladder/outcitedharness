from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CodeDocument:
    path: str
    content: str
    content_hash: str
    language: str


@dataclass(frozen=True)
class RepoSnapshot:
    repo_id: str
    source_host: str
    repo_root: str
    remote: str | None
    branch: str
    head: str
    dirty: bool
    state_hash: str
    file_hashes: dict[str, str]
    documents: tuple[CodeDocument, ...] = ()
    deleted: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticSlice:
    path: str
    symbol: str | None
    symbol_type: str | None
    start_line: int
    end_line: int
    text: str
    text_hash: str


@dataclass(frozen=True)
class SymbolRecord:
    path: str
    name: str
    kind: str
    line: int


@dataclass(frozen=True)
class ImportRecord:
    path: str
    module: str
    line: int


@dataclass(frozen=True)
class GCIHit:
    repo_id: str
    source_host: str
    repo_root: str
    revision: str
    state_hash: str
    path: str
    symbol: str | None
    symbol_type: str | None
    start_line: int
    end_line: int
    score: float
    match_type: str
    text: str


@dataclass(frozen=True)
class PreparedDocument:
    document: CodeDocument
    slices: tuple[SemanticSlice, ...]
    symbols: tuple[SymbolRecord, ...] = ()
    imports: tuple[ImportRecord, ...] = ()
    embeddings: tuple[tuple[float, ...], ...] = ()


@dataclass
class IndexStats:
    files: int = 0
    changed: int = 0
    unchanged: int = 0
    deleted: int = 0
    slices: int = 0
    embedded: int = 0
    encoder_calls: int = 0
    backoffs: int = 0
    latency_ms: list[float] = field(default_factory=list)
