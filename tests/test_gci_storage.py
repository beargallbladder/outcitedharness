from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from harness.gci.indexer import snapshot_state_hash
from harness.gci.models import CodeDocument, IndexStats, PreparedDocument, RepoSnapshot
from harness.gci.slicing import (
    MAX_SLICE_CHARS,
    SLICE_OVERLAP_CHARS,
    extract_imports,
    extract_symbols,
    semantic_slices,
)
from harness.gci.storage import GCIStorageError, GCIStore, assert_isolated_db


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _snapshot(
    content: str,
    *,
    state_hash: str = "state-1",
    documents: bool = True,
) -> RepoSnapshot:
    document = CodeDocument(
        path="src/scoring.py",
        content=content,
        content_hash=_hash(content),
        language="python",
    )
    return RepoSnapshot(
        repo_id="repo-a",
        source_host="m5",
        repo_root="/work/repo-a",
        remote="https://example.invalid/repo-a.git",
        branch="main",
        head="abc123",
        dirty=True,
        state_hash=state_hash,
        file_hashes={document.path: document.content_hash},
        documents=(document,) if documents else (),
    )


def _prepared(snapshot: RepoSnapshot, value: float = 1.0) -> list[PreparedDocument]:
    out = []
    for document in snapshot.documents:
        slices = semantic_slices(document.path, document.content, document.language)
        out.append(
            PreparedDocument(
                document=document,
                slices=tuple(slices),
                symbols=tuple(extract_symbols(document.path, document.content, document.language)),
                imports=tuple(extract_imports(document.path, document.content, document.language)),
                embeddings=tuple((value, 0.0, 0.0) for _ in slices),
            )
        )
    return out


def test_semantic_slices_honor_encoder_contract_and_retain_symbol():
    body = "def percentile_score(values):\n" + "\n".join(
        f"    value_{index} = values[{index} % 3]" for index in range(80)
    )
    slices = semantic_slices("src/scoring.py", body, "python")
    assert len(slices) > 1
    assert all(len(row.text) <= MAX_SLICE_CHARS for row in slices)
    assert all(row.symbol == "percentile_score" for row in slices)
    assert any(
        left.text[-SLICE_OVERLAP_CHARS:] == right.text[:SLICE_OVERLAP_CHARS]
        for left, right in zip(slices, slices[1:])
    )


def test_symbol_and_import_extraction():
    content = "import math\nfrom stats.core import percentile\n\nclass Score:\n    def run(self):\n        return math.nan\n"
    symbols = extract_symbols("score.py", content, "python")
    imports = extract_imports("score.py", content, "python")
    assert {(row.name, row.kind) for row in symbols} == {
        ("Score", "class"),
        ("run", "function"),
    }
    assert {row.module for row in imports} == {"math", "stats.core"}


def test_store_commits_generation_and_searches(tmp_path: Path):
    store = GCIStore(tmp_path / "gci.sqlite")
    snapshot = _snapshot(
        "import math\n\ndef percentile_score(values):\n    return math.floor(sum(values))\n"
    )
    generation = store.commit_generation(snapshot, _prepared(snapshot), IndexStats())
    assert generation > 0
    assert store.repo_manifest("repo-a")["files"] == snapshot.file_hashes
    assert store.symbol_search("percentile_score")[0].repo_root == "/work/repo-a"
    assert store.exact_search("math.floor")[0].path == "src/scoring.py"
    assert store.semantic_search([1.0, 0.0, 0.0])[0].match_type == "semantic"
    assert store.metrics()["files"] == 1


def test_failed_generation_preserves_previous_searchable_state(tmp_path: Path):
    store = GCIStore(tmp_path / "gci.sqlite")
    first = _snapshot("def old_score():\n    return 'old implementation remains searchable'\n")
    store.commit_generation(first, _prepared(first), IndexStats())
    changed = _snapshot(
        "def new_score():\n    return 'new implementation should not commit'\n",
        state_hash="state-2",
    )
    broken = _prepared(changed)
    broken[0] = PreparedDocument(
        document=broken[0].document,
        slices=broken[0].slices,
        symbols=broken[0].symbols,
        imports=broken[0].imports,
        embeddings=(),
    )
    with pytest.raises(GCIStorageError, match="slice/vector mismatch"):
        store.commit_generation(changed, broken, IndexStats())
    assert store.exact_search("old implementation")
    assert not store.exact_search("new implementation")
    assert store.repo_manifest("repo-a")["state_hash"] == "state-1"


def test_store_rejects_categoryrank_paths():
    for path in (
        Path("/home/samkim/semantic_search/code.sqlite"),
        Path("/data/categoryrank/gci.sqlite"),
    ):
        with pytest.raises(GCIStorageError, match="CategoryRank"):
            assert_isolated_db(path)


def test_identical_relative_paths_remain_distinct_across_repositories(tmp_path: Path):
    store = GCIStore(tmp_path / "gci.sqlite")
    first = _snapshot("def shared():\n    return 'first repository implementation'\n")
    second_base = _snapshot(
        "def shared():\n    return 'second repository implementation'\n",
        state_hash="placeholder",
    )
    second = replace(
        second_base,
        repo_id="repo-b",
        repo_root="/work/repo-b",
        head="def456",
        state_hash=snapshot_state_hash("def456", second_base.file_hashes),
    )
    first = replace(
        first,
        state_hash=snapshot_state_hash(first.head, first.file_hashes),
    )
    store.commit_generation(first, _prepared(first, 1.0), IndexStats())
    store.commit_generation(second, _prepared(second, 0.5), IndexStats())
    hits = store.symbol_search("shared")
    assert {hit.repo_id for hit in hits} == {"repo-a", "repo-b"}
    assert {hit.path for hit in hits} == {"src/scoring.py"}


def test_old_generations_are_bounded(tmp_path: Path):
    store = GCIStore(tmp_path / "gci.sqlite")
    for index in range(4):
        snapshot = _snapshot(
            f"def score():\n    return 'generation {index} implementation'\n",
            state_hash=f"state-{index}",
        )
        store.commit_generation(snapshot, _prepared(snapshot), IndexStats())
    conn = store.connect()
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM generations WHERE repo_id='repo-a'"
        ).fetchone()[0]
        fts_generations = conn.execute(
            "SELECT COUNT(DISTINCT generation) FROM gci_fts WHERE repo_id='repo-a'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 2
    assert fts_generations == 2
