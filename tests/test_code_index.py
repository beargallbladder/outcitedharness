import json
from pathlib import Path

from harness.task.code_index import (
    SCHEMA_VERSION,
    chunk_source,
    connect,
    gather_paths_for_intent,
    index_repos,
    query_index,
)


def _fake_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        seed = sum(text.encode()) % 97
        vec = [0.0] * 8
        vec[seed % 8] = 1.0
        if "score" in text.lower() or "listing" in text.lower():
            vec[0] = 2.0
        out.append(vec)
    return out


def _rank_embed(texts: list[str]) -> list[list[float]]:
    """repoB/wrongtree aligns with the query; repoA/righttree is weaker."""
    out = []
    for text in texts:
        lowered = text.lower()
        if "wrongtree" in lowered:
            out.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif "righttree" in lowered:
            out.append([0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif "listing" in lowered or "score" in lowered:
            out.append([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            out.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return out


def _write_py(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"def {marker}():\n    return '{marker} listing score padding for min chunk'\n"
    )


def test_chunk_source_splits_on_defs():
    chunks = chunk_source(
        "def one():\n    return 1  # keep this chunk above the min size floor\n\n"
        "def two():\n    return 2  # second function should start a new chunk\n"
    )
    assert len(chunks) >= 2
    assert any("def one" in body for _s, _e, body in chunks)
    assert any("def two" in body for _s, _e, body in chunks)


def test_index_skips_unchanged_and_retrieves(tmp_path: Path):
    repo = tmp_path / "locdna"
    src = repo / "apps" / "web" / "src"
    src.mkdir(parents=True)
    (src / "score.ts").write_text(
        "export function scoreListing(row) {\n  return row.score\n}\n"
    )
    (src / "other.ts").write_text(
        "export function noop() {\n  return null\n}\n// padding so the chunk clears the min size\n"
    )
    db = tmp_path / "code_index.sqlite"

    first = index_repos([repo], db, embed=_fake_embed)
    assert first["files"] == 2
    assert first["embedded"] >= 2
    second = index_repos([repo], db, embed=_fake_embed)
    assert second["unchanged"] == 2
    assert second["embedded"] == 0

    hits = query_index(
        "where do we score a listing", db, repo_root=repo, limit=4, embed=_fake_embed
    )
    assert hits
    assert any("score.ts" in hit.path for hit in hits)
    assert all(hit.repo_root == str(repo.resolve()) for hit in hits)
    paths = gather_paths_for_intent(
        "score a listing", db, workspace=repo, limit=4, embed=_fake_embed
    )
    assert "apps/web/src/score.ts" in paths
    assert "package.json" not in paths


def test_same_relative_path_stays_distinct(tmp_path: Path):
    repo_a = tmp_path / "repoA"
    repo_b = tmp_path / "repoB"
    _write_py(repo_a / "src" / "scoring.py", "score_a")
    _write_py(repo_b / "src" / "scoring.py", "score_b")
    db = tmp_path / "code_index.sqlite"
    index_repos([repo_a, repo_b], db, embed=_fake_embed)
    conn = connect(db)
    rows = conn.execute("SELECT repo_root, path FROM files ORDER BY repo_root").fetchall()
    conn.close()
    roots = {row[0] for row in rows}
    assert str(repo_a.resolve()) in roots
    assert str(repo_b.resolve()) in roots
    assert [row[1] for row in rows] == ["src/scoring.py", "src/scoring.py"]
    assert len(rows) == 2


def test_active_workspace_filter_ignores_higher_foreign_scores(tmp_path: Path):
    repo_a = tmp_path / "repoA"
    repo_b = tmp_path / "repoB"
    _write_py(repo_a / "src" / "scoring.py", "righttree_one")
    _write_py(repo_a / "src" / "other.py", "righttree_two")
    _write_py(repo_b / "src" / "scoring.py", "wrongtree_hot")
    _write_py(repo_b / "src" / "extra.py", "wrongtree_hotter")
    db = tmp_path / "code_index.sqlite"
    index_repos([repo_a, repo_b], db, embed=_rank_embed)

    hits = query_index("listing score", db, repo_root=repo_a, limit=6, embed=_rank_embed)
    assert hits
    assert all(hit.repo_root == str(repo_a.resolve()) for hit in hits)
    assert all("wrongtree" not in hit.text for hit in hits)
    paths = gather_paths_for_intent(
        "listing score", db, workspace=repo_a, limit=6, embed=_rank_embed
    )
    assert "src/scoring.py" in paths
    assert all(not path.startswith("repo") for path in paths)


def test_filter_before_topk_keeps_weaker_active_hits(tmp_path: Path):
    repo_a = tmp_path / "repoA"
    repo_b = tmp_path / "repoB"
    _write_py(repo_a / "services" / "engine" / "scoring" / "dna.py", "righttree_dna")
    _write_py(repo_a / "src" / "scoring.py", "righttree_score")
    for i in range(8):
        _write_py(repo_b / "src" / f"hot_{i}.py", f"wrongtree_{i}")
    db = tmp_path / "code_index.sqlite"
    index_repos([repo_a, repo_b], db, embed=_rank_embed)

    global_if_unfiltered = query_index(
        "listing score", db, repo_root=repo_b, limit=6, embed=_rank_embed
    )
    assert len(global_if_unfiltered) == 6
    assert all(hit.repo_root == str(repo_b.resolve()) for hit in global_if_unfiltered)

    hits = query_index("listing score", db, repo_root=repo_a, limit=6, embed=_rank_embed)
    assert hits
    assert all(hit.repo_root == str(repo_a.resolve()) for hit in hits)
    paths = gather_paths_for_intent(
        "listing score", db, workspace=repo_a, limit=6, embed=_rank_embed
    )
    assert "services/engine/scoring/dna.py" in paths
    assert "src/scoring.py" in paths
    assert not any(path.startswith("repoA/") for path in paths)
    assert not any("hot_" in path for path in paths)


def test_unknown_workspace_returns_no_semantic_hits(tmp_path: Path):
    repo_b = tmp_path / "repoB"
    _write_py(repo_b / "src" / "scoring.py", "wrongtree_only")
    db = tmp_path / "code_index.sqlite"
    index_repos([repo_b], db, embed=_rank_embed)
    missing = tmp_path / "not_indexed"
    missing.mkdir()
    assert query_index("listing score", db, repo_root=missing, limit=6, embed=_rank_embed) == []
    assert (
        gather_paths_for_intent("listing score", db, workspace=missing, embed=_rank_embed) == []
    )
    assert gather_paths_for_intent("listing score", db, workspace=None, embed=_rank_embed) == []


def test_unknown_workspace_keeps_nonsemantic_gather(tmp_path: Path):
    from harness.dispatch import default_gather_calls, merge_tool_catalog

    repo_b = tmp_path / "repoB"
    _write_py(repo_b / "src" / "scoring.py", "wrongtree_only")
    db = tmp_path / "code_index.sqlite"
    index_repos([repo_b], db, embed=_rank_embed)
    catalog = merge_tool_catalog(
        {
            "read_file": ("path",),
            "search_files": ("path", "regex", "query"),
            "list_files": ("path", "recursive"),
        }
    )
    missing = tmp_path / "not_indexed"
    missing.mkdir()
    calls = default_gather_calls(catalog, "listing score", workspace=missing)
    blob = " ".join(c["function"]["arguments"] for c in calls)
    assert "src/scoring.py" not in blob
    names = {c["function"]["name"] for c in calls}
    assert "search_files" in names or "list_files" in names


def test_foreign_repo_does_not_waste_gather_slot(tmp_path: Path, monkeypatch):
    locdna = tmp_path / "locationlocationlocation"
    harness = tmp_path / "Harnessv1"
    _write_py(locdna / "services" / "engine" / "catalog" / "cohort.py", "righttree_cohort")
    _write_py(harness / "harness" / "stats.py", "wrongtree_stats")
    db = tmp_path / "code_index.sqlite"
    index_repos([locdna, harness], db, embed=_rank_embed)

    paths = gather_paths_for_intent(
        "listing score", db, workspace=locdna, limit=6, embed=_rank_embed
    )
    assert "services/engine/catalog/cohort.py" in paths
    assert "harness/stats.py" not in paths
    assert all(not path.startswith("Harnessv1/") for path in paths)
    assert all(not path.startswith("locationlocationlocation/") for path in paths)

    from harness.dispatch import default_gather_calls, merge_tool_catalog

    monkeypatch.setattr("harness.task.code_index.default_index_path", lambda root=None: db)
    monkeypatch.setattr("harness.task.code_index.embed_texts", _rank_embed)
    catalog = merge_tool_catalog({"read_file": ("path",)})
    calls = default_gather_calls(catalog, "listing score", workspace=locdna)
    args = [json.loads(c["function"]["arguments"]) for c in calls]
    assert "harness/stats.py" not in [row.get("path") for row in args]
    assert any(row.get("path") == "services/engine/catalog/cohort.py" for row in args)


def test_legacy_schema_is_rebuilt(tmp_path: Path):
    db = tmp_path / "code_index.sqlite"
    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE files (repo TEXT, path TEXT, file_hash TEXT, PRIMARY KEY (repo, path))")
    conn.execute("INSERT INTO files VALUES ('Harnessv1', 'harness/stats.py', 'abc')")
    conn.commit()
    conn.close()
    repo = tmp_path / "repoA"
    _write_py(repo / "src" / "scoring.py", "righttree")
    index_repos([repo], db, embed=_fake_embed)
    conn = connect(db)
    version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
    cols = {row[1] for row in conn.execute("PRAGMA table_info(files)").fetchall()}
    conn.close()
    assert int(version) == SCHEMA_VERSION
    assert "repo_root" in cols
    assert "repo" not in cols


def test_gather_prepends_index_hits(monkeypatch):
    from harness.dispatch import default_gather_calls, merge_tool_catalog

    monkeypatch.setattr(
        "harness.task.code_index.gather_paths_for_intent",
        lambda *_a, **_k: ["apps/web/src/score.ts"],
    )
    catalog = merge_tool_catalog({"read_file": ("path",)})
    calls = default_gather_calls(catalog, "fix the listing score")
    blob = " ".join(c["function"]["arguments"] for c in calls)
    assert "apps/web/src/score.ts" in blob
    assert calls[0]["function"]["name"] == "read_file"


def test_cli_index_help():
    from typer.testing import CliRunner

    from harness.cli import app

    result = CliRunner().invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    assert "8800" in result.stdout
    result = CliRunner().invoke(app, ["retrieve", "--help"])
    assert result.exit_code == 0
    assert "workspace" in result.stdout.lower()
