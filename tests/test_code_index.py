from pathlib import Path

from harness.task.code_index import (
    chunk_source,
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

    hits = query_index("where do we score a listing", db, limit=4, embed=_fake_embed)
    assert hits
    assert any("score.ts" in hit.path for hit in hits)
    paths = gather_paths_for_intent("score a listing", db, limit=4, embed=_fake_embed)
    assert "apps/web/src/score.ts" in paths
    assert "package.json" not in paths


def test_missing_index_returns_empty(tmp_path: Path):
    assert query_index("anything", tmp_path / "missing.sqlite") == []
    assert gather_paths_for_intent("anything", tmp_path / "missing.sqlite") == []


def test_cli_index_help():
    from typer.testing import CliRunner

    from harness.cli import app

    result = CliRunner().invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    assert "8800" in result.stdout
    result = CliRunner().invoke(app, ["retrieve", "--help"])
    assert result.exit_code == 0
    assert "category" in result.stdout.lower() or "code index" in result.stdout.lower()


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
