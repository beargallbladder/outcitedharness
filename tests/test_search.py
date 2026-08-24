from pathlib import Path

from harness.task.search import choose_backend, search_code


def test_auto_picks_grep_for_identifiers():
    assert choose_backend("geocode", "auto") == "grep"
    assert choose_backend("def geocode", "auto") == "ast"
    assert choose_backend("why is geocoding slow", "auto") == "hybrid"


def test_ripgrep_finds_local_file(tmp_path: Path):
    src = tmp_path / "services" / "engine"
    src.mkdir(parents=True)
    (src / "geocode.py").write_text("def geocode(row):\n    return row\n")
    result = search_code("geocode", tmp_path, mode="grep")
    assert result.hits
    assert any("geocode.py" in h.path and "def geocode" in h.text for h in result.hits)


def test_semantic_is_unavailable_not_invented(tmp_path: Path):
    result = search_code("what does ingest mean", tmp_path, mode="semantic")
    assert result.hits == []
    assert "unavailable" in result.detail
