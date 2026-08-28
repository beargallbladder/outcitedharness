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


def test_semantic_calls_embedder(monkeypatch, tmp_path: Path):
    from harness.task import search as search_mod

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return (
                b'{"results":[{"rank":1,"text":"encoders","kim_slug":"encoders",'
                b'"source_artifact":"external_signals/x.json"}]}'
            )

    monkeypatch.setattr(search_mod.urllib.request, "urlopen", lambda *a, **k: _Resp())
    result = search_code("what does ingest mean", tmp_path, mode="semantic")
    assert result.backend == "semantic"
    assert result.hits
    assert result.hits[0].text == "encoders"
    assert "category index" in result.detail
    assert search_mod.embedder_thread_block("review this file") == ""
    block = search_mod.embedder_thread_block("which category keyword wins")
    assert "CR EMBEDDER" in block
    assert "encoders" in block
