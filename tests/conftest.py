import pytest


@pytest.fixture(autouse=True)
def _no_live_code_index(monkeypatch):
    """Keep default_gather off the live :8800 embedder unless a test patches it."""
    monkeypatch.setattr(
        "harness.task.code_index.gather_paths_for_intent",
        lambda *_a, **_k: [],
    )
