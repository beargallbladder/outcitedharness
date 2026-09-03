from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_hf_snapshot.py"
SPEC = importlib.util.spec_from_file_location("download_hf_snapshot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


def test_token_from_stdin_strips_newline_without_logging(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("hf_example-token\n"))
    assert downloader.token_from_stdin() == "hf_example-token"


def test_token_from_stdin_rejects_whitespace(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("bad token\n"))
    with pytest.raises(ValueError, match="malformed"):
        downloader.token_from_stdin()
