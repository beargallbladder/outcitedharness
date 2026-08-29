from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from harness.cli import app
from harness.sandbox.runtime import context_hash, stage_context


def test_stage_context_is_content_addressed_and_excludes_local_caches(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (source / "index.html").write_text("ready\n", encoding="utf-8")
    (source / "node_modules").mkdir()
    (source / "node_modules" / "large.js").write_text(
        "ignored\n", encoding="utf-8"
    )

    root = tmp_path / "runtime"
    staged = stage_context(source, root, "preview-1")

    assert staged == stage_context(source, root, "preview-1")
    assert context_hash(staged) == context_hash(source)
    assert (staged / "index.html").read_text(encoding="utf-8") == "ready\n"
    assert not (staged / "node_modules").exists()


def test_stage_context_rejects_symlinked_build_input(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (source / "outside").symlink_to(tmp_path / "not-in-context")

    with pytest.raises(ValueError, match="symlinked build input"):
        stage_context(source, tmp_path / "runtime", "preview-1")


def test_sandbox_cli_exposes_lifecycle_commands() -> None:
    result = CliRunner().invoke(app, ["sandbox", "--help"])

    assert result.exit_code == 0
    for command in ("up", "list", "status", "logs", "down", "unpublish", "gc"):
        assert command in result.stdout

    build_help = CliRunner().invoke(app, ["build", "--help"])
    assert build_help.exit_code == 0
    assert "preview" in build_help.stdout
