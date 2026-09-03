from pathlib import Path

import pytest

from harness.config import AppConfig, ModelConfig, Settings
from harness.optimize import (
    load_cases,
    parse_ranks,
    run_optimize,
    tokens_per_sec,
    tool_names,
)
from harness.providers.base import ChatResult
from harness.storage.db import Store


def _model(key: str) -> ModelConfig:
    return ModelConfig(
        key=key,
        tier=0,
        display_name=key,
        short_name=key,
        provider="openai_compatible",
        base_url=f"http://127.0.0.1/{key}",
        model=key,
    )


def _cfg(tmp_path: Path) -> AppConfig:
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    config_dir.joinpath("workers.yaml").write_text(
        """
workers:
  foreman:
    enabled: true
    role: foreman
    priority: 1
    model_key: asus2_qwen
    endpoint: http://127.0.0.1/asus2_qwen
"""
    )
    return AppConfig(
        root=tmp_path,
        settings=Settings(results_dir=tmp_path / "results", db_path=tmp_path / "h.db"),
        models={
            "dgx_qwen": _model("dgx_qwen"),
            "dgx2_qwen": _model("dgx2_qwen"),
            "asus_qwen": _model("asus_qwen"),
            "dgx3_qwen": _model("dgx3_qwen"),
            "asus2_qwen": _model("asus2_qwen"),
            "frontier": _model("frontier"),
        },
        pricing={},
    )


def test_load_repo_cases():
    from harness.config import find_project_root

    cases = load_cases(find_project_root())
    ids = [c.id for c in cases]
    assert ids == ["op01_read", "op02_cmd", "op03_pong"]
    assert cases[0].expect_tool == "read_file"
    assert cases[1].expect_tool == "execute_command"
    assert cases[2].expect_tool is None


def test_tokens_per_sec_and_tool_names():
    result = ChatResult(
        provider="openai_compatible",
        model="x",
        output_tokens=50,
        latency_ms=1000,
        tool_calls=[{"function": {"name": "read_file"}}],
    )
    assert tokens_per_sec(result) == 50.0
    assert tool_names(result) == ["read_file"]
    assert tokens_per_sec(ChatResult(provider="x", model="x")) is None


def test_parse_ranks():
    winner, ranks, reason = parse_ranks(
        'noise {"winner":"dgx_qwen","ranks":{"dgx_qwen":1,"asus_qwen":2},"reason":"tools"}',
        ["dgx_qwen", "asus_qwen"],
    )
    assert winner == "dgx_qwen"
    assert ranks == {"dgx_qwen": 1, "asus_qwen": 2}
    assert reason == "tools"
    assert parse_ranks("not json", ["dgx_qwen"]) == (None, {}, "not json")


@pytest.mark.asyncio
async def test_run_optimize_mocked(tmp_path: Path, monkeypatch):
    dest = tmp_path / "cases" / "fleet_optimize" / "op01_read"
    dest.mkdir(parents=True)
    dest.joinpath("case.yaml").write_text("id: op01_read\ntitle: read\nexpect_tool: read_file\n")
    dest.joinpath("prompt.md").write_text("read ARCHITECTURE.md")

    class Fake:
        def __init__(self, model: ModelConfig):
            self.model = model

        async def health(self, timeout_s: float):
            return True, "ok"

        async def chat(self, request):
            text = request.messages[-1].content
            if self.model.key == "asus2_qwen":
                if "Turn this into a worker packet" in text:
                    return ChatResult(
                        provider="openai_compatible",
                        model=self.model.model,
                        text="Intent: read the file\nConstraints: one tool\nWhat to return: heading",
                        input_tokens=20,
                        output_tokens=10,
                        latency_ms=80,
                    )
                return ChatResult(
                    provider="openai_compatible",
                    model=self.model.model,
                    text='{"winner":"dgx_qwen","ranks":{"dgx_qwen":1,"dgx2_qwen":2,"asus_qwen":3,"dgx3_qwen":4},"reason":"hit"}',
                    input_tokens=40,
                    output_tokens=20,
                    latency_ms=90,
                )
            return ChatResult(
                provider="openai_compatible",
                model=self.model.model,
                text="",
                input_tokens=30,
                output_tokens=8,
                latency_ms=120,
                tool_calls=[{"id": "1", "function": {"name": "read_file", "arguments": "{}"}}],
            )

    monkeypatch.setattr("harness.optimize.build_provider", Fake)
    report = await run_optimize(
        _cfg(tmp_path),
        worker_keys=["dgx_qwen", "dgx2_qwen", "asus_qwen", "dgx3_qwen"],
    )
    assert report.health["dgx_qwen"] == "ok"
    assert len(report.outcomes) == 1
    outcome = report.outcomes[0]
    assert outcome.winner == "dgx_qwen"
    assert all(s.tool_hit for s in outcome.shots)
    assert Path(report.json_path).exists()
    store = Store(tmp_path / "h.db")
    with store.connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    assert n == 4


def test_optimize_cli_help():
    from typer.testing import CliRunner

    from harness.cli import app

    result = CliRunner().invoke(app, ["optimize", "--help"])
    assert result.exit_code == 0
    assert "GB10" in result.stdout
