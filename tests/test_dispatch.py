from pathlib import Path

import pytest

from harness.config import AppConfig, ModelConfig, Settings
from harness.dispatch import parse_packets, run_dispatch, score_tool_hit
from harness.providers.base import ChatResult


@pytest.fixture(autouse=True)
def _clear_foreman_cache():
    from harness.dispatch import _foreman_cache

    _foreman_cache.clear()
    yield
    _foreman_cache.clear()


def _model(key: str) -> ModelConfig:
    return ModelConfig(
        key=key,
        tier=0 if not key.startswith("m5") else 1,
        display_name=key,
        short_name=key,
        provider="openai_compatible",
        base_url=f"http://127.0.0.1/{key}",
        model=key,
    )


def _cfg(tmp_path: Path) -> AppConfig:
    dest = tmp_path / "config"
    dest.mkdir()
    dest.joinpath("workers.yaml").write_text(
        """
workers:
  a:
    enabled: true
    role: coder
    model_key: dgx_qwen
  b:
    enabled: true
    role: coder
    model_key: dgx2_qwen
  fore:
    enabled: true
    role: foreman
    priority: 1
    model_key: m5_qwen
  peer:
    enabled: true
    role: foreman
    priority: 2
    model_key: asus2_qwen
  off:
    enabled: false
    role: coder
    model_key: asus_qwen
"""
    )
    return AppConfig(
        root=tmp_path,
        settings=Settings(results_dir=tmp_path / "results", db_path=tmp_path / "h.db"),
        models={
            "dgx_qwen": _model("dgx_qwen"),
            "dgx2_qwen": _model("dgx2_qwen"),
            "asus_qwen": _model("asus_qwen"),
            "m5_qwen": _model("m5_qwen"),
            "frontier": _model("frontier"),
        },
        pricing={},
    )


def _enable_test_critic(cfg: AppConfig) -> None:
    path = cfg.root / "config" / "workers.yaml"
    path.write_text(
        path.read_text()
        + """
  critic:
    enabled: true
    role: critic
    priority: 1
    model_key: asus3_nemotron
"""
    )
    cfg.models["asus3_nemotron"] = _model("asus3_nemotron")


def test_parse_packets_and_fallback():
    packets = parse_packets(
        'noise [{"id":"p1","title":"r","prompt":"read it","expect_tool":"read_file",'
        '"accept":{"invariants":["tool read_file"]}}]',
        "fallback",
        4,
    )
    assert len(packets) == 1
    assert packets[0].expect_tool == "read_file"
    assert "tool read_file" in packets[0].accept.invariants
    assert parse_packets("nope", "just do it", 4) == []
    assert parse_packets(
        '[{"id":"p1","title":"x","prompt":"do it"}]',
        "fallback",
        4,
    ) == []
    assert parse_packets(
        '[{"id":"p1","title":"x","prompt":"do it","expect_tool":"read_file"}]',
        "fallback",
        4,
    ) == []


def test_score_tool_hit():
    from harness.dispatch import Packet

    pkt = Packet(id="p1", title="t", prompt="x", expect_tool="read_file")
    ok = ChatResult(
        provider="x",
        model="x",
        tool_calls=[{"function": {"name": "read_file"}}],
    )
    assert score_tool_hit(pkt, ok, ["read_file"]) is True
    miss = ChatResult(provider="x", model="x", text="sure")
    assert score_tool_hit(pkt, miss, []) is False
    pong = Packet(id="p2", title="t", prompt="x", expect_tool=None)
    assert score_tool_hit(pong, ChatResult(provider="x", model="x", text="PONG"), []) is True
    assert score_tool_hit(
        pong,
        ChatResult(provider="x", model="x", text="", tool_calls=[{"function": {"name": "execute_command"}}]),
        ["execute_command"],
    ) is False


def test_action_binding_uses_only_available_edit_tool():
    from harness.dispatch import bind_action_calls

    calls = bind_action_calls(
        [
            {
                "name": "write_to_file",
                "arguments": {"path": "src/app.py", "content": "fixed"},
            },
            {"name": "run_commands", "arguments": {"commands": ["pytest"]}},
        ],
        {"editor": ("path", "content"), "run_commands": ("commands",)},
    )
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "editor"
    assert "src/app.py" in calls[0]["function"]["arguments"]


def test_action_binding_allows_five_distinct_files():
    from harness.dispatch import ACTION_MAX_CALLS, bind_action_calls

    raw = [
        {"name": "editor", "arguments": {"path": f"src/f{i}.py", "content": "x"}}
        for i in range(7)
    ]
    calls = bind_action_calls(raw, {"editor": ("path", "content")})
    assert ACTION_MAX_CALLS == 5
    assert len(calls) == 5
    paths = [c["function"]["arguments"] for c in calls]
    assert paths[0].count("src/f0.py") == 1
    dup = bind_action_calls(
        [
            {"name": "editor", "arguments": {"path": "src/a.py", "content": "1"}},
            {"name": "editor", "arguments": {"path": "src/a.py", "content": "2"}},
            {"name": "editor", "arguments": {"path": "src/b.py", "content": "3"}},
        ],
        {"editor": ("path", "content")},
    )
    assert len(dup) == 2


def test_change_job_strips_prose_invariants():
    from harness.dispatch import (
        AcceptSpec,
        Packet,
        is_change_job,
        is_prose_invariant,
        strip_prose_invariants,
    )

    assert is_change_job("fix the failing unit test")
    assert is_prose_invariant("text not yet")
    assert is_prose_invariant("text PONG")
    assert not is_prose_invariant("min_chars 40")
    assert not is_prose_invariant("text apps/web")
    packets = [
        Packet(
            id="p1",
            title="fix",
            prompt="patch it",
            accept=AcceptSpec(commands=("pytest",), invariants=("text not yet", "min_chars 10")),
        )
    ]
    cleaned = strip_prose_invariants(packets)
    assert cleaned[0].accept.invariants == ("min_chars 10",)
    assert cleaned[0].accept.commands == ("pytest",)


def test_score_invariants():
    from harness.dispatch import AcceptSpec, Packet, score_invariants

    pkt = Packet(
        id="p1",
        title="t",
        prompt="x",
        expect_tool="read_file",
        accept=AcceptSpec(invariants=("tool read_file", "no execute_command")),
    )
    ok = ChatResult(
        provider="x",
        model="x",
        text="read the file",
        tool_calls=[{"function": {"name": "read_file"}}],
    )
    assert score_invariants(pkt, ok, ["read_file"]) is True
    assert score_invariants(pkt, ok, ["read_file", "execute_command"]) is False
    tool_only = ChatResult(provider="x", model="x", tool_calls=[{"function": {"name": "read_file"}}])
    assert score_invariants(pkt, tool_only, ["read_file"]) is False


def test_min_chars_invariant():
    from harness.dispatch import AcceptSpec, Packet, score_invariants

    pkt = Packet(
        id="p1",
        title="review",
        prompt="review evidence",
        accept=AcceptSpec(invariants=("min_chars 12",)),
    )
    short = ChatResult(provider="x", model="x", text="too short")
    enough = ChatResult(provider="x", model="x", text="long enough answer")
    assert score_invariants(pkt, short, []) is False
    assert score_invariants(pkt, enough, []) is True


def test_review_grounding_rejects_unseen_paths_and_allows_honest_limits():
    from harness.dispatch import AcceptSpec, Packet, score_invariants

    packet = Packet(
        id="fallback-1",
        title="review",
        prompt=(
            "WORKSPACE EVIDENCE GATHERED BY CLINE:\n"
            "tool(read_files): FILE src/real.py\n"
            "1 | def work(): return 1"
        ),
        accept=AcceptSpec(invariants=("min_chars 20", "review_grounded")),
    )
    invented = ChatResult(
        provider="x",
        model="x",
        text="Critical: src/invented.py contains a proven SQL injection defect.",
    )
    grounded = ChatResult(
        provider="x",
        model="x",
        text="High: src/real.py has a concrete maintainability issue in work().",
    )
    honest = ChatResult(
        provider="x",
        model="x",
        text="No defect is proven by the supplied evidence.",
    )
    assert score_invariants(packet, invented, []) is False
    assert score_invariants(packet, grounded, []) is True
    assert score_invariants(packet, honest, []) is True


def test_critic_rejects_internally_contradictory_scores():
    from harness.dispatch import (
        AcceptSpec,
        Packet,
        Shot,
        _critic_scores_consistent,
    )

    shots = [
        Shot(
            packet=Packet(
                id=f"p{index}",
                title="x",
                prompt="x",
                accept=AcceptSpec(invariants=("min_chars 1",)),
            ),
            worker_id="w",
            model_key="m",
            result=ChatResult(provider="x", model="x", text="answer"),
            tokens_per_sec=None,
            tool_names=[],
            tool_hit=True,
            qa_pass=True,
            preview="answer",
        )
        for index in (1, 2)
    ]
    all_fail = {"p1": (False, "no"), "p2": (False, "no")}
    mixed = {"p1": (True, "yes"), "p2": (False, "no")}
    assert _critic_scores_consistent("proceed", all_fail, shots) is False
    assert _critic_scores_consistent("reject", all_fail, shots) is True
    assert _critic_scores_consistent("revise", mixed, shots) is True
    assert _critic_scores_consistent("proceed", {"p1": (True, "yes")}, shots) is False


def test_foreman_input_never_truncates_current_intent():
    from harness.dispatch import _foreman_input

    prompt = _foreman_input(
        "CURRENT REQUEST: review this codebase",
        "old-evidence-" * 4000,
        header="CLINE_TOOLS:\n- read_files: paths",
        limit=8000,
    )
    assert prompt.startswith("INTENT (the current job; never ignore):")
    assert "CURRENT REQUEST: review this codebase" in prompt
    assert "CLINE_TOOLS" in prompt
    assert "gathered evidence clipped" in prompt
    assert len(prompt) <= 8000


def test_evidence_fallback_fans_out_review_work():
    from harness.dispatch import (
        AcceptSpec,
        Packet,
        _is_review_job,
        fallback_packets,
        hydrate_packets,
    )

    packets = fallback_packets(
        "review the code base and find the biggest bugs",
        "tool(read_files): src/app.py\n" + ("def work(): pass\n" * 1000),
        8,
    )
    assert len(packets) == 4
    assert len({packet.id for packet in packets}) == 4
    assert all("WORKSPACE EVIDENCE GATHERED BY CLINE" in packet.prompt for packet in packets)
    assert all("tool(read_files): src/app.py" in packet.prompt for packet in packets)
    assert all(
        packet.accept.invariants == ("min_chars 120", "review_grounded")
        for packet in packets
    )
    assert fallback_packets("review the code", "", 8) == []
    assert _is_review_job("Return exactly QUALITY_CHECK_OK") is False

    planned = [
        Packet(
            id="p1",
            title="review app",
            prompt="Review src/app.py and cite concrete defects.",
            accept=AcceptSpec(invariants=("min_chars 120",)),
        )
    ]
    hydrated = hydrate_packets(
        planned,
        "tool(read_files): FILE src/app.py\n" + ("def work(): pass\n" * 100),
    )
    assert "WORKSPACE EVIDENCE GATHERED BY CLINE" in hydrated[0].prompt
    assert "def work()" in hydrated[0].prompt


@pytest.mark.asyncio
async def test_dispatch_leases_fallback_when_foreman_emits_no_packets(tmp_path: Path, monkeypatch):
    captured = {}

    class Healthy:
        def __init__(self, model: ModelConfig):
            self.model = model

        async def health(self, timeout_s: float):
            return True, "ok"

    async def no_packets(*args, **kwargs):
        return []

    async def capture_lease(pairs, packets):
        captured["packets"] = packets
        return []

    monkeypatch.setattr("harness.dispatch.build_provider", Healthy)
    monkeypatch.setattr("harness.dispatch._slice", no_packets)
    monkeypatch.setattr("harness.dispatch._lease", capture_lease)

    report = await run_dispatch(
        _cfg(tmp_path),
        "review the code base",
        thread="tool(read_files): src/app.py\n" + ("def work(): pass\n" * 1000),
    )
    assert report.slice_error == ""
    assert len(report.packets) == 4
    assert captured["packets"] == report.packets


@pytest.mark.asyncio
async def test_dispatch_excludes_coders_that_fail_health_probe(tmp_path: Path, monkeypatch):
    from harness.dispatch import AcceptSpec, Packet

    captured = {}

    class Health:
        def __init__(self, model: ModelConfig):
            self.model = model

        async def health(self, timeout_s: float):
            return (self.model.key != "dgx_qwen"), "down"

    async def one_packet(*args, **kwargs):
        return [
            Packet(
                id="p1",
                title="work",
                prompt="write a sufficiently detailed answer",
                accept=AcceptSpec(invariants=("min_chars 12",)),
            )
        ]

    async def capture_lease(pairs, packets):
        captured["workers"] = [worker.id for worker, _ in pairs]
        return []

    monkeypatch.setattr("harness.dispatch.build_provider", Health)
    monkeypatch.setattr("harness.dispatch._slice", one_packet)
    monkeypatch.setattr("harness.dispatch._lease", capture_lease)

    report = await run_dispatch(_cfg(tmp_path), "explain it")
    assert report.health["a"] == "down"
    assert report.health["b"] == "ok"
    assert captured["workers"] == ["b"]


@pytest.mark.asyncio
async def test_pick_foreman_falls_back_to_asus2(tmp_path: Path, monkeypatch):
    from harness.dispatch import pick_foreman

    probes: list[str] = []

    class Fake:
        def __init__(self, model: ModelConfig):
            self.model = model

        async def health(self, timeout_s: float):
            probes.append(self.model.key)
            return (self.model.key != "m5_qwen"), "down" if self.model.key == "m5_qwen" else "ok"

    monkeypatch.setattr("harness.dispatch.build_provider", Fake)
    cfg = _cfg(tmp_path)
    cfg.models["asus2_qwen"] = _model("asus2_qwen")

    picked = await pick_foreman(cfg)
    assert picked is not None
    key, model = picked
    assert key == "asus2_qwen"
    assert model.key == "asus2_qwen"

    # Second call inside the TTL uses the cached verdicts, no new probes.
    before = len(probes)
    picked = await pick_foreman(cfg)
    assert picked is not None and picked[0] == "asus2_qwen"
    assert len(probes) == before


@pytest.mark.asyncio
async def test_run_dispatch_fails_when_no_foreman_reachable(tmp_path: Path, monkeypatch):
    class Dead:
        def __init__(self, model: ModelConfig):
            self.model = model

        async def health(self, timeout_s: float):
            return False, "unreachable"

    monkeypatch.setattr("harness.dispatch.build_provider", Dead)
    cfg = _cfg(tmp_path)
    cfg.models["asus2_qwen"] = _model("asus2_qwen")
    with pytest.raises(ValueError, match="no foreman reachable"):
        await run_dispatch(cfg, "ping")


@pytest.mark.asyncio
async def test_run_dispatch_mocked(tmp_path: Path, monkeypatch):
    class Fake:
        def __init__(self, model: ModelConfig):
            self.model = model

        async def health(self, timeout_s: float):
            return True, "ok"

        async def chat(self, request):
            text = request.messages[-1].content
            if self.model.key == "m5_qwen":
                return ChatResult(
                    provider="openai_compatible",
                    model=self.model.model,
                    text=(
                        '[{"id":"p1","title":"one","prompt":"summarize ARCHITECTURE.md",'
                        '"expect_tool":null,"accept":{"invariants":["text ARCHITECTURE"]}},'
                        '{"id":"p2","title":"two","prompt":"count lines in ARCHITECTURE.md",'
                        '"expect_tool":null,"accept":{"invariants":["text lines"]}}]'
                    ),
                    latency_ms=50,
                    input_tokens=10,
                    output_tokens=20,
                )
            if "summarize" in text:
                body = "ARCHITECTURE.md is the live fleet table for this harness."
            else:
                body = "ARCHITECTURE.md has many lines; count them with wc -l."
            return ChatResult(
                provider="openai_compatible",
                model=self.model.model,
                text=body,
                latency_ms=100,
                input_tokens=30,
                output_tokens=8,
            )

    monkeypatch.setattr("harness.dispatch.build_provider", Fake)
    monkeypatch.setattr("harness.optimize.build_provider", Fake)
    report = await run_dispatch(_cfg(tmp_path), "measure ARCHITECTURE.md")
    assert len(report.packets) == 2
    assert len(report.shots) == 2
    assert {s.worker_id for s in report.shots} == {"a", "b"}
    assert all(s.tool_hit for s in report.shots)
    assert all(s.qa_pass for s in report.shots)
    assert Path(report.json_path).exists()


@pytest.mark.asyncio
async def test_run_dispatch_critic_can_fail_closed(tmp_path: Path, monkeypatch):
    dest = tmp_path / "config"
    dest.mkdir()
    dest.joinpath("workers.yaml").write_text(
        """
workers:
  a:
    enabled: true
    role: coder
    model_key: dgx_qwen
  fore:
    enabled: true
    role: foreman
    model_key: m5_qwen
  critic:
    enabled: true
    role: critic
    priority: 1
    model_key: asus3_nemotron
  glm:
    enabled: true
    role: critic
    priority: 2
    model_key: glm52
  nemo:
    enabled: true
    role: critic
    priority: 3
    model_key: nemotron_super
"""
    )
    cfg = AppConfig(
        root=tmp_path,
        settings=Settings(results_dir=tmp_path / "results", db_path=tmp_path / "h.db"),
        models={
            "dgx_qwen": _model("dgx_qwen"),
            "m5_qwen": _model("m5_qwen"),
            "asus3_nemotron": _model("asus3_nemotron"),
            "frontier": _model("frontier"),
        },
        pricing={},
    )

    class Fake:
        def __init__(self, model: ModelConfig):
            self.model = model

        async def health(self, timeout_s: float):
            return True, "ok"

        async def chat(self, request):
            if self.model.key == "m5_qwen":
                return ChatResult(
                    provider="openai_compatible",
                    model=self.model.model,
                    text=(
                        '[{"id":"p1","title":"one","prompt":"answer from the packet",'
                        '"expect_tool":null,"accept":{"invariants":["text file"]}}]'
                    ),
                )
            if self.model.key == "asus3_nemotron":
                return ChatResult(
                    provider="openai_compatible",
                    model=self.model.model,
                    text='{"verdict":"reject","shots":[{"id":"p1","pass":false,"why":"no evidence"}]}',
                )
            return ChatResult(
                provider="openai_compatible",
                model=self.model.model,
                text="I read the file. The contents are present.",
            )

    monkeypatch.setattr("harness.dispatch.build_provider", Fake)
    monkeypatch.setattr("harness.optimize.build_provider", Fake)
    report = await run_dispatch(cfg, "read it")
    assert report.shots[0].tool_hit is True
    assert report.shots[0].qa_pass is False
    assert report.critic_verdict == "reject"


@pytest.mark.asyncio
async def test_local_repair_succeeds_without_frontier(tmp_path: Path, monkeypatch):
    from harness.dispatch import AcceptSpec, Packet, Shot

    cfg = _cfg(tmp_path)
    _enable_test_critic(cfg)
    cfg.settings.local_revision_attempts = 1
    cfg.settings.auto_frontier_rescue = True
    packet = Packet(
        id="p1",
        title="answer",
        prompt="Use the supplied evidence.",
        accept=AcceptSpec(invariants=("min_chars 4",)),
    )
    leases = 0

    class Health:
        def __init__(self, model):
            self.model = model

        async def health(self, _timeout):
            return True, "ok"

    async def one_packet(*args, **kwargs):
        return [packet]

    async def lease(_pairs, packets):
        nonlocal leases
        leases += 1
        text = "unsupported answer" if leases == 1 else "correct grounded answer"
        return [
            Shot(
                packet=packets[0],
                worker_id="coder",
                model_key="dgx_qwen",
                result=ChatResult(provider="x", model="x", text=text),
                tokens_per_sec=None,
                tool_names=[],
                tool_hit=True,
                qa_pass=True,
                preview=text,
            )
        ]

    async def grade(_worker, _model, _intent, shots):
        ok = "correct" in shots[0].result.text
        verdict = "proceed" if ok else "reject"
        return verdict, "grade", {"p1": (ok, "grounded" if ok else "unsupported")}

    async def no_frontier(*args, **kwargs):
        raise AssertionError("frontier must not run after successful local repair")

    monkeypatch.setattr("harness.dispatch.build_provider", Health)
    monkeypatch.setattr("harness.dispatch._slice", one_packet)
    monkeypatch.setattr("harness.dispatch._lease", lease)
    monkeypatch.setattr("harness.dispatch._run_critic", grade)
    monkeypatch.setattr("harness.dispatch.run_rescue_text", no_frontier)

    report = await run_dispatch(cfg, "answer the question", thread="tool(read): evidence")
    assert report.local_rounds == 2
    assert report.shots[0].qa_pass is True
    assert report.frontier_run_id == ""
    assert len(report.attempt_history) == 2


@pytest.mark.asyncio
async def test_exhausted_local_repair_runs_one_verified_frontier_rescue(
    tmp_path: Path, monkeypatch
):
    from harness.dispatch import AcceptSpec, Packet, Shot
    from harness.rescue import RescueOutcome
    from harness.task.service import TaskService
    from harness.storage.db import Store

    cfg = _cfg(tmp_path)
    _enable_test_critic(cfg)
    cfg.settings.local_revision_attempts = 1
    cfg.settings.auto_frontier_rescue = True
    cfg.settings.max_frontier_calls_per_task = 1
    packet = Packet(
        id="p1",
        title="answer",
        prompt="Use the supplied evidence.",
        accept=AcceptSpec(invariants=("min_chars 4",)),
    )
    rescue_calls = 0

    class Health:
        def __init__(self, model):
            self.model = model

        async def health(self, _timeout):
            return True, "ok"

    async def one_packet(*args, **kwargs):
        return [packet]

    async def lease(_pairs, packets):
        return [
            Shot(
                packet=packets[0],
                worker_id="coder",
                model_key="dgx_qwen",
                result=ChatResult(provider="x", model="x", text="unsupported answer"),
                tokens_per_sec=None,
                tool_names=[],
                tool_hit=True,
                qa_pass=True,
                preview="unsupported answer",
            )
        ]

    async def grade(_worker, _model, _intent, shots):
        if shots[0].packet.id == "frontier-rescue":
            return "proceed", "valid", {"frontier-rescue": (True, "grounded")}
        return "reject", "invalid", {"p1": (False, "unsupported")}

    async def rescue(*args, **kwargs):
        nonlocal rescue_calls
        rescue_calls += 1
        return RescueOutcome(
            run_id="rescue-1",
            model_key="frontier",
            text="A verified frontier correction with concrete evidence.",
            latency_ms=100,
            input_tokens=20,
            output_tokens=10,
            estimated_cost=0.01,
            error=None,
            answer_path=tmp_path / "answer.txt",
        )

    monkeypatch.setattr("harness.dispatch.build_provider", Health)
    monkeypatch.setattr("harness.dispatch._slice", one_packet)
    monkeypatch.setattr("harness.dispatch._lease", lease)
    monkeypatch.setattr("harness.dispatch._run_critic", grade)
    monkeypatch.setattr("harness.dispatch.run_rescue_text", rescue)

    report = await run_dispatch(cfg, "answer the question", thread="tool(read): evidence")
    assert rescue_calls == 1
    assert report.frontier_verified is True
    assert report.frontier_run_id == "rescue-1"
    task = TaskService(Store(cfg.settings.db_path)).get(report.task_id)
    assert task.frontier_calls == 1
    assert task.final_outcome == "frontier_verified"


@pytest.mark.asyncio
async def test_critic_semantic_rejection_does_not_trigger_thinking_retry(monkeypatch):
    from harness.dispatch import AcceptSpec, Packet, Shot, _run_critic

    packet = Packet(
        id="p1",
        title="review",
        prompt="WORKSPACE EVIDENCE GATHERED BY CLINE:\nreturn status >= 500",
        accept=AcceptSpec(invariants=("min_chars 12",)),
    )
    shot = Shot(
        packet=packet,
        worker_id="coder",
        model_key="dgx_qwen",
        result=ChatResult(
            provider="x",
            model="x",
            text="The code fails over on every 4xx response.",
        ),
        tokens_per_sec=None,
        tool_names=[],
        tool_hit=True,
        qa_pass=True,
        preview="The code fails over on every 4xx response.",
    )
    calls = []

    async def grade(*args, **kwargs):
        calls.append(kwargs)
        return ChatResult(
            provider="x",
            model="x",
            text='{"verdict":"reject","shots":[{"id":"p1","pass":false,'
            '"why":"claim contradicts evidence"}]}',
        )

    monkeypatch.setattr("harness.dispatch._chat", grade)
    verdict, _text, scores = await _run_critic(
        None, _model("asus3_nemotron"), "review behavior", [shot]
    )
    assert verdict == "reject"
    assert scores == {"p1": (False, "claim contradicts evidence")}
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 200


def test_parse_critic_accepts_json_with_trailing_model_junk():
    from harness.dispatch import AcceptSpec, Packet, Shot, _parse_critic

    packet = Packet("p1", "review", "evidence", accept=AcceptSpec(invariants=("min_chars 1",)))
    shot = Shot(
        packet,
        "coder",
        "model",
        ChatResult(provider="x", model="x", text="answer"),
        None,
        [],
        True,
        True,
        "answer",
    )
    verdict, scores = _parse_critic(
        '{"verdict":"proceed","shots":[{"id":"p1","pass":true,"why":"grounded"}]}'
        "\nextra generated text {not json}",
        [shot],
    )
    assert verdict == "proceed"
    assert scores == {"p1": (True, "grounded")}


@pytest.mark.asyncio
async def test_invalid_multi_shot_grade_recovers_with_per_shot_grading(monkeypatch):
    from harness.dispatch import AcceptSpec, Packet, Shot, _grade_shots

    shots = [
        Shot(
            Packet(
                f"p{index}",
                "review",
                "WORKSPACE EVIDENCE GATHERED BY CLINE:\nsrc/app.py",
                accept=AcceptSpec(invariants=("min_chars 1",)),
            ),
            "coder",
            "model",
            ChatResult(provider="x", model="x", text="src/app.py is visible"),
            None,
            [],
            True,
            True,
            "src/app.py is visible",
        )
        for index in (1, 2)
    ]
    calls: list[list[str]] = []

    async def grade(_worker, _model, _intent, batch):
        ids = [shot.packet.id for shot in batch]
        calls.append(ids)
        if len(batch) > 1:
            return "insufficient", "truncated", {}
        return "proceed", "valid", {ids[0]: (True, "grounded")}

    monkeypatch.setattr("harness.dispatch._run_critic", grade)
    verdict, _text, critic_key, failures = await _grade_shots(
        [(None, _model("asus3_nemotron"))],
        "review the code",
        shots,
        allow_degraded=True,
    )
    assert calls == [["p1", "p2"], ["p1"], ["p2"]]
    assert verdict == "proceed"
    assert critic_key == "asus3_nemotron"
    assert failures == []
    assert all(shot.qa_pass for shot in shots)


@pytest.mark.asyncio
async def test_degraded_review_fails_closed_and_is_not_stitched(monkeypatch):
    from harness.dispatch import AcceptSpec, DispatchReport, Packet, Shot, _grade_shots
    from harness.gateway.orch import stitch_report

    answer = "Critical: src/invented.py contains a severe SQL injection. " * 3
    packet = Packet(
        "fallback-1",
        "review",
        "WORKSPACE EVIDENCE GATHERED BY CLINE:\nsrc/real.py",
        accept=AcceptSpec(invariants=("min_chars 20",)),
    )
    shot = Shot(
        packet,
        "coder",
        "model",
        ChatResult(provider="x", model="x", text=answer),
        None,
        [],
        True,
        True,
        answer,
    )

    async def invalid(*_args, **_kwargs):
        return "insufficient", "invalid JSON", {}

    monkeypatch.setattr("harness.dispatch._run_critic", invalid)
    verdict, _text, _key, _failures = await _grade_shots(
        [(None, _model("asus3_nemotron"))],
        "review the codebase",
        [shot],
        allow_degraded=True,
    )
    report = DispatchReport(
        run_id="degraded-review",
        intent="review the codebase",
        packets=[packet],
        shots=[shot],
        critic_verdict=verdict,
    )
    rendered = stitch_report(report)
    assert verdict == "degraded"
    assert shot.qa_pass is False
    assert "QA FAIL closed" in rendered
    assert "src/invented.py" not in rendered


@pytest.mark.asyncio
async def test_degraded_review_skips_revision_and_frontier(tmp_path: Path, monkeypatch):
    from harness.dispatch import Shot

    cfg = _cfg(tmp_path)
    _enable_test_critic(cfg)
    cfg.settings.local_revision_attempts = 2
    cfg.settings.auto_frontier_rescue = True
    lease_calls = 0
    rescue_calls = 0

    class Health:
        def __init__(self, model: ModelConfig):
            self.model = model

        async def health(self, _timeout_s: float):
            return True, "ok"

    async def lease(_pairs, packets):
        nonlocal lease_calls
        lease_calls += 1
        answer = "High: src/app.py contains a concrete issue in work(). " * 3
        return [
            Shot(
                packet,
                "coder",
                "model",
                ChatResult(provider="x", model="x", text=answer),
                None,
                [],
                True,
                True,
                answer,
            )
            for packet in packets
        ]

    async def invalid(*_args, **_kwargs):
        return "insufficient", "invalid JSON", {}

    async def rescue(*_args, **_kwargs):
        nonlocal rescue_calls
        rescue_calls += 1
        raise AssertionError("degraded review must not invoke frontier rescue")

    monkeypatch.setattr("harness.dispatch.build_provider", Health)
    monkeypatch.setattr("harness.dispatch._lease", lease)
    monkeypatch.setattr("harness.dispatch._run_critic", invalid)
    monkeypatch.setattr("harness.dispatch.run_rescue_text", rescue)

    report = await run_dispatch(
        cfg,
        "review the codebase",
        thread="tool(read_files): FILE src/app.py\n1 | def work(): return 1",
    )
    assert lease_calls == 1
    assert rescue_calls == 0
    assert report.local_rounds == 1
    assert report.critic_verdict == "degraded"
    assert not any(shot.qa_pass for shot in report.shots)
    assert report.frontier_run_id == ""


@pytest.mark.asyncio
async def test_run_dispatch_critic_down_serves_machine_pass_as_degraded(tmp_path: Path, monkeypatch):
    dest = tmp_path / "config"
    dest.mkdir()
    dest.joinpath("workers.yaml").write_text(
        """
workers:
  a:
    enabled: true
    role: coder
    model_key: dgx_qwen
  fore:
    enabled: true
    role: foreman
    model_key: m5_qwen
  critic:
    enabled: true
    role: researcher
    model_key: asus3_nemotron
"""
    )
    cfg = AppConfig(
        root=tmp_path,
        settings=Settings(results_dir=tmp_path / "results", db_path=tmp_path / "h.db"),
        models={
            "dgx_qwen": _model("dgx_qwen"),
            "m5_qwen": _model("m5_qwen"),
            "asus3_nemotron": _model("asus3_nemotron"),
            "frontier": _model("frontier"),
        },
        pricing={},
    )

    class Fake:
        def __init__(self, model: ModelConfig):
            self.model = model

        async def health(self, timeout_s: float):
            if self.model.key == "asus3_nemotron":
                return False, "down"
            return True, "ok"

        async def chat(self, request):
            if self.model.key == "m5_qwen":
                return ChatResult(
                    provider="openai_compatible",
                    model=self.model.model,
                    text=(
                        '[{"id":"p1","title":"one","prompt":"answer from the packet",'
                        '"expect_tool":null,"accept":{"invariants":["text file"]}}]'
                    ),
                )
            return ChatResult(
                provider="openai_compatible",
                model=self.model.model,
                text="I read the file. The contents are present.",
            )

    monkeypatch.setattr("harness.dispatch.build_provider", Fake)
    monkeypatch.setattr("harness.optimize.build_provider", Fake)
    report = await run_dispatch(cfg, "read it")
    assert report.shots[0].tool_hit is True
    assert report.shots[0].qa_pass is True
    assert report.shots[0].qa_why == "python-only; all critics unavailable or invalid"
    assert report.critic_verdict == "degraded"


@pytest.mark.asyncio
async def test_invalid_critic_output_falls_through_to_next_critic(tmp_path: Path, monkeypatch):
    from harness.dispatch import AcceptSpec, Packet, Shot

    dest = tmp_path / "config"
    dest.mkdir()
    dest.joinpath("workers.yaml").write_text(
        """
workers:
  coder:
    enabled: true
    role: coder
    model_key: dgx_qwen
  fore:
    enabled: true
    role: foreman
    model_key: m5_qwen
  critic:
    enabled: true
    role: critic
    priority: 1
    model_key: asus3_nemotron
  glm:
    enabled: true
    role: critic
    priority: 2
    model_key: glm52
  nemo:
    enabled: true
    role: critic
    priority: 3
    model_key: nemotron_super
"""
    )
    cfg = AppConfig(
        root=tmp_path,
        settings=Settings(results_dir=tmp_path / "results", db_path=tmp_path / "h.db"),
        models={
            key: _model(key)
            for key in ("dgx_qwen", "m5_qwen", "asus3_nemotron", "nemotron_super", "glm52")
        },
        pricing={},
    )

    class Health:
        def __init__(self, model: ModelConfig):
            self.model = model

        async def health(self, timeout_s: float):
            return (self.model.key != "asus3_nemotron"), "down"

    packet = Packet(
        id="p1",
        title="answer",
        prompt="write a grounded answer",
        accept=AcceptSpec(invariants=("min_chars 12",)),
    )
    shot = Shot(
        packet=packet,
        worker_id="coder",
        model_key="dgx_qwen",
        result=ChatResult(provider="x", model="x", text="a sufficiently long grounded answer"),
        tokens_per_sec=None,
        tool_names=[],
        tool_hit=True,
        qa_pass=True,
        preview="a sufficiently long grounded answer",
    )
    called = []

    async def one_packet(*args, **kwargs):
        return [packet]

    async def one_shot(*args, **kwargs):
        return [shot]

    async def grade(_worker, model, _intent, _shots):
        called.append(model.key)
        if model.key == "glm52":
            return "insufficient", "garbage", {}
        return "proceed", "valid", {"p1": (True, "grounded")}

    monkeypatch.setattr("harness.dispatch.build_provider", Health)
    monkeypatch.setattr("harness.dispatch._slice", one_packet)
    monkeypatch.setattr("harness.dispatch._lease", one_shot)
    monkeypatch.setattr("harness.dispatch._run_critic", grade)

    report = await run_dispatch(cfg, "explain the behavior")
    assert called == ["glm52", "nemotron_super"]
    assert report.critic_verdict == "proceed"
    assert report.shots[0].qa_pass is True


def test_stitch_report_fail_closed():
    from harness.dispatch import DispatchReport, Packet, Shot
    from harness.gateway.orch import stitch_report

    report = DispatchReport(
        run_id="dispatch-x",
        intent="read it",
        packets=[Packet(id="p1", title="one", prompt="read it")],
        shots=[
            Shot(
                packet=Packet(id="p1", title="one", prompt="read it"),
                worker_id="a",
                model_key="dgx_qwen",
                result=ChatResult(provider="x", model="x", text="nope"),
                tokens_per_sec=1.0,
                tool_names=[],
                tool_hit=False,
                qa_pass=False,
                preview="nope",
                qa_why="python accept miss",
            )
        ],
        critic_verdict="reject",
    )
    text = stitch_report(report)
    assert "QA FAIL closed" in text
    assert "nope" not in text


def test_verified_frontier_replaces_conflicting_partial_local_review():
    from harness.dispatch import DispatchReport, Packet, Shot
    from harness.gateway.orch import stitch_report

    packet = Packet(id="p1", title="review", prompt="review it")
    report = DispatchReport(
        run_id="dispatch-x",
        intent="review the codebase",
        packets=[packet],
        shots=[
            Shot(
                packet=packet,
                worker_id="a",
                model_key="dgx_qwen",
                result=ChatResult(
                    provider="x",
                    model="x",
                    text="Invented critical defect in src/fake.py.",
                ),
                tokens_per_sec=1.0,
                tool_names=[],
                tool_hit=True,
                qa_pass=True,
                preview="Invented critical defect in src/fake.py.",
            )
        ],
        critic_verdict="revise",
        frontier_run_id="rescue-1",
        frontier_verified=True,
        frontier_text="No defect is proven by the supplied evidence.",
    )
    text = stitch_report(report)
    assert "No defect is proven" in text
    assert "src/fake.py" not in text
    assert "Harness completion" not in text


def test_stitch_report_answers_not_tool_json():
    from harness.dispatch import DispatchReport, Packet, Shot
    from harness.gateway.orch import stitch_report

    report = DispatchReport(
        run_id="dispatch-x",
        intent="reach",
        packets=[Packet(id="p1", title="reach", prompt="fix reach")],
        shots=[
            Shot(
                packet=Packet(id="p1", title="reach", prompt="fix reach"),
                worker_id="a",
                model_key="dgx_qwen",
                result=ChatResult(
                    provider="x",
                    model="x",
                    text="CORRECTED TOTAL_REACH = 617M",
                    tool_calls=[{"function": {"name": "execute_command"}}],
                ),
                tokens_per_sec=50.0,
                tool_names=["execute_command"],
                tool_hit=True,
                qa_pass=True,
                preview="CORRECTED TOTAL_REACH = 617M",
            )
        ],
        critic_verdict="proceed",
    )
    text = stitch_report(report)
    assert "617M" in text
    assert "execute_command" not in text
    assert "chatcmpl-tool" not in text


def test_is_orch_echo():
    from harness.dispatch import is_orch_echo

    assert is_orch_echo("hello") is False
    dump = "Harness orch 20260826_164820_dispatch\ntester: 1/2 hit\ntools=execute_command"
    assert is_orch_echo(dump) is True


@pytest.mark.asyncio
async def test_orch_echo_does_not_lease(tmp_path: Path, monkeypatch):
    called = {"chat": 0}

    class Fake:
        def __init__(self, model: ModelConfig):
            self.model = model

        async def health(self, timeout_s: float):
            return True, "ok"

        async def chat(self, request):
            called["chat"] += 1
            return ChatResult(provider="x", model="x", text="nope")

    monkeypatch.setattr("harness.dispatch.build_provider", Fake)
    monkeypatch.setattr("harness.optimize.build_provider", Fake)
    report = await run_dispatch(
        _cfg(tmp_path),
        "Harness orch 20260826_164820_dispatch intent: x tester: 1/2 hit\ntools=execute_command",
    )
    assert report.slice_error
    assert report.shots == []
    assert called["chat"] == 0


def test_parse_foreman_plan_gather_and_dispatch():
    from harness.dispatch import bind_gather_calls, parse_foreman_plan

    mode, calls, packets = parse_foreman_plan(
        '{"mode":"gather","calls":[{"name":"read_file","arguments":{"path":"ARCHITECTURE.md"}}]}',
        4,
    )
    assert mode == "gather"
    assert calls[0]["name"] == "read_file"
    assert packets == []
    mode, calls, packets = parse_foreman_plan(
        '{"mode":"dispatch","packets":[{"id":"p1","title":"pong","prompt":"say PONG",'
        '"expect_tool":null,"accept":{"invariants":["text PONG"]}}]}',
        4,
    )
    assert mode == "dispatch"
    assert packets[0].id == "p1"
    catalog = {"read_file": ("path",), "write_to_file": ("path", "content")}
    bound = bind_gather_calls(
        [
            {"name": "read_file", "arguments": {"path": "ARCHITECTURE.md", "extra": 1}},
            {"name": "write_to_file", "arguments": {"path": "x", "content": "no"}},
        ],
        catalog,
    )
    assert len(bound) == 1
    assert bound[0]["function"]["name"] == "read_file"
    assert "ARCHITECTURE.md" in bound[0]["function"]["arguments"]
    assert "extra" not in bound[0]["function"]["arguments"]
    remapped = bind_gather_calls(
        [{"name": "readFile", "arguments": {"file_path": "lib/modelReach.ts"}}],
        catalog,
    )
    assert remapped[0]["function"]["name"] == "read_file"
    assert "lib/modelReach.ts" in remapped[0]["function"]["arguments"]


def test_gather_binds_to_real_cline_catalog():
    import json as _json

    from harness.dispatch import bind_gather_calls, default_gather_calls, merge_tool_catalog

    # The tool names this Cline actually exposes (from the AI_NoSuchToolError message).
    catalog = {
        "read_files": ("paths",),
        "search_codebase": ("query", "path"),
        "run_commands": ("commands",),
        "editor": ("path", "content"),
        "ask_question": ("question",),
        "fetch_web_content": ("url",),
    }
    merged = merge_tool_catalog(catalog)
    assert "read_file" not in merged  # never invent tools Cline does not have

    bound = bind_gather_calls(
        [
            {"name": "read_file", "arguments": {"path": "lib/modelReach.ts"}},
            {"name": "search_files", "arguments": {"path": ".", "regex": "TOTAL_REACH"}},
            {"name": "execute_command", "arguments": {"command": "ls -la"}},
            {"name": "editor", "arguments": {"path": "x", "content": "no"}},
            {"name": "ask_question", "arguments": {"question": "?"}},
        ],
        merged,
    )
    names = [c["function"]["name"] for c in bound]
    assert names == ["read_files", "search_codebase", "run_commands"]
    read_args = _json.loads(bound[0]["function"]["arguments"])
    assert read_args["paths"] == ["lib/modelReach.ts"]
    search_args = _json.loads(bound[1]["function"]["arguments"])
    assert search_args["query"] == "TOTAL_REACH"
    run_args = _json.loads(bound[2]["function"]["arguments"])
    assert run_args["commands"] == ["ls -la"]

    defaults = default_gather_calls(merged, "verify TOTAL_REACH consensus math")
    assert defaults
    default_names = {c["function"]["name"] for c in defaults}
    assert default_names <= {"read_files", "search_codebase", "run_commands"}
    # Empty catalog falls back to classic names so old Cline builds still work.
    classic = merge_tool_catalog({})
    assert "read_file" in classic


def test_frontend_job_gathers_apps_web_not_readme_only():
    from harness.dispatch import (
        AcceptSpec,
        Packet,
        default_gather_calls,
        evidence_covers_intent,
        is_frontend_job,
        merge_tool_catalog,
        sanitize_packets,
        thread_has_ui_source,
    )

    intent = "how about the front end code. is it clear, engaging?"
    assert is_frontend_job(intent)
    thin = (
        "tool(read_files): README.md\n- `apps/web` — thin debug / CRE surface\n"
        'tool(read_files): package.json\n{"name":"locationlocationlocation"}\n'
        "tool(read_files): apps/web/package.json\n{\"name\":\"@locdna/web\"}\n"
    )
    assert evidence_covers_intent(intent, thin) is False
    assert thread_has_ui_source(thin) is False
    assert evidence_covers_intent(
        intent,
        "tool(read_files): FILE apps/web/src/app/page.tsx\nexport default function Home() { return <main/> }",
    )

    catalog = merge_tool_catalog(
        {
            "read_files": ("paths",),
            "search_codebase": ("query", "path"),
            "run_commands": ("commands",),
            "list_files": ("path", "recursive"),
        }
    )
    calls = default_gather_calls(catalog, intent)
    blob = " ".join(c["function"]["arguments"] for c in calls)
    assert "apps/web" in blob
    assert "page.tsx" in blob

    lying = Packet(
        id="p1",
        title="Frontend condition assessment",
        prompt=(
            "Assess apps/web. You have NO file contents in this conversation, "
            "so do not invent specifics. Write that a real quality verdict is "
            "not yet possible."
        ),
        accept=AcceptSpec(invariants=("min_chars 120", "text not yet possible", "text apps/web")),
    )
    cleaned = sanitize_packets([lying], thin)
    assert "NO file contents" not in cleaned[0].prompt
    assert "not yet possible" not in cleaned[0].prompt.lower()
    assert all("not yet possible" not in inv for inv in cleaned[0].accept.invariants)
    assert "text apps/web" in cleaned[0].accept.invariants


def test_named_source_listing_is_not_coverage():
    from harness.dispatch import (
        default_gather_calls,
        evidence_covers_intent,
        merge_tool_catalog,
        needed_source_paths,
        thread_has_file_read,
    )

    intent = (
        "fix the failing unit test in test_add.py. add(1, 2) must return 3. "
        "Use pytest test_add.py as the acceptance command."
    )
    listing = (
        "assistant-tool list_files {\"path\": \".\", \"recursive\": false}\n"
        "tool(list_files): .git/\nadd.py\ntest_add.py\n"
        "assistant-tool run_commands {\"commands\": [\"ls -la\"]}\n"
        "tool(run_commands): add.py test_add.py README.md\n"
    )
    assert needed_source_paths(intent, listing) == ["test_add.py", "add.py"]
    assert thread_has_file_read(listing, "test_add.py") is False
    assert evidence_covers_intent(intent, listing) is False
    read = (
        'assistant-tool read_files {"paths":["test_add.py"]}\n'
        "tool(read_files): FILE test_add.py\nfrom add import add\n"
        'assistant-tool read_files {"paths":["add.py"]}\n'
        "tool(read_files): FILE add.py\ndef add(x, y):\n    return x - y\n"
    )
    assert evidence_covers_intent(intent, listing + read) is True

    catalog = merge_tool_catalog(
        {
            "read_files": ("paths",),
            "search_codebase": ("query", "path"),
            "run_commands": ("commands",),
            "list_files": ("path", "recursive"),
        }
    )
    calls = default_gather_calls(catalog, intent)
    blob = " ".join(c["function"]["arguments"] for c in calls)
    assert "test_add.py" in blob
    assert "add.py" in blob
    first = calls[0]["function"]["arguments"]
    assert "test_add.py" in first or "add.py" in first


def test_dispatch_cli_help():
    from typer.testing import CliRunner

    from harness.cli import app

    result = CliRunner().invoke(app, ["dispatch", "--help"])
    assert result.exit_code == 0
    assert "packets" in result.stdout

