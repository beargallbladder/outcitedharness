from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from harness.config import AppConfig, ModelConfig
from harness.cost import estimate_cost
from harness.providers.base import ChatMessage, ChatRequest
from harness.providers.factory import build_provider
from harness.serial.tools import ToolCall, parse_tool, run_tool
from harness.storage.db import Store, utcnow


SYSTEM = """You are editing a small isolated checkout of our production repo.
You may use exactly one tool per turn, in this XML form:

<tool name="read"><path>relative/path</path></tool>
<tool name="grep"><pattern>regex</pattern><path>optional/subdir</path></tool>
<tool name="strreplace"><path>relative/path</path><old>exact old text</old><new>replacement</new></tool>
<tool name="run"></tool>
<tool name="finish"><summary>what you changed</summary></tool>

Rules:
- `run` executes the hidden oracle. It starts FAIL. Your job is to make it PASS.
- Do not invent files outside the checkout. Do not rewrite tests to skip the bug.
- After `run` prints PASS, call finish.
- One tool only per message. No prose-only turns unless you are done — then finish.
"""


@dataclass
class SerialTicket:
    id: str
    title: str
    prompt: str
    repo: Path
    oracle: Path
    max_turns: int = 16


@dataclass
class SerialAttempt:
    model: ModelConfig
    verdict: str
    reason: str
    turns: int
    tools: list[str] = field(default_factory=list)
    check_output: str = ""
    estimated_cost: float | None = None
    latency_ms: float = 0
    workspace: Path | None = None


def load_ticket(path: Path) -> SerialTicket:
    import yaml

    raw = yaml.safe_load((path / "ticket.yaml").read_text())
    return SerialTicket(
        id=raw["id"],
        title=raw.get("title") or raw["id"],
        prompt=(path / "prompt.md").read_text(),
        repo=path / "repo",
        oracle=path / "oracle" / "check.py",
        max_turns=int(raw.get("max_turns") or 16),
    )


def discover_tickets(path: Path) -> list[SerialTicket]:
    target = path.resolve()
    if (target / "ticket.yaml").exists():
        return [load_ticket(target)]
    return [load_ticket(p.parent) for p in sorted(target.glob("*/ticket.yaml"))]


async def run_serial_attempt(
    cfg: AppConfig,
    ticket: SerialTicket,
    model: ModelConfig,
    dest: Path,
) -> SerialAttempt:
    workspace = dest / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(ticket.repo, workspace)
    provider = build_provider(model)
    messages = [
        ChatMessage(role="system", content=SYSTEM),
        ChatMessage(role="user", content=ticket.prompt.strip()),
    ]
    tools: list[str] = []
    latency = 0.0
    tokens_in = 0
    tokens_out = 0
    last_check = ""
    transcript: list[dict] = []

    for turn in range(1, ticket.max_turns + 1):
        request = ChatRequest(
            messages=messages,
            temperature=model.temperature,
            max_tokens=model.max_tokens or 1800,
            extra_body=model.extra_body,
            timeout_s=model.timeout_s or cfg.settings.default_timeout_s,
        )
        result = await provider.chat(request)
        latency += result.latency_ms
        tokens_in += result.input_tokens or 0
        tokens_out += result.output_tokens or 0
        text = result.text or ""
        transcript.append({"turn": turn, "role": "assistant", "text": text, "error": result.error})
        if result.error:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "transcript.json").write_text(json.dumps(transcript, indent=2)[:200000])
            return SerialAttempt(
                model=model,
                verdict="ERROR",
                reason=result.error,
                turns=turn,
                tools=tools,
                latency_ms=latency,
                workspace=workspace,
                estimated_cost=estimate_cost(cfg.pricing_for(model.key), tokens_in, tokens_out),
            )
        call = parse_tool(text)
        if call is None:
            messages.append(ChatMessage(role="assistant", content=text))
            messages.append(
                ChatMessage(
                    role="user",
                    content="No tool call parsed. Emit exactly one <tool name=...> block.",
                )
            )
            continue
        tools.append(call.name)
        if call.name == "finish":
            last_check = run_tool(workspace, ticket.oracle, ToolCall("run", {}))
            passed = last_check.startswith("PASS")
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "transcript.json").write_text(json.dumps(transcript, indent=2)[:200000])
            (dest / "check.txt").write_text(last_check)
            return SerialAttempt(
                model=model,
                verdict="PASS" if passed else "FAIL",
                reason="oracle PASS" if passed else "finished without oracle PASS",
                turns=turn,
                tools=tools,
                check_output=last_check,
                latency_ms=latency,
                workspace=workspace,
                estimated_cost=estimate_cost(cfg.pricing_for(model.key), tokens_in, tokens_out),
            )
        observation = run_tool(workspace, ticket.oracle, call)
        if call.name == "run":
            last_check = observation
            if observation.startswith("PASS"):
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "transcript.json").write_text(json.dumps(transcript, indent=2)[:200000])
                (dest / "check.txt").write_text(last_check)
                return SerialAttempt(
                    model=model,
                    verdict="PASS",
                    reason="oracle PASS",
                    turns=turn,
                    tools=tools,
                    check_output=last_check,
                    latency_ms=latency,
                    workspace=workspace,
                    estimated_cost=estimate_cost(cfg.pricing_for(model.key), tokens_in, tokens_out),
                )
        transcript.append({"turn": turn, "role": "tool", "name": call.name, "result": observation[:4000]})
        messages.append(ChatMessage(role="assistant", content=text))
        messages.append(ChatMessage(role="user", content=f"Tool {call.name} result:\n{observation}"))

    if not last_check:
        last_check = run_tool(workspace, ticket.oracle, ToolCall("run", {}))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "transcript.json").write_text(json.dumps(transcript, indent=2)[:200000])
    (dest / "check.txt").write_text(last_check or "")
    return SerialAttempt(
        model=model,
        verdict="PASS" if last_check.startswith("PASS") else "FAIL",
        reason="hit turn limit" if not last_check.startswith("PASS") else "oracle PASS",
        turns=ticket.max_turns,
        tools=tools,
        check_output=last_check,
        latency_ms=latency,
        workspace=workspace,
        estimated_cost=estimate_cost(cfg.pricing_for(model.key), tokens_in, tokens_out),
    )


async def run_serial(
    cfg: AppConfig,
    tickets: list[SerialTicket],
    only: list[str] | None = None,
) -> str:
    from harness.runner import make_run_id

    store = Store(cfg.settings.db_path)
    run_id = make_run_id("serial")
    models = cfg.models_for_mode("tournament", only=only)
    store.create_run(run_id, "serial", notes=f"tickets={len(tickets)}")
    root = cfg.settings.results_dir / "runs" / run_id

    for ticket in tickets:
        import asyncio

        async def one(model: ModelConfig) -> SerialAttempt:
            dest = root / ticket.id / model.key
            return await run_serial_attempt(cfg, ticket, model, dest)

        attempts = await asyncio.gather(*[one(m) for m in models])
        started = utcnow()
        for attempt in attempts:
            store.insert_model_result(
                {
                    "run_id": run_id,
                    "case_id": ticket.id,
                    "model_key": attempt.model.key,
                    "provider": attempt.model.provider,
                    "model": attempt.model.model,
                    "tier": attempt.model.tier,
                    "started_at": started,
                    "latency_ms": attempt.latency_ms,
                    "input_tokens": None,
                    "output_tokens": None,
                    "estimated_cost": attempt.estimated_cost,
                    "answer_path": str(attempt.workspace) if attempt.workspace else "",
                    "raw_path": str(root / ticket.id / attempt.model.key / "transcript.json"),
                    "error": None if attempt.verdict != "ERROR" else attempt.reason,
                    "verdict": attempt.verdict,
                    "evaluator": "serial_oracle",
                    "evaluation_detail": {
                        "reason": attempt.reason,
                        "turns": attempt.turns,
                        "tools": attempt.tools,
                        "check": attempt.check_output[-1500:],
                    },
                }
            )
        winner = next((a for a in sorted(attempts, key=lambda x: x.model.tier) if a.verdict == "PASS"), None)
        store.insert_case_run(
            {
                "run_id": run_id,
                "case_id": ticket.id,
                "mode": "serial",
                "minimum_model_that_solved": (
                    winner.model.short_name if winner else "NONE"
                ),
                "successful_tier": winner.model.tier if winner else None,
                "started_at": started,
                "finished_at": utcnow(),
            }
        )
    store.finish_run(run_id, len(tickets))
    return run_id
