"""Foreman plus parallel GB10 workers for model optimization."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from harness.config import AppConfig, ModelConfig
from harness.providers.base import ChatMessage, ChatRequest, ChatResult
from harness.providers.factory import build_provider
from harness.runner import make_run_id
from harness.storage.db import Store
from harness.task.models import AttemptRecord
from harness.task.service import TaskService

from harness.workers.registry import load_registry

SENIOR_KEY = "frontier"

WORKER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file by absolute path",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


@dataclass
class OptimizeCase:
    id: str
    title: str
    prompt: str
    expect_tool: str | None = None


@dataclass
class WorkerShot:
    model_key: str
    result: ChatResult
    tokens_per_sec: float | None
    tool_names: list[str]
    tool_hit: bool
    preview: str


@dataclass
class CaseOutcome:
    case: OptimizeCase
    packet: str
    shots: list[WorkerShot]
    winner: str | None
    ranks: dict[str, int]
    reason: str


@dataclass
class OptimizeReport:
    run_id: str
    outcomes: list[CaseOutcome] = field(default_factory=list)
    health: dict[str, str] = field(default_factory=dict)
    senior_text: str = ""
    json_path: str = ""


def tokens_per_sec(result: ChatResult) -> float | None:
    if not result.output_tokens or result.latency_ms <= 0:
        return None
    return result.output_tokens / (result.latency_ms / 1000.0)


def tool_names(result: ChatResult) -> list[str]:
    names: list[str] = []
    for call in result.tool_calls:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = fn.get("name") or call.get("name")
        if name:
            names.append(str(name))
    return names


def load_cases(root: Path, only: list[str] | None = None) -> list[OptimizeCase]:
    folder = root / "cases" / "fleet_optimize"
    cases: list[OptimizeCase] = []
    if not folder.is_dir():
        return cases
    wanted = set(only) if only else None
    for path in sorted(folder.glob("*/case.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        case_id = str(raw.get("id") or path.parent.name)
        if wanted is not None and case_id not in wanted:
            continue
        prompt_path = path.parent / "prompt.md"
        cases.append(
            OptimizeCase(
                id=case_id,
                title=str(raw.get("title") or path.parent.name),
                prompt=prompt_path.read_text() if prompt_path.exists() else str(raw.get("prompt") or ""),
                expect_tool=raw.get("expect_tool"),
            )
        )
    return cases


def _preview(result: ChatResult, limit: int = 600) -> str:
    if result.error:
        return f"ERROR {result.error}"
    if result.tool_calls:
        return json.dumps(result.tool_calls, default=str)[:limit]
    return (result.text or "")[:limit]


async def _chat(
    model: ModelConfig,
    messages: list[ChatMessage],
    extra: dict[str, Any] | None = None,
    max_tokens: int | None = None,
) -> ChatResult:
    body = dict(model.extra_body)
    if extra:
        body.update(extra)
    req = ChatRequest(
        messages=messages,
        temperature=model.temperature,
        max_tokens=max_tokens or model.max_tokens or 512,
        extra_body=body,
        timeout_s=model.timeout_s,
    )
    return await build_provider(model).chat(req)


async def _foreman_packet(foreman: ModelConfig, case: OptimizeCase) -> str:
    result = await _chat(
        foreman,
        [
            ChatMessage(
                role="system",
                content=(
                    "You write short work packets for coder boxes. "
                    "Output ONLY the packet. No preamble. Under 1200 characters."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "Turn this into a worker packet with Intent, Constraints, and What to return.\n\n"
                    f"{case.prompt}"
                ),
            ),
        ],
    )
    if result.error or not (result.text or "").strip():
        return case.prompt
    return result.text.strip()[:2000]


def parse_ranks(text: str, keys: list[str]) -> tuple[str | None, dict[str, int], str]:
    blob = text or ""
    match = re.search(r"\{.*\}", blob, flags=re.S)
    if not match:
        return None, {}, blob[:400]
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, {}, blob[:400]
    ranks = {str(k): int(v) for k, v in (data.get("ranks") or {}).items() if str(k) in keys}
    winner = data.get("winner")
    winner_s = str(winner) if winner in keys else None
    return winner_s, ranks, str(data.get("reason") or "")[:400]


def _configured_foreman(cfg: AppConfig) -> ModelConfig | None:
    for worker in load_registry(cfg.root).pool("foreman"):
        if worker.model_key:
            model = cfg.models.get(worker.model_key)
            if model is not None and model.enabled:
                return model
    return None


async def _foreman_rank(
    foreman: ModelConfig, case: OptimizeCase, shots: list[WorkerShot]
) -> tuple[str | None, dict[str, int], str]:
    payload = []
    for shot in shots:
        payload.append(
            {
                "model_key": shot.model_key,
                "error": shot.result.error,
                "tool_names": shot.tool_names,
                "tool_hit": shot.tool_hit,
                "latency_ms": round(shot.result.latency_ms, 1),
                "preview": shot.preview,
            }
        )
    result = await _chat(
        foreman,
        [
            ChatMessage(
                role="system",
                content=(
                    'Score coder answers. JSON only: '
                    '{"winner":"model_key","ranks":{"k":1},"reason":"..."}. '
                    "Rank 1 is best. Prefer correct tool_calls over prose."
                ),
            ),
            ChatMessage(
                role="user",
                content=f"Case {case.id}: {case.title}\nExpect tool: {case.expect_tool}\n\n{json.dumps(payload)}",
            ),
        ],
    )
    return parse_ranks(result.text or "", [s.model_key for s in shots])


async def _senior_note(senior: ModelConfig, report: OptimizeReport) -> str:
    compact = [
        {
            "id": o.case.id,
            "winner": o.winner,
            "workers": [
                {
                    "model_key": s.model_key,
                    "latency_ms": round(s.result.latency_ms, 1),
                    "tok_s": round(s.tokens_per_sec, 1) if s.tokens_per_sec else None,
                    "tool_hit": s.tool_hit,
                    "error": s.result.error,
                }
                for s in o.shots
            ],
        }
        for o in report.outcomes
    ]
    result = await _chat(
        senior,
        [
            ChatMessage(
                role="system",
                content=(
                    "You are the senior reviewer. Six sentences max. "
                    "Say which box is fastest, which misses tools, and one tuning move."
                ),
            ),
            ChatMessage(role="user", content=json.dumps({"health": report.health, "cases": compact})),
        ],
        max_tokens=400,
    )
    if result.error:
        return f"ERROR {result.error}"
    return (result.text or "").strip()


async def run_optimize(
    cfg: AppConfig,
    *,
    worker_keys: list[str] | None = None,
    use_foreman: bool = True,
    use_senior: bool = False,
    only: list[str] | None = None,
) -> OptimizeReport:
    if worker_keys is None:
        worker_keys = [w.model_key for w in load_registry(cfg.root).pool("coder") if w.model_key]
    keys = list(worker_keys)
    missing = [k for k in keys if k not in cfg.models]
    if missing:
        raise ValueError(f"unknown workers: {missing}")
    workers = [cfg.models[k] for k in keys if cfg.models[k].enabled]
    if not workers:
        raise ValueError("no enabled workers")

    cases = load_cases(cfg.root, only=only)
    if not cases:
        raise ValueError("no fleet_optimize cases")

    foreman = _configured_foreman(cfg) if use_foreman else None
    if use_foreman and (foreman is None or not foreman.enabled):
        raise ValueError("no enabled foreman worker is configured")
    senior = cfg.models.get(SENIOR_KEY) if use_senior else None
    if use_senior and (senior is None or not senior.enabled):
        raise ValueError("senior frontier is not enabled")

    run_id = make_run_id("optimize")
    report = OptimizeReport(run_id=run_id)
    cfg.settings.results_dir.mkdir(parents=True, exist_ok=True)
    store = Store(cfg.settings.db_path)
    svc = TaskService(store)
    task = svc.start(f"optimize {run_id}")

    health_models = list(workers)
    if foreman:
        health_models.append(foreman)
    if senior:
        health_models.append(senior)
    for model in health_models:
        ok, detail = await build_provider(model).health(cfg.settings.health_timeout_s)
        report.health[model.key] = "ok" if ok else detail

    for case in cases:
        packet = case.prompt
        if foreman:
            packet = await _foreman_packet(foreman, case)

        async def one(model: ModelConfig, packet_text: str = packet) -> WorkerShot:
            result = await _chat(
                model,
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "You are a coder worker. Prefer a tool call over prose when a tool fits. "
                            "Keep answers short."
                        ),
                    ),
                    ChatMessage(role="user", content=packet_text),
                ],
                {"tools": WORKER_TOOLS, "tool_choice": "auto"},
            )
            names = tool_names(result)
            if case.expect_tool:
                hit = case.expect_tool in names
            else:
                hit = (not names) and bool((result.text or "").strip()) and not result.error
            return WorkerShot(
                model_key=model.key,
                result=result,
                tokens_per_sec=tokens_per_sec(result),
                tool_names=names,
                tool_hit=hit,
                preview=_preview(result),
            )

        shots = list(await asyncio.gather(*[one(m) for m in workers]))
        winner, ranks, reason = None, {}, ""
        if foreman:
            winner, ranks, reason = await _foreman_rank(foreman, case, shots)
        report.outcomes.append(
            CaseOutcome(case=case, packet=packet, shots=shots, winner=winner, ranks=ranks, reason=reason)
        )
        for i, shot in enumerate(shots):
            last = i == len(shots) - 1 and case is cases[-1] and not use_senior
            svc.record(
                AttemptRecord(
                    task_id=task.task_id,
                    attempt=0,
                    worker=shot.model_key,
                    result="success" if not shot.result.error else "failed",
                    tool_calls=len(shot.tool_names),
                    input_tokens=shot.result.input_tokens,
                    output_tokens=shot.result.output_tokens,
                    ttft_ms=shot.result.latency_ms,
                    tokens_per_sec=shot.tokens_per_sec,
                ),
                close=last,
            )

    if senior:
        report.senior_text = await _senior_note(senior, report)
        svc.record(
            AttemptRecord(
                task_id=task.task_id,
                attempt=0,
                worker=senior.key,
                result="success" if not report.senior_text.startswith("ERROR") else "failed",
                output_tokens=None,
            ),
            close=True,
        )

    dest = cfg.settings.results_dir / f"{run_id}.json"
    dest.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "health": report.health,
                "senior": report.senior_text,
                "cases": [
                    {
                        "id": o.case.id,
                        "winner": o.winner,
                        "ranks": o.ranks,
                        "reason": o.reason,
                        "packet_chars": len(o.packet),
                        "workers": [
                            {
                                "model_key": s.model_key,
                                "latency_ms": s.result.latency_ms,
                                "input_tokens": s.result.input_tokens,
                                "output_tokens": s.result.output_tokens,
                                "tokens_per_sec": s.tokens_per_sec,
                                "tool_names": s.tool_names,
                                "tool_hit": s.tool_hit,
                                "error": s.result.error,
                            }
                            for s in o.shots
                        ],
                    }
                    for o in report.outcomes
                ],
            },
            indent=2,
        )
    )
    report.json_path = str(dest)
    return report
