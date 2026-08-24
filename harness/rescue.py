"""Quality-escalate to frontier with a constructed packet, not a Cline dump."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.config import AppConfig
from harness.cost import estimate_cost
from harness.providers.base import ChatMessage, ChatRequest
from harness.providers.factory import build_provider
from harness.storage.db import Store, utcnow

SECTIONS = (
    "TASK",
    "RELEVANT ARCHITECTURE",
    "FILES",
    "OBSERVED FAILURE",
    "ATTEMPTS",
    "TEST EVIDENCE",
    "FOREMAN HYPOTHESIS",
    "QUESTION",
)

PACKET_TEMPLATE = """# TASK

# RELEVANT ARCHITECTURE

# FILES

# OBSERVED FAILURE

# ATTEMPTS

# TEST EVIDENCE

# FOREMAN HYPOTHESIS

# QUESTION
root cause + next execution plan for the Spark coder
"""

SYSTEM = (
    "You are the senior reviewer. You receive a constructed packet only — "
    "not a chat transcript. Reply with:\n"
    "1. Root cause (one short paragraph)\n"
    "2. Next execution plan for the Spark coder: files to touch, commands to run, "
    "and the acceptance check\n"
    "Do not ask follow-up questions. Do not invent a new conversation."
)


class PacketError(ValueError):
    pass


def missing_sections(text: str) -> list[str]:
    upper = text.upper()
    missing = []
    for name in SECTIONS:
        if name not in upper:
            missing.append(name)
    return missing


def load_packet(path: Path) -> str:
    text = path.read_text()
    missing = missing_sections(text)
    if missing:
        raise PacketError(
            "Packet is missing sections: " + ", ".join(missing) + ". "
            "Use `harness rescue --template`."
        )
    if len(text) > 20_000:
        raise PacketError(
            f"Packet is {len(text)} chars. Keep it under 20k — do not paste a Cline thread."
        )
    return text


@dataclass
class RescueOutcome:
    run_id: str
    model_key: str
    text: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost: float | None
    error: str | None
    answer_path: Path


async def run_rescue(
    cfg: AppConfig,
    packet_path: Path,
    model_key: str = "frontier",
    store: Store | None = None,
) -> RescueOutcome:
    store = store or Store(cfg.settings.db_path)
    packet = load_packet(packet_path)
    model = cfg.models.get(model_key)
    if model is None or not model.enabled:
        raise PacketError(f"Model '{model_key}' is missing or disabled")

    from harness.runner import make_run_id, write_artifacts
    from harness.evaluation.base import EvalResult

    run_id = make_run_id("rescue")
    store.create_run(run_id, "rescue", notes=str(packet_path))
    provider = build_provider(model)
    request = ChatRequest(
        messages=[
            ChatMessage(role="system", content=SYSTEM),
            ChatMessage(role="user", content=packet),
        ],
        temperature=model.temperature or 0,
        max_tokens=model.max_tokens or 2048,
        extra_body=dict(model.extra_body),
        timeout_s=model.timeout_s,
    )
    started = utcnow()
    result = await provider.chat(request)
    evaluation = EvalResult(
        verdict="PARTIAL" if result.text and not result.error else "FAIL",
        evaluator="rescue",
        reason="senior packet reply" if result.text else (result.error or "empty"),
        format_ok=bool(result.text),
        correctness_ok=None,
    )
    answer_path, _raw = write_artifacts(
        cfg.settings.results_dir, run_id, packet_path.stem, model.key, result, evaluation
    )
    cost = estimate_cost(cfg.pricing_for(model.key), result.input_tokens, result.output_tokens)
    store.insert_model_result(
        {
            "run_id": run_id,
            "case_id": packet_path.stem,
            "model_key": model.key,
            "provider": result.provider,
            "model": result.model,
            "tier": model.tier,
            "started_at": started,
            "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "estimated_cost": cost,
            "answer_path": str(answer_path),
            "raw_path": str(_raw),
            "error": result.error,
            "verdict": evaluation.verdict,
            "evaluator": evaluation.evaluator,
            "evaluation_detail": {
                "reason": evaluation.reason,
                "packet": str(packet_path),
            },
        }
    )
    store.finish_run(run_id, 1)
    return RescueOutcome(
        run_id=run_id,
        model_key=model.key,
        text=result.text or "",
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        estimated_cost=cost,
        error=result.error,
        answer_path=answer_path,
    )
