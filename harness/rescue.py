"""Quality-escalate to frontier with a constructed packet, not a Cline dump."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

AUTO_SYSTEM = (
    "You are the frontier rescue worker. Local models already attempted this task and failed QA. "
    "Return the finished answer or concrete patch the user needs, not a discussion of the process. "
    "Use only facts and source evidence in the packet. Address the critic failures explicitly, "
    "include exact files and verification commands when code changes are required, and never claim "
    "that tests ran unless TEST EVIDENCE proves it. Do not ask follow-up questions."
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


def validate_packet(text: str, max_chars: int = 20_000) -> str:
    text = text.strip()
    missing = missing_sections(text)
    if missing:
        raise PacketError("Packet is missing sections: " + ", ".join(missing))
    if len(text) > max_chars:
        label = "20k" if max_chars == 20_000 else str(max_chars)
        raise PacketError(f"Packet is {len(text)} chars. Keep it under {label}.")
    return text


def load_packet(path: Path) -> str:
    try:
        return validate_packet(path.read_text())
    except PacketError as exc:
        raise PacketError(f"{exc} Use `harness rescue --template`.") from exc


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    marker = "\n\n[... clipped by harness ...]\n\n"
    head = (limit - len(marker)) // 2
    return text[:head] + marker + text[-(limit - len(marker) - head) :]


def build_auto_rescue_packet(
    intent: str,
    thread: str,
    attempts: list[dict[str, Any]],
    critic_text: str,
    *,
    max_chars: int = 20_000,
) -> str:
    """Build a bounded evidence packet after local solving is exhausted."""
    attempt_lines = []
    for row in attempts:
        attempt_lines.append(
            "\n".join(
                (
                    f"packet={row.get('packet', '')} worker={row.get('worker', '')}",
                    f"qa={row.get('qa', '')} reason={row.get('why', '')}",
                    f"answer:\n{_clip(str(row.get('answer') or ''), 2400)}",
                )
            )
        )
    fixed = (
        f"# TASK\n{_clip(intent, 3000)}\n\n"
        "# RELEVANT ARCHITECTURE\n"
        "Cursor/Cline has repository tools. Local foreman, coders, and critic attempted the task "
        "before this single frontier rescue.\n\n"
        "# FILES\nUse only file paths present in TEST EVIDENCE.\n\n"
        f"# OBSERVED FAILURE\n{_clip(critic_text, 2500) or 'Local QA rejected the attempted answer.'}\n\n"
        f"# ATTEMPTS\n{_clip(chr(10).join(attempt_lines), 6500) or '(none)'}\n\n"
        "# TEST EVIDENCE\n"
    )
    question = (
        "\n\n# FOREMAN HYPOTHESIS\n"
        "The local answer is incomplete, unsupported, or failed its acceptance checks.\n\n"
        "# QUESTION\n"
        "Produce the finished evidence-grounded answer. If changes are needed, provide an exact "
        "patch plan with files, edits, commands, and acceptance checks."
    )
    room = max(0, max_chars - len(fixed) - len(question))
    packet = fixed + _clip(thread, room) + question
    return validate_packet(packet, max_chars=max_chars)



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
    return await _run_rescue_packet(
        cfg,
        packet,
        case_id=packet_path.stem,
        notes=str(packet_path),
        model_key=model_key,
        store=store,
        system=SYSTEM,
    )


async def run_rescue_text(
    cfg: AppConfig,
    packet: str,
    *,
    case_id: str,
    model_key: str | None = None,
    store: Store | None = None,
) -> RescueOutcome:
    """Run one automatic frontier rescue without writing an intermediate packet."""
    store = store or Store(cfg.settings.db_path)
    packet = validate_packet(packet, max_chars=cfg.settings.frontier_max_input_chars)
    return await _run_rescue_packet(
        cfg,
        packet,
        case_id=case_id,
        notes=f"automatic rescue for {case_id}",
        model_key=model_key or cfg.settings.frontier_model_key,
        store=store,
        system=AUTO_SYSTEM,
        max_tokens=cfg.settings.frontier_max_output_tokens,
    )


async def _run_rescue_packet(
    cfg: AppConfig,
    packet: str,
    *,
    case_id: str,
    notes: str,
    model_key: str,
    store: Store,
    system: str,
    max_tokens: int | None = None,
) -> RescueOutcome:
    model = cfg.models.get(model_key)
    if model is None or not model.enabled:
        raise PacketError(f"Model '{model_key}' is missing or disabled")

    from harness.runner import make_run_id, write_artifacts
    from harness.evaluation.base import EvalResult

    run_id = make_run_id("rescue")
    store.create_run(run_id, "rescue", notes=notes)
    provider = build_provider(model)
    request = ChatRequest(
        messages=[
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=packet),
        ],
        temperature=model.temperature or 0,
        max_tokens=max_tokens or model.max_tokens or 2048,
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
        cfg.settings.results_dir, run_id, case_id, model.key, result, evaluation
    )
    cost = estimate_cost(cfg.pricing_for(model.key), result.input_tokens, result.output_tokens)
    store.insert_model_result(
        {
            "run_id": run_id,
            "case_id": case_id,
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
                "packet": notes,
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
