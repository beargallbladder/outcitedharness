from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.cases.prompt import build_prompt
from harness.cases.schema import Case
from harness.config import AppConfig, ModelConfig
from harness.cost import estimate_cost
from harness.evaluation import evaluate
from harness.evaluation.base import EvalResult
from harness.providers.base import ChatMessage, ChatRequest, ChatResult
from harness.providers.factory import build_provider
from harness.storage.db import Store, utcnow


def make_run_id(mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{mode}"


@dataclass
class ModelAttempt:
    model: ModelConfig
    result: ChatResult
    evaluation: EvalResult
    estimated_cost: float | None
    started_at: str
    answer_path: Path
    raw_path: Path


def write_artifacts(
    root: Path,
    run_id: str,
    case_id: str,
    model_key: str,
    result: ChatResult,
    evaluation: EvalResult,
    seed: int | None = None,
) -> tuple[Path, Path]:
    dest = root / "runs" / run_id / case_id / model_key
    if seed is not None:
        dest = dest / f"seed_{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    answer_path = dest / "answer.txt"
    raw_path = dest / "raw.json"
    meta_path = dest / "meta.json"
    answer_path.write_text(result.text or "")
    raw_path.write_text(json.dumps(result.raw_response, indent=2, default=str))
    meta_path.write_text(
        json.dumps(
            {
                "provider": result.provider,
                "model": result.model,
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "error": result.error,
                "evaluation": {
                    "verdict": evaluation.verdict,
                    "evaluator": evaluation.evaluator,
                    "reason": evaluation.reason,
                    "format_ok": evaluation.format_ok,
                    "correctness_ok": evaluation.correctness_ok,
                    "detail": evaluation.detail,
                    "seed": seed,
                },
            },
            indent=2,
            default=str,
        )
    )
    return answer_path, raw_path


async def invoke_model(
    cfg: AppConfig,
    store: Store,
    run_id: str,
    case: Case,
    model: ModelConfig,
    seed: int | None = None,
    prompt_override: str | None = None,
) -> ModelAttempt:
    packet = build_prompt(case, cfg.settings, model)
    messages: list[ChatMessage] = []
    if packet.system:
        messages.append(ChatMessage(role="system", content=packet.system))
    messages.append(ChatMessage(role="user", content=prompt_override or packet.user))

    request = ChatRequest(
        messages=messages,
        temperature=model.temperature,
        max_tokens=model.max_tokens,
        extra_body=model.extra_body,
        timeout_s=model.timeout_s or cfg.settings.default_timeout_s,
        seed=seed,
    )
    started_at = utcnow()
    provider = build_provider(model)
    try:
        result = await asyncio.wait_for(provider.chat(request), timeout=request.timeout_s)
    except asyncio.TimeoutError:
        result = ChatResult(
            provider=model.provider,
            model=model.model,
            latency_ms=request.timeout_s * 1000,
            error=f"TimeoutError: exceeded {request.timeout_s}s hard cap",
        )
    evaluation = evaluate(case, result.text or "", error=result.error)
    cost = estimate_cost(
        cfg.pricing_for(model.key),
        result.input_tokens,
        result.output_tokens,
    )
    answer_path, raw_path = write_artifacts(
        cfg.settings.results_dir,
        run_id,
        case.id,
        model.key,
        result,
        evaluation,
        seed=seed,
    )
    store.insert_model_result(
        {
            "run_id": run_id,
            "case_id": case.id,
            "model_key": model.key,
            "provider": result.provider,
            "model": result.model,
            "tier": model.tier,
            "started_at": started_at,
            "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "estimated_cost": cost,
            "answer_path": str(answer_path),
            "raw_path": str(raw_path),
            "error": result.error,
            "verdict": evaluation.verdict,
            "evaluator": evaluation.evaluator,
            "evaluation_detail": {
                "reason": evaluation.reason,
                "detail": evaluation.detail,
                "format_ok": evaluation.format_ok,
                "correctness_ok": evaluation.correctness_ok,
                "seed": seed,
            },
        }
    )
    return ModelAttempt(
        model=model,
        result=result,
        evaluation=evaluation,
        estimated_cost=cost,
        started_at=started_at,
        answer_path=answer_path,
        raw_path=raw_path,
    )


def summarize_attempts(case: Case, mode: str, attempts: list[ModelAttempt]) -> dict[str, Any]:
    winner = next((a for a in attempts if a.evaluation.verdict == "PASS"), None)
    total_ms = sum(a.result.latency_ms for a in attempts)
    costs = [a.estimated_cost for a in attempts if a.estimated_cost is not None]
    total_cost = sum(costs) if costs else None

    if winner:
        prior = attempts[: attempts.index(winner)]
        waste_ms = sum(a.result.latency_ms for a in prior)
        waste_costs = [a.estimated_cost for a in prior if a.estimated_cost is not None]
        waste_cost = sum(waste_costs) if waste_costs else 0.0
        failed_tiers = len(prior)
        min_model = winner.model.short_name
        successful_tier = winner.model.tier
    else:
        waste_ms = total_ms
        waste_cost = total_cost
        failed_tiers = len(attempts)
        min_model = "NONE"
        successful_tier = None

    return {
        "run_id": None,
        "case_id": case.id,
        "mode": mode,
        "minimum_model_that_solved": min_model,
        "successful_tier": successful_tier,
        "total_escalation_latency_ms": total_ms,
        "total_escalation_cost": total_cost,
        "wasted_latency_before_success_ms": waste_ms,
        "wasted_cost_before_success": waste_cost,
        "failed_tiers": failed_tiers,
        "started_at": attempts[0].started_at if attempts else utcnow(),
        "finished_at": utcnow(),
    }
