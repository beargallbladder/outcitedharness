#!/usr/bin/env python3
"""Resume the chair miss-pack after voiding a hung Flash call."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from harness.cases.loader import discover_cases
from harness.config import load_config
from harness.runner import invoke_model, summarize_attempts
from harness.storage.db import Store, utcnow

RUN_ID = "20260823_000733_tournament"
VOID_CASE = "hc13_symlink_flip_macos"
VOID_MODEL = "deepseek_flash"
ONLY = ["deepseek_flash", "hy3", "minimax", "glm52", "frontier"]
ROOT = Path(__file__).resolve().parents[1]


def void_flash(cfg, store: Store) -> None:
    dest = cfg.settings.results_dir / "runs" / RUN_ID / VOID_CASE / VOID_MODEL
    dest.mkdir(parents=True, exist_ok=True)
    answer = dest / "answer.txt"
    raw = dest / "raw.json"
    meta = dest / "meta.json"
    answer.write_text("")
    raw.write_text(json.dumps({"void": True, "reason": "operator void: hung past timeout"}, indent=2))
    detail = {
        "reason": "operator void: hung past 240s timeout; killed after ~1h",
        "detail": {},
        "format_ok": None,
        "correctness_ok": None,
        "seed": None,
    }
    meta.write_text(
        json.dumps(
            {
                "provider": "openai_compatible",
                "model": cfg.models[VOID_MODEL].model,
                "latency_ms": None,
                "input_tokens": None,
                "output_tokens": None,
                "error": "VOID: hung past timeout, operator killed",
                "evaluation": {
                    "verdict": "VOID",
                    "evaluator": "human",
                    "reason": detail["reason"],
                    "format_ok": None,
                    "correctness_ok": None,
                    "detail": {},
                    "seed": None,
                },
            },
            indent=2,
        )
    )
    existing = store.model_results(RUN_ID, VOID_CASE)
    if any(r["model_key"] == VOID_MODEL for r in existing):
        print(f"VOID row already present for {VOID_CASE}/{VOID_MODEL}")
        return
    model = cfg.models[VOID_MODEL]
    store.insert_model_result(
        {
            "run_id": RUN_ID,
            "case_id": VOID_CASE,
            "model_key": VOID_MODEL,
            "provider": model.provider,
            "model": model.model,
            "tier": model.tier,
            "started_at": utcnow(),
            "latency_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "estimated_cost": None,
            "answer_path": str(answer),
            "raw_path": str(raw),
            "error": "VOID: hung past timeout, operator killed",
            "verdict": "VOID",
            "evaluator": "human",
            "evaluation_detail": detail,
        }
    )
    print(f"VOID {VOID_CASE}/{VOID_MODEL}")


def ensure_case_run(store: Store, case_id: str) -> None:
    existing = [r for r in store.case_runs(RUN_ID) if r["case_id"] == case_id]
    if existing:
        store.recompute_case_run(RUN_ID, case_id)
        return
    rows = store.model_results(RUN_ID, case_id)
    if not rows:
        return
    store.insert_case_run(
        {
            "run_id": RUN_ID,
            "case_id": case_id,
            "mode": "tournament",
            "minimum_model_that_solved": "NONE",
            "successful_tier": None,
            "total_escalation_latency_ms": sum((r["latency_ms"] or 0) for r in rows),
            "total_escalation_cost": None,
            "wasted_latency_before_success_ms": 0,
            "wasted_cost_before_success": None,
            "failed_tiers": len(rows),
            "started_at": rows[0]["started_at"],
            "finished_at": utcnow(),
        }
    )
    store.recompute_case_run(RUN_ID, case_id)


async def resume() -> None:
    cfg = load_config()
    store = Store(cfg.settings.db_path)
    void_flash(cfg, store)
    ensure_case_run(store, VOID_CASE)

    done = {(r["case_id"], r["model_key"]) for r in store.model_results(RUN_ID)}
    cases = discover_cases(ROOT / "cases" / "eval_chair")
    models = [cfg.models[k] for k in ONLY]
    remaining = [c for c in cases if any((c.id, m.key) not in done for m in models)]
    print(f"remaining cases: {[c.id for c in remaining]}")

    for case in remaining:
        need = [m for m in models if (case.id, m.key) not in done]
        print(f"running {case.id}: {[m.key for m in need]}", flush=True)
        raw = await asyncio.gather(
            *[invoke_model(cfg, store, RUN_ID, case, model) for model in need]
        )
        attempts = sorted(raw, key=lambda a: (a.model.tier, a.model.key))
        print(
            "  "
            + "  ".join(f"{a.model.key}={a.evaluation.verdict}" for a in attempts),
            flush=True,
        )
        if not any(r["case_id"] == case.id for r in store.case_runs(RUN_ID)):
            summary = summarize_attempts(case, "tournament", attempts)
            summary["run_id"] = RUN_ID
            store.insert_case_run(summary)
        else:
            store.recompute_case_run(RUN_ID, case.id)

    store.finish_run(RUN_ID, len(discover_cases(ROOT / "cases" / "eval_chair")))
    print("chair resume finished")


if __name__ == "__main__":
    asyncio.run(resume())
