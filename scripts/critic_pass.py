#!/usr/bin/env python3
"""Second pass: same model, missed rubric groups as a checklist."""

from __future__ import annotations

import argparse
import asyncio
import json

from harness.cases.loader import discover_cases
from harness.cases.prompt import build_prompt
from harness.config import load_config
from harness.runner import invoke_model, make_run_id, summarize_attempts
from harness.storage.db import Store, utcnow


def missed_from(detail: dict) -> list[list[str]]:
    raw = detail.get("detail") or {}
    if isinstance(raw, str):
        raw = json.loads(raw)
    groups = []
    for item in raw.get("missed_groups") or []:
        groups.append([str(x) for x in item.get("group") or []])
    return groups


def critic_prompt(original: str, prior: str, missed: list[list[str]]) -> str:
    bullets = []
    for group in missed:
        bullets.append("- " + " / ".join(group[:6]))
    checklist = "\n".join(bullets) or "- (complete any unanswered question from the original task)"
    return (
        original.strip()
        + "\n\n## Completeness pass\n"
        "Your previous answer is below. It missed these required points "
        "(any one synonym per line is enough). Keep what was already correct. "
        "Add only the missing points.\n\n"
        f"{checklist}\n\n## Previous answer\n\n{prior.strip()}\n"
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-run", required=True)
    parser.add_argument("--cases", default="cases/eval_live")
    parser.add_argument("--only", default="dgx_qwen,glm52,hy3,deepseek_retest,minimax")
    args = parser.parse_args()
    only = [k.strip() for k in args.only.split(",") if k.strip()]

    cfg = load_config()
    store = Store(cfg.settings.db_path)
    cases = {c.id: c for c in discover_cases(cfg.root / args.cases)}
    models = {m.key: m for m in cfg.models_for_mode("tournament", only=only)}
    rows = store.model_results(args.from_run)
    run_id = make_run_id("critic")
    store.create_run(run_id, "critic", notes=f"from={args.from_run}")

    by_case: dict[str, list] = {cid: [] for cid in cases}
    for row in rows:
        key = row["model_key"]
        cid = row["case_id"]
        if key not in models or cid not in cases:
            continue
        if row["verdict"] not in ("PARTIAL", "FAIL"):
            continue
        detail = json.loads(row["evaluation_detail"] or "{}")
        missed = missed_from(detail)
        if row["evaluator"] != "keyword_rubric" and not missed:
            continue
        prior = ""
        if row["answer_path"]:
            from pathlib import Path

            prior = Path(row["answer_path"]).read_text()
        case = cases[cid]
        original = build_prompt(case, cfg.settings, models[key]).user
        packet = critic_prompt(original, prior, missed)
        attempt = await invoke_model(
            cfg, store, run_id, case, models[key], prompt_override=packet
        )
        by_case[cid].append(attempt)
        print(f"{cid} {key} {row['verdict']} -> {attempt.evaluation.verdict} {attempt.evaluation.reason}")

    for case in cases.values():
        attempts = by_case[case.id]
        if not attempts:
            continue
        summary = summarize_attempts(case, "critic", attempts)
        summary["run_id"] = run_id
        if not summary["started_at"]:
            summary["started_at"] = utcnow()
        store.insert_case_run(summary)
    store.finish_run(run_id, len(cases))
    print(f"critic run {run_id}")


if __name__ == "__main__":
    asyncio.run(main())
