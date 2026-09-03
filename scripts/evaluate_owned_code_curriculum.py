#!/usr/bin/env python3
"""Evaluate exact repair of held-out executable curriculum tasks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import urlparse

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_owned_code_curriculum_dataset import load_manifest
from harness.training.security import assert_no_secrets, redact_text


SCHEMA = "harness.owned-code-curriculum-evaluation.v1"
_DIFF_PATH = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_cases(dataset_root: Path, split: str, limit: int | None) -> list[dict[str, Any]]:
    manifest = load_manifest(dataset_root / "manifest.json")
    path = dataset_root / "canonical" / f"{split}.jsonl"
    receipt = manifest["artifacts"].get(f"canonical/{split}.jsonl")
    if (
        not path.is_file()
        or path.is_symlink()
        or receipt is None
        or path.stat().st_size != receipt["bytes"]
        or _sha256(path.read_bytes()) != receipt["sha256"]
    ):
        raise ValueError("curriculum split is missing or changed")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("curriculum evaluation split is empty")
    if limit is not None:
        if limit < 1:
            raise ValueError("evaluation limit must be positive")
        rows = rows[:limit]
    return rows


def _extract_patch(text: str, expected_path: str) -> str:
    value = text.strip()
    fenced = re.search(r"```(?:diff)?\s*\n(.*?)```", value, re.DOTALL)
    if fenced:
        value = fenced.group(1).strip()
    start = value.find("diff --git ")
    if start >= 0:
        patch = value[start:].strip() + "\n"
    else:
        header = f"--- a/{expected_path}\n+++ b/{expected_path}\n"
        start = value.find(header)
        if start < 0:
            raise ValueError("response has no unified Git diff")
        patch = (
            f"diff --git a/{expected_path} b/{expected_path}\n"
            + value[start:].strip()
            + "\n"
        )
    paths = _DIFF_PATH.findall(patch)
    if not paths or any(
        before != expected_path or after != expected_path
        for before, after in paths
    ):
        raise ValueError("candidate patch modifies an unexpected path")
    assert_no_secrets(patch, field="curriculum evaluation patch")
    return patch


def _apply_and_match(
    *,
    expected_path: str,
    mutant_source: str,
    expected_source_sha256: str,
    patch: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="harness-curriculum-eval-") as raw:
        root = Path(raw)
        target = root / expected_path
        target.parent.mkdir(parents=True)
        target.write_text(mutant_source, encoding="utf-8")
        result = subprocess.run(
            [
                "git",
                "apply",
                "--recount",
                "--whitespace=nowarn",
                "-",
            ],
            cwd=root,
            input=patch,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return {
                "passed": False,
                "patch_applied": False,
                "detail": redact_text(result.stderr)[-2000:],
                "result_sha256": None,
            }
        actual = _sha256(target.read_bytes())
        return {
            "passed": actual == expected_source_sha256,
            "patch_applied": True,
            "detail": (
                "exact canonical source restored"
                if actual == expected_source_sha256
                else "patch applied but canonical source digest differs"
            ),
            "result_sha256": actual,
        }


async def _evaluate_case(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    base_url: str,
    model: str,
    max_completion_tokens: int,
    case: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    async with semaphore:
        response = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You repair owned Python repositories. Return only one "
                            "unified Git diff for the requested file."
                        ),
                    },
                    {"role": "user", "content": case["prompt"]},
                ],
                "temperature": 0,
                "seed": 0,
                "max_tokens": max_completion_tokens,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "preserve_thinking": False,
                },
            },
        )
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    payload = response.json()
    text = str(payload["choices"][0]["message"].get("content") or "")
    assert_no_secrets(text, field="curriculum model response")
    error = None
    patch = ""
    try:
        path = str(case["lineage_id"]).split(":file:", 1)[1]
        patch = _extract_patch(text, path)
        result = await asyncio.to_thread(
            _apply_and_match,
            expected_path=path,
            mutant_source=str(case["mutant_source"]),
            expected_source_sha256=str(case["source_file_sha256"]),
            patch=patch,
        )
    except Exception as exc:
        error = redact_text(f"{type(exc).__name__}: {exc}")
        result = {
            "passed": False,
            "patch_applied": False,
            "detail": error,
            "result_sha256": None,
        }
    usage = payload.get("usage") if isinstance(payload, dict) else {}
    return {
        "case_id": case["pair_id"],
        "event_id": case["event_id"],
        "lineage_id": case["lineage_id"],
        "mutation_operator": case["mutation_operator"],
        "passed": result["passed"],
        "patch_applied": result["patch_applied"],
        "detail": result["detail"],
        "error": error,
        "latency_ms": round(latency_ms, 3),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "patch_sha256": _sha256(patch.encode()),
        "result_sha256": result["result_sha256"],
    }


async def evaluate(
    *,
    cases: list[dict[str, Any]],
    base_url: str,
    model: str,
    concurrency: int,
    timeout: float,
    max_completion_tokens: int = 2048,
    api_key: str | None = None,
) -> dict[str, Any]:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base URL must be an HTTP endpoint")
    if not 1 <= concurrency <= 16:
        raise ValueError("concurrency must be between one and 16")
    if not 1 <= max_completion_tokens <= 4096:
        raise ValueError("completion token budget must be between one and 4096")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        headers=headers,
    ) as client:
        models = await client.get(f"{base_url.rstrip('/')}/models")
        models.raise_for_status()
        names = {
            str(row.get("id") or "")
            for row in models.json().get("data", [])
            if isinstance(row, dict)
        }
        if model not in names:
            raise ValueError("configured model is not exposed by the endpoint")
        semaphore = asyncio.Semaphore(concurrency)
        rows = await asyncio.gather(
            *(
                _evaluate_case(
                    client,
                    semaphore,
                    base_url=base_url,
                    model=model,
                    max_completion_tokens=max_completion_tokens,
                    case=case,
                )
                for case in cases
            )
        )
    latencies = [row["latency_ms"] for row in rows]
    passed = sum(row["passed"] for row in rows)
    patch_applied = sum(row["patch_applied"] for row in rows)
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "model": model,
        "model_endpoint_sha256": _sha256(base_url.encode()),
        "generation_config": {
            "temperature": 0,
            "seed": 0,
            "max_completion_tokens": max_completion_tokens,
            "thinking": False,
        },
        "sample_count": len(rows),
        "passed": passed,
        "patch_applied": patch_applied,
        "verified_success_rate": passed / len(rows),
        "patch_application_rate": patch_applied / len(rows),
        "median_latency_ms": median(latencies),
        "p95_latency_ms": sorted(latencies)[round((len(latencies) - 1) * 0.95)],
        "cases": rows,
    }
    core["evidence_sha256"] = _sha256(_canonical(core))
    return core


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("curriculum evaluation output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o444)
    finally:
        temporary.unlink(missing_ok=True)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--api-key-env")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    api_key = (
        os.environ.get(arguments.api_key_env)
        if arguments.api_key_env
        else None
    )
    if arguments.api_key_env and not api_key:
        raise ValueError("configured API key environment variable is unavailable")
    manifest = load_manifest(arguments.dataset / "manifest.json")
    result = await evaluate(
        cases=_load_cases(arguments.dataset, arguments.split, arguments.limit),
        base_url=arguments.base_url,
        model=arguments.model,
        concurrency=arguments.concurrency,
        timeout=arguments.timeout,
        max_completion_tokens=arguments.max_completion_tokens,
        api_key=api_key,
    )
    result.pop("evidence_sha256", None)
    result["dataset_manifest_sha256"] = manifest["core_sha256"]
    result["split"] = arguments.split
    result["evidence_sha256"] = _sha256(_canonical(result))
    _write_once(arguments.output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "model",
                    "sample_count",
                    "passed",
                    "patch_applied",
                    "verified_success_rate",
                    "patch_application_rate",
                    "p95_latency_ms",
                    "evidence_sha256",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
