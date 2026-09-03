#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible coding model on a frozen repair fixture."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
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

from harness.training.security import assert_no_secrets, redact_text


FIXTURE_SCHEMA = "harness.coding-repair-fixture.v1"
RESULT_SCHEMA = "harness.coding-repair-evaluation.v1"
NETWORK_DENY_PROFILE = "(version 1)(allow default)(deny network*)"
_DIFF_PATH = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_fixture(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("coding fixture must be a regular file")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != FIXTURE_SCHEMA:
        raise ValueError("unsupported coding fixture")
    expected = value.get("core_sha256")
    core = {key: item for key, item in value.items() if key != "core_sha256"}
    if expected != _sha256(_canonical(core)):
        raise ValueError("coding fixture digest mismatch")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) < 5:
        raise ValueError("coding fixture requires at least five cases")
    identifiers = [row.get("case_id") for row in cases if isinstance(row, dict)]
    if (
        len(identifiers) != len(cases)
        or len(set(identifiers)) != len(cases)
        or any(not isinstance(value, str) or not value for value in identifiers)
    ):
        raise ValueError("coding fixture case IDs are malformed")
    return value


def _extract_patch(text: str) -> str:
    value = text.strip()
    fenced = re.search(r"```(?:diff)?\s*\n(.*?)```", value, re.DOTALL)
    if fenced:
        value = fenced.group(1).strip()
    start = value.find("diff --git ")
    if start >= 0:
        value = value[start:].strip() + "\n"
    else:
        start = value.find("--- a/solution.py\n+++ b/solution.py\n")
        if start < 0:
            raise ValueError("response has no unified git diff")
        value = (
            "diff --git a/solution.py b/solution.py\n"
            + value[start:].strip()
            + "\n"
        )
    paths = _DIFF_PATH.findall(value)
    if not paths or any(
        before != "solution.py" or after != "solution.py"
        for before, after in paths
    ):
        raise ValueError("candidate patch may modify only solution.py")
    assert_no_secrets(value, field="coding evaluation patch")
    return value


def _sandbox_command(workspace: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "test_solution.py",
        "hidden_test_solution.py",
    ]
    if platform.system() == "Darwin":
        return ["/usr/bin/sandbox-exec", "-p", NETWORK_DENY_PROFILE, *command]
    bubblewrap = shutil.which("bwrap")
    if platform.system() == "Linux" and bubblewrap:
        return [
            bubblewrap,
            "--unshare-net",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--bind",
            str(workspace),
            str(workspace),
            "--tmpfs",
            "/tmp",
            "--chdir",
            str(workspace),
            *command,
        ]
    raise RuntimeError("coding evaluation requires sandbox-exec or bubblewrap")


def _run_tests(case: dict[str, Any], patch: str | None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="harness-coding-eval-") as raw:
        root = Path(raw)
        files = case.get("files")
        if not isinstance(files, dict) or set(files) != {
            "solution.py",
            "test_solution.py",
        }:
            raise ValueError("coding fixture files are malformed")
        for relative, content in files.items():
            if not isinstance(content, str):
                raise ValueError("coding fixture source must be text")
            (root / relative).write_text(content)
        if patch is not None:
            checked = subprocess.run(
                [
                    "git",
                    "apply",
                    "--check",
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
            if checked.returncode:
                return {
                    "passed": False,
                    "returncode": checked.returncode,
                    "detail": redact_text(checked.stderr)[-2000:],
                    "patch_applied": False,
                }
            applied = subprocess.run(
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
            if applied.returncode:
                raise RuntimeError("patch changed between git apply checks")
        hidden = case.get("hidden_test")
        if not isinstance(hidden, str):
            raise ValueError("coding fixture hidden test is malformed")
        (root / "hidden_test_solution.py").write_text(hidden)
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": raw,
            "TMPDIR": raw,
            "CI": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            result = subprocess.run(
                _sandbox_command(root),
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "returncode": None,
                "detail": "test timeout",
                "patch_applied": patch is not None,
            }
        detail = redact_text((result.stderr or result.stdout)[-2000:])
        assert_no_secrets(detail, field="coding evaluation test output")
        return {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "detail": detail,
            "patch_applied": patch is not None,
        }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile)]


async def _evaluate_case(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    case: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    baseline = await asyncio.to_thread(_run_tests, case, None)
    if baseline["passed"]:
        raise ValueError(f"{case['case_id']}: frozen parent unexpectedly passes")
    prompt = (
        f"{case['prompt']}\n\n"
        "Repository files:\n"
        + "\n\n".join(
            f"--- {path} ---\n{content}"
            for path, content in sorted(case["files"].items())
        )
        + "\n\nReturn only a unified git diff modifying solution.py."
    )
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
                            "You repair Python repositories. Return only the "
                            "requested unified git diff."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "seed": 0,
                "max_tokens": 4096,
            },
        )
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    payload = response.json()
    text = str(payload["choices"][0]["message"].get("content") or "")
    assert_no_secrets(text, field="coding evaluation model response")
    error = None
    try:
        patch = _extract_patch(text)
        result = await asyncio.to_thread(_run_tests, case, patch)
    except Exception as exc:
        patch = ""
        error = redact_text(f"{type(exc).__name__}: {exc}")
        result = {
            "passed": False,
            "returncode": None,
            "detail": error,
            "patch_applied": False,
        }
    usage = payload.get("usage") if isinstance(payload, dict) else {}
    return {
        "case_id": case["case_id"],
        "task_class": case["task_class"],
        "passed": result["passed"],
        "patch_applied": result["patch_applied"],
        "returncode": result["returncode"],
        "detail": result["detail"],
        "error": error,
        "latency_ms": round(latency_ms, 3),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "patch_sha256": _sha256(patch.encode()),
    }


async def evaluate(
    *,
    fixture: dict[str, Any],
    base_url: str,
    model: str,
    concurrency: int,
    timeout: float,
) -> dict[str, Any]:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base URL must be an HTTP endpoint")
    assert_no_secrets(base_url, field="coding evaluation base URL")
    assert_no_secrets(model, field="coding evaluation model")
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        semaphore = asyncio.Semaphore(concurrency)
        rows = await asyncio.gather(
            *(
                _evaluate_case(
                    client,
                    base_url=base_url,
                    model=model,
                    case=case,
                    semaphore=semaphore,
                )
                for case in fixture["cases"]
            )
        )
    latencies = [row["latency_ms"] for row in rows]
    passed = sum(row["passed"] for row in rows)
    core: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "fixture_sha256": fixture["core_sha256"],
        "model": model,
        "model_endpoint_sha256": _sha256(base_url.encode()),
        "sample_count": len(rows),
        "passed": passed,
        "verified_success_rate": passed / len(rows),
        "median_latency_ms": median(latencies),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "critical_regressions": 0,
        "cases": rows,
    }
    core["evidence_sha256"] = _sha256(_canonical(core))
    return core


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("coding evaluation output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o444)
    finally:
        temporary.unlink(missing_ok=True)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300)
    arguments = parser.parse_args()
    if not 1 <= arguments.concurrency <= 4:
        raise ValueError("concurrency must be between one and four")
    result = await evaluate(
        fixture=load_fixture(arguments.fixture),
        base_url=arguments.base_url,
        model=arguments.model,
        concurrency=arguments.concurrency,
        timeout=arguments.timeout,
    )
    _write_once(arguments.output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "model",
                    "sample_count",
                    "passed",
                    "verified_success_rate",
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
