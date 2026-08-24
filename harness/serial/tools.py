from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


TOOL_RE = re.compile(
    r"<tool\s+name=['\"](\w+)['\"]\s*>(.*?)</tool>",
    re.IGNORECASE | re.DOTALL,
)
TAG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)


@dataclass
class ToolCall:
    name: str
    args: dict[str, str]


def parse_tool(text: str) -> ToolCall | None:
    match = TOOL_RE.search(text or "")
    if match:
        name = match.group(1).strip().lower()
        args = {m.group(1).lower(): m.group(2) for m in TAG_RE.finditer(match.group(2))}
        return ToolCall(name=name, args=args)
    finish = re.search(r"\b(DONE|FINISHED)\b", text or "", re.I)
    if finish and "<tool" not in (text or "").lower():
        return ToolCall(name="finish", args={"summary": (text or "")[-1500:]})
    return None


def resolve_in(workspace: Path, rel: str) -> Path:
    raw = (rel or "").strip().lstrip("/")
    if not raw or ".." in Path(raw).parts:
        raise ValueError(f"illegal path: {rel!r}")
    target = (workspace / raw).resolve()
    root = workspace.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes workspace: {rel}")
    return target


def run_tool(workspace: Path, oracle: Path, call: ToolCall) -> str:
    name = call.name
    if name == "read":
        path = resolve_in(workspace, call.args.get("path", ""))
        if not path.is_file():
            return f"ERROR: missing file {path.relative_to(workspace)}"
        text = path.read_text()
        if len(text) > 20_000:
            text = text[:20_000] + "\n...[truncated]..."
        return text
    if name == "grep":
        pattern = call.args.get("pattern", "")
        if not pattern:
            return "ERROR: pattern required"
        rel = (call.args.get("path") or "").strip()
        roots = [resolve_in(workspace, rel)] if rel else [workspace]
        regex = re.compile(pattern)
        hits: list[str] = []
        for root in roots:
            files = [root] if root.is_file() else sorted(root.rglob("*"))
            for file in files:
                if not file.is_file():
                    continue
                try:
                    lines = file.read_text().splitlines()
                except Exception:
                    continue
                for idx, line in enumerate(lines, 1):
                    if regex.search(line):
                        loc = file.relative_to(workspace)
                        hits.append(f"{loc}:{idx}:{line}")
                        if len(hits) >= 40:
                            return "\n".join(hits)
        return "\n".join(hits) if hits else "NO MATCHES"
    if name == "strreplace":
        path = resolve_in(workspace, call.args.get("path", ""))
        old = call.args.get("old", "")
        new = call.args.get("new", "")
        if not path.is_file():
            return f"ERROR: missing file {path.relative_to(workspace)}"
        text = path.read_text()
        count = text.count(old)
        if count == 0:
            return "ERROR: old text not found"
        if count > 1:
            return f"ERROR: old text matched {count} times; make it unique"
        path.write_text(text.replace(old, new, 1))
        return f"OK replaced 1 occurrence in {path.relative_to(workspace)}"
    if name == "run":
        completed = subprocess.run(
            ["python3", str(oracle)],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env={**__import__("os").environ, "HARNESS_WORKSPACE": str(workspace)},
        )
        out = (completed.stdout + completed.stderr).strip()
        status = "PASS" if completed.returncode == 0 else "FAIL"
        return f"{status} (exit {completed.returncode})\n{out[-4000:]}"
    if name == "finish":
        return "FINISH"
    return f"ERROR: unknown tool {name}. Use read, grep, strreplace, run, finish."
