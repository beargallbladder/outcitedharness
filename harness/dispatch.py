"""M5 carves packets; idle coder boxes take them. Nemotron QA. Not Cline. Not harness-auto."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from harness.config import AppConfig, ModelConfig
from harness.optimize import (
    SENIOR_KEY,
    _chat,
    tokens_per_sec,
    tool_names,
)
from harness.providers.base import ChatMessage, ChatResult
from harness.providers.factory import build_provider
from harness.rescue import PacketError, build_auto_rescue_packet, run_rescue_text
from harness.runner import make_run_id
from harness.storage.db import Store
from harness.task.models import AttemptRecord, Evidence
from harness.task.service import TaskService
from harness.workers.registry import Worker, load_registry

FOREMAN_KEY = "m5_qwen"
_FOREMAN_PROBE_TIMEOUT_S = 4.0
_FOREMAN_CACHE_TTL_S = 20.0
_foreman_cache: dict[str, tuple[float, bool]] = {}

FOREMAN_SYSTEM = (
    "You are the harness foreman. Split work into packets coder boxes can finish in text. "
    "JSON array only. No preamble. No lease without accept. "
    "Coder boxes cannot see the user's disk and cannot run Cline tools. "
    "Do not emit find, ls, glob, grep, or /testbed packets. "
    "The INTENT block is the job. RECENT HARNESS DISPATCH is metrics only, not the job. "
    "Never assert workspace facts (git state, test results, file contents) that are not "
    "in THREAD tool results from THIS conversation; inventing them is the worst failure. "
    "Never put the expected answer sentence inside a packet prompt: a packet the coder "
    "can satisfy by echoing its own prompt is invalid, emit [] instead. "
    "If INTENT asks to read, edit, or review a repo and INTENT/THREAD have no file contents, emit []. "
    "A path that appears in THREAD (listing, README, package.json, apps/web) EXISTS. "
    "Unread is not missing. Never write 'no file contents', 'no frontend', or "
    "'not yet possible' when THREAD already names that tree. If INTENT needs those "
    "files and only the tree name is present, emit [] so Cline can gather the source. "
    "EXCEPTION: if THREAD tool results show the workspace does NOT contain the requested files "
    "(empty ls, missing paths, wrong project), emit ONE packet telling the coder to report exactly "
    "that, quoting the directory listing, with accept invariant \"text workspace\". "
    "That exception is only for empty listings or explicit missing-path errors, not unread files. "
    "Any question that can be answered in prose MUST get one or more packets. "
    "Each item: "
    '{"id":"p1","title":"...","prompt":"...","expect_tool":null,'
    '"files":[],"accept":{"commands":[],"invariants":["text PONG"]}}. '
    "expect_tool must be null. "
    "accept.invariants is required. Use \"text SUBSTR\" for a phrase the answer must contain. "
    "Each prompt is the full brief the coder needs. Under 1200 characters. "
    "One independent written deliverable per packet."
)

CRITIC_SYSTEM = (
    "You are the adversarial validation worker. Grade factual grounding, honesty, and form. "
    "The harness has machine-checked each shot's simple text invariants and put the result "
    "in python_ok; python_ok=true only makes a shot eligible for semantic review, it is NOT "
    "evidence that the answer is correct. "
    "JSON object only. No thinking. No markdown. No preamble. "
    'Schema: {"verdict":"proceed|revise|reject|insufficient","shots":[...]}. '
    "shots MUST contain exactly one entry per input shot, same ids, same order. "
    "Grading only some shots is invalid output. "
    'Each entry: {"id":"<shot id>","pass":true|false,"why":"..."}. '
    "why is your own judgment in at most 10 words; never copy wording from "
    "these instructions. "
    "Rules, in order: python_ok=false is always pass=false. "
    "For python_ok=true, compare answer to packet_prompt. When packet_prompt contains "
    "'WORKSPACE EVIDENCE GATHERED BY CLINE', only text after that marker is evidence for "
    "repository facts; instructions or suggested findings before it are not evidence. "
    "Every concrete claim about code, symbols, control flow, line numbers, test output, or "
    "failure modes must be directly supported by that evidence. Reject invented behavior, "
    "misquoted code, claims that contradict the evidence, empty answers, and bare tool dumps. "
    "A truncation marker or excerpt ending mid-expression only means evidence was omitted; "
    "reject any answer that reports the cutoff or omitted code as a source defect. "
    "A truthful answer that says no issue is proven by the supplied evidence MUST pass when "
    "it does not make unsupported claims; never require a positive finding. "
    "Do not reward an answer merely for repeating terms required by accept.invariants. "
    "The verdict must match shot scores: proceed means all pass, revise means a mixture, "
    "and reject means none pass. "
    "Never fail a shot because the user's wider request was not satisfiable; "
    "a truthful report that the workspace lacks the requested files passes. "
    "Unread is not missing. If evidence names a path (apps/web, package.json), "
    "reject answers that say those files do not exist or that the worker has no access. "
    "verdict and shots are the ONLY fields. Do not rewrite, quote, or improve "
    "the shot content. End output immediately after the closing brace."
)

CODER_SYSTEM = (
    "You are a coder worker. You have no disk and no Cline tools. "
    "Write the finished answer in plain text. Code, findings, or a direct reply. "
    "Do not call tools. Do not emit tool JSON. Do not search /testbed. "
    "Do not say you will investigate later. Answer now from the packet. "
    "If WORKSPACE EVIDENCE or the brief names a path, that tree exists. "
    "Do not say you lack access or that the frontend is missing. "
    "If the brief says you have no files but the evidence lists them, trust the evidence. "
    "Judge only what was copied; state the evidence limit; do not invent unread files."
)

FOREMAN_ORCH_SYSTEM = (
    "You are the harness foreman. Cline has hands on the user's disk. Coder boxes are blind. "
    "JSON object only. No preamble. No thinking. "
    "If the repo is needed and THREAD is missing those files, gather with Cline tools: "
    '{"mode":"gather","calls":[{"name":"read_file","arguments":{"path":"..."}}]}. '
    "Use only the listed Cline tool names. Prefer read_file, list_files, search_files. "
    "If INTENT asks to sync, pull, checkout, or run git or any shell command in the "
    "workspace, gather with the execute/run tool; coders cannot touch the repo. "
    "If THREAD shows a successful edit/write tool result but no later test or verification "
    "output, gather exactly one execute/run call for the narrowest relevant test. Never "
    "declare a code change complete before command evidence returns. If verification fails, "
    "dispatch the failure evidence for local repair. "
    "Never assert workspace facts (git state, test results, file contents) that are not "
    "in THREAD tool results from THIS conversation. "
    "Never put the expected answer sentence inside a packet prompt: a packet the coder "
    "can satisfy by echoing its own prompt is invalid. "
    "Max 8 calls. No writes. No attempt_completion. No /testbed. "
    "A listed path EXISTS. Unread source on that tree is a gather trigger, not a dispatch. "
    "If INTENT is frontend/UI/clarity/engagement and THREAD names apps/web but has no "
    ".tsx/component source, gather those files; do not dispatch a 'cannot assess' packet. "
    "Never write 'no file contents', 'no frontend', or 'not yet possible' when THREAD "
    "already names that tree. Never require those phrases as accept invariants. "
    "If THREAD already has enough file contents, or the question needs no repo, dispatch: "
    '{"mode":"dispatch","packets":[{"id":"p1","title":"...","prompt":"...","expect_tool":null,'
    '"files":[],"accept":{"commands":[],"invariants":["text PONG"]}}]}. '
    "If THREAD tool results show the workspace does NOT contain the requested files, dispatch ONE "
    "packet: report the mismatch, quote the listing, accept invariant \"text workspace\". "
    "That missing-files packet is only for empty ls or explicit missing-path errors. "
    "Copy needed file excerpts into packet.prompt. expect_tool must be null. "
    "accept.invariants is required. Use \"text SUBSTR\"."
)

# Concepts the harness understands → tool names Cline variants use for them.
# We NEVER emit a name that is not in the catalog Cline sent this turn.
CONCEPT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "read": ("read_file", "read_files"),
    "search": ("search_files", "search_codebase", "codebase_search", "grep", "glob"),
    "list": ("list_files", "list_dir", "list_code_definition_names"),
    "run": ("execute_command", "run_commands", "run_command", "bash", "shell"),
    "edit": ("editor", "apply_diff", "replace_in_file", "write_to_file"),
}

TOOL_CONCEPT: dict[str, str] = {
    name: concept for concept, names in CONCEPT_CANDIDATES.items() for name in names
}
TOOL_CONCEPT.update(
    {
        "read": "read",
        "readfile": "read",
        "list": "list",
        "listfiles": "list",
        "search": "search",
        "searchfiles": "search",
        "execute": "run",
        "run": "run",
        "edit": "edit",
        "apply": "edit",
    }
)

GATHER_TOOLS = frozenset(TOOL_CONCEPT)

WRITE_TOOLS = frozenset(
    {
        "write_to_file",
        "replace_in_file",
        "editor",
        "apply_diff",
        "attempt_completion",
        "new_task",
        "browser_action",
        "fetch_web_content",
        "ask_question",
        "ask_followup_question",
        "plan_mode_response",
    }
)

ACTION_MAX_CALLS = 5
ACTION_SYSTEM = (
    "You are the harness execution integrator. Cursor/Cline has the workspace tools; local "
    "workers are blind and supplied an accepted solution. Return JSON only. If the solution "
    "contains enough exact information to change the workspace, return "
    '{"mode":"act","calls":[{"name":"an available edit tool","arguments":{...}}]}. '
    "Emit 1 to 5 edit calls this turn, one distinct file per call, only paths present in "
    "the solution. Tests happen after Cline returns those tool results. "
    "Use only a tool name and argument properties in AVAILABLE TOOLS. Never invent paths, "
    "edits, or code absent from the accepted solution and evidence. If this is written work, "
    "a review, or the patch is not exact enough to apply safely, return "
    '{"mode":"complete","calls":[]}.'
)

# Only used when Cline sent no tool schema at all.
DEFAULT_CLINE_TOOLS: dict[str, tuple[str, ...]] = {
    "read_file": ("path",),
    "search_files": ("path", "regex", "file_pattern"),
    "execute_command": ("command",),
}

ARG_ALIASES = {
    "file_path": "path",
    "filepath": "path",
    "rel_path": "path",
    "uri": "path",
    "cmd": "command",
    "command_line": "command",
    "pattern": "regex",
}


@dataclass
class AcceptSpec:
    commands: tuple[str, ...] = ()
    invariants: tuple[str, ...] = ()

    def ready(self) -> bool:
        return bool(self.invariants or self.commands)


@dataclass
class Packet:
    id: str
    title: str
    prompt: str
    expect_tool: str | None = None
    files: tuple[str, ...] = ()
    accept: AcceptSpec = field(default_factory=AcceptSpec)


@dataclass
class Shot:
    packet: Packet
    worker_id: str
    model_key: str
    result: ChatResult
    tokens_per_sec: float | None
    tool_names: list[str]
    tool_hit: bool
    qa_pass: bool
    preview: str
    qa_why: str = ""


@dataclass
class DispatchReport:
    run_id: str
    intent: str
    packets: list[Packet] = field(default_factory=list)
    shots: list[Shot] = field(default_factory=list)
    health: dict[str, str] = field(default_factory=dict)
    senior_text: str = ""
    critic_text: str = ""
    critic_verdict: str = ""
    critic_key: str = ""
    json_path: str = ""
    slice_error: str = ""
    foreman_key: str = ""
    task_id: str = ""
    local_rounds: int = 0
    attempt_history: list[Shot] = field(default_factory=list)
    frontier_text: str = ""
    frontier_run_id: str = ""
    frontier_model_key: str = ""
    frontier_verified: bool = False
    frontier_why: str = ""
    frontier_cost: float | None = None


def score_tool_hit(packet: Packet, result: ChatResult, names: list[str]) -> bool:
    if result.error:
        return False
    if packet.expect_tool:
        return packet.expect_tool in names
    return bool((result.text or "").strip())


def score_invariants(packet: Packet, result: ChatResult, names: list[str]) -> bool:
    text = (result.text or "").strip()
    if not text:
        return False
    if not score_tool_hit(packet, result, names):
        return False
    for raw in packet.accept.invariants:
        inv = raw.strip()
        if not inv:
            continue
        lower = inv.lower()
        if lower.startswith("tool "):
            if inv.split(None, 1)[1] not in names:
                return False
        elif lower.startswith("no "):
            if inv.split(None, 1)[1] in names:
                return False
        elif lower.startswith("text "):
            parts = inv.split(None, 1)
            if len(parts) < 2 or parts[1].lower() not in text.lower():
                return False
        elif lower.startswith("min_chars "):
            parts = inv.split(None, 1)
            try:
                minimum = int(parts[1])
            except (IndexError, ValueError):
                return False
            if len(text) < minimum:
                return False
        else:
            if inv.lower() not in text.lower() and inv not in names:
                return False
    return True


def _accept_from_row(row: dict[str, Any], expect: str | None) -> AcceptSpec:
    raw = row.get("accept") if isinstance(row.get("accept"), dict) else {}
    commands = tuple(str(x).strip() for x in (raw.get("commands") or []) if str(x).strip())
    invariants = [str(x).strip() for x in (raw.get("invariants") or []) if str(x).strip()]
    if expect and f"tool {expect}" not in invariants:
        invariants.insert(0, f"tool {expect}")
    return AcceptSpec(commands=commands, invariants=tuple(invariants))


def parse_packets(text: str, fallback: str, limit: int) -> list[Packet]:
    blob = text or ""
    match = re.search(r"\[.*\]", blob, flags=re.S)
    rows: list[Any] = []
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                rows = data
        except json.JSONDecodeError:
            rows = []
    packets: list[Packet] = []
    for i, row in enumerate(rows[:limit]):
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "").strip()
        if not prompt:
            continue
        expect = row.get("expect_tool")
        expect_s = str(expect).strip() if expect not in (None, "", "null", "none") else None
        files = tuple(str(x) for x in (row.get("files") or []) if str(x).strip())
        if not isinstance(row.get("accept"), dict):
            continue
        accept = _accept_from_row(row, expect_s)
        if not accept.ready():
            continue
        packets.append(
            Packet(
                id=str(row.get("id") or f"p{i+1}"),
                title=str(row.get("title") or f"packet {i+1}"),
                prompt=prompt[:16000],
                expect_tool=expect_s,
                files=files,
                accept=accept,
            )
        )
    if packets:
        return packets
    return []


def parse_foreman_plan(text: str, limit: int) -> tuple[str, list[dict[str, Any]], list[Packet]]:
    blob = text or ""
    obj_match = re.search(r"\{.*\}", blob, flags=re.S)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            mode = str(data.get("mode") or "").strip().lower()
            raw_calls = data.get("calls") if isinstance(data.get("calls"), list) else []
            calls = [row for row in raw_calls if isinstance(row, dict)]
            raw_packets = data.get("packets")
            packets: list[Packet] = []
            if isinstance(raw_packets, list):
                packets = parse_packets(json.dumps(raw_packets), "", limit)
            if mode == "gather":
                return "gather", calls, []
            if mode == "dispatch" or packets:
                return "dispatch", [], packets
    packets = parse_packets(blob, "", limit)
    if packets:
        return "dispatch", [], packets
    return "dispatch", [], []


def _resolve_gather_name(name: str, catalog: dict[str, tuple[str, ...]]) -> str | None:
    """Map a requested tool name onto a name that exists in THIS Cline's catalog."""
    write_lower = {t.lower() for t in WRITE_TOOLS}
    by_lower = {key.lower(): key for key in catalog}
    normalized = name.lower().replace("-", "_")
    direct = by_lower.get(normalized)
    if direct is not None:
        if direct.lower() in write_lower:
            return None
        if direct.lower() in GATHER_TOOLS:
            return direct
        return None
    concept = TOOL_CONCEPT.get(normalized)
    if not concept:
        return None
    for candidate in CONCEPT_CANDIDATES[concept]:
        bound = by_lower.get(candidate)
        if bound is not None and bound.lower() not in write_lower:
            return bound
    return None


def _fit_args(args: dict[str, Any], props: tuple[str, ...]) -> dict[str, Any]:
    remapped: dict[str, Any] = {}
    for key, value in args.items():
        remapped[ARG_ALIASES.get(key, key)] = value
    if not props:
        return remapped
    out = {k: v for k, v in remapped.items() if k in props}
    # Bridge singular/plural and regex/query differences across Cline versions.
    if "paths" in props and "paths" not in out and "path" in remapped:
        out["paths"] = [remapped["path"]]
    if "path" in props and "path" not in out and "paths" in remapped and isinstance(remapped["paths"], list):
        out["path"] = remapped["paths"][0] if remapped["paths"] else "."
    if "commands" in props and "commands" not in out and "command" in remapped:
        out["commands"] = [remapped["command"]]
    if "command" in props and "command" not in out and "commands" in remapped and isinstance(remapped["commands"], list):
        out["command"] = "; ".join(str(c) for c in remapped["commands"])
    if "query" in props and "query" not in out and "regex" in remapped:
        out["query"] = remapped["regex"]
    if "regex" in props and "regex" not in out and "query" in remapped:
        out["regex"] = remapped["query"]
    return out or remapped


def bind_gather_calls(raw: list[dict[str, Any]], catalog: dict[str, tuple[str, ...]], limit: int = 8) -> list[dict[str, Any]]:
    from harness.gateway.qwen_tools import openai_tool_call

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw[: limit * 2]:
        name = str(row.get("name") or row.get("function") or "").strip()
        if not name:
            continue
        bound = _resolve_gather_name(name, catalog)
        if not bound:
            continue
        args = row.get("arguments") if isinstance(row.get("arguments"), dict) else row.get("parameters")
        if not isinstance(args, dict):
            args = {}
        args = _fit_args(args, catalog.get(bound) or ())
        key = f"{bound}:{json.dumps(args, sort_keys=True)}"
        if key in seen:
            continue
        seen.add(key)
        out.append(openai_tool_call(bound, args))
        if len(out) >= limit:
            break
    return out


def _resolve_action_name(name: str, catalog: dict[str, tuple[str, ...]]) -> str | None:
    by_lower = {key.lower(): key for key in catalog}
    normalized = name.lower().replace("-", "_")
    direct = by_lower.get(normalized)
    if direct and direct.lower() in WRITE_TOOLS:
        return direct
    concept = TOOL_CONCEPT.get(normalized)
    if concept != "edit":
        return None
    for candidate in CONCEPT_CANDIDATES[concept]:
        bound = by_lower.get(candidate)
        if bound:
            return bound
    return None


def bind_action_calls(
    raw: list[dict[str, Any]],
    catalog: dict[str, tuple[str, ...]],
    limit: int = ACTION_MAX_CALLS,
) -> list[dict[str, Any]]:
    from harness.gateway.qwen_tools import openai_tool_call

    out: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for row in raw:
        name = str(row.get("name") or row.get("function") or "").strip()
        bound = _resolve_action_name(name, catalog)
        if not bound:
            continue
        args = row.get("arguments") if isinstance(row.get("arguments"), dict) else {}
        fitted = _fit_args(args, catalog.get(bound) or ())
        path = str(fitted.get("path") or fitted.get("file_path") or "")
        if path:
            if path in seen_paths:
                continue
            seen_paths.add(path)
        out.append(openai_tool_call(bound, fitted))
        if len(out) >= limit:
            break
    return out


_STOPWORDS = {
    "this", "that", "from", "with", "have", "need", "find", "file", "files",
    "quote", "code", "report", "working", "workspace", "every", "claim",
    "final", "answer", "rules", "order", "search", "missing", "which", "say",
}

_FRONTEND_JOB_RE = re.compile(
    r"front[\s-]*end|frontend|\bapps/web\b|\bpage\.tsx\b|\bnext\.js\b",
    re.IGNORECASE,
)
_UI_SOURCE_RE = re.compile(
    r"apps/web/[^\s\"']+\.(tsx|jsx|vue|css|scss)|"
    r"\b(?:page|layout|PropertyDashboard)\.(tsx|jsx)\b|"
    r"from ['\"]react['\"]|"
    r"from ['\"]next/",
    re.IGNORECASE,
)
_DENIAL_SENTENCE_RE = re.compile(
    r"(?is)"
    r"(?:you have\s+no\s+file\s+contents[^.!?\n]*[.!?]?\s*)|"
    r"(?:no\s+frontend\s+source\s+has\s+been\s+read[^.!?\n]*[.!?]?\s*)|"
    r"(?:no\s+file\s+contents\s+in\s+this\s+(?:conversation|session)[^.!?\n]*[.!?]?\s*)|"
    r"(?:a\s+real\s+quality\s+verdict\s+is\s+not\s+yet\s+possible[^.!?\n]*[.!?]?\s*)|"
    r"(?:so\s+do\s+not\s+invent\s+specifics[^.!?\n]*[.!?]?\s*)"
)
_DENIAL_INVARIANT_RE = re.compile(
    r"(?i)not yet possible|not yet inspected|no frontend|no file contents"
)


def is_frontend_job(intent: str) -> bool:
    return bool(_FRONTEND_JOB_RE.search(intent or ""))


def thread_has_ui_source(thread: str) -> bool:
    return bool(_UI_SOURCE_RE.search(thread or ""))


def thread_lists_frontend(thread: str) -> bool:
    text = thread or ""
    return "apps/web" in text or "@locdna/web" in text


def evidence_covers_intent(intent: str, thread: str) -> bool:
    """True when THREAD already has the files this INTENT needs to answer.

    A frontend/UX question is not covered by README or package.json. Those
    prove the tree exists; they do not let a blind coder judge the UI.
    """
    if not is_frontend_job(intent):
        return True
    return thread_has_ui_source(thread)


def default_gather_calls(catalog: dict[str, tuple[str, ...]], intent: str) -> list[dict[str, Any]]:
    tokens = list(dict.fromkeys(
        w for w in re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", intent) if w.lower() not in _STOPWORDS
    ))
    regex = "|".join(tokens[:6]) if tokens else "."
    # Semantic search tools reject regex syntax; give them plain words.
    query = " ".join(tokens[:6]) if tokens else "main entry point"
    generic = [
        {"name": "search_files", "arguments": {"path": ".", "regex": regex, "query": query}},
        {"name": "list_files", "arguments": {"path": ".", "recursive": False}},
        {
            "name": "execute_command",
            "arguments": {
                "command": (
                    "ls -la; find . -maxdepth 3 -type f "
                    "! -path './node_modules/*' ! -path './.git/*' "
                    "| head -80"
                )
            },
        },
        {"name": "read_file", "arguments": {"path": "README.md"}},
        {"name": "read_file", "arguments": {"path": "package.json"}},
    ]
    frontend = [
        {"name": "list_files", "arguments": {"path": "apps/web", "recursive": True}},
        {"name": "read_file", "arguments": {"path": "apps/web/package.json"}},
        {"name": "read_file", "arguments": {"path": "apps/web/src/app/page.tsx"}},
        {"name": "read_file", "arguments": {"path": "apps/web/src/app/layout.tsx"}},
        {
            "name": "search_files",
            "arguments": {
                "path": "apps/web",
                "regex": "export default|function ",
                "query": "page layout dashboard component",
                "file_pattern": "*.tsx",
            },
        },
        {
            "name": "execute_command",
            "arguments": {
                "command": (
                    "find apps/web -type f "
                    "\\( -name '*.tsx' -o -name '*.jsx' -o -name '*.css' \\) "
                    "! -path '*/node_modules/*' | head -80"
                )
            },
        },
    ]
    raw = frontend + generic if is_frontend_job(intent) else generic
    try:
        from harness.task.code_index import gather_paths_for_intent

        extra = [
            {"name": "read_file", "arguments": {"path": path}}
            for path in gather_paths_for_intent(intent, limit=6)
        ]
        if extra:
            raw = extra + raw
    except Exception:
        pass
    return bind_gather_calls(raw, catalog)


def merge_tool_catalog(catalog: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    # Never invent tools: only fall back to the classic names when Cline sent nothing.
    if catalog:
        return dict(catalog)
    return dict(DEFAULT_CLINE_TOOLS)


def _clip_context(text: str, limit: int) -> str:
    """Keep both ends of gathered evidence within a prompt budget."""
    text = text.strip()
    if len(text) <= limit:
        return text
    marker = "\n\n[... gathered evidence clipped ...]\n\n"
    head = max(0, (limit - len(marker)) // 2)
    tail = max(0, limit - len(marker) - head)
    return text[:head] + marker + text[-tail:]


def _foreman_input(intent: str, context: str, *, header: str = "", limit: int) -> str:
    """Build a bounded prompt that can never truncate away the current job.

    The old shape was THREAD + INTENT followed by ``user[:limit]``. A few
    successful Cline gather rounds made THREAD exceed the limit and silently
    removed INTENT, so the foreman returned no packets and no coder was leased.
    """
    intent_block = f"INTENT (the current job; never ignore):\n{intent.strip()[:4000]}"
    fixed = "\n\n".join(part.strip() for part in (intent_block, header) if part.strip())
    remaining = max(0, limit - len(fixed) - len("\n\nTHREAD EVIDENCE:\n"))
    evidence = _clip_context(context, remaining) if remaining else ""
    return f"{fixed}\n\nTHREAD EVIDENCE:\n{evidence}"[:limit]


async def plan_orch(
    foreman: ModelConfig,
    intent: str,
    thread: str,
    catalog: dict[str, tuple[str, ...]],
    limit: int,
    gather_round: int,
) -> tuple[str, list[dict[str, Any]], list[Packet]]:
    tool_lines = "\n".join(f"- {name}: {', '.join(props) or '(no params)'}" for name, props in catalog.items()) or "(none)"
    header = (
        f"GATHER_ROUND: {gather_round}/4\n"
        f"CLINE_TOOLS:\n{tool_lines}"
    )
    user = _foreman_input(intent, thread, header=header, limit=16000)
    result = await _chat(
        foreman,
        [
            ChatMessage(role="system", content=FOREMAN_ORCH_SYSTEM),
            ChatMessage(role="user", content=user),
        ],
        max_tokens=1600,
    )
    return parse_foreman_plan(result.text or "", limit)


async def plan_actions(
    foreman: ModelConfig,
    intent: str,
    thread: str,
    accepted_solution: str,
    catalog: dict[str, tuple[str, ...]],
    tools: list[Any],
    working_set: str = "",
) -> list[dict[str, Any]]:
    """Convert an accepted local solution into up to five Cline edit calls."""
    available = {
        name: list(props)
        for name, props in catalog.items()
        if name.lower() in WRITE_TOOLS
    }
    if not any(
        name.lower() in WRITE_TOOLS for name in available
    ):
        return []
    schemas = json.dumps(tools, default=str)[:10_000]
    user = (
        f"INTENT:\n{intent[:3000]}\n\n"
        f"ACCEPTED LOCAL SOLUTION:\n{_clip_context(accepted_solution, 9000)}\n\n"
        f"WORKING SET:\n{_clip_context(working_set, 3000)}\n\n"
        f"WORKSPACE EVIDENCE:\n{_clip_context(thread, 8000)}\n\n"
        f"AVAILABLE TOOL PROPERTIES:\n{json.dumps(available)}\n\n"
        f"AVAILABLE TOOL SCHEMAS:\n{schemas}"
    )
    result = await _chat(
        foreman,
        [
            ChatMessage(role="system", content=ACTION_SYSTEM),
            ChatMessage(role="user", content=user),
        ],
        max_tokens=1200,
    )
    match = re.search(r"\{.*\}", result.text or "", flags=re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict) or str(data.get("mode") or "").lower() != "act":
        return []
    raw = data.get("calls") if isinstance(data.get("calls"), list) else []
    return bind_action_calls(
        [row for row in raw if isinstance(row, dict)],
        catalog,
        limit=ACTION_MAX_CALLS,
    )


async def pick_foreman(cfg: AppConfig) -> tuple[str, ModelConfig] | None:
    """First healthy foreman by workers.yaml priority.

    Health verdicts are cached briefly so every orch turn does not re-probe
    a dead first choice and eat the connect timeout.
    """
    now = time.monotonic()
    timeout = min(cfg.settings.health_timeout_s, _FOREMAN_PROBE_TIMEOUT_S)
    registry = load_registry(cfg.root)
    for worker in registry.pool("foreman"):
        key = worker.model_key or ""
        model = cfg.models.get(key)
        if model is None or not model.enabled:
            continue
        cached = _foreman_cache.get(key)
        if cached and now - cached[0] < _FOREMAN_CACHE_TTL_S:
            ok = cached[1]
        else:
            ok, _detail = await build_provider(model).health(timeout)
            _foreman_cache[key] = (now, ok)
        if ok:
            return key, model
    return None


def coder_models(cfg: AppConfig, worker_keys: list[str] | None = None) -> list[tuple[Worker, ModelConfig]]:
    registry = load_registry(cfg.root)
    if worker_keys:
        wanted = list(worker_keys)
        out: list[tuple[Worker, ModelConfig]] = []
        by_key = {w.model_key: w for w in registry.pool("coder")}
        for key in wanted:
            if key not in cfg.models or not cfg.models[key].enabled:
                raise ValueError(f"unknown or disabled model {key}")
            worker = by_key.get(key)
            if worker is None:
                worker = Worker(
                    id=key,
                    enabled=True,
                    model_key=key,
                    endpoint=cfg.models[key].base_url,
                    capabilities=("coding",),
                    failover_order=None,
                    role="coder",
                )
            out.append((worker, cfg.models[key]))
        return out
    rows = []
    for worker in registry.pool("coder"):
        model = cfg.models.get(worker.model_key or "")
        if model and model.enabled:
            rows.append((worker, model))
    if not rows:
        raise ValueError("no enabled coder boxes in config/workers.yaml")
    return rows


def critic_model(cfg: AppConfig) -> tuple[Worker, ModelConfig] | None:
    models = critic_models(cfg)
    return models[0] if models else None


def critic_models(cfg: AppConfig) -> list[tuple[Worker, ModelConfig]]:
    registry = load_registry(cfg.root)
    workers = registry.pool("critic") or registry.pool("researcher")
    out: list[tuple[Worker, ModelConfig]] = []
    for worker in workers:
        model = cfg.models.get(worker.model_key or "")
        if model and model.enabled:
            out.append((worker, model))
    return out


def is_orch_echo(intent: str) -> bool:
    if "Harness orch" not in intent:
        return False
    return "tools=" in intent or "tester:" in intent or "Running coder pool" in intent


def _recent_runs(cfg: AppConfig) -> str:
    folder = cfg.settings.results_dir
    if not folder.is_dir():
        return ""
    rows: list[str] = []
    for path in sorted(folder.glob("*_dispatch.json"), reverse=True)[:4]:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        shots = data.get("shots") or []
        rates = [s.get("tokens_per_sec") for s in shots if isinstance(s, dict) and s.get("tokens_per_sec")]
        rate = ",".join(f"{float(r):.0f}" for r in rates[:4])
        # Never include prior intents here: the foreman has lifted old intent
        # text into new packets as fabricated context. Numbers only.
        rows.append(f"{data.get('run_id')} shots={len(shots)} tok/s={rate or '-'}")
    if not rows:
        return ""
    return "PRIOR HARNESS METRICS (not the user job; use only if INTENT asks about Cline/harness speed):\n" + "\n".join(rows)


async def _slice(foreman: ModelConfig, intent: str, limit: int, extra: str = "") -> list[Packet]:
    packets: list[Packet] = []
    user = _foreman_input(intent, extra, limit=8000)
    repair = ""
    for _ in range(2):
        result = await _chat(
            foreman,
            [
                ChatMessage(role="system", content=FOREMAN_SYSTEM + f" Emit 0 to {limit} packets."),
                ChatMessage(role="user", content=(user + repair)[:9000]),
            ],
            max_tokens=1600,
        )
        packets = parse_packets(result.text or "", intent, limit)
        if packets:
            return packets
        repair = (
            "\n\nREPAIR: Your previous response produced zero valid packets. "
            "The current INTENT is present above. If THREAD EVIDENCE is non-empty, "
            "emit at least one evidence-grounded written-work packet. Every packet "
            'needs accept.invariants; use "min_chars 120" for an analytical answer.'
        )
    return []


def _is_review_job(intent: str) -> bool:
    return bool(
        re.search(
            r"\b(?:review|audit|code\s*base|codebase|bugs?|architecture|quality)\b",
            intent,
            flags=re.IGNORECASE,
        )
    )


def is_change_job(intent: str) -> bool:
    return bool(
        re.search(
            r"\b(?:fix|implement|add|remove|change|update|edit|refactor|build|repair)\b",
            intent,
            flags=re.IGNORECASE,
        )
    )


def is_prose_invariant(inv: str) -> bool:
    """True when an accept rule is a required phrase, not a machine check."""
    lower = (inv or "").strip().lower()
    if lower.startswith(("min_chars ", "tool ", "no ")):
        return False
    if lower.startswith("text "):
        body = inv.split(None, 1)[1] if " " in inv.strip() else ""
        if "/" in body or re.search(r"\.(py|ts|tsx|js|jsx|go|rs)$", body):
            return False
        return True
    return False


def strip_prose_invariants(packets: list[Packet]) -> list[Packet]:
    """Change jobs cannot be accepted by echoing words like 'not yet'."""
    for packet in packets:
        kept = tuple(inv for inv in packet.accept.invariants if not is_prose_invariant(inv))
        if not kept:
            kept = ("min_chars 40",)
        if kept != packet.accept.invariants:
            packet.accept = AcceptSpec(commands=packet.accept.commands, invariants=kept)
    return packets


def fallback_packets(intent: str, thread: str, limit: int) -> list[Packet]:
    """Lease useful work when the foreman fails to serialize valid packets.

    This is deliberately evidence-bound: it never gives blind workers a repo
    task without Cline tool results. Broad review requests fan out across four
    independent review dimensions so the coder pool is actually used.
    """
    evidence = thread.strip()
    if not evidence:
        return []
    review_job = _is_review_job(intent)
    focuses = (
        "correctness, data integrity, and concrete logic defects",
        "security, unsafe assumptions, and failure handling",
        "tests, build/deploy behavior, observability, and operational risks",
        "architecture, maintainability, coupling, and missing boundaries",
    )
    count = min(len(focuses), limit) if review_job else 1
    blocks = [
        block.strip()
        for block in re.split(r"(?=\ntool\([^)]+\):)", "\n" + evidence)
        if block.strip()
    ]
    if len(blocks) >= count:
        chunks = ["\n".join(blocks[index::count]) for index in range(count)]
    else:
        chunk_size = max(1, (len(evidence) + count - 1) // count)
        chunks = [
            evidence[index * chunk_size : (index + 1) * chunk_size]
            for index in range(count)
        ]
    packets: list[Packet] = []
    for index in range(count):
        chunk = chunks[index]
        if not chunk:
            chunk = evidence
        focus = focuses[index] if review_job else "the user's requested deliverable"
        prompt = (
            f"USER REQUEST:\n{intent[:1200]}\n\n"
            f"YOUR FOCUS:\n{focus}\n\n"
            "WORKSPACE EVIDENCE GATHERED BY CLINE:\n"
            f"{_clip_context(chunk, 13000)}\n\n"
            "Produce finished written work now. Base every repository claim only on "
            "the evidence above; cite paths or symbols shown there. State evidence "
            "limits plainly. Do not invent files, commands, test results, or wider "
            "coverage. A truncation marker or an excerpt ending mid-expression is not "
            "a source-code defect; never report omitted text as broken code. Only report "
            "a defect when visible code proves it. For a review, lead with concrete "
            "findings ordered by severity."
        )
        packets.append(
            Packet(
                id=f"fallback-{index + 1}",
                title=f"Evidence-backed {focus}",
                prompt=prompt[:16000],
                accept=AcceptSpec(invariants=("min_chars 120",)),
            )
        )
    return packets


def hydrate_packets(packets: list[Packet], thread: str) -> list[Packet]:
    """Attach gathered Cline evidence to foreman-authored packet briefs.

    The foreman sometimes emitted a good title and acceptance rule but only
    named a file instead of copying its contents. Blind coder boxes then either
    refused the job or invented code. Select evidence around any path named in
    the packet; otherwise distribute the gathered thread across packets.
    """
    evidence = thread.strip()
    if not evidence or not packets:
        return packets
    path_rx = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+")
    span = max(1, (len(evidence) + len(packets) - 1) // len(packets))
    for index, packet in enumerate(packets):
        if "WORKSPACE EVIDENCE GATHERED BY CLINE:" in packet.prompt:
            continue
        candidates = path_rx.findall(packet.prompt)
        position = next((evidence.find(path) for path in candidates if evidence.find(path) >= 0), -1)
        if position >= 0:
            start = max(0, position - 200)
            selected = evidence[start : start + 13000]
        else:
            selected = evidence[index * span : (index + 1) * span] or evidence
        suffix = (
            "\n\nWORKSPACE EVIDENCE GATHERED BY CLINE "
            "(the only source of repository facts):\n"
        )
        room = 16000 - len(packet.prompt) - len(suffix)
        if room > 200:
            packet.prompt = packet.prompt + suffix + _clip_context(selected, room)
    return packets


def packets_claim_unread(packets: list[Packet]) -> bool:
    for packet in packets:
        text = packet.prompt.lower()
        if "no file contents" in text or "not yet possible" in text:
            return True
        if "no frontend source" in text:
            return True
        if any(_DENIAL_INVARIANT_RE.search(inv) for inv in packet.accept.invariants):
            return True
    return False


def sanitize_packets(packets: list[Packet], thread: str) -> list[Packet]:
    """Stop a listed tree from being briefed as missing or unread.

    Cline access is not the same as packet evidence, but a path that already
    appears in THREAD must not produce a 'no frontend' / 'NO file contents'
    instruction. That packet is what made the coder contradict the workspace.
    """
    if not packets or not thread_lists_frontend(thread):
        return packets
    for packet in packets:
        stripped = _DENIAL_SENTENCE_RE.sub("", packet.prompt).strip()
        if stripped != packet.prompt.strip():
            packet.prompt = (
                "The workspace contains the named frontend tree. Assess from "
                "WORKSPACE EVIDENCE. Do not say the frontend is missing or that "
                "you lack access.\n\n"
                + stripped
            )
        kept = tuple(
            inv for inv in packet.accept.invariants if not _DENIAL_INVARIANT_RE.search(inv)
        )
        if not kept:
            kept = ("min_chars 80",)
        if kept != packet.accept.invariants:
            packet.accept = AcceptSpec(commands=packet.accept.commands, invariants=kept)
    return packets


async def _run_shot(worker: Worker, model: ModelConfig, packet: Packet) -> Shot:
    result = await _chat(
        model,
        [
            ChatMessage(role="system", content=CODER_SYSTEM),
            ChatMessage(role="user", content=packet.prompt),
        ],
        max_tokens=2048,
    )
    names = tool_names(result)
    text = (result.text or "").strip()
    lint = score_invariants(packet, result, names)
    preview = f"ERROR {result.error}" if result.error else text[:2000]
    return Shot(
        packet=packet,
        worker_id=worker.id,
        model_key=model.key,
        result=result,
        tokens_per_sec=tokens_per_sec(result),
        tool_names=names,
        tool_hit=score_tool_hit(packet, result, names),
        qa_pass=lint,
        preview=preview,
        qa_why="python accept" if lint else "python accept miss",
    )


async def _lease(pairs: list[tuple[Worker, ModelConfig]], packets: list[Packet]) -> list[Shot]:
    idle = list(pairs)
    pending = list(packets)
    shots: list[Shot] = []
    inflight: dict[asyncio.Task[Shot], tuple[Worker, ModelConfig]] = {}
    while pending or inflight:
        while idle and pending:
            worker, model = idle.pop(0)
            packet = pending.pop(0)
            task = asyncio.create_task(_run_shot(worker, model, packet))
            inflight[task] = (worker, model)
        if not inflight:
            break
        done, _ = await asyncio.wait(inflight.keys(), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            worker, model = inflight.pop(task)
            shots.append(await task)
            idle.append((worker, model))
    return shots


def _parse_critic(text: str, shots: list[Shot]) -> tuple[str, dict[str, tuple[bool, str]]]:
    blob = text or ""
    match = re.search(r"\{.*\}", blob, flags=re.S)
    data = None
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
    if isinstance(data, dict):
        verdict = str(data.get("verdict") or "insufficient").strip().lower()
        if verdict not in {"proceed", "revise", "reject", "insufficient"}:
            verdict = "insufficient"
        by_id: dict[str, tuple[bool, str]] = {}
        rows = data.get("shots") if isinstance(data.get("shots"), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("id") or "").strip()
            if not sid:
                continue
            why = str(row.get("why") or "")[:160]
            by_id[sid] = (bool(row.get("pass")), why)
        return verdict, by_id
    # Salvage path: local critics sometimes append junk fields or run past the
    # token cap, leaving unclosed JSON. The verdict and per-shot rows come
    # first, so pull them out with regex instead of failing every shot.
    verdict_m = re.search(r'"verdict"\s*:\s*"(proceed|revise|reject|insufficient)"', blob)
    rows_rx = re.findall(
        r'\{\s*"id"\s*:\s*"([^"]+)"\s*,\s*"pass"\s*:\s*(true|false)\s*,\s*"why"\s*:\s*"([^"]*)"',
        blob,
    )
    if not verdict_m or not rows_rx:
        return "insufficient", {}
    by_id = {sid: (flag == "true", why[:160]) for sid, flag, why in rows_rx}
    return verdict_m.group(1), by_id


def _critic_scores_consistent(
    verdict: str,
    by_id: dict[str, tuple[bool, str]],
    shots: list[Shot],
) -> bool:
    expected = {shot.packet.id for shot in shots}
    if set(by_id) != expected or not expected:
        return False
    passed = sum(1 for ok, _ in by_id.values() if ok)
    if verdict == "proceed":
        return passed == len(expected)
    if verdict == "revise":
        return 0 < passed < len(expected)
    if verdict == "reject":
        return passed == 0
    return verdict == "insufficient"


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return text


def _critic_extra(model: ModelConfig, thinking: bool) -> dict:
    extra = dict(model.extra_body)
    if model.key != "asus3_nemotron":
        return extra
    kwargs = dict(extra.get("chat_template_kwargs") or {})
    kwargs["enable_thinking"] = thinking
    extra["chat_template_kwargs"] = kwargs
    extra["enable_thinking"] = thinking
    return extra


async def _run_critic(
    worker: Worker,
    model: ModelConfig,
    intent: str,
    shots: list[Shot],
) -> tuple[str, str, dict[str, tuple[bool, str]]]:
    compact = {
        "intent": intent[:1500],
        "shots": [
            {
                "id": s.packet.id,
                "worker": s.worker_id,
                "accept": {
                    "commands": list(s.packet.accept.commands),
                    "invariants": list(s.packet.accept.invariants),
                },
                "files": list(s.packet.files),
                "tools": s.tool_names,
                "python_ok": s.qa_pass,
                "error": s.result.error,
                # Semantic QA needs the same evidence the blind coder saw.
                # Without packet_prompt, a fluent invented review is
                # indistinguishable from a grounded one.
                "packet_prompt": s.packet.prompt[:6000],
                "answer": (s.result.text or "")[:6000],
            }
            for s in shots
        ],
    }
    messages = [
        ChatMessage(role="system", content=CRITIC_SYSTEM),
        ChatMessage(role="user", content=json.dumps(compact)),
    ]
    result = await _chat(
        model,
        messages,
        _critic_extra(model, thinking=False),
        # One row per shot plus verdict; 220 flat truncated multi-shot grades.
        max_tokens=min(1200, 120 + 80 * len(shots)),
    )
    if result.error:
        return "insufficient", f"ERROR {result.error}", {}
    text = _strip_thinking((result.text or "").strip())
    verdict, by_id = _parse_critic(text, shots)
    if by_id and not _critic_scores_consistent(verdict, by_id, shots):
        return "insufficient", text, {}
    # Do not retry semantic rejections with thinking enabled. python_ok only
    # checks simple invariants; a critic rejection after python_ok=true is the
    # semantic gate doing its job, not a reason to generate thousands of
    # reasoning tokens. The old retry made every real rejection take 2+ minutes.
    return verdict, text, by_id


async def _grade_shots(
    critic_candidates: list[tuple[Worker, ModelConfig]],
    intent: str,
    shots: list[Shot],
    *,
    allow_degraded: bool,
) -> tuple[str, str, str, list[str]]:
    """Apply the first structurally valid critic grade to a set of shots."""
    failures: list[str] = []
    python_ok = {shot.packet.id: shot.qa_pass for shot in shots}
    for c_worker, c_model in critic_candidates:
        verdict, text, by_id = await _run_critic(c_worker, c_model, intent, shots)
        if not by_id:
            failures.append(f"{c_model.key}: invalid grading")
            continue
        for shot in shots:
            passed, why = by_id.get(shot.packet.id, (False, "critic omitted shot"))
            shot.qa_pass = bool(python_ok.get(shot.packet.id) and passed)
            shot.qa_why = why or shot.qa_why
        return verdict, text, c_model.key, failures

    detail = "; ".join(failures) or "all critic backends unavailable"
    if allow_degraded:
        for shot in shots:
            if shot.qa_pass:
                shot.qa_why = "python-only; all critics unavailable or invalid"
        return (
            "degraded",
            f"{detail}; serving only machine-accepted answers as explicitly unverified",
            "",
            failures,
        )
    for shot in shots:
        shot.qa_pass = False
        shot.qa_why = "frontier answer could not be independently verified"
    return "insufficient", detail, "", failures


def _revision_packets(shots: list[Shot], round_number: int) -> list[Packet]:
    packets: list[Packet] = []
    for shot in shots:
        previous = _clip_context(shot.result.text or f"ERROR {shot.result.error}", 3000)
        base = _clip_context(shot.packet.prompt, 11000)
        prompt = (
            f"{base}\n\nLOCAL REVISION ROUND {round_number}:\n"
            f"The previous answer failed QA: {shot.qa_why or 'acceptance failure'}.\n"
            f"PREVIOUS ANSWER:\n{previous}\n\n"
            "Return a corrected finished answer. Fix the stated failure without inventing "
            "repository facts or claiming tests that are not in the supplied evidence."
        )
        packets.append(
            Packet(
                id=shot.packet.id,
                title=f"{shot.packet.title} (local repair {round_number})",
                prompt=prompt[:16000],
                expect_tool=shot.packet.expect_tool,
                files=shot.packet.files,
                accept=shot.packet.accept,
            )
        )
    return packets


def _aggregate_verdict(shots: list[Shot], degraded: bool = False) -> str:
    if degraded:
        return "degraded"
    passed = sum(1 for shot in shots if shot.qa_pass)
    if passed == len(shots) and shots:
        return "proceed"
    if passed:
        return "revise"
    return "reject"


def _shot_evidence(shots: list[Shot]) -> list[dict[str, Any]]:
    return [
        {
            "packet": shot.packet.id,
            "worker": shot.worker_id,
            "qa": shot.qa_pass,
            "why": shot.qa_why,
            "answer": shot.result.text or f"ERROR {shot.result.error}",
        }
        for shot in shots
    ]


def _write_error_report(cfg: AppConfig, report: DispatchReport, svc: TaskService, task) -> DispatchReport:
    dest = cfg.settings.results_dir / f"{report.run_id}.json"
    dest.write_text(
        json.dumps({"run_id": report.run_id, "intent": report.intent, "error": report.slice_error}, indent=2)
    )
    report.json_path = str(dest)
    svc.set_stage(task.task_id, "failed", report.slice_error)
    svc.record(
        AttemptRecord(task_id=task.task_id, attempt=0, worker="foreman", result="failed"),
        close=True,
    )
    return report


async def run_dispatch(
    cfg: AppConfig,
    intent: str,
    *,
    worker_keys: list[str] | None = None,
    use_senior: bool = False,
    thread: str = "",
    packets: list[Packet] | None = None,
) -> DispatchReport:
    intent = intent.strip()
    if not intent:
        raise ValueError("intent required")
    pairs = coder_models(cfg, worker_keys)
    picked = await pick_foreman(cfg)
    if picked is None:
        raise ValueError("no foreman reachable; configured foreman pool is down or disabled")
    foreman_key, foreman = picked
    senior = cfg.models.get(SENIOR_KEY) if use_senior else None
    if use_senior and (senior is None or not senior.enabled):
        raise ValueError("senior frontier is not enabled")
    critic_candidates: list[tuple[Worker, ModelConfig]] = []

    run_id = make_run_id("dispatch")
    report = DispatchReport(run_id=run_id, intent=intent)
    cfg.settings.results_dir.mkdir(parents=True, exist_ok=True)
    store = Store(cfg.settings.db_path)
    svc = TaskService(store)
    task = svc.start(intent)
    report.task_id = task.task_id
    svc.set_stage(task.task_id, "planning")

    if is_orch_echo(intent):
        report.slice_error = (
            "this user message is a previous harness-orch dump, not a new job. "
            "Ask the actual question."
        )
        return _write_error_report(cfg, report, svc, task)

    healthy_pairs: list[tuple[Worker, ModelConfig]] = []
    for worker, model in pairs:
        ok, detail = await build_provider(model).health(cfg.settings.health_timeout_s)
        report.health[worker.id] = "ok" if ok else detail
        if ok:
            healthy_pairs.append((worker, model))
    pairs = healthy_pairs
    report.foreman_key = foreman_key
    report.health[f"foreman:{foreman_key}"] = "ok"
    for c_worker, c_model in critic_models(cfg):
        ok, detail = await build_provider(c_model).health(cfg.settings.health_timeout_s)
        report.health[f"critic:{c_worker.id}"] = "ok" if ok else detail
        if ok:
            critic_candidates.append((c_worker, c_model))

    if not pairs:
        report.slice_error = "no coder backend passed its live health probe; no lease attempted"
        return _write_error_report(cfg, report, svc, task)

    limit = min(8, max(len(pairs), len(pairs) * 2))
    extra_parts = [p for p in (thread.strip(), _recent_runs(cfg)) if p]
    if thread.strip() and _is_review_job(intent):
        # Review packets must be neutral. A generative foreman has repeatedly
        # seeded its packet prompts with supposed bugs, causing blind coders to
        # echo those claims and making fabricated findings look like consensus.
        report.packets = fallback_packets(intent, thread, limit)
    elif packets:
        report.packets = list(packets)
    else:
        packets = await _slice(foreman, intent, limit, extra="\n\n".join(extra_parts))
        report.packets = packets
    if not report.packets and thread.strip():
        report.packets = fallback_packets(intent, thread, limit)
    if report.packets and thread.strip():
        report.packets = hydrate_packets(report.packets, thread)
        report.packets = sanitize_packets(report.packets, thread)
    if report.packets and is_change_job(intent):
        report.packets = strip_prose_invariants(report.packets)
    if not report.packets:
        report.slice_error = (
            "no usable workspace evidence reached dispatch and the foreman emitted "
            "no accept-ready packet; no coder lease. This is a harness gather/protocol "
            "failure, not proof that the wrong folder is open."
        )
        return _write_error_report(cfg, report, svc, task)

    svc.set_stage(task.task_id, "local_solve")
    initial_shots = await _lease(pairs, report.packets)
    report.local_rounds = 1
    all_attempts = list(initial_shots)
    current = {shot.packet.id: shot for shot in initial_shots}

    verdict, critic_text, critic_key, _critic_failures = await _grade_shots(
        critic_candidates,
        intent,
        initial_shots,
        allow_degraded=True,
    )
    report.critic_verdict = verdict
    report.critic_text = critic_text
    report.critic_key = critic_key
    svc.add_evidence(
        Evidence(
            task_id=task.task_id,
            kind="critic_grade",
            payload={"round": 1, "verdict": verdict, "critic_key": critic_key},
        )
    )

    for revision in range(1, cfg.settings.local_revision_attempts + 1):
        failed = [shot for shot in current.values() if not shot.qa_pass]
        if not failed:
            break
        svc.set_stage(task.task_id, "local_repair")
        repair_packets = _revision_packets(failed, revision)
        # Rotate the pool so a rejected answer is likely retried by a different
        # local model while preserving bounded, deterministic work.
        rotated_pairs = pairs[revision % len(pairs) :] + pairs[: revision % len(pairs)]
        repaired = await _lease(rotated_pairs, repair_packets)
        report.local_rounds += 1
        all_attempts.extend(repaired)
        r_verdict, r_text, r_key, _ = await _grade_shots(
            critic_candidates,
            intent,
            repaired,
            allow_degraded=True,
        )
        for shot in repaired:
            current[shot.packet.id] = shot
        report.critic_text = r_text
        report.critic_key = r_key or report.critic_key
        report.critic_verdict = _aggregate_verdict(
            list(current.values()),
            degraded=r_verdict == "degraded",
        )
        svc.add_evidence(
            Evidence(
                task_id=task.task_id,
                kind="critic_grade",
                payload={
                    "round": revision + 1,
                    "verdict": r_verdict,
                    "critic_key": r_key,
                },
            )
        )

    report.shots = [
        current[packet.id] for packet in report.packets if packet.id in current
    ]
    report.attempt_history = all_attempts
    failed_final = [shot for shot in report.shots if not shot.qa_pass]

    if (
        failed_final
        and cfg.settings.auto_frontier_rescue
        and svc.claim_frontier(task.task_id, cfg.settings.max_frontier_calls_per_task)
    ):
        try:
            rescue_packet = build_auto_rescue_packet(
                intent,
                thread,
                _shot_evidence(
                    [
                        shot
                        for shot in all_attempts
                        if shot.packet.id in {failed.packet.id for failed in failed_final}
                    ]
                ),
                report.critic_text,
                max_chars=cfg.settings.frontier_max_input_chars,
            )
            outcome = await run_rescue_text(
                cfg,
                rescue_packet,
                case_id=task.task_id,
                store=store,
            )
            report.frontier_run_id = outcome.run_id
            report.frontier_model_key = outcome.model_key
            report.frontier_cost = outcome.estimated_cost
            report.frontier_text = outcome.text.strip()
            if outcome.error or not report.frontier_text:
                report.frontier_why = outcome.error or "empty frontier answer"
            else:
                frontier_packet = Packet(
                    id="frontier-rescue",
                    title="Frontier rescue",
                    prompt=rescue_packet,
                    accept=AcceptSpec(invariants=("min_chars 20",)),
                )
                frontier_result = ChatResult(
                    provider="frontier",
                    model=outcome.model_key,
                    text=report.frontier_text,
                    input_tokens=outcome.input_tokens,
                    output_tokens=outcome.output_tokens,
                    latency_ms=outcome.latency_ms,
                    error=outcome.error,
                )
                frontier_shot = Shot(
                    packet=frontier_packet,
                    worker_id="frontier",
                    model_key=outcome.model_key,
                    result=frontier_result,
                    tokens_per_sec=tokens_per_sec(frontier_result),
                    tool_names=[],
                    tool_hit=True,
                    qa_pass=True,
                    preview=report.frontier_text[:2000],
                )
                f_verdict, _f_text, f_key, _ = await _grade_shots(
                    critic_candidates,
                    intent,
                    [frontier_shot],
                    allow_degraded=False,
                )
                report.frontier_verified = bool(
                    frontier_shot.qa_pass and f_verdict == "proceed"
                )
                report.frontier_why = frontier_shot.qa_why
                svc.add_evidence(
                    Evidence(
                        task_id=task.task_id,
                        kind="frontier_rescue",
                        payload={
                            "run_id": outcome.run_id,
                            "model_key": outcome.model_key,
                            "critic_key": f_key,
                            "verified": report.frontier_verified,
                            "cost": outcome.estimated_cost,
                        },
                    )
                )
        except PacketError as exc:
            report.frontier_why = str(exc)
        except Exception as exc:
            report.frontier_why = f"{type(exc).__name__}: {exc}"

    for i, shot in enumerate(all_attempts):
        last = (
            i == len(all_attempts) - 1
            and not use_senior
            and not report.frontier_run_id
        )
        svc.record(
            AttemptRecord(
                task_id=task.task_id,
                attempt=i,
                worker=shot.worker_id,
                result="success" if shot.qa_pass and not shot.result.error else "failed",
                tool_calls=len(shot.tool_names),
                input_tokens=shot.result.input_tokens,
                output_tokens=shot.result.output_tokens,
                ttft_ms=shot.result.latency_ms,
                tokens_per_sec=shot.tokens_per_sec,
            ),
            close=last,
        )

    if senior:
        compact = [
            {
                "packet": s.packet.id,
                "worker": s.worker_id,
                "qa": s.qa_pass,
                "hit": s.tool_hit,
                "ms": round(s.result.latency_ms, 1),
                "tools": s.tool_names,
                "error": s.result.error,
            }
            for s in report.shots
        ]
        result = await _chat(
            senior,
            [
                ChatMessage(
                    role="system",
                    content="Senior reviewer. Six sentences max. Who passed QA, who missed, one next packet.",
                ),
                ChatMessage(role="user", content=json.dumps({"intent": intent, "shots": compact})),
            ],
            max_tokens=400,
        )
        report.senior_text = f"ERROR {result.error}" if result.error else (result.text or "").strip()
        svc.record(
            AttemptRecord(
                task_id=task.task_id,
                attempt=0,
                worker=senior.key,
                result="success" if not report.senior_text.startswith("ERROR") else "failed",
            ),
            close=True,
        )

    local_complete = bool(report.shots) and all(shot.qa_pass for shot in report.shots)
    solved = local_complete or report.frontier_verified
    svc.finish(
        task.task_id,
        solved,
        "local" if local_complete else ("frontier_verified" if solved else "unresolved"),
    )

    dest = cfg.settings.results_dir / f"{run_id}.json"
    dest.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "task_id": report.task_id,
                "intent": intent,
                "health": report.health,
                "local_rounds": report.local_rounds,
                "senior": report.senior_text,
                "critic_verdict": report.critic_verdict,
                "critic_key": report.critic_key,
                "critic": report.critic_text,
                "slice_error": report.slice_error,
                "frontier": {
                    "run_id": report.frontier_run_id,
                    "model_key": report.frontier_model_key,
                    "verified": report.frontier_verified,
                    "why": report.frontier_why,
                    "estimated_cost": report.frontier_cost,
                    "text": report.frontier_text[:2000],
                },
                "packets": [
                    {
                        "id": p.id,
                        "title": p.title,
                        "expect_tool": p.expect_tool,
                        "files": list(p.files),
                        "accept": {
                            "commands": list(p.accept.commands),
                            "invariants": list(p.accept.invariants),
                        },
                        "prompt": p.prompt,
                    }
                    for p in report.packets
                ],
                "shots": [
                    {
                        "packet": s.packet.id,
                        "worker": s.worker_id,
                        "model_key": s.model_key,
                        "latency_ms": s.result.latency_ms,
                        "input_tokens": s.result.input_tokens,
                        "output_tokens": s.result.output_tokens,
                        "tokens_per_sec": s.tokens_per_sec,
                        "tool_names": s.tool_names,
                        "tool_hit": s.tool_hit,
                        "qa_pass": s.qa_pass,
                        "qa_why": s.qa_why,
                        "text": (s.result.text or "")[:800],
                        "error": s.result.error,
                    }
                    for s in report.shots
                ],
                "attempt_history": [
                    {
                        "packet": s.packet.id,
                        "worker": s.worker_id,
                        "model_key": s.model_key,
                        "qa_pass": s.qa_pass,
                        "qa_why": s.qa_why,
                        "latency_ms": s.result.latency_ms,
                        "input_tokens": s.result.input_tokens,
                        "output_tokens": s.result.output_tokens,
                        "text": (s.result.text or "")[:800],
                        "error": s.result.error,
                    }
                    for s in report.attempt_history
                ],
            },
            indent=2,
        )
    )
    report.json_path = str(dest)
    return report
