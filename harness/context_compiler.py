"""Deterministic, budgeted coder context compiled from persistent loop state."""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from harness.orch_loop import LoopState, WorkingFile

CHARS_PER_TOKEN = 4
DEFAULT_CONTEXT_TOKENS = 6_000
MIN_CONTEXT_TOKENS = 1_024
_SLICE_MARKER = "\n[... exact content sliced by ContextCompiler ...]\n"


@dataclass(frozen=True)
class CompiledContext:
    text: str
    budget_tokens: int
    estimated_tokens: int
    included_paths: tuple[str, ...]
    omitted_paths: tuple[str, ...]
    digest: str


def _clip_exact(text: str, limit: int, *, tail: bool = False) -> str:
    value = text or ""
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    if limit <= len(_SLICE_MARKER) + 16:
        return value[-limit:] if tail else value[:limit]
    if tail:
        return _SLICE_MARKER + value[-(limit - len(_SLICE_MARKER)) :]
    head = (limit - len(_SLICE_MARKER)) * 2 // 3
    return value[:head] + _SLICE_MARKER + value[-(limit - len(_SLICE_MARKER) - head) :]


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    name = Path(lowered).name
    return (
        "/tests/" in f"/{lowered}"
        or name.startswith("test_")
        or name.endswith(("_test.py", ".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx"))
    )


def _file_block(path: str, snapshot: WorkingFile, content_limit: int) -> str:
    content = _clip_exact(snapshot.content, max(0, content_limit))
    return f"FILE: {path}\nCONTENT_HASH: {snapshot.content_hash}\n{content}"


def _render_files(
    state: LoopState,
    paths: Iterable[str],
    limit: int,
    *,
    current_changed: bool = False,
) -> tuple[str, list[str]]:
    ordered = list(dict.fromkeys(paths))
    if not ordered or limit <= 0:
        return "", []
    headers: list[tuple[str, WorkingFile | None, str]] = []
    for path in ordered:
        snapshot = state.working_set.files_read.get(path)
        if current_changed and path in state.working_set.stale_files:
            headers.append((path, None, f"FILE: {path}\nCURRENT CONTENT: refresh pending"))
        elif snapshot is not None:
            headers.append(
                (
                    path,
                    snapshot,
                    f"FILE: {path}\nCONTENT_HASH: {snapshot.content_hash}\n",
                )
            )
    if not headers:
        return "", []
    separator_chars = 2 * max(0, len(headers) - 1)
    fixed = sum(len(header) for _path, _snapshot, header in headers) + separator_chars
    if fixed >= limit:
        body = "\n".join(f"FILE: {path}" for path, _snapshot, _header in headers)
        return _clip_exact(body, limit), [path for path, _snapshot, _header in headers]
    remaining = limit - fixed
    blocks: list[str] = []
    included: list[str] = []
    snapshots_left = sum(snapshot is not None for _path, snapshot, _header in headers)
    for path, snapshot, header in headers:
        included.append(path)
        if snapshot is None:
            blocks.append(header)
            continue
        share = remaining // max(1, snapshots_left)
        take = min(len(snapshot.content), share)
        blocks.append(_file_block(path, snapshot, take))
        remaining -= take
        snapshots_left -= 1
    return "\n\n".join(blocks), included


def _test_relevance(state: LoopState, path: str) -> int:
    evidence = "\n".join(
        [
            state.working_set.objective or state.intent,
            state.last_cmd or "",
            state.stderr_tail,
            state.stdout_tail,
        ]
    ).lower()
    score = 0
    if path.lower() in evidence:
        score += 100
    if Path(path).name.lower() in evidence:
        score += 50
    stem = Path(path).stem.lower().removeprefix("test_").removesuffix("_test")
    for changed in state.working_set.files_changed:
        if stem and stem == Path(changed).stem.lower():
            score += 25
    return score


def _other_relevance(state: LoopState, path: str) -> tuple[int, str]:
    objective_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", state.intent)
    }
    path_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", path)
    }
    score = len(objective_tokens & path_tokens)
    failure = f"{state.stderr_tail}\n{state.stdout_tail}".lower()
    if path.lower() in failure:
        score += 100
    if Path(path).name.lower() in failure:
        score += 50
    snapshot = state.working_set.files_read.get(path)
    if snapshot:
        score += 5 * sum(
            bool(re.search(rf"\b{re.escape(symbol)}\b", snapshot.content))
            for symbol in state.expansion_symbols
        )
    if path in state.semantic_expansion_paths:
        score += 2
    return (-score, path)


def _attempts_text(state: LoopState) -> str:
    attempts = list(state.attempt_summaries)
    if attempts and (state.last_exit is not None or state.timed_out):
        attempts = attempts[:-1]
    rows: list[str] = []
    for row in attempts[-4:]:
        changed = ", ".join(str(path) for path in row.get("changed_files") or []) or "(none)"
        rows.append(
            f"{row.get('iteration', '?')}. command={row.get('command') or '(none)'} "
            f"exit={row.get('exit_code') if row.get('exit_code') is not None else 'unknown'} "
            f"changed={changed} failure={row.get('failure') or '(empty)'}"
        )
    return "\n".join(rows)


def _failure_text(state: LoopState, limit: int) -> str:
    metadata = (
        f"command: {state.last_cmd or '(none)'}\n"
        f"exit: {state.last_exit if state.last_exit is not None else 'unknown'}\n"
        f"timed_out: {state.timed_out}\n"
        "exact output tail:\n"
    )
    if limit <= len(metadata):
        return _clip_exact(metadata, limit)
    output = (
        state.stderr_tail
        or state.stdout_tail
        or "(none; no verification failure yet)"
    )
    return metadata + _clip_exact(output, limit - len(metadata), tail=True)


def _allocate_required(
    lengths: list[int],
    available: int,
) -> list[int]:
    weights = (0.08, 0.07, 0.18, 0.39, 0.18)
    allocations = [
        min(length, int(max(0, available) * weight))
        for length, weight in zip(lengths, weights)
    ]
    remaining = max(0, available - sum(allocations))
    for index in (3, 2, 4, 0, 1):
        extra = min(lengths[index] - allocations[index], remaining)
        allocations[index] += max(0, extra)
        remaining -= max(0, extra)
    return allocations


def _attr(value: object) -> str:
    return html.escape(str(value), quote=True)


def _tag(name: str, body: str, **attrs: object) -> str:
    suffix = "".join(f' {key}="{_attr(value)}"' for key, value in attrs.items())
    return f"<{name}{suffix}>\n{body}\n</{name}>"


def _current_file_blocks(
    state: LoopState,
    paths: list[str],
    limit: int,
) -> tuple[str, list[str]]:
    if not paths or limit <= 0:
        return "", []
    fixed_blocks: list[tuple[str, str, WorkingFile | None]] = []
    for path in paths:
        snapshot = state.working_set.files_read.get(path)
        if path in state.working_set.stale_files:
            opening = (
                f'<CURRENT_FILE path="{_attr(path)}" status="refresh_pending">'
            )
            fixed_blocks.append((path, opening, None))
        elif snapshot is None:
            opening = (
                f'<CURRENT_FILE path="{_attr(path)}" status="missing_after_mutation">'
            )
            fixed_blocks.append((path, opening, None))
        else:
            opening = (
                f'<CURRENT_FILE path="{_attr(path)}" '
                f'hash="{_attr(snapshot.content_hash)}">'
            )
            fixed_blocks.append((path, opening, snapshot))
    fixed = sum(
        len(opening) + len("\n\n</CURRENT_FILE>")
        for _path, opening, _snapshot in fixed_blocks
    ) + 2 * max(0, len(fixed_blocks) - 1)
    if fixed >= limit:
        path_only = "\n".join(
            f'<CURRENT_FILE path="{_attr(path)}" status="omitted_by_budget" />'
            for path, _opening, _snapshot in fixed_blocks
        )
        return _clip_exact(path_only, limit), [path for path, _opening, _snapshot in fixed_blocks]
    remaining = limit - fixed
    with_content = sum(snapshot is not None for _path, _opening, snapshot in fixed_blocks)
    blocks: list[str] = []
    for _path, opening, snapshot in fixed_blocks:
        if snapshot is None:
            blocks.append(f"{opening}\n\n</CURRENT_FILE>")
            continue
        share = remaining // max(1, with_content)
        content = _clip_exact(snapshot.content, min(len(snapshot.content), share))
        blocks.append(f"{opening}\n{content}\n</CURRENT_FILE>")
        remaining -= len(content)
        with_content -= 1
    return "\n\n".join(blocks), [path for path, _opening, _snapshot in fixed_blocks]


def _evidence_block(tag: str, path: str, snapshot: WorkingFile) -> str:
    return _tag(
        tag,
        snapshot.content,
        path=path,
        hash=snapshot.content_hash,
    )


def _packet_wrapper(
    body: str,
    *,
    phase: str,
    budget_tokens: int,
    budget_chars: int,
) -> tuple[str, int]:
    closing = "</CODER_CONTEXT>"
    template = (
        f'<CODER_CONTEXT phase="{_attr(phase)}" '
        f'budget_tokens="{budget_tokens}" budget_chars="{budget_chars}" '
        'used_chars="{used:010d}" estimated_tokens="{estimated:010d}">\n'
    )
    probe = template.format(used=0, estimated=0) + body + "\n" + closing
    used = len(probe)
    estimated = (used + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN
    text = template.format(used=used, estimated=estimated) + body + "\n" + closing
    return text, estimated


def compile_coder_context(
    state: LoopState,
    *,
    phase: str = "repair",
    budget_tokens: int = DEFAULT_CONTEXT_TOKENS,
    instruction: str = "",
    legacy_thread: str = "",
) -> CompiledContext:
    """Compile authoritative current state into the exact bounded coder packet."""
    tokens = max(MIN_CONTEXT_TOKENS, int(budget_tokens))
    budget = tokens * CHARS_PER_TOKEN
    changed = sorted(dict.fromkeys(state.working_set.files_changed))
    changed_set = set(changed)
    all_paths = sorted(state.working_set.files_read)

    semantic = set(state.semantic_expansion_paths)
    expanded = sorted(
        path
        for path in state.expansion_paths
        if path in state.working_set.files_read
        and path not in changed_set
        and path not in semantic
    )
    expanded_set = set(expanded)
    tests = sorted(
        (
            path
            for path in all_paths
            if path not in changed_set
            and path not in expanded_set
            and _is_test_path(path)
            and _test_relevance(state, path) > 0
        ),
        key=lambda path: (-_test_relevance(state, path), path),
    )
    reserved = changed_set | set(tests) | set(expanded)
    others = sorted(
        (
            path
            for path in all_paths
            if path not in reserved
            and (phase == "apply" or _other_relevance(state, path)[0] < 0)
        ),
        key=lambda path: _other_relevance(state, path),
    )

    objective = state.working_set.objective or state.intent or "(none)"
    commands = state.working_set.acceptance_commands
    acceptance = "\n".join(
        f'<COMMAND required="true">{command}</COMMAND>' for command in commands
    ) or '<COMMAND required="true" status="missing" />'
    acceptance += (
        "\n<INVARIANT>Every command exits 0 against the same current file state.</INVARIANT>"
    )
    failure_output = (
        state.stderr_tail
        or state.stdout_tail
        or "(none; no verification failure yet)"
    )
    diff = state.working_set.current_diff or "(none)"
    diff_hash = state.active_diff_hash or state.last_diff_hash or ""

    wrapper_probe, _ = _packet_wrapper(
        "",
        phase=phase,
        budget_tokens=tokens,
        budget_chars=budget,
    )
    body_budget = max(0, budget - len(wrapper_probe))
    instruction_section = _tag("INSTRUCTION", instruction) if instruction else ""

    def required_sections(
        objective_limit: int,
        acceptance_limit: int,
        failure_limit: int,
        changed_limit: int,
        diff_limit: int,
    ) -> tuple[list[str], list[str]]:
        failure_body = _clip_exact(failure_output, failure_limit, tail=True)
        current_files, current_included = _current_file_blocks(
            state,
            changed,
            changed_limit,
        )
        sections = [
            _tag("OBJECTIVE", _clip_exact(objective, objective_limit)),
            _tag("ACCEPTANCE", _clip_exact(acceptance, acceptance_limit)),
            _tag(
                "LATEST_FAILURE",
                failure_body,
                command=state.last_cmd or "",
                exit_code=(
                    state.last_exit
                    if state.last_exit is not None
                    else "unknown"
                ),
                timed_out=str(state.timed_out).lower(),
            ),
            current_files
            or '<CURRENT_FILE status="none" />',
            _tag("CURRENT_DIFF", _clip_exact(diff, diff_limit), hash=diff_hash),
        ]
        return sections, current_included

    full_current, _ = _current_file_blocks(state, changed, 10**9)
    full_required = [
        _tag("OBJECTIVE", objective),
        _tag("ACCEPTANCE", acceptance),
        _tag(
            "LATEST_FAILURE",
            failure_output,
            command=state.last_cmd or "",
            exit_code=state.last_exit if state.last_exit is not None else "unknown",
            timed_out=str(state.timed_out).lower(),
        ),
        full_current or '<CURRENT_FILE status="none" />',
        _tag("CURRENT_DIFF", diff, hash=diff_hash),
    ]
    reserved_instruction = len(instruction_section) + (2 if instruction_section else 0)
    if len("\n\n".join(full_required)) + reserved_instruction <= body_budget:
        sections = full_required
        changed_included = changed
    else:
        compact_limits = (
            min(len(objective), 1_200),
            min(len(acceptance), 1_200),
            min(len(failure_output), 4_000),
            10**9,
            min(len(diff), 4_000),
        )
        compact_without_current, _ = required_sections(
            compact_limits[0],
            compact_limits[1],
            compact_limits[2],
            0,
            compact_limits[4],
        )
        non_current = len("\n\n".join(compact_without_current)) + reserved_instruction
        changed_room = max(128, body_budget - non_current)
        sections, changed_included = required_sections(
            compact_limits[0],
            compact_limits[1],
            compact_limits[2],
            changed_room,
            compact_limits[4],
        )
        while len("\n\n".join(sections)) + reserved_instruction > body_budget:
            overflow = len("\n\n".join(sections)) + reserved_instruction - body_budget
            changed_room = max(64, changed_room - overflow - 8)
            sections, changed_included = required_sections(
                min(compact_limits[0], 400),
                min(compact_limits[1], 400),
                min(compact_limits[2], 900),
                changed_room,
                min(compact_limits[4], 600),
            )
            if changed_room == 64:
                break
    included = list(changed_included)

    def append_full_block(block: str, path: str | None = None) -> bool:
        candidate = "\n\n".join([*sections, block])
        if len(candidate) + reserved_instruction > body_budget:
            return False
        sections.append(block)
        if path and path not in included:
            included.append(path)
        return True

    # Optional evidence is all-or-nothing and never displaces current changed source.
    for path in expanded:
        snapshot = state.working_set.files_read[path]
        append_full_block(_evidence_block("EXPANDED_EVIDENCE", path, snapshot), path)
    for path in tests:
        snapshot = state.working_set.files_read[path]
        append_full_block(_evidence_block("RELEVANT_TEST", path, snapshot), path)
    for path in others:
        snapshot = state.working_set.files_read[path]
        append_full_block(_evidence_block("WORKING_FILE", path, snapshot), path)

    attempts = _attempts_text(state)
    if attempts:
        append_full_block(_tag("PREVIOUS_ATTEMPTS", attempts))
    if (
        legacy_thread.strip()
        and not state.working_set.files_read
        and not state.working_set.current_diff
    ):
        append_full_block(
            _tag("LEGACY_RECOVERY", _clip_exact(legacy_thread, 1_000))
        )
    if instruction_section:
        sections.append(instruction_section)

    body = "\n\n".join(sections)
    if len(body) > body_budget:
        body = _clip_exact(body, body_budget)
    text, estimated = _packet_wrapper(
        body,
        phase=phase,
        budget_tokens=tokens,
        budget_chars=budget,
    )
    included_set = set(included)
    omitted = tuple(path for path in all_paths if path not in included_set)
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    return CompiledContext(
        text=text,
        budget_tokens=tokens,
        estimated_tokens=estimated,
        included_paths=tuple(included),
        omitted_paths=omitted,
        digest=digest,
    )


def compile_context(
    state: LoopState,
    *,
    budget_tokens: int = DEFAULT_CONTEXT_TOKENS,
    instruction: str = "",
) -> CompiledContext:
    """Backward-compatible entry point for callers compiled before provenance tags."""
    return compile_coder_context(
        state,
        phase="repair" if state.last_exit is not None else "apply",
        budget_tokens=budget_tokens,
        instruction=instruction,
    )
