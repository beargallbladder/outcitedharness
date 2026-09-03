from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.shadow.models import (
    HookRecord,
    ShadowPolicy,
    ShadowTask,
    canonical_json,
    safe_identifier,
)
from harness.shadow.objects import ShadowObjectStore
from harness.shadow.policy import (
    canonical_relative_path,
    load_policy,
    path_allowed,
    sanitize_payload,
    source_allowed,
)
from harness.shadow.repository import capture_repository_snapshot, discover_repository
from harness.shadow.runner import shadow_actionable
from harness.shadow.spool import ShadowSpool
from harness.training.security import assert_no_secrets, redact_text


PROMPT_EVENTS = frozenset({"beforeSubmitPrompt"})
PATH_EVENTS = frozenset({"beforeReadFile", "afterFileEdit"})


def default_spool_root() -> Path:
    return Path(
        os.environ.get("HARNESS_SHADOW_SPOOL") or "~/.harness/shadow"
    ).expanduser()


def _first(payload: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = payload.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                candidate = _first(item, ("text", "content", "prompt"))
                if candidate is not None:
                    parts.append(_text(candidate))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        candidate = _first(value, ("text", "content", "prompt", "message"))
        return _text(candidate) if candidate is not None else ""
    return ""


def extract_prompt(payload: dict[str, Any]) -> str:
    value = _first(
        payload,
        (
            "prompt",
            "user_prompt",
            "userPrompt",
            "message",
            "content",
            "text",
        ),
    )
    return _text(value).strip()


def extract_session_id(payload: dict[str, Any]) -> str:
    value = _first(
        payload,
        (
            "conversation_id",
            "conversationId",
            "session_id",
            "sessionId",
            "composer_id",
            "composerId",
        ),
    )
    return safe_identifier(value, fallback="session-unknown")


def extract_generation_id(payload: dict[str, Any]) -> str | None:
    value = _first(
        payload,
        (
            "generation_id",
            "generationId",
            "request_id",
            "requestId",
        ),
    )
    return safe_identifier(value, fallback="") or None


def correlation_id(repository_id: str, session_id: str) -> str:
    digest = hashlib.sha256(f"{repository_id}\0{session_id}".encode()).hexdigest()
    return f"cursor:{digest[:40]}"


def _event_record(
    *,
    event_type: str,
    task_id: str | None,
    correlation: str,
    payload: dict[str, Any],
    created_at: datetime,
) -> HookRecord:
    payload_sha256 = hashlib.sha256(canonical_json(payload)).hexdigest()
    event_id = f"hook-{uuid.uuid4().hex}"
    return HookRecord(
        event_id=event_id,
        task_id=task_id,
        correlation_id=correlation,
        event_type=safe_identifier(event_type, fallback="unknown"),
        payload=payload,
        payload_sha256=payload_sha256,
        created_at=created_at,
    )


def _attachment_receipts(
    repository_root: Path,
    policy: ShadowPolicy,
    attachments: Any,
) -> list[dict[str, Any]]:
    receipts = []
    for row in attachments if isinstance(attachments, list) else []:
        if not isinstance(row, dict):
            continue
        receipt: dict[str, Any] = {"type": str(row.get("type") or "unknown")[:64]}
        candidate = row.get("file_path")
        try:
            relative = canonical_relative_path(repository_root, str(candidate))
            source = repository_root / relative
            if (
                not path_allowed(policy, relative)
                or source.is_symlink()
                or not source.is_file()
                or source.stat().st_size > 10_000_000
            ):
                raise ValueError("attachment is outside capture policy")
            data = source.read_bytes()
            receipt.update(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "byte_size": len(data),
                }
            )
        except (OSError, ValueError):
            receipt["omitted"] = True
        receipts.append(receipt)
    return receipts


def _minimize_payload(
    event_type: str,
    payload: dict[str, Any],
    *,
    repository_root: Path,
    policy: ShadowPolicy,
) -> dict[str, Any]:
    reduced = dict(payload)
    reduced.pop("transcript_path", None)
    attachments = reduced.pop("attachments", None)
    if attachments is not None:
        reduced["attachment_receipts"] = _attachment_receipts(
            repository_root,
            policy,
            attachments,
        )
    if event_type == "beforeReadFile" and "content" in reduced:
        content = str(reduced.pop("content") or "")
        reduced["content_sha256"] = hashlib.sha256(content.encode()).hexdigest()
        reduced["content_bytes"] = len(content.encode())
    if event_type == "afterFileEdit" and "edits" in reduced:
        edits = reduced.pop("edits")
        reduced["edits_sha256"] = hashlib.sha256(canonical_json(edits)).hexdigest()
        reduced["edit_count"] = len(edits) if isinstance(edits, list) else 0
    return reduced


def capture_hook_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    repository_root: Path | None = None,
    spool_root: Path | None = None,
) -> str | None:
    requested_root = repository_root or Path.cwd()
    root = discover_repository(requested_root)
    policy = load_policy(root)
    if policy is None:
        return None

    session_id = extract_session_id(payload)
    correlation = correlation_id(policy.repository_id, session_id)
    if event_type in PATH_EVENTS:
        candidate = _first(payload, ("file_path", "filePath", "path"))
        if candidate is None:
            return None
        try:
            relative = canonical_relative_path(root, str(candidate))
        except ValueError:
            return None
        if not path_allowed(policy, relative):
            return None

    minimized = _minimize_payload(
        event_type,
        payload,
        repository_root=root,
        policy=policy,
    )
    sanitized = sanitize_payload(minimized)
    if not isinstance(sanitized, dict):
        sanitized = {"payload": sanitized}
    generation_id = extract_generation_id(payload)
    created_at = datetime.now(timezone.utc)
    spool_path = (spool_root or default_spool_root()).expanduser()
    spool = ShadowSpool(spool_path)

    if event_type in PROMPT_EVENTS:
        prompt = extract_prompt(payload)
        if not prompt:
            raise ValueError("Cursor prompt event contains no user prompt")
        assert_no_secrets(prompt, field="Cursor shadow prompt")
        prompt = redact_text(prompt)
        if not source_allowed(policy, prompt):
            return None
        if not shadow_actionable(prompt):
            sanitized["shadow_disposition"] = "ignored_non_actionable"
            spool.append_hook(
                _event_record(
                    event_type=event_type,
                    task_id=None,
                    correlation=correlation,
                    payload=sanitized,
                    created_at=created_at,
                )
            )
            return None
        snapshot = capture_repository_snapshot(
            root,
            policy,
            ShadowObjectStore(spool_path),
        )
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        identity = canonical_json(
            {
                "repository_id": policy.repository_id,
                "session_id": session_id,
                "generation_id": generation_id,
                "prompt_sha256": prompt_sha256,
                "state_sha256": snapshot.state_sha256,
            }
        )
        task_id = f"shadow-{hashlib.sha256(identity).hexdigest()[:40]}"
        task = ShadowTask(
            task_id=task_id,
            correlation_id=correlation,
            session_id=session_id,
            generation_id=generation_id,
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            policy=policy,
            snapshot=snapshot,
            created_at=created_at,
        )
        spool.enqueue(task)
        sanitized["shadow_task_id"] = task_id
        sanitized["repository_state_sha256"] = snapshot.state_sha256
        spool.append_hook(
            _event_record(
                event_type=event_type,
                task_id=task_id,
                correlation=correlation,
                payload=sanitized,
                created_at=created_at,
            )
        )
        return task_id

    task_id = spool.task_for_correlation(
        repository_id=policy.repository_id,
        correlation_id=correlation,
    )
    if event_type == "stop" and task_id is not None:
        final_snapshot = capture_repository_snapshot(
            root,
            policy,
            ShadowObjectStore(spool_path),
        )
        sanitized["repository_state"] = final_snapshot.model_dump(
            mode="json",
            exclude_none=True,
        )
    spool.append_hook(
        _event_record(
            event_type=event_type,
            task_id=task_id,
            correlation=correlation,
            payload=sanitized,
            created_at=created_at,
        )
    )
    return task_id


def read_hook_input(max_bytes: int = 4_000_000) -> dict[str, Any]:
    data = os.read(0, max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Cursor hook payload exceeds the input limit")
    value = json.loads(data or b"{}")
    if not isinstance(value, dict):
        raise ValueError("Cursor hook input must be a JSON object")
    return value
