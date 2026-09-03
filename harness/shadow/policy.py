from __future__ import annotations

import fnmatch
import json
import os
import stat
from pathlib import Path
from typing import Any

from harness.shadow.models import ModelRuntime, ShadowPolicy
from harness.training.models import SourceKind, is_excluded_learning_source
from harness.training.security import redact_text


POLICY_NAME = ".harness-shadow.json"
DEFAULT_RUNTIME_PATH = Path("~/.harness/shadow/runtime.json")
_SENSITIVE_KEYS = {
    "apikey",
    "accesstoken",
    "refreshtoken",
    "authtoken",
    "authorization",
    "clientsecret",
    "cookie",
    "credential",
    "githubtoken",
    "hftoken",
    "openaitoken",
    "anthropictoken",
    "password",
    "passwd",
    "privatekey",
    "secret",
    "token",
    "xapikey",
}


def _load_regular_json(path: Path, label: str) -> dict[str, Any]:
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_file():
        raise ValueError(f"{label} must be a regular file")
    info = expanded.stat()
    if info.st_uid != os.geteuid() or not stat.S_ISREG(info.st_mode):
        raise PermissionError(f"{label} must be owned by the current user")
    value = json.loads(expanded.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def load_policy(repository_root: Path) -> ShadowPolicy | None:
    path = repository_root.resolve() / POLICY_NAME
    if not path.exists():
        return None
    return ShadowPolicy.model_validate(_load_regular_json(path, "shadow policy"))


def load_runtime(path: Path | None = None) -> ModelRuntime:
    target = path or Path(
        os.environ.get("HARNESS_SHADOW_RUNTIME") or DEFAULT_RUNTIME_PATH
    )
    return ModelRuntime.model_validate(_load_regular_json(target, "shadow runtime"))


def canonical_relative_path(repository_root: Path, candidate: Path | str) -> str:
    root = repository_root.resolve()
    path = Path(candidate)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError("path escapes the opted-in repository")
    relative = resolved.relative_to(root).as_posix()
    if not relative or relative == "." or ".." in Path(relative).parts:
        raise ValueError("path must name a repository child")
    return relative


def path_allowed(policy: ShadowPolicy, relative_path: str) -> bool:
    path = relative_path.replace("\\", "/").strip("/")
    if not path or ".." in Path(path).parts:
        return False
    allowed = any(
        pattern == "."
        or fnmatch.fnmatch(path, pattern)
        or fnmatch.fnmatch(f"/{path}", pattern)
        for pattern in policy.allowed_paths
    )
    excluded = any(
        fnmatch.fnmatch(path, pattern)
        or fnmatch.fnmatch(f"/{path}", pattern)
        or fnmatch.fnmatch(Path(path).name, pattern)
        for pattern in policy.excluded_paths
    )
    return allowed and not excluded


def source_allowed(policy: ShadowPolicy, *material: str) -> bool:
    if (
        policy.owner == "self"
        and policy.data_use == "shadow_learning"
        and policy.authorization_scope == "owned_repository_cursor_shadow"
    ):
        return True
    return not is_excluded_learning_source(
        SourceKind.OTHER,
        f"repository://{policy.repository_id}",
        "\n".join(material),
    )


def sanitize_payload(value: Any, *, max_string_chars: int = 40_000) -> Any:
    def clean(item: Any, depth: int) -> Any:
        if depth > 12:
            return "[TRUNCATED_DEPTH]"
        if isinstance(item, str):
            text = redact_text(item)
            if len(text) > max_string_chars:
                return text[:max_string_chars] + "\n[TRUNCATED]"
            return text
        if isinstance(item, dict):
            output: dict[str, Any] = {}
            for raw_key, child in list(item.items())[:512]:
                key = str(raw_key)[:256]
                normalized = "".join(character for character in key.casefold() if character.isalnum())
                if normalized in _SENSITIVE_KEYS:
                    output[key] = "redacted"
                else:
                    output[key] = clean(child, depth + 1)
            return output
        if isinstance(item, (list, tuple)):
            return [clean(child, depth + 1) for child in list(item)[:512]]
        if item is None or isinstance(item, (bool, int, float)):
            return item
        return redact_text(str(item))[:max_string_chars]

    return clean(value, 0)
