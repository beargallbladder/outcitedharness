from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from harness.shadow.models import (
    RepositorySnapshot,
    ShadowPolicy,
    UntrackedFile,
    canonical_json,
)
from harness.shadow.objects import ShadowObjectStore
from harness.shadow.policy import canonical_relative_path, path_allowed
from harness.training.security import assert_no_secrets, find_secrets


class SnapshotError(RuntimeError):
    pass


def _git(root: Path, *arguments: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        input=input_text,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(Path.home()),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        },
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:500]
        raise SnapshotError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout


def discover_repository(path: Path) -> Path:
    requested = path.expanduser().resolve()
    output = _git(requested, "rev-parse", "--show-toplevel").strip()
    root = Path(output).resolve()
    if (
        not root.is_dir()
        or root.is_symlink()
        or root.stat().st_uid != os.geteuid()
    ):
        raise SnapshotError("repository root must be an owned regular directory")
    return root


def _nul_paths(value: str) -> list[str]:
    return [row for row in value.split("\0") if row]


def capture_repository_snapshot(
    repository_root: Path,
    policy: ShadowPolicy,
    object_store: ShadowObjectStore,
) -> RepositorySnapshot:
    root = discover_repository(repository_root)
    revision = _git(root, "rev-parse", "HEAD").strip()
    if len(revision) not in {40, 64}:
        raise SnapshotError("repository HEAD is not an immutable Git revision")

    changed = _nul_paths(
        _git(
            root,
            "diff",
            "--name-only",
            "-z",
            "--no-ext-diff",
            "HEAD",
            "--",
        )
    )
    allowed_patches = []
    omitted = 0
    for value in changed:
        relative = canonical_relative_path(root, value)
        if path_allowed(policy, relative):
            patch = _git(
                root,
                "diff",
                "--binary",
                "--no-ext-diff",
                "HEAD",
                "--",
                relative,
            )
            if find_secrets(patch):
                omitted += 1
            else:
                allowed_patches.append(patch)
        else:
            omitted += 1
    dirty_patch = "".join(allowed_patches)
    if "GIT binary patch" in dirty_patch or "new file mode 120000" in dirty_patch:
        raise SnapshotError("binary and symlink dirty changes are not shadowable")
    if len(dirty_patch.encode()) > policy.max_dirty_patch_bytes:
        raise SnapshotError("dirty patch exceeds the repository shadow policy")
    assert_no_secrets(dirty_patch, field="shadow dirty patch")

    untracked: list[UntrackedFile] = []
    untracked_total = 0
    for value in _nul_paths(
        _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    ):
        relative = canonical_relative_path(root, value)
        if not path_allowed(policy, relative):
            omitted += 1
            continue
        source = root / relative
        info = source.lstat()
        if not stat.S_ISREG(info.st_mode) or source.is_symlink():
            raise SnapshotError(f"untracked path is not a regular file: {relative}")
        if info.st_size > policy.max_untracked_file_bytes:
            raise SnapshotError(f"untracked file exceeds policy limit: {relative}")
        untracked_total += info.st_size
        if untracked_total > policy.max_untracked_total_bytes:
            raise SnapshotError("untracked files exceed the aggregate policy limit")
        try:
            content = source.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SnapshotError(
                f"untracked file is not UTF-8 text: {relative}"
            ) from exc
        if find_secrets(content):
            omitted += 1
            continue
        digest, size, object_path = object_store.put_text(content)
        untracked.append(
            UntrackedFile(
                path=relative,
                sha256=digest,
                byte_size=size,
                object_path=object_path,
            )
        )

    captured_at = datetime.now(timezone.utc)
    dirty_digest = hashlib.sha256(dirty_patch.encode()).hexdigest()
    state = {
        "repository_id": policy.repository_id,
        "revision": revision,
        "dirty_patch_sha256": dirty_digest,
        "untracked_files": [
            row.model_dump(mode="json", exclude_none=True)
            for row in sorted(untracked, key=lambda item: item.path)
        ],
        "omitted_path_count": omitted,
    }
    return RepositorySnapshot(
        repository_id=policy.repository_id,
        repository_root=root,
        revision=revision,
        dirty_patch=dirty_patch,
        dirty_patch_sha256=dirty_digest,
        untracked_files=tuple(sorted(untracked, key=lambda item: item.path)),
        omitted_path_count=omitted,
        state_sha256=hashlib.sha256(canonical_json(state)).hexdigest(),
        captured_at=captured_at,
    )
