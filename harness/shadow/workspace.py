from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

from harness.shadow.models import RepositorySnapshot, ShadowPolicy, canonical_json
from harness.shadow.objects import ShadowObjectStore
from harness.shadow.policy import canonical_relative_path, path_allowed
from harness.training.security import assert_no_secrets


_DIFF_PATH = re.compile(r"(?m)^diff --git a/(.+?) b/(.+?)$")


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": tempfile.gettempdir(),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "CI": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        },
    )


def _require(result: subprocess.CompletedProcess[str], operation: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:1000]
        raise RuntimeError(f"{operation} failed: {detail}")
    return result.stdout


def _safe_destination(root: Path, relative: str) -> Path:
    target = (root / relative).resolve(strict=False)
    if not target.is_relative_to(root.resolve()):
        raise ValueError("workspace path escapes the isolated checkout")
    current = root.resolve()
    for part in Path(relative).parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("workspace path traverses a symlink")
    return target


def materialize_snapshot(
    snapshot: RepositorySnapshot,
    policy: ShadowPolicy,
    *,
    work_root: Path,
    object_store: ShadowObjectStore,
    workspace_id: str,
) -> Path:
    root = work_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", workspace_id):
        raise ValueError("workspace ID is invalid")
    destination = root / workspace_id
    if destination.exists() or destination.is_symlink():
        raise ValueError("isolated shadow workspace already exists")
    temporary = Path(tempfile.mkdtemp(prefix=".materialize-", dir=root))
    workspace = temporary / "workspace"
    try:
        _require(
            _run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--shared",
                    "--no-checkout",
                    str(snapshot.repository_root),
                    str(workspace),
                ],
                timeout=120,
            ),
            "shadow clone",
        )
        _require(
            _run(
                ["git", "checkout", "--quiet", "--detach", snapshot.revision],
                cwd=workspace,
            ),
            "shadow checkout",
        )
        if snapshot.dirty_patch:
            _require(
                _run(
                    ["git", "apply", "--whitespace=nowarn", "-"],
                    cwd=workspace,
                    input_text=snapshot.dirty_patch,
                ),
                "shadow dirty-patch replay",
            )
        for item in snapshot.untracked_files:
            if not path_allowed(policy, item.path):
                raise ValueError("snapshot contains a path denied by current policy")
            content = object_store.read_text(item.sha256, item.object_path)
            target = _safe_destination(workspace, item.path)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        _require(
            _run(["git", "add", "-A", "--"], cwd=workspace),
            "shadow parent-state staging",
        )
        os.replace(workspace, destination)
        temporary.rmdir()
        return destination
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def transition_to_snapshot(
    workspace_root: Path,
    policy: ShadowPolicy,
    *,
    parent: RepositorySnapshot,
    final: RepositorySnapshot,
    object_store: ShadowObjectStore,
) -> None:
    if (
        parent.repository_id != final.repository_id
        or parent.revision != final.revision
    ):
        raise ValueError("snapshot transition crosses repository lineage")
    workspace = ShadowWorkspace(workspace_root, policy)
    if parent.dirty_patch:
        _require(
            _run(
                ["git", "apply", "--reverse", "--whitespace=nowarn", "-"],
                cwd=workspace.root,
                input_text=parent.dirty_patch,
            ),
            "shadow parent-patch reversal",
        )
    for item in parent.untracked_files:
        _relative, target = workspace._path(item.path)
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise ValueError("parent untracked path changed type during replay")
            target.unlink()
    if final.dirty_patch:
        workspace.apply_patch(final.dirty_patch)
    for item in final.untracked_files:
        if not path_allowed(policy, item.path):
            raise PermissionError("final snapshot contains a denied path")
        content = object_store.read_text(item.sha256, item.object_path)
        _relative, target = workspace._path(item.path)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise ValueError("final untracked path changed type during replay")
            target.unlink()
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())


class ShadowWorkspace:
    def __init__(self, root: Path, policy: ShadowPolicy):
        self.root = root.resolve()
        self.policy = policy
        if (
            self.root.is_symlink()
            or not self.root.is_dir()
            or self.root.stat().st_uid != os.geteuid()
        ):
            raise ValueError("shadow workspace must be an owned directory")

    def _path(self, value: str) -> tuple[str, Path]:
        relative = canonical_relative_path(self.root, value)
        if not path_allowed(self.policy, relative):
            raise PermissionError(f"path is outside the shadow policy: {relative}")
        path = _safe_destination(self.root, relative)
        return relative, path

    def list_files(self, pattern: str = "") -> str:
        output = _require(
            _run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=self.root),
            "list files",
        )
        rows = [
            path
            for path in output.splitlines()
            if path_allowed(self.policy, path)
            and (not pattern or fnmatch_path(path, pattern))
        ]
        return "\n".join(rows[:2000])

    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        max_lines: int = 400,
    ) -> str:
        relative, source = self._path(path)
        if (
            source.is_symlink()
            or not source.is_file()
            or not stat.S_ISREG(source.stat().st_mode)
        ):
            raise ValueError(f"not a regular file: {relative}")
        if source.stat().st_size > 1_000_000:
            raise ValueError(f"file exceeds the read limit: {relative}")
        text = source.read_text(encoding="utf-8")
        assert_no_secrets(text, field=f"shadow read {relative}")
        rows = text.splitlines()
        start = max(0, int(start_line) - 1)
        limit = max(1, min(int(max_lines), 1000))
        selected = rows[start : start + limit]
        return "\n".join(
            f"{number}:{line}"
            for number, line in enumerate(selected, start=start + 1)
        )

    def search_files(self, pattern: str, path: str = ".") -> str:
        if not pattern or len(pattern) > 300:
            raise ValueError("search pattern must contain 1-300 characters")
        expression = re.compile(pattern)
        prefix = "" if path in {"", "."} else canonical_relative_path(self.root, path)
        matches = []
        for relative in self.list_files().splitlines():
            if prefix and relative != prefix and not relative.startswith(prefix + "/"):
                continue
            source = self.root / relative
            if source.is_symlink() or not source.is_file() or source.stat().st_size > 1_000_000:
                continue
            try:
                text = source.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            assert_no_secrets(text, field=f"shadow search {relative}")
            for line_number, line in enumerate(text.splitlines(), 1):
                if expression.search(line):
                    matches.append(f"{relative}:{line_number}:{line}")
                    if len(matches) >= 300:
                        return "\n".join(matches)
        return "\n".join(matches)

    def apply_patch(self, patch: str) -> str:
        if not patch.lstrip().startswith("diff --git "):
            raise ValueError("patch must be a unified Git diff")
        if len(patch.encode()) > self.policy.max_dirty_patch_bytes:
            raise ValueError("patch exceeds the shadow policy limit")
        if any(
            marker in patch
            for marker in (
                "GIT binary patch",
                "new file mode 120000",
                "Subproject commit ",
            )
        ):
            raise ValueError("binary, symlink, and submodule patches are forbidden")
        rows = _DIFF_PATH.findall(patch)
        if not rows:
            raise ValueError("patch contains no Git file headers")
        for left, right in rows:
            for value in (left, right):
                relative = canonical_relative_path(self.root, value)
                if not path_allowed(self.policy, relative):
                    raise PermissionError(f"patch path is denied: {relative}")
        assert_no_secrets(patch, field="Qwen shadow patch")
        _require(
            _run(["git", "apply", "--check", "-"], cwd=self.root, input_text=patch),
            "shadow patch check",
        )
        _require(
            _run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                cwd=self.root,
                input_text=patch,
            ),
            "shadow patch apply",
        )
        return "PATCH_APPLIED"

    def diff(self) -> str:
        tracked = _require(
            _run(
                ["git", "diff", "--binary", "--no-ext-diff", "--"],
                cwd=self.root,
            ),
            "shadow diff",
        )
        additions = []
        untracked = _require(
            _run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=self.root,
            ),
            "shadow untracked diff inventory",
        )
        for relative in untracked.splitlines():
            if not path_allowed(self.policy, relative):
                continue
            source = self.root / relative
            if (
                source.is_symlink()
                or not source.is_file()
                or source.stat().st_size > self.policy.max_untracked_file_bytes
            ):
                raise ValueError(f"new shadow file is not admissible: {relative}")
            result = _run(
                ["git", "diff", "--no-index", "--binary", "--", "/dev/null", relative],
                cwd=self.root,
            )
            if result.returncode not in {0, 1}:
                _require(result, f"shadow new-file diff {relative}")
            additions.append(result.stdout)
        value = tracked + "".join(additions)
        assert_no_secrets(value, field="shadow workspace diff")
        return value

    def state_sha256(self) -> str:
        diff = self.diff()
        untracked = []
        for relative in _require(
            _run(["git", "ls-files", "--others", "--exclude-standard"], cwd=self.root),
            "shadow untracked inventory",
        ).splitlines():
            if not path_allowed(self.policy, relative):
                continue
            source = self.root / relative
            if source.is_file() and not source.is_symlink():
                untracked.append(
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                )
        return hashlib.sha256(
            canonical_json(
                {
                    "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                    "untracked": untracked,
                }
            )
        ).hexdigest()


def fnmatch_path(path: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatch(path, pattern)
