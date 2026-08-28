"""Task-scoped file checkpoints and conflict-safe explicit rollback."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

CHECKPOINT_VERSION = 1
DEFAULT_MAX_FILE_BYTES = 1_000_000


class CheckpointError(RuntimeError):
    pass


class RollbackConflict(CheckpointError):
    def __init__(self, paths: list[str]):
        self.paths = sorted(dict.fromkeys(paths))
        super().__init__("workspace changed after task checkpoint: " + ", ".join(self.paths))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _snapshot_state_hash(files: dict[str, FileSnapshot]) -> str:
    digest = hashlib.sha256()
    for path, snapshot in sorted(files.items()):
        digest.update(path.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(b"1" if snapshot.exists else b"0")
        digest.update(b"\0")
        digest.update(snapshot.content_hash.encode())
        digest.update(b"\0")
        digest.update(str(snapshot.mode if snapshot.exists else 0).encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _safe_relpath(raw: str) -> str:
    value = str(raw or "").strip().replace("\\", "/")
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CheckpointError(f"unsafe checkpoint path: {raw!r}")
    return path.as_posix()


def _run_git(root: Path, argv: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *argv],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (proc.stdout or "").strip()


def _atomic_write(path: Path, data: bytes, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class FileSnapshot:
    path: str
    exists: bool
    content_hash: str = ""
    size: int = 0
    mode: int = 0
    tracked: bool = False
    dirty: bool = False
    untracked: bool = False
    status: str = ""
    missing_parents: tuple[str, ...] = ()
    captured_before_iteration: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FileSnapshot:
        return cls(
            path=str(raw.get("path") or ""),
            exists=bool(raw.get("exists")),
            content_hash=str(raw.get("content_hash") or ""),
            size=int(raw.get("size") or 0),
            mode=int(raw.get("mode") or 0),
            tracked=bool(raw.get("tracked")),
            dirty=bool(raw.get("dirty")),
            untracked=bool(raw.get("untracked")),
            status=str(raw.get("status") or ""),
            missing_parents=tuple(str(p) for p in (raw.get("missing_parents") or [])),
            captured_before_iteration=int(raw.get("captured_before_iteration") or 0),
        )


@dataclass
class CheckpointManifest:
    task_id: str
    run_id: str
    kind: str
    number: int
    repo_root: str
    intent: str
    head: str = ""
    active_diff_hash: str = ""
    created_at: str = field(default_factory=_now)
    files: dict[str, FileSnapshot] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["version"] = CHECKPOINT_VERSION
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CheckpointManifest:
        files = raw.get("files") if isinstance(raw.get("files"), dict) else {}
        return cls(
            task_id=str(raw.get("task_id") or ""),
            run_id=str(raw.get("run_id") or ""),
            kind=str(raw.get("kind") or ""),
            number=int(raw.get("number") or 0),
            repo_root=str(raw.get("repo_root") or ""),
            intent=str(raw.get("intent") or ""),
            head=str(raw.get("head") or ""),
            active_diff_hash=str(raw.get("active_diff_hash") or ""),
            created_at=str(raw.get("created_at") or ""),
            files={
                str(path): FileSnapshot.from_dict(value)
                for path, value in files.items()
                if isinstance(value, dict)
            },
        )


@dataclass(frozen=True)
class RollbackPreview:
    task_id: str
    run_id: str
    repo_root: str
    restore: tuple[str, ...]
    remove: tuple[str, ...]
    conflicts: tuple[str, ...]


@dataclass(frozen=True)
class RollbackResult:
    task_id: str
    run_id: str
    restored: tuple[str, ...]
    removed: tuple[str, ...]
    removed_dirs: tuple[str, ...]
    audit_path: str


class CheckpointStore:
    def __init__(self, root: Path, *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES):
        self.root = Path(root).expanduser().resolve()
        self.max_file_bytes = max(1, int(max_file_bytes))

    def _run_dir(self, task_id: str, run_id: str) -> Path:
        if not task_id or not run_id or "/" in task_id or "/" in run_id:
            raise CheckpointError("invalid checkpoint task/run id")
        return self.root / task_id / run_id

    def _manifest_path(self, task_id: str, run_id: str, number: int) -> Path:
        run_dir = self._run_dir(task_id, run_id)
        return (
            run_dir / "baseline.json"
            if number == 0
            else run_dir / "checkpoints" / f"{number}.json"
        )

    def manifest_path(self, task_id: str, run_id: str, number: int) -> Path:
        return self._manifest_path(task_id, run_id, number)

    def _blob_path(self, task_id: str, run_id: str, digest: str) -> Path:
        return self._run_dir(task_id, run_id) / "blobs" / digest

    @contextmanager
    def lock(self, task_id: str, run_id: str) -> Iterator[None]:
        run_dir = self._run_dir(task_id, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        lock_path = run_dir / ".lock"
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _workspace_path(self, root: Path, rel: str) -> Path:
        safe = _safe_relpath(rel)
        candidate = root.joinpath(*PurePosixPath(safe).parts)
        current = root
        for part in PurePosixPath(safe).parts:
            current = current / part
            if current.is_symlink():
                raise CheckpointError(f"symlink checkpoint path refused: {safe}")
            if not current.exists():
                break
        try:
            candidate.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise CheckpointError(f"checkpoint path escapes workspace: {safe}") from exc
        return candidate

    def _git_metadata(self, root: Path, rel: str) -> tuple[bool, str]:
        tracked = bool(_run_git(root, ["ls-files", "--error-unmatch", "--", rel]))
        status_text = _run_git(
            root,
            ["status", "--porcelain=v1", "--untracked-files=all", "--", rel],
        )
        status_value = status_text.splitlines()[0][:2].strip() if status_text else ""
        return tracked, status_value

    def _missing_parents(self, root: Path, path: Path) -> tuple[str, ...]:
        missing: list[str] = []
        parent = path.parent
        while parent != root and root in parent.parents:
            if parent.exists():
                break
            missing.append(parent.relative_to(root).as_posix())
            parent = parent.parent
        return tuple(missing)

    def _capture_file(
        self,
        task_id: str,
        run_id: str,
        root: Path,
        rel: str,
        *,
        before_iteration: int,
        git_metadata: bool,
    ) -> FileSnapshot:
        safe = _safe_relpath(rel)
        path = self._workspace_path(root, safe)
        tracked, status_value = self._git_metadata(root, safe) if git_metadata else (False, "")
        missing_parents = self._missing_parents(root, path)
        try:
            if not path.exists():
                return FileSnapshot(
                    path=safe,
                    exists=False,
                    tracked=tracked,
                    dirty=bool(status_value and status_value != "??"),
                    untracked=status_value == "??",
                    status=status_value,
                    missing_parents=missing_parents,
                    captured_before_iteration=before_iteration,
                )
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise CheckpointError(f"checkpoint supports regular files only: {safe}")
            if info.st_size > self.max_file_bytes:
                raise CheckpointError(
                    f"checkpoint file exceeds {self.max_file_bytes} bytes: {safe}"
                )
            data = path.read_bytes()
        except CheckpointError:
            raise
        except OSError as exc:
            raise CheckpointError(f"cannot snapshot checkpoint path {safe}: {exc}") from exc
        if len(data) > self.max_file_bytes:
            raise CheckpointError(
                f"checkpoint file exceeds {self.max_file_bytes} bytes: {safe}"
            )
        digest = _digest(data)
        blob = self._blob_path(task_id, run_id, digest)
        if not blob.exists():
            _atomic_write(blob, data, 0o600)
        return FileSnapshot(
            path=safe,
            exists=True,
            content_hash=digest,
            size=len(data),
            mode=stat.S_IMODE(info.st_mode),
            tracked=tracked,
            dirty=bool(status_value and status_value != "??"),
            untracked=status_value == "??",
            status=status_value,
            missing_parents=missing_parents,
            captured_before_iteration=before_iteration,
        )

    def _write_manifest(self, manifest: CheckpointManifest) -> Path:
        path = self._manifest_path(
            manifest.task_id,
            manifest.run_id,
            manifest.number,
        )
        payload = json.dumps(
            manifest.to_dict(),
            indent=2,
            sort_keys=True,
        ).encode()
        _atomic_write(path, payload, 0o600)
        return path

    def load_manifest(
        self,
        task_id: str,
        run_id: str,
        number: int,
    ) -> CheckpointManifest | None:
        path = self._manifest_path(task_id, run_id, number)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"invalid checkpoint manifest: {path}") from exc
        if not isinstance(raw, dict) or int(raw.get("version") or 0) != CHECKPOINT_VERSION:
            raise CheckpointError(f"unsupported checkpoint manifest: {path}")
        return CheckpointManifest.from_dict(raw)

    def capture_baseline(
        self,
        *,
        task_id: str,
        run_id: str,
        repo_root: Path,
        intent: str,
        paths: list[str],
        before_iteration: int,
    ) -> CheckpointManifest:
        root = Path(repo_root).expanduser().resolve()
        if not root.is_dir():
            raise CheckpointError(f"workspace does not exist: {root}")
        with self.lock(task_id, run_id):
            manifest = self.load_manifest(task_id, run_id, 0) or CheckpointManifest(
                task_id=task_id,
                run_id=run_id,
                kind="baseline",
                number=0,
                repo_root=str(root),
                intent=intent,
                head=_run_git(root, ["rev-parse", "HEAD"]),
            )
            if Path(manifest.repo_root).resolve() != root:
                raise CheckpointError("checkpoint workspace changed")
            for raw in paths:
                rel = _safe_relpath(raw)
                if rel in manifest.files:
                    continue
                manifest.files[rel] = self._capture_file(
                    task_id,
                    run_id,
                    root,
                    rel,
                    before_iteration=before_iteration,
                    git_metadata=True,
                )
            self._write_manifest(manifest)
            return manifest

    def record_checkpoint(
        self,
        *,
        task_id: str,
        run_id: str,
        number: int,
    ) -> CheckpointManifest:
        if number <= 0:
            raise CheckpointError("checkpoint number must be positive")
        with self.lock(task_id, run_id):
            baseline = self.load_manifest(task_id, run_id, 0)
            if baseline is None:
                raise CheckpointError("task baseline is missing")
            root = Path(baseline.repo_root).resolve()
            files = {
                path: self._capture_file(
                    task_id,
                    run_id,
                    root,
                    path,
                    before_iteration=snapshot.captured_before_iteration,
                    git_metadata=False,
                )
                for path, snapshot in sorted(baseline.files.items())
            }
            manifest = CheckpointManifest(
                task_id=task_id,
                run_id=run_id,
                kind="checkpoint",
                number=number,
                repo_root=str(root),
                intent=baseline.intent,
                head=baseline.head,
                active_diff_hash=_snapshot_state_hash(files),
                files=files,
            )
            self._write_manifest(manifest)
            return manifest

    def latest_checkpoint(
        self,
        task_id: str,
        run_id: str,
    ) -> CheckpointManifest | None:
        directory = self._run_dir(task_id, run_id) / "checkpoints"
        if not directory.exists():
            return None
        numbers = [
            int(path.stem)
            for path in directory.glob("*.json")
            if path.stem.isdigit()
        ]
        return self.load_manifest(task_id, run_id, max(numbers)) if numbers else None

    def _blob(self, task_id: str, run_id: str, snapshot: FileSnapshot) -> bytes:
        if not snapshot.exists or not snapshot.content_hash:
            return b""
        path = self._blob_path(task_id, run_id, snapshot.content_hash)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise CheckpointError(f"checkpoint blob is missing: {snapshot.path}") from exc
        if _digest(data) != snapshot.content_hash:
            raise CheckpointError(f"checkpoint blob hash mismatch: {snapshot.path}")
        return data

    @staticmethod
    def _same_file(left: FileSnapshot, right: FileSnapshot) -> bool:
        return (
            left.exists == right.exists
            and left.content_hash == right.content_hash
            and (not left.exists or left.mode == right.mode)
        )

    def rollback_preview(self, task_id: str, run_id: str) -> RollbackPreview:
        if (self._run_dir(task_id, run_id) / "rollback.json").exists():
            raise CheckpointError("task checkpoint was already rolled back")
        baseline = self.load_manifest(task_id, run_id, 0)
        latest = self.latest_checkpoint(task_id, run_id)
        if baseline is None or latest is None:
            raise CheckpointError("baseline or latest checkpoint is missing")
        root = Path(baseline.repo_root).resolve()
        conflicts: list[str] = []
        restore: list[str] = []
        remove: list[str] = []
        for path, expected in sorted(latest.files.items()):
            current = self._capture_file(
                task_id,
                run_id,
                root,
                path,
                before_iteration=expected.captured_before_iteration,
                git_metadata=False,
            )
            if not self._same_file(current, expected):
                conflicts.append(path)
            elif baseline.files[path].exists:
                restore.append(path)
            else:
                remove.append(path)
        return RollbackPreview(
            task_id=task_id,
            run_id=run_id,
            repo_root=str(root),
            restore=tuple(restore),
            remove=tuple(remove),
            conflicts=tuple(conflicts),
        )

    def _rollback_evidence(
        self,
        task_id: str,
        run_id: str,
        *,
        status: str,
        reason: str,
        conflicts: list[str] | None = None,
    ) -> Path:
        payload = {
            "task_id": task_id,
            "run_id": run_id,
            "status": status,
            "reason": reason,
            "conflicts": sorted(conflicts or []),
            "created_at": _now(),
        }
        path = (
            self._run_dir(task_id, run_id)
            / "rollback-attempts"
            / f"{uuid.uuid4().hex}.json"
        )
        _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True).encode(), 0o600)
        return path

    def record_rollback_refusal(
        self,
        task_id: str,
        run_id: str,
        *,
        reason: str,
        conflicts: list[str] | None = None,
    ) -> Path:
        with self.lock(task_id, run_id):
            return self._rollback_evidence(
                task_id,
                run_id,
                status="refused",
                reason=reason,
                conflicts=conflicts,
            )

    def _restore_snapshot(
        self,
        task_id: str,
        run_id: str,
        root: Path,
        snapshot: FileSnapshot,
    ) -> None:
        path = self._workspace_path(root, snapshot.path)
        if snapshot.exists:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(
                path,
                self._blob(task_id, run_id, snapshot),
                snapshot.mode,
            )
        elif path.exists():
            path.unlink()

    def rollback(self, task_id: str, run_id: str) -> RollbackResult:
        with self.lock(task_id, run_id):
            if (self._run_dir(task_id, run_id) / "rollback.json").exists():
                raise CheckpointError("task checkpoint was already rolled back")
            baseline = self.load_manifest(task_id, run_id, 0)
            latest = self.latest_checkpoint(task_id, run_id)
            if baseline is None or latest is None:
                raise CheckpointError("baseline or latest checkpoint is missing")
            root = Path(baseline.repo_root).resolve()
            current: dict[str, FileSnapshot] = {}
            conflicts: list[str] = []
            for path, expected in sorted(latest.files.items()):
                observed = self._capture_file(
                    task_id,
                    run_id,
                    root,
                    path,
                    before_iteration=expected.captured_before_iteration,
                    git_metadata=False,
                )
                current[path] = observed
                if not self._same_file(observed, expected):
                    conflicts.append(path)
            if conflicts:
                self._rollback_evidence(
                    task_id,
                    run_id,
                    status="refused",
                    reason="workspace changed after task checkpoint",
                    conflicts=conflicts,
                )
                raise RollbackConflict(conflicts)

            restored: list[str] = []
            removed: list[str] = []
            changed: list[str] = []
            candidate_dirs = sorted(
                {
                    parent
                    for snapshot in baseline.files.values()
                    if not snapshot.exists
                    for parent in snapshot.missing_parents
                },
                key=lambda value: (-len(PurePosixPath(value).parts), value),
            )
            candidate_dir_modes: dict[str, int] = {}
            for rel in candidate_dirs:
                directory = self._workspace_path(root, rel)
                if directory.is_dir():
                    candidate_dir_modes[rel] = stat.S_IMODE(directory.stat().st_mode)
            audit_path = self._run_dir(task_id, run_id) / "rollback.json"
            removed_dirs: list[str] = []
            try:
                for path, original in sorted(baseline.files.items()):
                    observed_now = self._capture_file(
                        task_id,
                        run_id,
                        root,
                        path,
                        before_iteration=original.captured_before_iteration,
                        git_metadata=False,
                    )
                    if not self._same_file(observed_now, current[path]):
                        raise RollbackConflict([path])
                    self._restore_snapshot(task_id, run_id, root, original)
                    changed.append(path)
                    (restored if original.exists else removed).append(path)
                for rel in candidate_dirs:
                    directory = self._workspace_path(root, rel)
                    try:
                        directory.rmdir()
                    except OSError:
                        continue
                    removed_dirs.append(rel)
                audit = {
                    "task_id": task_id,
                    "run_id": run_id,
                    "rolled_back_at": _now(),
                    "from_checkpoint": latest.number,
                    "restored": restored,
                    "removed": removed,
                    "removed_dirs": removed_dirs,
                }
                _atomic_write(
                    audit_path,
                    json.dumps(audit, indent=2, sort_keys=True).encode(),
                    0o600,
                )
            except Exception as exc:
                compensation_errors: list[str] = []
                for rel in sorted(
                    removed_dirs,
                    key=lambda value: (len(PurePosixPath(value).parts), value),
                ):
                    try:
                        directory = self._workspace_path(root, rel)
                        directory.mkdir(exist_ok=True)
                        os.chmod(directory, candidate_dir_modes[rel])
                    except Exception as compensation_exc:
                        compensation_errors.append(f"{rel}/: {compensation_exc}")
                for path in reversed(changed):
                    try:
                        self._restore_snapshot(task_id, run_id, root, current[path])
                    except Exception as compensation_exc:
                        compensation_errors.append(f"{path}: {compensation_exc}")
                reason = (
                    str(exc)
                    if isinstance(exc, RollbackConflict)
                    else f"rollback I/O failure: {exc}"
                )
                if compensation_errors:
                    reason += "; compensation failure: " + "; ".join(compensation_errors)
                try:
                    self._rollback_evidence(
                        task_id,
                        run_id,
                        status="refused",
                        reason=reason,
                        conflicts=exc.paths if isinstance(exc, RollbackConflict) else None,
                    )
                except OSError as evidence_exc:
                    reason += f"; could not record refusal: {evidence_exc}"
                if isinstance(exc, RollbackConflict):
                    raise
                raise CheckpointError(reason) from exc

            return RollbackResult(
                task_id=task_id,
                run_id=run_id,
                restored=tuple(restored),
                removed=tuple(removed),
                removed_dirs=tuple(removed_dirs),
                audit_path=str(audit_path),
            )
