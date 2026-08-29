from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from .models import SandboxRecord, SandboxState


class RegistryError(RuntimeError):
    pass


class RecordExistsError(RegistryError):
    pass


class RecordNotFoundError(RegistryError):
    pass


class JsonSandboxRegistry:
    """Atomic process-safe JSON persistence for sandbox lifecycle records."""

    VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            raise ValueError("registry path must be absolute")
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self._thread_lock = threading.RLock()

    def add(
        self,
        record: SandboxRecord,
        *,
        max_active: int | None = None,
    ) -> None:
        def mutate(rows: dict[str, SandboxRecord]) -> None:
            existing = rows.get(record.sandbox_id)
            if existing is not None and existing.state is not SandboxState.REMOVED:
                raise RecordExistsError(
                    f"sandbox record already exists: {record.sandbox_id}"
                )
            if max_active is not None:
                if max_active < 1:
                    raise ValueError("max_active must be positive")
                active = sum(
                    item.state is not SandboxState.REMOVED
                    for item in rows.values()
                )
                if active >= max_active:
                    raise RecordExistsError(
                        f"active sandbox limit reached: {max_active}"
                    )
            rows[record.sandbox_id] = record

        self._mutate(mutate)

    def put(self, record: SandboxRecord) -> None:
        def mutate(rows: dict[str, SandboxRecord]) -> None:
            if record.sandbox_id not in rows:
                raise RecordNotFoundError(record.sandbox_id)
            rows[record.sandbox_id] = record

        self._mutate(mutate)

    def update(
        self,
        sandbox_id: str,
        update: Callable[[SandboxRecord], SandboxRecord],
    ) -> SandboxRecord:
        result: SandboxRecord | None = None

        def mutate(rows: dict[str, SandboxRecord]) -> None:
            nonlocal result
            try:
                current = rows[sandbox_id]
            except KeyError as exc:
                raise RecordNotFoundError(sandbox_id) from exc
            result = update(current)
            if result.sandbox_id != sandbox_id:
                raise RegistryError("record update may not change sandbox_id")
            rows[sandbox_id] = result

        self._mutate(mutate)
        assert result is not None
        return result

    def get(self, sandbox_id: str) -> SandboxRecord | None:
        with self._locked():
            return self._read().get(sandbox_id)

    def require(self, sandbox_id: str) -> SandboxRecord:
        record = self.get(sandbox_id)
        if record is None:
            raise RecordNotFoundError(sandbox_id)
        return record

    def list(self) -> tuple[SandboxRecord, ...]:
        with self._locked():
            return tuple(
                sorted(self._read().values(), key=lambda row: row.sandbox_id)
            )

    def _mutate(
        self, mutation: Callable[[dict[str, SandboxRecord]], None]
    ) -> None:
        with self._locked():
            rows = self._read()
            mutation(rows)
            self._write(rows)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with self.lock_path.open("a+", encoding="utf-8") as lock:
                os.chmod(self.lock_path, 0o600)
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read(self) -> dict[str, SandboxRecord]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise RegistryError("sandbox registry root is malformed")
            if raw.get("version") != self.VERSION:
                raise RegistryError("unsupported sandbox registry version")
            records = raw.get("records")
            if not isinstance(records, dict):
                raise RegistryError("sandbox registry records are malformed")
            parsed = {
                str(key): SandboxRecord.from_dict(value)
                for key, value in records.items()
            }
            if any(key != record.sandbox_id for key, record in parsed.items()):
                raise RegistryError("sandbox registry keys do not match records")
            return parsed
        except RegistryError:
            raise
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RegistryError(f"cannot read sandbox registry: {exc}") from exc

    def _write(self, rows: dict[str, SandboxRecord]) -> None:
        payload = {
            "version": self.VERSION,
            "records": {
                key: rows[key].to_dict() for key in sorted(rows)
            },
        }
        data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        temporary: str | None = None
        try:
            fd, temporary = tempfile.mkstemp(
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                text=True,
            )
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            raise RegistryError(f"cannot write sandbox registry: {exc}") from exc
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
