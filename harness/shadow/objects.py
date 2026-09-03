from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path

from harness.training.security import assert_no_secrets


class ShadowObjectStore:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.objects = self.root / "objects" / "sha256"
        self.objects.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        os.chmod(self.root / "objects", 0o700)
        os.chmod(self.objects, 0o700)

    def put_text(self, content: str) -> tuple[str, int, str]:
        assert_no_secrets(content, field="shadow snapshot object")
        data = content.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        directory = self.objects / digest[:2]
        directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
        destination = directory / digest
        if destination.exists():
            self._verify(destination, digest, len(data))
        else:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{digest}.",
                suffix=".tmp",
                dir=directory,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    os.fchmod(handle.fileno(), 0o600)
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    self._verify(destination, digest, len(data))
            finally:
                temporary.unlink(missing_ok=True)
        return digest, len(data), destination.relative_to(self.root).as_posix()

    def read_text(self, digest: str, relative_path: str) -> str:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("object digest must be lowercase SHA-256")
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root) or path.is_symlink():
            raise ValueError("object path escapes the shadow store")
        self._verify(path, digest, path.stat().st_size)
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _verify(path: Path, digest: str, size: int) -> None:
        if path.is_symlink() or not path.is_file():
            raise ValueError("shadow object is not a regular file")
        info = path.stat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_size != size
            or info.st_mode & 0o077
        ):
            raise ValueError("shadow object metadata is invalid")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("shadow object digest mismatch")
