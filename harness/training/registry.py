from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from harness.training.models import TrainingManifest


class RegistryError(RuntimeError):
    pass


class ManifestConflictError(RegistryError):
    pass


class ManifestIntegrityError(RegistryError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def manifest_digest(manifest: TrainingManifest) -> str:
    return hashlib.sha256(
        canonical_json(manifest.model_dump(mode="json"))
    ).hexdigest()


class ManifestRegistry:
    """Filesystem registry with immutable, checksum-verified atomic entries."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.manifests_dir = self.root / "manifests"
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, manifest_id: str) -> Path:
        # Validation also prevents traversal if this method is reached via get().
        if (
            not manifest_id
            or len(manifest_id) > 128
            or manifest_id[0] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            or any(
                char
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for char in manifest_id
            )
        ):
            raise ValueError("invalid manifest_id")
        return self.manifests_dir / f"{manifest_id}.json"

    def register(self, manifest: TrainingManifest) -> str:
        destination = self._path(manifest.manifest_id)
        digest = manifest_digest(manifest)
        envelope = {
            "registry_schema_version": 1,
            "manifest_sha256": digest,
            "manifest": manifest.model_dump(mode="json"),
        }
        payload = canonical_json(envelope) + b"\n"

        if destination.exists():
            current = self.get(manifest.manifest_id)
            if manifest_digest(current) == digest:
                return digest
            raise ManifestConflictError(
                f"manifest {manifest.manifest_id!r} is immutable and already exists"
            )

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{manifest.manifest_id}.",
            suffix=".tmp",
            dir=self.manifests_dir,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            try:
                # Hard-link creation is atomic and refuses to replace a concurrent
                # writer's entry. The temporary inode is removed afterward.
                os.link(temporary, destination)
            except FileExistsError:
                current = self.get(manifest.manifest_id)
                if manifest_digest(current) != digest:
                    raise ManifestConflictError(
                        f"concurrent conflicting manifest {manifest.manifest_id!r}"
                    )
            self._fsync_directory()
        finally:
            temporary.unlink(missing_ok=True)
        return digest

    def get(self, manifest_id: str) -> TrainingManifest:
        path = self._path(manifest_id)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope.get("registry_schema_version") != 1:
                raise ValueError("unsupported registry schema")
            manifest = TrainingManifest.model_validate(envelope["manifest"])
            expected = str(envelope["manifest_sha256"])
        except FileNotFoundError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ManifestIntegrityError(
                f"manifest {manifest_id!r} is not a valid registry entry"
            ) from exc
        actual = manifest_digest(manifest)
        if expected != actual:
            raise ManifestIntegrityError(
                f"manifest {manifest_id!r} checksum mismatch"
            )
        if manifest.manifest_id != manifest_id:
            raise ManifestIntegrityError("manifest identity does not match filename")
        return manifest

    def list(self) -> tuple[TrainingManifest, ...]:
        return tuple(
            self.get(path.stem)
            for path in sorted(self.manifests_dir.glob("*.json"))
            if not path.name.startswith(".")
        )

    def _fsync_directory(self) -> None:
        descriptor = os.open(self.manifests_dir, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    # Explicit aliases make call sites read naturally.
    put = register
    load = get
    list_manifests = list
