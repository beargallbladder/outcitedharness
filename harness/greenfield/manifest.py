from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from harness.greenfield.models import (
    GreenfieldDiscovery,
    GreenfieldManifest,
    MilestonePlan,
    ProductSpec,
)


class ManifestDriftError(RuntimeError):
    pass


def canonical_hash(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def destination_fingerprint(destination: Path) -> str:
    path = destination.expanduser().absolute()
    parent = path.parent.resolve()
    data: dict[str, Any] = {
        "path": str(parent / path.name),
        "parent": str(parent),
        "name": path.name,
        "exists": path.exists() or path.is_symlink(),
    }
    if data["exists"]:
        info = path.lstat()
        data.update(
            {
                "mode": stat.S_IFMT(info.st_mode),
                "inode": info.st_ino,
                "device": info.st_dev,
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "symlink": path.is_symlink(),
            }
        )
        if path.is_dir() and not path.is_symlink():
            data["children"] = sorted(item.name for item in path.iterdir())
    return canonical_hash(data)


def build_manifest(
    *,
    run_id: str,
    spec: ProductSpec,
    plan: MilestonePlan,
    discovery: GreenfieldDiscovery,
    destination: str,
    destination_fingerprint_value: str,
) -> GreenfieldManifest:
    return GreenfieldManifest(
        run_id=run_id,
        project_name=spec.project_name,
        stack=spec.stack,
        runtime=spec.runtime,
        package_manager=spec.package_manager,
        approved_dependencies=spec.approved_dependencies,
        destination=destination,
        destination_fingerprint=destination_fingerprint_value,
        spec_hash=canonical_hash(spec),
        plan_hash=canonical_hash(plan),
        discovery_hash=canonical_hash(discovery),
    )


def assert_manifest_unchanged(
    manifest: GreenfieldManifest,
    spec: ProductSpec,
    plan: MilestonePlan,
    discovery: GreenfieldDiscovery,
) -> None:
    actual = {
        "spec": canonical_hash(spec),
        "plan": canonical_hash(plan),
        "discovery": canonical_hash(discovery),
    }
    expected = {
        "spec": manifest.spec_hash,
        "plan": manifest.plan_hash,
        "discovery": manifest.discovery_hash,
    }
    drift = [name for name in expected if expected[name] != actual[name]]
    if drift:
        raise ManifestDriftError(
            "approved greenfield manifest drifted: " + ", ".join(drift)
        )


def assert_destination_unchanged(manifest: GreenfieldManifest) -> None:
    current = destination_fingerprint(Path(manifest.destination))
    if current != manifest.destination_fingerprint:
        raise ManifestDriftError("destination changed after planning")


def safe_project_name(value: str) -> str:
    name = value.strip().lower()
    if (
        not name
        or len(name) > 64
        or name[0] in ".-"
        or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in name)
        or ".." in name
        or os.sep in name
    ):
        raise ValueError("project name must be a safe lowercase slug")
    return name
