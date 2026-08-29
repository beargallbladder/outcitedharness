from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.training.models import (
    Artifact,
    DataUse,
    RunStatus,
    SourceKind,
    SourceProvenance,
    TrainingManifest,
    TrainingRun,
)
from harness.training.registry import (
    ManifestConflictError,
    ManifestIntegrityError,
    ManifestRegistry,
)


NOW = datetime(2026, 8, 29, tzinfo=timezone.utc)


def _source(*, record_id: str = "source-1") -> SourceProvenance:
    return SourceProvenance(
        source_kind=SourceKind.HARNESS,
        source_uri="harness://runs/run-1/case-1",
        source_record_id=record_id,
        collected_at=NOW,
        content_sha256=hashlib.sha256(record_id.encode()).hexdigest(),
        lineage_id="case-1",
        license="internal-approved",
    )


def _manifest(*, manifest_id: str = "manifest-1", record_id: str = "source-1"):
    source = _source(record_id=record_id)
    artifact = Artifact(
        artifact_id="pairs-1",
        kind="text_pairs",
        uri="file:///exports/pairs.jsonl",
        sha256="1" * 64,
        byte_size=10,
        media_type="application/jsonl",
        provenance=source,
    )
    run = TrainingRun(
        run_id="run-1",
        status=RunStatus.COMPLETE,
        started_at=NOW,
        finished_at=NOW,
        source_ids=(record_id,),
        artifact_ids=("pairs-1",),
    )
    return TrainingManifest(
        manifest_id=manifest_id,
        created_at=NOW,
        run=run,
        sources=(source,),
        artifacts=(artifact,),
    )


def test_mutable_source_must_be_retrieval_only():
    with pytest.raises(ValidationError, match="retrieval_only"):
        SourceProvenance(
            source_kind="categoryrank",
            source_uri="snapshot://categoryrank/2026-W34",
            source_record_id="snapshot-1",
            collected_at=NOW,
            content_sha256="a" * 64,
            lineage_id="categoryrank",
            license="internal",
            mutable_facts=True,
            data_use=DataUse.TRAINING,
        )


def test_manifest_requires_closed_references():
    manifest = _manifest()
    raw = manifest.model_dump()
    raw["run"]["artifact_ids"] = ("missing",)
    with pytest.raises(ValidationError, match="dangling"):
        TrainingManifest.model_validate(raw)


def test_registry_is_idempotent_immutable_and_integrity_checked(tmp_path: Path):
    registry = ManifestRegistry(tmp_path / "registry")
    manifest = _manifest()
    digest = registry.register(manifest)
    assert len(digest) == 64
    assert registry.register(manifest) == digest
    assert registry.get(manifest.manifest_id) == manifest
    assert registry.list() == (manifest,)

    with pytest.raises(ManifestConflictError):
        registry.register(_manifest(record_id="source-2"))

    path = registry.manifests_dir / "manifest-1.json"
    envelope = json.loads(path.read_text())
    envelope["manifest"]["run"]["run_id"] = "tampered"
    path.write_text(json.dumps(envelope))
    with pytest.raises(ManifestIntegrityError, match="checksum"):
        registry.get("manifest-1")


def test_registry_rejects_traversal(tmp_path: Path):
    registry = ManifestRegistry(tmp_path)
    with pytest.raises(ValueError):
        registry.get("../manifest")
