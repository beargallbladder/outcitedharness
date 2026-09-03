"""Deterministic identities and immutable bundles for electronics claims."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from harness.electronics.models import (
    ClaimClass,
    EntityReference,
    EvidenceReference,
    FactClaim,
    ModelIdentity,
)


BUNDLE_SCHEMA = "harness.electronics-claim-bundle.v1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def stable_id(prefix: str, value: Any) -> str:
    if prefix not in {"claim", "verify", "admit", "pair", "bundle"}:
        raise ValueError("unsupported stable ID prefix")
    return f"{prefix}-{hashlib.sha256(canonical_json(value)).hexdigest()[:32]}"


def claim_identity_payload(
    *,
    entity: EntityReference,
    field: str,
    value: Any,
    unit: str | None,
    conditions: Mapping[str, Any],
    claim_class: ClaimClass,
    extraction_method: str,
    evidence: tuple[EvidenceReference, ...],
    source_claim_ids: tuple[str, ...],
    model: ModelIdentity | None,
) -> dict[str, Any]:
    return {
        "entity": entity.model_dump(mode="json", by_alias=True),
        "field": field,
        "value": value,
        "unit": unit,
        "conditions": dict(conditions),
        "claim_class": claim_class.value,
        "extraction_method": extraction_method,
        "evidence": [
            item.model_dump(mode="json", by_alias=True) for item in evidence
        ],
        "source_claim_ids": list(source_claim_ids),
        "model": (
            model.model_dump(mode="json", by_alias=True)
            if model is not None
            else None
        ),
    }


def make_claim(
    *,
    entity: EntityReference,
    field: str,
    value: Any,
    claim_class: ClaimClass,
    extraction_method: str,
    evidence: tuple[EvidenceReference, ...],
    created_at: datetime,
    unit: str | None = None,
    conditions: Mapping[str, Any] | None = None,
    source_claim_ids: tuple[str, ...] = (),
    model: ModelIdentity | None = None,
) -> FactClaim:
    condition_values = dict(conditions or {})
    identity = claim_identity_payload(
        entity=entity,
        field=field,
        value=value,
        unit=unit,
        conditions=condition_values,
        claim_class=claim_class,
        extraction_method=extraction_method,
        evidence=evidence,
        source_claim_ids=source_claim_ids,
        model=model,
    )
    return FactClaim(
        claim_id=stable_id("claim", identity),
        entity=entity,
        field=field,
        value=value,
        unit=unit,
        conditions=condition_values,
        claim_class=claim_class,
        extraction_method=extraction_method,
        evidence=evidence,
        source_claim_ids=source_claim_ids,
        model=model,
        created_at=created_at,
    )


def claim_sha256(claim: FactClaim) -> str:
    return hashlib.sha256(
        canonical_json(claim.model_dump(mode="json", by_alias=True))
    ).hexdigest()


def verify_claim_identity(claim: FactClaim) -> None:
    expected = stable_id(
        "claim",
        claim_identity_payload(
            entity=claim.entity,
            field=claim.field,
            value=claim.value,
            unit=claim.unit,
            conditions=claim.conditions,
            claim_class=claim.claim_class,
            extraction_method=claim.extraction_method,
            evidence=claim.evidence,
            source_claim_ids=claim.source_claim_ids,
            model=claim.model,
        ),
    )
    if claim.claim_id != expected:
        raise ValueError(f"claim identity mismatch: {claim.claim_id}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def seal_claim_bundle(
    destination: Path,
    claims: Iterable[FactClaim],
    *,
    source_receipts: Mapping[str, Any],
    created_at: datetime,
) -> dict[str, Any]:
    output = destination.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"claim bundle already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        claims_path = temporary / "claims.jsonl"
        field_counts: Counter[str] = Counter()
        class_counts: Counter[str] = Counter()
        entity_grains: Counter[str] = Counter()
        seen: set[str] = set()
        count = 0
        with claims_path.open("wb") as handle:
            for claim in claims:
                verify_claim_identity(claim)
                if claim.claim_id in seen:
                    raise ValueError(f"duplicate claim_id: {claim.claim_id}")
                seen.add(claim.claim_id)
                handle.write(
                    canonical_json(
                        claim.model_dump(mode="json", by_alias=True)
                    )
                    + b"\n"
                )
                count += 1
                field_counts[claim.field] += 1
                class_counts[claim.claim_class.value] += 1
                entity_grains[claim.entity.grain.value] += 1
            handle.flush()
            os.fsync(handle.fileno())
        claims_digest = _sha256(claims_path)
        core: dict[str, Any] = {
            "schema": BUNDLE_SCHEMA,
            "sources": dict(source_receipts),
            "artifacts": {
                "claims.jsonl": {
                    "sha256": claims_digest,
                    "bytes": claims_path.stat().st_size,
                }
            },
            "counts": {
                "claims": count,
                "claim_classes": dict(sorted(class_counts.items())),
                "entity_grains": dict(sorted(entity_grains.items())),
                "fields": dict(sorted(field_counts.items())),
            },
        }
        core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
        manifest = {
            "created_at": created_at.isoformat(),
            **core,
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("wb") as handle:
            handle.write(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        for path in (claims_path, manifest_path):
            os.chmod(path, 0o444)
        os.chmod(temporary, 0o555)
        os.rename(temporary, output)
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return manifest
    except BaseException:
        try:
            os.chmod(temporary, 0o755)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_claim_bundle(path: Path) -> dict[str, Any]:
    root = path.expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    claims_path = root / "claims.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("claim bundle schema is not supported")
    expected_artifact = manifest.get("artifacts", {}).get("claims.jsonl", {})
    if expected_artifact.get("sha256") != _sha256(claims_path):
        raise ValueError("claim bundle artifact hash mismatch")
    if expected_artifact.get("bytes") != claims_path.stat().st_size:
        raise ValueError("claim bundle artifact size mismatch")
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_at", "evidence_sha256"}
    }
    if manifest.get("evidence_sha256") != hashlib.sha256(
        canonical_json(core)
    ).hexdigest():
        raise ValueError("claim bundle evidence digest mismatch")
    count = 0
    seen: set[str] = set()
    with claims_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                claim = FactClaim.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"invalid claim at line {line_number}"
                ) from exc
            verify_claim_identity(claim)
            if claim.claim_id in seen:
                raise ValueError("claim bundle contains duplicate claim IDs")
            seen.add(claim.claim_id)
            count += 1
    if count != manifest.get("counts", {}).get("claims"):
        raise ValueError("claim bundle count mismatch")
    return manifest


__all__ = [
    "BUNDLE_SCHEMA",
    "canonical_json",
    "claim_sha256",
    "make_claim",
    "seal_claim_bundle",
    "stable_id",
    "verify_claim_bundle",
    "verify_claim_identity",
]
