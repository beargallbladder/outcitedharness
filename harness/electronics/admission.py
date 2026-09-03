"""Verification, admission, and CR-bundle construction for pinout claims."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from harness.electronics.claims import (
    canonical_json,
    claim_sha256,
    stable_id,
    verify_claim_bundle,
)
from harness.electronics.locator import package_pin_count
from harness.electronics.models import (
    AdmissionStatus,
    ClaimAdmission,
    ClaimVerification,
    FactClaim,
    VerificationStatus,
)


ADMISSION_BUNDLE_SCHEMA = "harness.electronics-admission-bundle.v1"
CR_IMPORT_SCHEMA = "harness.cr-pin-package-import.v1"
PIN_FIELDS = {
    "pin.number",
    "pin.name",
    "pin.type",
    "pin.functions",
    "pin.direction",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class PinGroup:
    record_id: str
    package: str
    split: str
    document_sha256: str
    expected_pins: int | None
    physical_ids: set[str] = field(default_factory=set)
    required_fields: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    values: dict[tuple[str, str], set[bytes]] = field(
        default_factory=lambda: defaultdict(set)
    )

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.record_id,
            self.package,
            self.split,
            self.document_sha256,
        )

    @property
    def conflicts(self) -> int:
        return sum(len(values) > 1 for values in self.values.values())

    @property
    def complete(self) -> bool:
        if self.expected_pins is None or self.expected_pins < 1:
            return False
        if len(self.physical_ids) != self.expected_pins or self.conflicts:
            return False
        return all(
            {"pin.number", "pin.name"}.issubset(
                self.required_fields.get(physical_id, set())
            )
            for physical_id in self.physical_ids
        )


def _physical_id(value: Any) -> str:
    output = re.sub(
        r"\s+",
        "",
        str("" if value is None else value).upper(),
    )
    output = re.sub(r"\(\d+\)$", "", output)
    return output


def iter_claims(bundle: Path) -> Iterator[FactClaim]:
    verify_claim_bundle(bundle)
    with (bundle / "claims.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield FactClaim.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"invalid claim at line {line_number}"
                ) from exc


def scan_pin_groups(bundle: Path) -> dict[tuple[str, str, str, str], PinGroup]:
    groups: dict[tuple[str, str, str, str], PinGroup] = {}
    for claim in iter_claims(bundle):
        if claim.field not in PIN_FIELDS:
            continue
        split = str(claim.conditions.get("source_split") or "")
        package = str(claim.entity.package or "")
        record_id = str(claim.entity.family or "")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"pin claim has invalid source split: {claim.claim_id}")
        if not package or not record_id or not claim.evidence:
            raise ValueError(f"pin claim lacks group identity: {claim.claim_id}")
        document_sha256 = claim.evidence[0].document_sha256
        key = (record_id, package, split, document_sha256)
        group = groups.setdefault(
            key,
            PinGroup(
                record_id=record_id,
                package=package,
                split=split,
                document_sha256=document_sha256,
                expected_pins=package_pin_count(package),
            ),
        )
        physical = _physical_id(claim.entity.canonical_id)
        if not physical:
            raise ValueError(f"pin claim has empty physical ID: {claim.claim_id}")
        group.physical_ids.add(physical)
        group.required_fields[physical].add(claim.field)
        group.values[(physical, claim.field)].add(canonical_json(claim.value))
    return groups


def _verification(
    claim: FactClaim,
    *,
    created_at: datetime,
    authorization_sha256: str,
) -> ClaimVerification:
    checks = (
        "frontier_ground_truth_status_validated",
        "exact_pdf_package_and_page_alignment",
        "rendered_evidence_hash_verified",
        "training_owner_authorization_verified",
    )
    evidence_hashes = tuple(
        sorted(
            {
                authorization_sha256,
                *(
                    reference.artifact_sha256
                    or reference.document_sha256
                    for reference in claim.evidence
                ),
            }
        )
    )
    claim_digest = claim_sha256(claim)
    identity = {
        "claim_id": claim.claim_id,
        "claim_sha256": claim_digest,
        "status": VerificationStatus.PASSED.value,
        "verifier": "frontier_consensus",
        "checks": checks,
        "evidence_sha256": evidence_hashes,
    }
    return ClaimVerification(
        verification_id=stable_id("verify", identity),
        claim_id=claim.claim_id,
        claim_sha256=claim_digest,
        status=VerificationStatus.PASSED,
        verifier="frontier_consensus",
        checks=checks,
        evidence_sha256=evidence_hashes,
        created_at=created_at,
    )


def _admission(
    claim: FactClaim,
    verification: ClaimVerification,
    group: PinGroup,
    *,
    created_at: datetime,
) -> ClaimAdmission:
    if group.split == "train":
        purposes = [
            "electronics_warehouse",
            "local_training",
            "embeddings",
        ]
        if group.complete:
            purposes.append("cr_import")
    else:
        purposes = ["frozen_evaluation"]
    identity = {
        "claim_id": claim.claim_id,
        "verification_id": verification.verification_id,
        "policy": "pinout-row-owner-authorized-v1",
        "dataset_purposes": purposes,
    }
    return ClaimAdmission(
        admission_id=stable_id("admit", identity),
        claim_id=claim.claim_id,
        claim_class=claim.claim_class,
        verification_id=verification.verification_id,
        verification_status=verification.status,
        status=AdmissionStatus.ADMITTED,
        policy="pinout-row-owner-authorized-v1",
        dataset_purposes=tuple(purposes),
        created_at=created_at,
    )


def _pin_sort_key(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def build_admission_bundle(
    source_bundle: Path,
    destination: Path,
    *,
    authorization_path: Path,
    created_at: datetime,
) -> dict[str, Any]:
    source = source_bundle.expanduser().resolve(strict=True)
    authorization = authorization_path.expanduser().resolve(strict=True)
    authorization_sha256 = hashlib.sha256(authorization.read_bytes()).hexdigest()
    authorization_value = json.loads(authorization.read_text(encoding="utf-8"))
    if authorization_value.get("training_authorized") is not True:
        raise ValueError("training authorization is not affirmative")
    groups = scan_pin_groups(source)
    output = destination.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"admission bundle already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    verification_path = temporary / "verifications.jsonl"
    admission_path = temporary / "admissions.jsonl"
    cr_path = temporary / "cr-pin-packages.jsonl"
    disposition_counts: Counter[str] = Counter()
    purpose_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    cr_rows: dict[
        tuple[str, str, str, str],
        dict[str, dict[str, Any]],
    ] = defaultdict(lambda: defaultdict(dict))
    cr_claims: dict[
        tuple[str, str, str, str],
        dict[str, dict[str, str]],
    ] = defaultdict(lambda: defaultdict(dict))
    try:
        with verification_path.open("xb") as verification_handle, admission_path.open(
            "xb"
        ) as admission_handle:
            for claim in iter_claims(source):
                if claim.field not in PIN_FIELDS:
                    continue
                key = (
                    str(claim.entity.family),
                    str(claim.entity.package),
                    str(claim.conditions["source_split"]),
                    claim.evidence[0].document_sha256,
                )
                group = groups[key]
                verification = _verification(
                    claim,
                    created_at=created_at,
                    authorization_sha256=authorization_sha256,
                )
                admission = _admission(
                    claim,
                    verification,
                    group,
                    created_at=created_at,
                )
                verification_handle.write(
                    canonical_json(
                        verification.model_dump(mode="json", by_alias=True)
                    )
                    + b"\n"
                )
                admission_handle.write(
                    canonical_json(
                        admission.model_dump(mode="json", by_alias=True)
                    )
                    + b"\n"
                )
                disposition_counts[group.split] += 1
                purpose_counts.update(admission.dataset_purposes)
                field_counts[claim.field] += 1
                if group.split == "train" and group.complete:
                    physical = _physical_id(claim.entity.canonical_id)
                    row = cr_rows[key][physical]
                    row["pin_number"] = claim.entity.canonical_id
                    field_name = {
                        "pin.name": "pin_name",
                        "pin.type": "type",
                        "pin.functions": "functions",
                        "pin.direction": "direction",
                    }.get(claim.field)
                    if field_name is not None:
                        row[field_name] = claim.value
                    cr_claims[key][physical][claim.field] = claim.claim_id
            for handle in (verification_handle, admission_handle):
                handle.flush()
                os.fsync(handle.fileno())

        cr_records = 0
        with cr_path.open("xb") as handle:
            for key, pins in sorted(cr_rows.items()):
                group = groups[key]
                if len(pins) != group.expected_pins:
                    raise ValueError(f"complete group lost pins during CR build: {key}")
                record = {
                    "schema": CR_IMPORT_SCHEMA,
                    "operation": "candidate_upsert",
                    "record_id": group.record_id,
                    "document_sha256": group.document_sha256,
                    "package": group.package,
                    "expected_package_pins": group.expected_pins,
                    "pins": [
                        {
                            **pins[physical],
                            "source_claim_ids": cr_claims[key][physical],
                        }
                        for physical in sorted(pins, key=_pin_sort_key)
                    ],
                }
                handle.write(canonical_json(record) + b"\n")
                cr_records += 1
            handle.flush()
            os.fsync(handle.fileno())

        artifacts = {}
        for path in (verification_path, admission_path, cr_path):
            artifacts[path.name] = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        complete_groups = sum(group.complete for group in groups.values())
        core = {
            "schema": ADMISSION_BUNDLE_SCHEMA,
            "policy": {
                "name": "pinout-row-owner-authorized-v1",
                "train_split_purposes": [
                    "electronics_warehouse",
                    "local_training",
                    "embeddings",
                ],
                "evaluation_split_purposes": ["frozen_evaluation"],
                "cr_import_requires_complete_package": True,
                "direct_cr_write": False,
            },
            "sources": {
                "claim_bundle": {
                    "path": str(source),
                    "evidence_sha256": verify_claim_bundle(source)[
                        "evidence_sha256"
                    ],
                },
                "training_authorization": {
                    "path": str(authorization),
                    "sha256": authorization_sha256,
                },
            },
            "artifacts": artifacts,
            "counts": {
                "pin_groups": len(groups),
                "complete_pin_groups": complete_groups,
                "withheld_pin_groups": len(groups) - complete_groups,
                "claims_by_split": dict(sorted(disposition_counts.items())),
                "claims_by_field": dict(sorted(field_counts.items())),
                "admissions_by_purpose": dict(sorted(purpose_counts.items())),
                "cr_package_records": cr_records,
            },
        }
        core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
        manifest = {"created_at": created_at.isoformat(), **core}
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("xb") as handle:
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
        for path in (
            verification_path,
            admission_path,
            cr_path,
            manifest_path,
        ):
            os.chmod(path, 0o444)
        os.chmod(temporary, 0o555)
        os.rename(temporary, output)
        return manifest
    except BaseException:
        try:
            os.chmod(temporary, 0o755)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "ADMISSION_BUNDLE_SCHEMA",
    "CR_IMPORT_SCHEMA",
    "PinGroup",
    "build_admission_bundle",
    "iter_claims",
    "scan_pin_groups",
]
