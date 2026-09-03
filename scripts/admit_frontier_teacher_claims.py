#!/usr/bin/env python3
"""Admit source-grounded frontier claims to training and embedding only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.electronics.admission import iter_claims
from harness.electronics.claims import (
    canonical_json,
    claim_sha256,
    stable_id,
    verify_claim_bundle,
)
from harness.electronics.frontier_batch import FrontierTeacherVerification
from harness.electronics.models import (
    AdmissionStatus,
    ClaimAdmission,
    ClaimVerification,
    VerificationStatus,
)


def _write(path: Path, values: list[dict[str, Any]]) -> dict[str, Any]:
    payload = b"".join(canonical_json(value) + b"\n" for value in values)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-bundle", type=Path, required=True)
    parser.add_argument("--teacher-verifications", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    claim_root = args.claim_bundle.expanduser().resolve(strict=True)
    claim_manifest = verify_claim_bundle(claim_root)
    claims = {claim.claim_id: claim for claim in iter_claims(claim_root)}
    teacher_path = args.teacher_verifications.expanduser().resolve(strict=True)
    teacher_by_claim = {}
    with teacher_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            teacher = FrontierTeacherVerification.model_validate_json(line)
            if teacher.status != "passed":
                continue
            for claim_id in teacher.claim_ids:
                if claim_id in teacher_by_claim:
                    raise ValueError(f"claim has duplicate teacher proof: {claim_id}")
                teacher_by_claim[claim_id] = teacher
    if set(teacher_by_claim) != set(claims):
        raise ValueError("claim bundle and passed teacher claims differ")

    now = datetime.now(timezone.utc)
    verifications = []
    admissions = []
    policy = "frontier-teacher-source-grounded-v1"
    for claim_id in sorted(claims):
        claim = claims[claim_id]
        teacher = teacher_by_claim[claim_id]
        verification_core = {
            "claim_id": claim_id,
            "claim_sha256": claim_sha256(claim),
            "status": VerificationStatus.PASSED.value,
            "verifier": "deterministic",
            "checks": list(teacher.checks),
            "evidence_sha256": list(teacher.evidence_sha256),
        }
        verification = ClaimVerification(
            verification_id=stable_id("verify", verification_core),
            claim_id=claim_id,
            claim_sha256=verification_core["claim_sha256"],
            status=VerificationStatus.PASSED,
            verifier="deterministic",
            checks=tuple(teacher.checks),
            evidence_sha256=tuple(teacher.evidence_sha256),
            created_at=now,
        )
        admission_core = {
            "claim_id": claim_id,
            "claim_class": claim.claim_class.value,
            "verification_id": verification.verification_id,
            "verification_status": VerificationStatus.PASSED.value,
            "status": AdmissionStatus.ADMITTED.value,
            "policy": policy,
            "dataset_purposes": [
                "electronics_warehouse",
                "local_training",
                "embeddings",
            ],
        }
        admission = ClaimAdmission(
            admission_id=stable_id("admit", admission_core),
            claim_id=claim_id,
            claim_class=claim.claim_class,
            verification_id=verification.verification_id,
            verification_status=VerificationStatus.PASSED,
            status=AdmissionStatus.ADMITTED,
            policy=policy,
            dataset_purposes=(
                "electronics_warehouse",
                "local_training",
                "embeddings",
            ),
            created_at=now,
        )
        verifications.append(verification.model_dump(mode="json", by_alias=True))
        admissions.append(admission.model_dump(mode="json", by_alias=True))

    output = args.output_directory.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        artifacts = {
            "verifications.jsonl": _write(
                temporary / "verifications.jsonl",
                verifications,
            ),
            "admissions.jsonl": _write(
                temporary / "admissions.jsonl",
                admissions,
            ),
        }
        core = {
            "schema": (
                "harness.electronics-frontier-claim-admission-bundle.v1"
            ),
            "policy": {
                "name": policy,
                "source_row_grounding_required": True,
                "cr_import": False,
                "direct_cr_write": False,
            },
            "sources": {
                "claim_bundle": {
                    "path": str(claim_root),
                    "evidence_sha256": claim_manifest["evidence_sha256"],
                },
                "teacher_verifications": {
                    "path": str(teacher_path),
                    "sha256": hashlib.sha256(
                        teacher_path.read_bytes()
                    ).hexdigest(),
                },
            },
            "counts": {
                "claims": len(claims),
                "verifications": len(verifications),
                "admissions": len(admissions),
                "embedding_claims": len(admissions),
                "local_training_claims": len(admissions),
                "cr_import_claims": 0,
            },
            "artifacts": artifacts,
        }
        core["evidence_sha256"] = hashlib.sha256(
            canonical_json(core)
        ).hexdigest()
        manifest = {
            "created_at": now.isoformat(),
            **core,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(manifest_path, 0o444)
        os.chmod(temporary, 0o555)
        os.rename(temporary, output)
    except BaseException:
        os.chmod(temporary, 0o755)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
