from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from harness.electronics.admission import build_admission_bundle, scan_pin_groups
from harness.electronics.claims import make_claim, seal_claim_bundle
from harness.electronics.models import (
    ClaimClass,
    EntityGrain,
    EntityReference,
    EvidenceKind,
    EvidenceReference,
)


def _claims():
    created_at = datetime(2026, 9, 1, tzinfo=timezone.utc)
    evidence = (
        EvidenceReference(
            kind=EvidenceKind.IMAGE_REGION,
            document_sha256="a" * 64,
            source_uri="file:///corpus/atom1.pdf",
            artifact_sha256="b" * 64,
            page_1based=4,
            bbox=(1, 2, 3, 4),
        ),
    )
    for pin_no, pin_name in ((1, "PA0"), (2, "VSS")):
        entity = EntityReference(
            entity_id=f"pin:acme:atom1:lqfp2:{pin_no}",
            grain=EntityGrain.PIN_OR_BALL,
            canonical_id=str(pin_no),
            vendor="acme",
            family="acme_atom1",
            package="LQFP2",
        )
        conditions = {
            "source_split": "train",
            "source_example_id": f"row-{pin_no}",
            "package": "LQFP2",
        }
        yield make_claim(
            entity=entity,
            field="pin.number",
            value=pin_no,
            claim_class=ClaimClass.VISIBLE_FACT,
            extraction_method="pymupdf_vision",
            evidence=evidence,
            conditions=conditions,
            created_at=created_at,
        )
        yield make_claim(
            entity=entity,
            field="pin.name",
            value=pin_name,
            claim_class=ClaimClass.VISIBLE_FACT,
            extraction_method="pymupdf_vision",
            evidence=evidence,
            conditions=conditions,
            created_at=created_at,
        )


def test_complete_train_package_is_admitted_to_cr_bundle(tmp_path: Path):
    source = tmp_path / "claims"
    seal_claim_bundle(
        source,
        list(_claims()),
        source_receipts={"fixture": True},
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps({"training_authorized": True}),
        encoding="utf-8",
    )

    groups = scan_pin_groups(source)
    assert len(groups) == 1
    assert next(iter(groups.values())).complete is True

    destination = tmp_path / "admissions"
    manifest = build_admission_bundle(
        source,
        destination,
        authorization_path=authorization,
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )

    assert manifest["counts"]["complete_pin_groups"] == 1
    assert manifest["counts"]["cr_package_records"] == 1
    assert manifest["counts"]["admissions_by_purpose"]["local_training"] == 4
    assert manifest["counts"]["admissions_by_purpose"]["cr_import"] == 4
    record = json.loads(
        (destination / "cr-pin-packages.jsonl").read_text(encoding="utf-8")
    )
    assert record["operation"] == "candidate_upsert"
    assert len(record["pins"]) == 2
