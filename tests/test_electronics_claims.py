from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.electronics.claims import (
    make_claim,
    seal_claim_bundle,
    verify_claim_bundle,
)
from harness.electronics.models import (
    ClaimClass,
    EntityGrain,
    EntityReference,
    EvidenceKind,
    EvidenceReference,
)


def _claim():
    return make_claim(
        entity=EntityReference(
            entity_id="opn:acme:atom1",
            grain=EntityGrain.OPN,
            canonical_id="ATOM1",
            vendor="acme",
        ),
        field="clock.max_frequency",
        value=100,
        unit="MHz",
        claim_class=ClaimClass.VISIBLE_FACT,
        extraction_method="pymupdf_text",
        evidence=(
            EvidenceReference(
                kind=EvidenceKind.TEXT_SPAN,
                document_sha256="a" * 64,
                source_uri="file:///corpus/atom1.pdf",
                page_1based=10,
                quoted_text="Maximum frequency 100 MHz",
            ),
        ),
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


def test_claim_identity_does_not_depend_on_capture_time():
    first = _claim()
    second = first.model_copy(
        update={"created_at": datetime(2026, 9, 2, tzinfo=timezone.utc)}
    )

    assert first.claim_id == second.claim_id


def test_claim_bundle_is_hash_bound_and_immutable(tmp_path: Path):
    output = tmp_path / "bundle"
    manifest = seal_claim_bundle(
        output,
        [_claim()],
        source_receipts={"registry_sha256": "b" * 64},
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )

    verified = verify_claim_bundle(output)
    assert manifest == verified
    assert verified["counts"]["claims"] == 1
    assert (output / "claims.jsonl").stat().st_mode & 0o777 == 0o444
    with pytest.raises(ValueError, match="already exists"):
        seal_claim_bundle(
            output,
            [_claim()],
            source_receipts={},
            created_at=datetime.now(timezone.utc),
        )
