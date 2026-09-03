from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from harness.electronics.claims import make_claim
from harness.electronics.embeddings import (
    claim_embedding_text,
    seal_embedding_sidecar,
)
from harness.electronics.models import (
    ClaimClass,
    EntityGrain,
    EntityReference,
    EvidenceKind,
    EvidenceReference,
)


@dataclass
class _Response:
    vectors: tuple[tuple[float, ...], ...]


class _Encoder:
    model = "fixture-bge"

    def embed(self, texts):
        return _Response(tuple((1.0, 0.0, float(index)) for index, _ in enumerate(texts)))


def _claim():
    return make_claim(
        entity=EntityReference(
            entity_id="pin:acme:atom1:lqfp2:1",
            grain=EntityGrain.PIN_OR_BALL,
            canonical_id="1",
            vendor="acme",
            family="ATOM1",
            package="LQFP2",
        ),
        field="pin.functions",
        value=["ADC0", "GPIO"],
        claim_class=ClaimClass.SEMANTIC_LABEL,
        extraction_method="pymupdf_table",
        evidence=(
            EvidenceReference(
                kind=EvidenceKind.TABLE_CELL,
                document_sha256="a" * 64,
                source_uri="file:///corpus/atom1.pdf",
                page_1based=3,
                table_index=0,
                row_index=1,
                column_index=2,
                quoted_text="ADC0 / GPIO",
            ),
        ),
        conditions={"package": "LQFP2"},
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


def test_claim_embedding_text_preserves_entity_and_fact():
    text = claim_embedding_text(_claim())
    assert "family: ATOM1" in text
    assert "package: LQFP2" in text
    assert "field: pin.functions" in text
    assert 'value: ["ADC0","GPIO"]' in text


def test_embedding_sidecar_is_binary_hash_bound(tmp_path: Path):
    output = tmp_path / "embeddings"
    manifest = seal_embedding_sidecar(
        output,
        [_claim()],
        encoder=_Encoder(),
        batch_size=1,
        source_receipts={"admission": "b" * 64},
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )

    assert manifest["counts"]["embeddings"] == 1
    assert manifest["format"]["dimensions"] == 3
    assert (output / "vectors.f32").stat().st_size == 12
    index = json.loads((output / "index.jsonl").read_text())
    assert index["claim_id"] == _claim().claim_id
    assert index["offset_bytes"] == 0
