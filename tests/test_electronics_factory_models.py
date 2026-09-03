from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from harness.electronics.models import (
    AdmissionStatus,
    ClaimClass,
    ClaimAdmission,
    ClaimVerification,
    EntityGrain,
    EntityReference,
    EvidenceKind,
    EvidenceReference,
    FactClaim,
    ModelIdentity,
    PairCapability,
    PairDisposition,
    PairModality,
    PreferenceTrainingPairCandidate,
    TrainingPairCandidate,
    VerificationStatus,
)


SHA = "a" * 64
CLAIM_ID = "claim-" + ("b" * 32)


def _entity() -> EntityReference:
    return EntityReference(
        entity_id="pin:acme:atom1:lqfp2:1",
        grain=EntityGrain.PIN_OR_BALL,
        canonical_id="1",
        vendor="acme",
        family="ATOM1",
        package="LQFP2",
    )


def _evidence() -> EvidenceReference:
    return EvidenceReference(
        kind=EvidenceKind.TABLE_CELL,
        document_sha256=SHA,
        source_uri="file:///corpus/atom1.pdf",
        page_1based=12,
        table_index=0,
        row_index=1,
        column_index=2,
        quoted_text="PA0",
    )


def test_visible_fact_requires_page_level_evidence():
    with pytest.raises(ValidationError, match="require source evidence"):
        FactClaim(
            claim_id=CLAIM_ID,
            entity=_entity(),
            field="pin.name",
            value="PA0",
            claim_class=ClaimClass.VISIBLE_FACT,
            extraction_method="pymupdf_table",
            created_at=datetime.now(timezone.utc),
        )


def test_table_evidence_requires_complete_coordinates():
    with pytest.raises(ValidationError, match="table, row, and column"):
        EvidenceReference(
            kind=EvidenceKind.TABLE_CELL,
            document_sha256=SHA,
            source_uri="file:///corpus/atom1.pdf",
            page_1based=12,
            table_index=0,
            row_index=1,
        )


def test_anthropic_teacher_is_bound_to_batch_request():
    with pytest.raises(ValidationError, match="batch_id"):
        ModelIdentity(
            provider="anthropic",
            model="claude-sonnet",
            request_sha256=SHA,
        )


def test_vision_training_pair_is_explicitly_for_local_training():
    teacher = ModelIdentity(
        provider="anthropic",
        model="claude-sonnet",
        request_sha256=SHA,
        response_id="msg_1",
        batch_id="msgbatch_1",
    )
    pair = TrainingPairCandidate(
        pair_id="pair-" + ("c" * 32),
        capability=PairCapability.PIN_OR_BALL,
        modality=PairModality.VISION,
        prompt="Extract this physical pin row.",
        response='{"pin_number":"1","pin_name":"PA0"}',
        source_claim_ids=(CLAIM_ID,),
        lineage_ids=(SHA,),
        image_uris=("file:///evidence/row.png",),
        image_sha256=("d" * 64,),
        teacher=teacher,
        disposition=PairDisposition.ADMITTED,
    )

    assert pair.purpose == "local_training_pair_generation"
    assert pair.teacher is not None
    assert pair.teacher.batch_id == "msgbatch_1"


def test_preference_pair_preserves_local_and_frontier_answers():
    teacher = ModelIdentity(
        provider="anthropic",
        model="claude-sonnet",
        request_sha256=SHA,
        response_id="msg_1",
        batch_id="msgbatch_1",
    )
    local = ModelIdentity(
        provider="local",
        model="qwen3-vl",
        request_sha256="b" * 64,
    )
    pair = PreferenceTrainingPairCandidate(
        pair_id="pair-" + ("d" * 32),
        capability=PairCapability.PIN_SEMANTICS,
        prompt="Extract the visible pin semantics.",
        chosen_response='{"pins":[{"pin_no":1,"name":"PA0"}]}',
        rejected_response='{"pins":[]}',
        chosen_source_sha256="c" * 64,
        rejected_source_sha256="d" * 64,
        source_claim_ids=(CLAIM_ID,),
        lineage_ids=(SHA,),
        image_uris=("file:///evidence/page.png",),
        image_sha256=("e" * 64,),
        local_model=local,
        teacher=teacher,
    )

    assert pair.training_format == "vision_dpo"
    assert pair.local_model.model == "qwen3-vl"
    assert pair.teacher.batch_id == "msgbatch_1"


def test_derived_recommendation_cannot_masquerade_as_visible_fact():
    with pytest.raises(ValidationError, match="require source claims"):
        FactClaim(
            claim_id=CLAIM_ID,
            entity=EntityReference(
                entity_id="family:acme:atom1",
                grain=EntityGrain.FAMILY,
                canonical_id="ATOM1",
                vendor="acme",
            ),
            field="competition.recommended_alternative",
            value="OTHER1",
            claim_class=ClaimClass.DERIVED_RECOMMENDATION,
            extraction_method="parametric_rule",
            evidence=(_evidence(),),
            created_at=datetime.now(timezone.utc),
        )


def test_admission_requires_passed_claim_verification():
    verification = ClaimVerification(
        verification_id="verify-" + ("e" * 32),
        claim_id=CLAIM_ID,
        claim_sha256=SHA,
        status=VerificationStatus.QUARANTINED,
        verifier="frontier_consensus",
        checks=("schema_valid",),
        evidence_sha256=(SHA,),
        reason="teacher disagrees with visible source",
        created_at=datetime.now(timezone.utc),
    )

    with pytest.raises(ValidationError, match="requires passed"):
        ClaimAdmission(
            admission_id="admit-" + ("f" * 32),
            claim_id=CLAIM_ID,
            claim_class=ClaimClass.SEMANTIC_LABEL,
            verification_id=verification.verification_id,
            verification_status=verification.status,
            status=AdmissionStatus.ADMITTED,
            policy="electronics-admission-v1",
            dataset_purposes=("local_training",),
            created_at=datetime.now(timezone.utc),
        )
