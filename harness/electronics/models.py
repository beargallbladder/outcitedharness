"""Strict contracts for evidence-backed electronics extraction."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, Field(min_length=1)]


class EntityGrain(str, Enum):
    DOCUMENT = "document"
    SERIES = "series"
    FAMILY = "family"
    BASE_PART = "base_part"
    OPN = "opn"
    PACKAGE = "package"
    PIN_OR_BALL = "pin_or_ball"


class ClaimClass(str, Enum):
    VISIBLE_FACT = "visible_fact"
    SEMANTIC_LABEL = "semantic_label"
    DERIVED_RECOMMENDATION = "derived_recommendation"


class EvidenceKind(str, Enum):
    SOURCE_RECORD = "source_record"
    TEXT_SPAN = "text_span"
    TABLE_CELL = "table_cell"
    IMAGE_REGION = "image_region"


class PairCapability(str, Enum):
    PAGE_LOCATION = "page_location"
    PIN_OR_BALL = "pin_or_ball"
    PIN_SEMANTICS = "pin_semantics"
    PARAMETRICS = "parametrics"
    SERIES_SUMMARY = "series_summary"
    OPN_DECODER = "opn_decoder"


class PairModality(str, Enum):
    TEXT = "text"
    VISION = "vision"


class PairDisposition(str, Enum):
    CANDIDATE = "candidate"
    ADMITTED = "admitted"
    QUARANTINED = "quarantined"


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class AdmissionStatus(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


def _validate_json_value(value: Any, *, field: str) -> Any:
    def visit(item: Any, path: str) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{field}{path} contains a non-finite number")
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(f"{field}{path} contains an invalid object key")
                visit(child, f"{path}.{key}")
            return
        raise ValueError(f"{field}{path} is not JSON-compatible")

    visit(value, "")
    # This also catches objects with surprising JSON encoders.
    json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


class SourceDocument(StrictModel):
    schema_name: Literal["harness.electronics-source-document.v1"] = Field(
        default="harness.electronics-source-document.v1",
        alias="schema",
        serialization_alias="schema",
    )
    document_sha256: Sha256
    source_uri: NonEmpty
    byte_size: Annotated[int, Field(gt=0)]
    vendor: str | None = None
    media_type: Literal["application/pdf"] = "application/pdf"
    page_count: Annotated[int | None, Field(gt=0)] = None
    record_ids: tuple[NonEmpty, ...] = ()

    @field_validator("source_uri")
    @classmethod
    def require_uri(cls, value: str) -> str:
        if "://" not in value:
            raise ValueError("source_uri must include a URI scheme")
        return value

    @field_validator("record_ids")
    @classmethod
    def unique_record_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("record_ids must be unique")
        return value


class EntityReference(StrictModel):
    entity_id: Annotated[
        str,
        Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/#@+-]{0,255}$"),
    ]
    grain: EntityGrain
    canonical_id: NonEmpty
    vendor: str | None = None
    family: str | None = None
    package: str | None = None
    parent_ids: tuple[NonEmpty, ...] = ()

    @field_validator("parent_ids")
    @classmethod
    def unique_parents(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("parent_ids must be unique")
        return value

    @model_validator(mode="after")
    def package_grain_requires_package(self) -> EntityReference:
        if self.grain in {EntityGrain.PACKAGE, EntityGrain.PIN_OR_BALL}:
            if not self.package:
                raise ValueError("package and pin/ball entities require package identity")
        return self


class EvidenceReference(StrictModel):
    kind: EvidenceKind
    document_sha256: Sha256
    source_uri: NonEmpty
    artifact_sha256: Sha256 | None = None
    page_1based: Annotated[int | None, Field(gt=0)] = None
    bbox: tuple[float, float, float, float] | None = None
    table_index: Annotated[int | None, Field(ge=0)] = None
    row_index: Annotated[int | None, Field(ge=0)] = None
    column_index: Annotated[int | None, Field(ge=0)] = None
    quoted_text: str | None = None

    @field_validator("source_uri")
    @classmethod
    def require_uri(cls, value: str) -> str:
        if "://" not in value:
            raise ValueError("source_uri must include a URI scheme")
        return value

    @field_validator("bbox")
    @classmethod
    def valid_bbox(
        cls,
        value: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        x0, y0, x1, y1 = value
        if not all(math.isfinite(item) for item in value) or x1 <= x0 or y1 <= y0:
            raise ValueError("bbox must contain finite increasing coordinates")
        return value

    @model_validator(mode="after")
    def evidence_coordinates_are_complete(self) -> EvidenceReference:
        if self.kind is not EvidenceKind.SOURCE_RECORD and self.page_1based is None:
            raise ValueError("page-level evidence requires page_1based")
        if self.kind is EvidenceKind.TABLE_CELL and (
            self.table_index is None
            or self.row_index is None
            or self.column_index is None
        ):
            raise ValueError("table-cell evidence requires table, row, and column indexes")
        if self.kind is EvidenceKind.IMAGE_REGION and self.bbox is None:
            raise ValueError("image-region evidence requires bbox")
        if self.quoted_text is not None and not self.quoted_text.strip():
            raise ValueError("quoted_text cannot be blank")
        return self


class ModelIdentity(StrictModel):
    provider: NonEmpty
    model: NonEmpty
    revision: str | None = None
    request_sha256: Sha256
    response_id: str | None = None
    batch_id: str | None = None

    @model_validator(mode="after")
    def anthropic_teacher_uses_batch(self) -> ModelIdentity:
        if self.provider.casefold() == "anthropic" and not self.batch_id:
            raise ValueError("Anthropic teacher identity requires a batch_id")
        return self


class FactClaim(StrictModel):
    schema_name: Literal["harness.electronics-fact-claim.v1"] = Field(
        default="harness.electronics-fact-claim.v1",
        alias="schema",
        serialization_alias="schema",
    )
    claim_id: Annotated[
        str,
        Field(pattern=r"^claim-[0-9a-f]{32}$"),
    ]
    entity: EntityReference
    field: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]{0,127}$")]
    value: Any
    unit: str | None = None
    conditions: dict[str, Any] = Field(default_factory=dict)
    claim_class: ClaimClass
    extraction_method: NonEmpty
    evidence: tuple[EvidenceReference, ...] = ()
    source_claim_ids: tuple[Annotated[str, Field(pattern=r"^claim-[0-9a-f]{32}$")], ...] = ()
    model: ModelIdentity | None = None
    created_at: datetime

    @field_validator("value")
    @classmethod
    def json_value(cls, value: Any) -> Any:
        return _validate_json_value(value, field="value")

    @field_validator("conditions")
    @classmethod
    def json_conditions(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_value(value, field="conditions")

    @field_validator("created_at")
    @classmethod
    def aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator("source_claim_ids")
    @classmethod
    def unique_source_claims(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("source_claim_ids must be unique")
        return value

    @model_validator(mode="after")
    def enforce_claim_class(self) -> FactClaim:
        if self.claim_class in {
            ClaimClass.VISIBLE_FACT,
            ClaimClass.SEMANTIC_LABEL,
        } and not self.evidence:
            raise ValueError("visible and semantic claims require source evidence")
        if (
            self.claim_class is ClaimClass.DERIVED_RECOMMENDATION
            and not self.source_claim_ids
        ):
            raise ValueError("derived recommendations require source claims")
        if self.unit is not None and not self.unit.strip():
            raise ValueError("unit cannot be blank")
        if self.model is not None and not re.search(
            r"(model|vision|frontier|llm|vlm|teacher)",
            self.extraction_method,
            re.IGNORECASE,
        ):
            raise ValueError("model identity requires a model-based extraction method")
        return self


class TrainingPairCandidate(StrictModel):
    schema_name: Literal["harness.electronics-training-pair.v1"] = Field(
        default="harness.electronics-training-pair.v1",
        alias="schema",
        serialization_alias="schema",
    )
    pair_id: Annotated[str, Field(pattern=r"^pair-[0-9a-f]{32}$")]
    purpose: Literal["local_training_pair_generation"] = (
        "local_training_pair_generation"
    )
    capability: PairCapability
    modality: PairModality
    prompt: NonEmpty
    response: NonEmpty
    source_claim_ids: tuple[
        Annotated[str, Field(pattern=r"^claim-[0-9a-f]{32}$")], ...
    ]
    lineage_ids: tuple[NonEmpty, ...]
    image_uris: tuple[NonEmpty, ...] = ()
    image_sha256: tuple[Sha256, ...] = ()
    teacher: ModelIdentity | None = None
    disposition: PairDisposition = PairDisposition.CANDIDATE
    quarantine_reason: str | None = None

    @model_validator(mode="after")
    def validate_pair(self) -> TrainingPairCandidate:
        if not self.source_claim_ids or not self.lineage_ids:
            raise ValueError("training pairs require source claims and lineages")
        if len(set(self.source_claim_ids)) != len(self.source_claim_ids):
            raise ValueError("source_claim_ids must be unique")
        if len(set(self.lineage_ids)) != len(self.lineage_ids):
            raise ValueError("lineage_ids must be unique")
        if len(self.image_uris) != len(self.image_sha256):
            raise ValueError("each image URI requires a SHA-256")
        if self.modality is PairModality.VISION and not self.image_uris:
            raise ValueError("vision pairs require images")
        if self.modality is PairModality.TEXT and self.image_uris:
            raise ValueError("text pairs cannot include images")
        if self.disposition is PairDisposition.QUARANTINED:
            if not self.quarantine_reason:
                raise ValueError("quarantined pairs require a reason")
        elif self.quarantine_reason is not None:
            raise ValueError("only quarantined pairs may have a quarantine reason")
        return self


class PreferenceTrainingPairCandidate(StrictModel):
    """Vision preference pair: frontier-grounded answer over local attempt."""

    schema_name: Literal[
        "harness.electronics-preference-training-pair.v1"
    ] = Field(
        default="harness.electronics-preference-training-pair.v1",
        alias="schema",
        serialization_alias="schema",
    )
    pair_id: Annotated[str, Field(pattern=r"^pair-[0-9a-f]{32}$")]
    purpose: Literal["local_training_pair_generation"] = (
        "local_training_pair_generation"
    )
    training_format: Literal["vision_dpo"] = "vision_dpo"
    capability: PairCapability
    prompt: NonEmpty
    chosen_response: NonEmpty
    rejected_response: NonEmpty
    chosen_source_sha256: Sha256
    rejected_source_sha256: Sha256
    source_claim_ids: tuple[
        Annotated[str, Field(pattern=r"^claim-[0-9a-f]{32}$")], ...
    ]
    lineage_ids: tuple[NonEmpty, ...]
    image_uris: tuple[NonEmpty, ...]
    image_sha256: tuple[Sha256, ...]
    local_model: ModelIdentity
    teacher: ModelIdentity
    disposition: PairDisposition = PairDisposition.ADMITTED

    @model_validator(mode="after")
    def validate_preference_pair(self) -> PreferenceTrainingPairCandidate:
        if self.chosen_response == self.rejected_response:
            raise ValueError("chosen and rejected responses must differ")
        if not self.source_claim_ids or not self.lineage_ids:
            raise ValueError("preference pairs require claims and lineages")
        if not self.image_uris:
            raise ValueError("vision preference pairs require images")
        if len(self.image_uris) != len(self.image_sha256):
            raise ValueError("each image URI requires a SHA-256")
        if self.local_model.provider.casefold() != "local":
            raise ValueError("rejected response must come from a local model")
        if self.teacher.provider.casefold() != "anthropic":
            raise ValueError("chosen response must come from Anthropic batch")
        if self.disposition is not PairDisposition.ADMITTED:
            raise ValueError("only admitted preference pairs may be exported")
        return self


class ClaimVerification(StrictModel):
    schema_name: Literal["harness.electronics-claim-verification.v1"] = Field(
        default="harness.electronics-claim-verification.v1",
        alias="schema",
        serialization_alias="schema",
    )
    verification_id: Annotated[str, Field(pattern=r"^verify-[0-9a-f]{32}$")]
    claim_id: Annotated[str, Field(pattern=r"^claim-[0-9a-f]{32}$")]
    claim_sha256: Sha256
    status: VerificationStatus
    verifier: Literal[
        "deterministic",
        "cross_source",
        "frontier_consensus",
        "human",
    ]
    checks: tuple[NonEmpty, ...]
    evidence_sha256: tuple[Sha256, ...]
    reason: str | None = None
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def verification_has_decision_evidence(self) -> ClaimVerification:
        if not self.checks or not self.evidence_sha256:
            raise ValueError("verification requires checks and evidence hashes")
        if len(set(self.evidence_sha256)) != len(self.evidence_sha256):
            raise ValueError("evidence_sha256 must be unique")
        if self.status is VerificationStatus.PASSED:
            if self.reason is not None:
                raise ValueError("passed verification cannot include a failure reason")
        elif not self.reason:
            raise ValueError("failed or quarantined verification requires a reason")
        return self


class ClaimAdmission(StrictModel):
    schema_name: Literal["harness.electronics-claim-admission.v1"] = Field(
        default="harness.electronics-claim-admission.v1",
        alias="schema",
        serialization_alias="schema",
    )
    admission_id: Annotated[str, Field(pattern=r"^admit-[0-9a-f]{32}$")]
    claim_id: Annotated[str, Field(pattern=r"^claim-[0-9a-f]{32}$")]
    claim_class: ClaimClass
    verification_id: Annotated[str, Field(pattern=r"^verify-[0-9a-f]{32}$")]
    verification_status: VerificationStatus
    status: AdmissionStatus
    policy: NonEmpty
    dataset_purposes: tuple[
        Literal[
            "electronics_warehouse",
            "local_training",
            "frozen_evaluation",
            "cr_import",
            "embeddings",
        ],
        ...,
    ]
    reason: str | None = None
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def only_verified_claims_are_admitted(self) -> ClaimAdmission:
        if len(set(self.dataset_purposes)) != len(self.dataset_purposes):
            raise ValueError("dataset_purposes must be unique")
        if self.status is AdmissionStatus.ADMITTED:
            if self.verification_status is not VerificationStatus.PASSED:
                raise ValueError("admission requires passed claim verification")
            if not self.dataset_purposes:
                raise ValueError("admission requires at least one dataset purpose")
            if self.reason is not None:
                raise ValueError("admitted claims cannot include a rejection reason")
        elif not self.reason:
            raise ValueError("rejected or quarantined claims require a reason")
        return self


__all__ = [
    "AdmissionStatus",
    "ClaimClass",
    "ClaimAdmission",
    "ClaimVerification",
    "EntityGrain",
    "EntityReference",
    "EvidenceKind",
    "EvidenceReference",
    "FactClaim",
    "ModelIdentity",
    "PairCapability",
    "PairDisposition",
    "PairModality",
    "PreferenceTrainingPairCandidate",
    "SourceDocument",
    "TrainingPairCandidate",
    "VerificationStatus",
]
