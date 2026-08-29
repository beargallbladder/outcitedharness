from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from harness.training.security import assert_no_secrets, assert_value_no_secrets


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, Field(min_length=1)]


class SourceKind(str, Enum):
    HARNESS = "harness"
    DESIGNWINS = "designwins"
    GIT = "git"
    CATEGORYRANK = "categoryrank"
    OTHER = "other"


class DataUse(str, Enum):
    TRAINING = "training"
    RETRIEVAL_ONLY = "retrieval_only"
    QUARANTINE = "quarantine"


class FactValue(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class RunStatus(str, Enum):
    BUILDING = "building"
    COMPLETE = "complete"
    FAILED = "failed"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SourceProvenance(StrictModel):
    """Complete, immutable identity for a source record."""

    source_kind: SourceKind
    source_uri: NonEmpty
    source_record_id: NonEmpty
    collected_at: datetime
    content_sha256: Sha256
    lineage_id: NonEmpty
    license: NonEmpty
    revision: str | None = None
    mutable_facts: bool = False
    data_use: DataUse = DataUse.TRAINING

    @field_validator("collected_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("collected_at must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def mutable_facts_are_retrieval_only(self) -> SourceProvenance:
        if "://" not in self.source_uri:
            raise ValueError("source_uri must include a URI scheme")
        assert_no_secrets(self.source_uri, field="source_uri")
        if self.mutable_facts and self.data_use is not DataUse.RETRIEVAL_ONLY:
            raise ValueError("raw mutable facts must be marked retrieval_only")
        if self.source_kind is SourceKind.GIT and self.revision is not None:
            if (
                len(self.revision) not in {40, 64}
                or any(char not in "0123456789abcdef" for char in self.revision)
            ):
                raise ValueError("git revision must be a full lowercase commit hash")
        return self


class Artifact(StrictModel):
    artifact_id: NonEmpty
    kind: NonEmpty
    uri: NonEmpty
    sha256: Sha256
    byte_size: Annotated[int, Field(ge=0)]
    media_type: NonEmpty
    provenance: SourceProvenance
    data_use: DataUse = DataUse.TRAINING
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_source_restrictions(self) -> Artifact:
        assert_value_no_secrets(self.metadata, field="artifact.metadata")
        if self.provenance.data_use is not DataUse.TRAINING and (
            self.data_use is DataUse.TRAINING
        ):
            raise ValueError("artifact cannot be more permissive than its source")
        return self


class TrainingRun(StrictModel):
    run_id: NonEmpty
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None = None
    source_ids: tuple[NonEmpty, ...]
    artifact_ids: tuple[NonEmpty, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("started_at", "finished_at")
    @classmethod
    def timestamps_are_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("run timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def complete_run_has_end(self) -> TrainingRun:
        assert_value_no_secrets(self.metadata, field="run.metadata")
        if self.status is RunStatus.COMPLETE and self.finished_at is None:
            raise ValueError("complete run requires finished_at")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        return self


class TrainingManifest(StrictModel):
    schema_version: Literal[1] = 1
    manifest_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
    created_at: datetime
    run: TrainingRun
    sources: tuple[SourceProvenance, ...]
    artifacts: tuple[Artifact, ...]

    @field_validator("created_at")
    @classmethod
    def created_at_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def references_are_closed(self) -> TrainingManifest:
        source_ids = {row.source_record_id for row in self.sources}
        artifact_ids = {row.artifact_id for row in self.artifacts}
        if len(source_ids) != len(self.sources):
            raise ValueError("duplicate source_record_id")
        if len(artifact_ids) != len(self.artifacts):
            raise ValueError("duplicate artifact_id")
        missing_sources = set(self.run.source_ids) - source_ids
        missing_artifacts = set(self.run.artifact_ids) - artifact_ids
        if missing_sources or missing_artifacts:
            raise ValueError(
                f"manifest has dangling references: sources={sorted(missing_sources)}, "
                f"artifacts={sorted(missing_artifacts)}"
            )
        sources_by_id = {row.source_record_id: row for row in self.sources}
        for artifact in self.artifacts:
            canonical_source = sources_by_id.get(artifact.provenance.source_record_id)
            if canonical_source is None:
                raise ValueError("artifact provenance source is absent from manifest")
            if artifact.provenance != canonical_source:
                raise ValueError("artifact provenance differs from manifest source")
        return self


class TextPair(StrictModel):
    pair_id: NonEmpty
    prompt: NonEmpty
    response: NonEmpty
    provenance: SourceProvenance
    label: FactValue = FactValue.POSITIVE
    data_use: DataUse = DataUse.TRAINING
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_training_pair(self) -> TextPair:
        assert_no_secrets(self.prompt, field="prompt")
        assert_no_secrets(self.response, field="response")
        assert_value_no_secrets(self.metadata, field="pair.metadata")
        if self.provenance.data_use is not self.data_use:
            raise ValueError("pair data_use must match provenance data_use")
        return self


class VisionPair(TextPair):
    image_uris: tuple[NonEmpty, ...]
    image_sha256: tuple[Sha256, ...]

    @model_validator(mode="after")
    def image_provenance_is_complete(self) -> VisionPair:
        if not self.image_uris:
            raise ValueError("vision pair requires at least one image")
        if len(self.image_uris) != len(self.image_sha256):
            raise ValueError("each image URI requires a sha256")
        return self


class TestEvidence(StrictModel):
    command: NonEmpty
    status: Literal["pass", "fail", "unknown"]
    output_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def known_status_has_evidence(self) -> TestEvidence:
        assert_no_secrets(self.command, field="test command")
        if self.status != "unknown" and self.output_sha256 is None:
            raise ValueError("known test status requires output_sha256")
        return self


class GitCandidate(StrictModel):
    candidate_id: NonEmpty
    problem: NonEmpty
    patch: NonEmpty
    tests: tuple[TestEvidence, ...]
    provenance: SourceProvenance
    data_use: Literal[DataUse.QUARANTINE] = DataUse.QUARANTINE
    quarantine_reason: NonEmpty
    approved_for_training: Literal[False] = False

    @model_validator(mode="after")
    def quarantine_and_scan(self) -> GitCandidate:
        if self.provenance.source_kind is not SourceKind.GIT:
            raise ValueError("git candidate requires git provenance")
        if self.provenance.data_use is not DataUse.QUARANTINE:
            raise ValueError("git candidate provenance must be quarantined")
        if not self.patch.lstrip().startswith("diff --git "):
            raise ValueError("patch must be a unified git diff")
        assert_no_secrets(self.problem, field="problem")
        assert_no_secrets(self.patch, field="patch")
        return self
