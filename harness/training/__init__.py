"""Offline, provenance-first training data control plane.

This package has no production database integration. Raw mutable sources remain
retrieval-only, and generic git material remains quarantined until separately
reviewed.
"""

from harness.training.categoryrank import (
    CATEGORY_MENTIONS_SCHEMA,
    CATEGORY_SENTINELS,
    CategoryAlias,
    CategoryMentionV2,
    CategoryRankSource,
    CategorySuccessor,
    filter_category_mentions,
    is_category_sentinel,
    resolve_category_successor,
)
from harness.training.exporters import (
    ExportValidationError,
    export_harness_pass_pairs,
    extract_git_candidates,
    load_designwins_text_pairs,
    load_designwins_vision_pairs,
    load_native_designwins_text_pairs,
    load_native_designwins_vision_pairs,
    write_pairs_jsonl,
)
from harness.training.hygiene import (
    SecretDetectedError,
    assert_no_secrets,
    deduplicate,
    deduplicate_pairs,
    find_secrets,
    redact_text,
)
from harness.training.models import (
    Artifact,
    DataUse,
    FactValue,
    GitCandidate,
    RunStatus,
    SourceKind,
    SourceProvenance,
    TestEvidence,
    TextPair,
    TrainingManifest,
    TrainingRun,
    VisionPair,
)
from harness.training.registry import (
    ManifestConflictError,
    ManifestIntegrityError,
    ManifestRegistry,
    manifest_digest,
)
from harness.training.split import (
    Split,
    SplitRatios,
    assert_no_lineage_leakage,
    grouped_lineage_split,
    grouped_temporal_split,
    known_labels,
)

__all__ = [
    "Artifact",
    "CATEGORY_MENTIONS_SCHEMA",
    "CATEGORY_SENTINELS",
    "CategoryAlias",
    "CategoryMentionV2",
    "CategoryRankSource",
    "CategorySuccessor",
    "DataUse",
    "ExportValidationError",
    "FactValue",
    "GitCandidate",
    "ManifestConflictError",
    "ManifestIntegrityError",
    "ManifestRegistry",
    "RunStatus",
    "SecretDetectedError",
    "SourceKind",
    "SourceProvenance",
    "Split",
    "SplitRatios",
    "TestEvidence",
    "TextPair",
    "TrainingManifest",
    "TrainingRun",
    "VisionPair",
    "assert_no_lineage_leakage",
    "assert_no_secrets",
    "deduplicate",
    "deduplicate_pairs",
    "export_harness_pass_pairs",
    "extract_git_candidates",
    "filter_category_mentions",
    "find_secrets",
    "grouped_lineage_split",
    "grouped_temporal_split",
    "is_category_sentinel",
    "known_labels",
    "load_designwins_text_pairs",
    "load_designwins_vision_pairs",
    "load_native_designwins_text_pairs",
    "load_native_designwins_vision_pairs",
    "manifest_digest",
    "redact_text",
    "resolve_category_successor",
    "write_pairs_jsonl",
]
