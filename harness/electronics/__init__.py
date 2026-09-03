"""Source-grounded electronics document processing."""

from .locator import (
    LocateResult,
    locate_pin_definition_pages,
    validate_physical_pin_truth,
)
from .models import (
    ClaimAdmission,
    ClaimClass,
    ClaimVerification,
    EntityGrain,
    EvidenceReference,
    FactClaim,
    TrainingPairCandidate,
)

__all__ = [
    "ClaimAdmission",
    "ClaimClass",
    "ClaimVerification",
    "EntityGrain",
    "EvidenceReference",
    "FactClaim",
    "LocateResult",
    "TrainingPairCandidate",
    "locate_pin_definition_pages",
    "validate_physical_pin_truth",
]
