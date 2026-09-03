from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CapabilityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str = Field(min_length=1)
    verified_examples: int = Field(ge=0)
    checkpoint_format: str = Field(min_length=1)
    completed_gates: frozenset[str] = frozenset()
    evidence_sha256: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def evidence_digests_are_complete(self) -> CapabilityEvidence:
        for gate in self.completed_gates:
            digest = self.evidence_sha256.get(gate)
            if (
                digest is None
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(
                    f"completed gate {gate!r} requires a lowercase SHA-256 digest"
                )
        return self


class CapabilityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str
    qualified: bool
    missing_gates: tuple[str, ...]
    reasons: tuple[str, ...]


def qualify_target(
    ladder: dict[str, Any],
    evidence: CapabilityEvidence,
) -> CapabilityDecision:
    targets = ladder.get("targets")
    if not isinstance(targets, dict) or evidence.target not in targets:
        raise KeyError(evidence.target)
    target = targets[evidence.target]
    required = tuple(str(gate) for gate in target.get("required_gates") or ())
    missing = tuple(gate for gate in required if gate not in evidence.completed_gates)
    reasons: list[str] = []
    minimum_examples = int(target["minimum_verified_examples"])
    if evidence.verified_examples < minimum_examples:
        reasons.append(
            f"requires {minimum_examples} verified examples; "
            f"found {evidence.verified_examples}"
        )
    expected_format = str(target["checkpoint_format"])
    if evidence.target == "qwen3_coder_next_80b":
        lowered = evidence.checkpoint_format.casefold()
        if "nvfp4" in lowered or "serving" in lowered:
            reasons.append("NVFP4 serving artifacts are not trainable checkpoints")
        if evidence.checkpoint_format not in {
            "bf16_trainable",
            "fp16_trainable",
            "bnb_4bit_trainable",
        }:
            reasons.append(
                f"checkpoint format {evidence.checkpoint_format!r} "
                "is not an approved 80B trainable format"
            )
        if target.get("distributed_required") and "two_node_fsdp_qlora" in missing:
            reasons.append("two-node FSDP-QLoRA has not passed load/step/save/resume")
    elif expected_format == "trainable_or_supported_4bit" and (
        evidence.checkpoint_format
        not in {
            "bf16_trainable",
            "fp16_trainable",
            "supported_4bit",
            "trainable_or_supported_4bit",
        }
    ):
        reasons.append(
            f"checkpoint format {evidence.checkpoint_format!r} "
            "is not a supported trainable format"
        )
    elif expected_format != "trainable_or_supported_4bit" and (
        evidence.checkpoint_format != expected_format
    ):
        reasons.append(
            f"checkpoint format {evidence.checkpoint_format!r} "
            f"does not match {expected_format!r}"
        )
    if missing:
        reasons.append(f"missing gates: {', '.join(missing)}")
    return CapabilityDecision(
        target=evidence.target,
        qualified=not reasons,
        missing_gates=missing,
        reasons=tuple(reasons),
    )
