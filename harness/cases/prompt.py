from __future__ import annotations

from dataclasses import dataclass

from harness.cases.loader import collect_binary_inputs, collect_text_evidence
from harness.cases.schema import Case
from harness.config import ModelConfig, Settings


@dataclass
class PromptPacket:
    system: str | None
    user: str
    skipped_binaries: list[str]


def build_prompt(case: Case, settings: Settings, model: ModelConfig) -> PromptPacket:
    """Build a semantically identical task packet for every model.

    Provider-specific wire formatting happens later. The task and evidence
    stay the same. Native image attachment is gated on capability flags.
    """
    parts = [case.prompt.strip()]
    evidence = collect_text_evidence(case)
    if evidence:
        parts.append("\n\n## Input evidence\n")
        for name, content in evidence:
            parts.append(f"\n### {name}\n\n```\n{content.rstrip()}\n```\n")

    binaries = collect_binary_inputs(case)
    skipped: list[str] = []
    if binaries and not model.capabilities.vision:
        skipped = [p.name for p in binaries]
        names = ", ".join(skipped)
        parts.append(
            "\n\n## Binary inputs (not attached)\n"
            f"This model is configured without native vision. "
            f"These files were not sent: {names}.\n"
            "Use the textual evidence above if present.\n"
        )

    system = case.system_prompt or settings.system_prompt or None
    return PromptPacket(system=system, user="".join(parts), skipped_binaries=skipped)
