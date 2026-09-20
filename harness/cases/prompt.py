from __future__ import annotations

import base64
from dataclasses import dataclass, field

from harness.cases.loader import collect_binary_inputs, collect_text_evidence
from harness.cases.schema import Case
from harness.config import ModelConfig, Settings
from harness.providers.base import ImageAttachment


IMAGE_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


@dataclass
class PromptPacket:
    system: str | None
    user: str
    skipped_binaries: list[str]
    images: list[ImageAttachment] = field(default_factory=list)


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
    images: list[ImageAttachment] = []
    if binaries and not model.capabilities.vision:
        skipped = [p.name for p in binaries]
        names = ", ".join(skipped)
        parts.append(
            "\n\n## Binary inputs (not attached)\n"
            f"This model is configured without native vision. "
            f"These files were not sent: {names}.\n"
            "Use the textual evidence above if present.\n"
        )
    elif binaries:
        for path in binaries:
            mime = IMAGE_MIME_BY_SUFFIX.get(path.suffix.lower())
            if mime is None:
                skipped.append(path.name)
                continue
            data_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            images.append(ImageAttachment(mime_type=mime, data_b64=data_b64))
        if skipped:
            names = ", ".join(skipped)
            parts.append(
                "\n\n## Binary inputs (not attached)\n"
                f"Image files were sent as native attachments; "
                f"these non-image files were not sent: {names}.\n"
            )

    system = case.system_prompt or settings.system_prompt or None
    return PromptPacket(
        system=system,
        user="".join(parts),
        skipped_binaries=skipped,
        images=images,
    )
