"""Datasheet coverage ontology: canonical pillars, states, and manifests.

Implements the accounting layer of the Datasheet Coverage Ontology PRD:
every processed document is scored not by "how much did we get" but by
"of what we targeted, what did we locate, extract, and validate".

The pillar ontology is adopted in full as canonical naming; extraction
lanes map onto pillars incrementally. States never exaggerate: a pillar
only reaches VALIDATED through grounded teacher verification, and
CONFLICT never silently collapses into VALIDATED.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable, Mapping

__all__ = [
    "CAPABILITY_PILLARS",
    "PILLARS",
    "STATES",
    "build_coverage_manifest",
]

# Canonical pillar ontology (PRD section 3). Priority weights per PRD
# section 9: P0=5, P1=2, P2=1.
PILLARS: dict[str, dict[str, Any]] = {
    "P01": {"name": "Document Identity & Provenance", "weight": 5},
    "P02": {"name": "Product Identity, OPN & Ordering", "weight": 5},
    "P03": {"name": "Family / Variant Matrix", "weight": 5},
    "P04": {"name": "Functional Features & Architecture", "weight": 5},
    "P05": {"name": "Package & Mechanical", "weight": 5},
    "P06": {"name": "Pin / Ball / Pad Mapping", "weight": 5},
    "P07": {"name": "Pin Function & Multiplexing Semantics", "weight": 5},
    "P08": {"name": "Absolute Maximum Ratings & Operating Envelope", "weight": 5},
    "P09": {"name": "DC Electrical Characteristics", "weight": 5},
    "P10": {"name": "AC, Dynamic & Timing Characteristics", "weight": 5},
    "P11": {"name": "Power, Reset, Clocking & Sequencing", "weight": 5},
    "P12": {"name": "Thermal & Derating", "weight": 5},
    "P13": {"name": "Functional-Block / Peripheral Performance", "weight": 5},
    "P14": {"name": "Memory, Configuration, Boot & Security", "weight": 2},
    "P15": {"name": "Typical Performance & Characterization Curves", "weight": 2},
    "P16": {"name": "Application & Design Implementation Guidance", "weight": 2},
    "P17": {"name": "Qualification, Reliability & Compliance", "weight": 2},
    "P18": {"name": "Ecosystem & Development Support", "weight": 1},
    "P19": {"name": "Revision & Change History", "weight": 2},
}

# Extraction lanes currently implemented, mapped to canonical pillars.
# "conflates" records known ontology debt for honest reporting.
CAPABILITY_PILLARS: dict[str, dict[str, Any]] = {
    "pin_or_ball": {"pillar": "P06", "conflates": []},
    "pin_semantics": {"pillar": "P07", "conflates": ["P06"]},
    "parametrics": {"pillar": "P09", "conflates": ["P08", "P10", "P13"]},
    "series_summary": {"pillar": "P04", "conflates": []},
    "opn_decoder": {"pillar": "P02", "conflates": []},
}

# Pillar state machine (PRD section 6), in order of progression.
STATES = (
    "NOT_APPLICABLE",
    "EXPECTED_NOT_FOUND",
    "LOCATED",
    "EXTRACTED",
    "NORMALIZED",
    "PROVENANCED",
    "VALIDATED",
    "CONFLICT",
)

_PROGRESSION = {name: rank for rank, name in enumerate(STATES[:-1])}


def _advance(current: str | None, proposed: str) -> str:
    """Move a pillar forward through the state machine.

    CONFLICT is sticky: once evidence disagrees, later successes must not
    silently mask it (PRD: CONFLICT never collapses to VALIDATED).
    """
    if current == "CONFLICT" or proposed == "CONFLICT":
        return "CONFLICT"
    if current is None:
        return proposed
    if _PROGRESSION[proposed] > _PROGRESSION[current]:
        return proposed
    return current


def build_coverage_manifest(
    work_items: Iterable[Mapping[str, Any]],
    local_results: Iterable[Mapping[str, Any]],
    candidates: Iterable[Mapping[str, Any]],
    verified_outcomes: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Roll sealed run artifacts into a per-document coverage manifest.

    State assignment is strictly evidence-backed:
    - LOCATED: a structurally gated work item targeted the document page.
    - EXTRACTED: the local model produced a parseable, schema-valid result.
    - CONFLICT: local verification found cross-source disagreement.
    - VALIDATED: a grounded teacher outcome admitted the extraction.
    Without teacher verification the maximum reachable state is EXTRACTED.
    """
    work_by_id: dict[str, Mapping[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}

    def _doc(sha: str) -> dict[str, Any]:
        return documents.setdefault(
            sha,
            {
                "document_sha256": sha,
                "vendor": None,
                "pillars": {},
            },
        )

    def _pillar(sha: str, capability: str) -> dict[str, Any]:
        mapping = CAPABILITY_PILLARS[capability]
        entry = _doc(sha)["pillars"].setdefault(
            mapping["pillar"],
            {
                "capability": capability,
                "conflates": mapping["conflates"],
                "state": None,
                "targeted_pages": 0,
                "extracted_pages": 0,
                "attempt_statuses": defaultdict(int),
                "validated_items": 0,
                "quarantined_items": 0,
            },
        )
        return entry

    for item in work_items:
        capability = str(item["capability"])
        if capability not in CAPABILITY_PILLARS:
            continue
        sha = str(item["document_sha256"])
        work_by_id[str(item["work_id"])] = item
        document = _doc(sha)
        if item.get("vendor"):
            document["vendor"] = item["vendor"]
        pillar = _pillar(sha, capability)
        pillar["targeted_pages"] += 1
        pillar["state"] = _advance(pillar["state"], "LOCATED")

    for row in local_results:
        item = work_by_id.get(str(row.get("work_id")))
        if item is None:
            continue
        pillar = _pillar(str(item["document_sha256"]), str(item["capability"]))
        pillar["extracted_pages"] += 1
        pillar["state"] = _advance(pillar["state"], "EXTRACTED")

    for candidate in candidates:
        sha = str(candidate.get("document_sha256", ""))
        capability = str(candidate.get("capability", ""))
        if capability not in CAPABILITY_PILLARS or sha not in documents:
            continue
        pillar = _pillar(sha, capability)
        for attempt in candidate.get("local_attempts", []):
            status = str(attempt.get("status", "unknown"))
            pillar["attempt_statuses"][status] += 1
            if status == "cross_source_disagreement":
                pillar["state"] = _advance(pillar["state"], "CONFLICT")

    for outcome in verified_outcomes:
        item = work_by_id.get(str(outcome.get("work_id")))
        sha = str(outcome.get("document_sha256", ""))
        capability = str(outcome.get("capability", ""))
        if item is not None:
            sha = str(item["document_sha256"])
            capability = str(item["capability"])
        if capability not in CAPABILITY_PILLARS or sha not in documents:
            continue
        pillar = _pillar(sha, capability)
        disposition = str(outcome.get("disposition", ""))
        if disposition == "admitted":
            pillar["validated_items"] += 1
            pillar["state"] = _advance(pillar["state"], "VALIDATED")
        elif disposition in {"quarantined", "rejected"}:
            pillar["quarantined_items"] += 1
            pillar["state"] = _advance(pillar["state"], "CONFLICT")

    # Serialize defaultdicts and compute the weighted coverage index.
    weighted_total = 0
    weighted_validated = 0
    weighted_extracted = 0
    state_counts: dict[str, int] = defaultdict(int)
    for document in documents.values():
        for pillar_id, pillar in document["pillars"].items():
            pillar["attempt_statuses"] = dict(
                sorted(pillar["attempt_statuses"].items())
            )
            weight = PILLARS[pillar_id]["weight"]
            weighted_total += weight
            state = pillar["state"]
            state_counts[state] += 1
            if state == "VALIDATED":
                weighted_validated += weight
            if state in {"EXTRACTED", "NORMALIZED", "PROVENANCED", "VALIDATED"}:
                weighted_extracted += weight

    return {
        "schema": "harness.electronics-coverage-manifest.v1",
        "pillar_ontology": {
            key: {"name": value["name"], "weight": value["weight"]}
            for key, value in PILLARS.items()
        },
        "capability_pillars": CAPABILITY_PILLARS,
        "documents": dict(sorted(documents.items())),
        "summary": {
            "documents": len(documents),
            "document_pillar_states": dict(sorted(state_counts.items())),
            "weighted_applicable": weighted_total,
            "weighted_extracted": weighted_extracted,
            "weighted_validated": weighted_validated,
            "extraction_coverage_index": (
                round(weighted_extracted / weighted_total, 4)
                if weighted_total
                else None
            ),
            "validated_coverage_index": (
                round(weighted_validated / weighted_total, 4)
                if weighted_total
                else None
            ),
        },
    }


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
