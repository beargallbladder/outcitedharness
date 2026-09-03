#!/usr/bin/env python3
"""Admit verified DesignWins chunks and register frozen v4 splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from harness.storage.db import Store
from harness.training.ledger import (
    ArtifactPayload,
    LearningLedger,
    VerificationPayload,
)
from harness.training.models import LearningEvent, SourceKind, TextPair
from harness.training.queue import DatasetMember, DatasetVersionRegistry
from harness.training.registry import canonical_json
from harness.training.split import Split


MARKER = "\n\nPIN TABLE TEXT:\n"
EXPECTED_HOLDOUT_COUNTS = {Split.VALIDATION: 127, Split.TEST: 141}
POLICY_VERSION = "designwins-grounded-chunks-v1"
ADMISSION_REASON = (
    "deterministic source-grounded chunk of an admitted owned DesignWins pair"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_event_id(pair: TextPair) -> str:
    digest = hashlib.sha256(
        f"{pair.pair_id}\0{pair.provenance.content_sha256}".encode()
    ).hexdigest()
    return f"designwins-{digest[:32]}"


def _chunk_event_id(pair: TextPair, chunk_manifest_sha256: str) -> str:
    digest = hashlib.sha256(
        (
            f"{pair.pair_id}\0{pair.provenance.content_sha256}\0"
            f"{chunk_manifest_sha256}"
        ).encode()
    ).hexdigest()
    return f"designwins-v4-{digest[:32]}"


def _load_pairs(path: Path) -> list[TextPair]:
    pairs: list[TextPair] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                pairs.append(TextPair.model_validate_json(line))
    return pairs


def _pin_counter(
    response: str,
    *,
    require_unique_physical_pins: bool = True,
) -> Counter[bytes]:
    value = json.loads(response)
    pins = value.get("pins") if isinstance(value, dict) else None
    if not isinstance(pins, list) or not pins:
        raise ValueError("response must contain at least one pin")
    names = [pin.get("name") for pin in pins if isinstance(pin, dict)]
    if (
        len(names) != len(pins)
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("response pin names must be present")
    physical_pins = {
        (
            str(pin["name"]),
            json.dumps(pin.get("pin_no"), sort_keys=True, separators=(",", ":")),
        )
        for pin in pins
    }
    if require_unique_physical_pins and len(physical_pins) != len(pins):
        raise ValueError("response contains a duplicate physical pin")
    return Counter(canonical_json(pin) for pin in pins)


def validate_chunk(
    child: TextPair,
    parent: TextPair,
    llamafactory_record: dict[str, Any],
    *,
    cutoff_len: int,
) -> dict[str, Any]:
    if child.metadata.get("parent_pair_id") != parent.pair_id:
        raise ValueError("chunk parent identity mismatch")
    if child.data_use.value != "training" or parent.data_use.value != "training":
        raise ValueError("chunk and parent must be approved training records")
    if child.provenance.lineage_id != parent.provenance.lineage_id:
        raise ValueError("chunk lineage differs from its parent")
    for field in ("source_kind", "source_uri", "license", "revision"):
        if getattr(child.provenance, field) != getattr(parent.provenance, field):
            raise ValueError(f"chunk provenance {field} differs from its parent")
    expected_content_sha256 = hashlib.sha256(
        canonical_json({"prompt": child.prompt, "response": child.response})
    ).hexdigest()
    if child.provenance.content_sha256 != expected_content_sha256:
        raise ValueError("chunk content digest is invalid")
    if MARKER not in child.prompt or MARKER not in parent.prompt:
        raise ValueError("chunk or parent lacks pin-table source marker")
    start = child.metadata.get("source_start")
    end = child.metadata.get("source_end")
    tokens = child.metadata.get("sequence_tokens")
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or not isinstance(tokens, int)
        or start < 0
        or end <= start
        or tokens > cutoff_len
    ):
        raise ValueError("chunk boundaries or token count are invalid")
    parent_source = parent.prompt.split(MARKER, 1)[1]
    child_source = child.prompt.split(MARKER, 1)[1]
    if child_source != parent_source[start:end].strip():
        raise ValueError("chunk source is not the declared parent substring")
    child_pins = _pin_counter(child.response)
    parent_pins = _pin_counter(
        parent.response,
        require_unique_physical_pins=False,
    )
    if child_pins - parent_pins:
        raise ValueError("chunk response is not a subset of the parent response")
    for pin_bytes in child_pins:
        name = str(json.loads(pin_bytes).get("name") or "")
        if not name or name.casefold() not in child_source.casefold():
            raise ValueError("chunk response contains a source-ungrounded pin")
    if llamafactory_record != {
        "instruction": child.prompt,
        "input": "",
        "output": child.response,
    }:
        raise ValueError("canonical and LLaMA Factory records differ")
    return {
        "parent_pair_id": parent.pair_id,
        "parent_content_sha256": parent.provenance.content_sha256,
        "source_start": start,
        "source_end": end,
        "sequence_tokens": tokens,
    }


def _require_parent_admission(
    store: Store,
    pair: TextPair,
) -> tuple[str, str]:
    event_id = _source_event_id(pair)
    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT a.artifact_id
            FROM learning_admissions d
            JOIN learning_artifacts a ON a.event_id = d.event_id
            WHERE d.event_id = ? AND d.decision = 'eligible'
              AND a.kind = 'canonical_response'
            """,
            (event_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"parent pair {pair.pair_id!r} is not admitted")
    return event_id, str(row["artifact_id"])


def register(args: argparse.Namespace) -> dict[str, Any]:
    store = Store(args.database)
    ledger = LearningLedger(store, args.artifact_root)
    chunk_manifest = json.loads(args.chunk_manifest.read_text(encoding="utf-8"))
    sequence_audit = json.loads(args.sequence_audit.read_text(encoding="utf-8"))
    if chunk_manifest.get("schema") != "harness.dataset.designwins-chunked.v1":
        raise ValueError("unexpected chunk-manifest schema")
    if sequence_audit.get("schema") != "harness.designwins-sequence-length-audit.v1":
        raise ValueError("unexpected sequence-audit schema")
    chunk_sha256 = _sha256(args.chunked_train)
    llama_sha256 = _sha256(args.llamafactory_train)
    artifacts = chunk_manifest.get("artifacts", {})
    if (
        artifacts.get("canonical/train.jsonl", {}).get("sha256") != chunk_sha256
        or artifacts.get("llamafactory/designwins_text_train.json", {}).get(
            "sha256"
        )
        != llama_sha256
    ):
        raise ValueError("chunk manifest does not bind both training artifacts")
    if (
        chunk_manifest.get("source_sha256")
        != _sha256(args.source_dataset_root / "train.jsonl")
    ):
        raise ValueError("chunk manifest does not bind the admitted parent split")
    if sequence_audit.get("dataset_sha256") != llama_sha256:
        raise ValueError("sequence audit does not bind the LLaMA Factory dataset")
    cutoff = int(chunk_manifest.get("cutoff_len", 0))
    cutoff_result = sequence_audit.get("cutoffs", {}).get(str(cutoff), {})
    if (
        cutoff < 1
        or cutoff_result.get("truncated_records") != 0
        or cutoff_result.get("truncated_rate") != 0
    ):
        raise ValueError("sequence audit does not prove zero truncation")

    source_root = args.source_dataset_root
    train_parents = {
        pair.pair_id: pair
        for pair in _load_pairs(source_root / "train.jsonl")
    }
    chunks = _load_pairs(args.chunked_train)
    llama_records = json.loads(args.llamafactory_train.read_text(encoding="utf-8"))
    if (
        not isinstance(llama_records, list)
        or len(chunks) != len(llama_records)
        or len(chunks) != chunk_manifest.get("chunks")
        or len(chunks) != sequence_audit.get("records")
    ):
        raise ValueError("chunk record counts do not agree")

    members: list[DatasetMember] = []
    chunk_manifest_sha256 = _sha256(args.chunk_manifest)
    sequence_audit_sha256 = _sha256(args.sequence_audit)
    for child, llama_record in zip(chunks, llama_records, strict=True):
        parent_id = child.metadata.get("parent_pair_id")
        parent = train_parents.get(str(parent_id))
        if parent is None:
            raise ValueError(f"unknown chunk parent {parent_id!r}")
        proof = validate_chunk(
            child,
            parent,
            llama_record,
            cutoff_len=cutoff,
        )
        parent_event_id, _parent_artifact_id = _require_parent_admission(
            store, parent
        )
        proof.update(
            {
                "parent_event_id": parent_event_id,
                "chunk_manifest_sha256": chunk_manifest_sha256,
                "sequence_audit_sha256": sequence_audit_sha256,
            }
        )
        event_id = _chunk_event_id(child, chunk_manifest_sha256)
        capture = ledger.capture(
            LearningEvent(
                event_id=event_id,
                event_type="verified_designwins_chunk",
                source_kind=SourceKind.DESIGNWINS,
                source_uri=child.provenance.source_uri,
                source_revision=chunk_manifest_sha256,
                lineage_id=child.provenance.lineage_id,
                authorization_scope=child.provenance.license,
                created_at=child.provenance.collected_at,
                metadata={
                    "pair_id": child.pair_id,
                    "parent_pair_id": parent.pair_id,
                    "parent_event_id": parent_event_id,
                    "data_use": "training",
                    "disposition": "verified",
                    "sequence_tokens": proof["sequence_tokens"],
                },
            ),
            [
                ArtifactPayload(kind="prompt", content=child.prompt),
                ArtifactPayload(
                    kind="canonical_response",
                    content=child.response,
                    media_type="application/json",
                ),
                ArtifactPayload(
                    kind="transformation_proof",
                    content=json.dumps(proof, sort_keys=True),
                    media_type="application/json",
                ),
            ],
            [
                VerificationPayload(
                    kind="deterministic_grounded_chunk",
                    status="pass",
                    verifier="scripts/register_designwins_v4_dataset.py",
                    output_kind="transformation_proof",
                    metadata={
                        "chunk_manifest_sha256": chunk_manifest_sha256,
                        "sequence_audit_sha256": sequence_audit_sha256,
                    },
                )
            ],
        )
        ledger.admit_verified_event(
            event_id,
            capture.verifications[0].verification_id,
            policy_version=POLICY_VERSION,
            reason=ADMISSION_REASON,
        )
        response_artifact = next(
            row for row in capture.artifacts if row.kind == "canonical_response"
        )
        members.append(
            DatasetMember(
                event_id=event_id,
                artifact_id=response_artifact.artifact_id,
                split=Split.TRAIN,
                lineage_id=child.provenance.lineage_id,
                source_document_sha256=parent.provenance.content_sha256,
            )
        )

    counts = {Split.TRAIN.value: len(chunks)}
    for split, expected_count in EXPECTED_HOLDOUT_COUNTS.items():
        holdout = _load_pairs(source_root / f"{split.value}.jsonl")
        if len(holdout) != expected_count:
            raise ValueError(
                f"{split.value} has {len(holdout)} records; expected {expected_count}"
            )
        for pair in holdout:
            event_id, artifact_id = _require_parent_admission(store, pair)
            members.append(
                DatasetMember(
                    event_id=event_id,
                    artifact_id=artifact_id,
                    split=split,
                    lineage_id=pair.provenance.lineage_id,
                    source_document_sha256=pair.provenance.content_sha256,
                )
            )
        counts[split.value] = len(holdout)

    source_revision = hashlib.sha256(
        canonical_json(
            {
                "chunk_manifest_sha256": chunk_manifest_sha256,
                "sequence_audit_sha256": sequence_audit_sha256,
                "validation_sha256": _sha256(source_root / "validation.jsonl"),
                "test_sha256": _sha256(source_root / "test.jsonl"),
            }
        )
    ).hexdigest()
    dataset_manifest_sha256 = DatasetVersionRegistry(store).create(
        dataset_version_id=args.dataset_version_id,
        name="designwins-text",
        version="v4-20260831",
        source_revision=source_revision,
        split_policy={
            "kind": "parent_lineage_preserved_chunking",
            "leakage_keys": ["lineage_id", "source_document_sha256"],
            "chunk_manifest_sha256": chunk_manifest_sha256,
            "sequence_audit_sha256": sequence_audit_sha256,
        },
        members=members,
    )
    return {
        "schema": "harness.designwins-v4-registration.v1",
        "dataset_version_id": args.dataset_version_id,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_revision": source_revision,
        "counts": counts,
        "admitted_chunks": len(chunks),
        "chunk_manifest_sha256": chunk_manifest_sha256,
        "sequence_audit_sha256": sequence_audit_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--source-dataset-root", required=True, type=Path)
    parser.add_argument("--chunked-train", required=True, type=Path)
    parser.add_argument("--llamafactory-train", required=True, type=Path)
    parser.add_argument("--chunk-manifest", required=True, type=Path)
    parser.add_argument("--sequence-audit", required=True, type=Path)
    parser.add_argument(
        "--dataset-version-id",
        default="designwins-text-v4-20260831",
    )
    return parser.parse_args()


def main() -> int:
    result = register(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
