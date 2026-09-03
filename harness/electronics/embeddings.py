"""Immutable BGE sidecars for admitted electronics claims."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from harness.electronics.claims import canonical_json
from harness.electronics.models import FactClaim


EMBEDDING_SCHEMA = "harness.electronics-embedding-sidecar.v1"


class Encoder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> Any: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def claim_embedding_text(claim: FactClaim) -> str:
    identity = claim.entity
    parts = [
        f"vendor: {identity.vendor or 'unknown'}",
        f"grain: {identity.grain.value}",
        f"entity: {identity.canonical_id}",
    ]
    if identity.family:
        parts.append(f"family: {identity.family}")
    if identity.package:
        parts.append(f"package: {identity.package}")
    parts.extend(
        [
            f"field: {claim.field}",
            "value: "
            + json.dumps(
                claim.value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        ]
    )
    if claim.unit:
        parts.append(f"unit: {claim.unit}")
    if claim.conditions:
        parts.append(
            "conditions: "
            + json.dumps(
                claim.conditions,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    return "\n".join(parts)


def seal_embedding_sidecar(
    destination: Path,
    claims: Iterable[FactClaim],
    *,
    encoder: Encoder,
    batch_size: int,
    source_receipts: dict[str, Any],
    created_at: datetime,
) -> dict[str, Any]:
    if not 1 <= batch_size <= 256:
        raise ValueError("batch_size must be between 1 and 256")
    output = destination.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"embedding sidecar already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    vectors_path = temporary / "vectors.f32"
    index_path = temporary / "index.jsonl"
    dimensions: int | None = None
    count = 0
    seen: set[str] = set()

    def flush(
        pending_claims: Sequence[FactClaim],
        pending_texts: list[str],
        vector_handle: Any,
        index_handle: Any,
    ) -> None:
        nonlocal count, dimensions
        if not pending_claims:
            return
        response = encoder.embed(pending_texts)
        vectors = response.vectors
        if len(vectors) != len(pending_claims):
            raise ValueError("encoder returned the wrong number of vectors")
        for claim, text, vector in zip(
            pending_claims,
            pending_texts,
            vectors,
            strict=True,
        ):
            values = tuple(float(item) for item in vector)
            if not values or not all(math.isfinite(item) for item in values):
                raise ValueError("encoder returned an invalid vector")
            if dimensions is None:
                dimensions = len(values)
            elif len(values) != dimensions:
                raise ValueError("encoder dimensions changed within sidecar")
            offset = vector_handle.tell()
            vector_handle.write(struct.pack(f"<{len(values)}f", *values))
            index_handle.write(
                canonical_json(
                    {
                        "claim_id": claim.claim_id,
                        "offset_bytes": offset,
                        "dimensions": len(values),
                        "text_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                    }
                )
                + b"\n"
            )
            count += 1

    try:
        pending_claims: list[FactClaim] = []
        pending_texts: list[str] = []
        with vectors_path.open("xb") as vector_handle, index_path.open(
            "xb"
        ) as index_handle:
            for claim in claims:
                if claim.claim_id in seen:
                    raise ValueError(f"duplicate embedding claim: {claim.claim_id}")
                seen.add(claim.claim_id)
                pending_claims.append(claim)
                pending_texts.append(claim_embedding_text(claim))
                if len(pending_claims) == batch_size:
                    flush(
                        pending_claims,
                        pending_texts,
                        vector_handle,
                        index_handle,
                    )
                    pending_claims = []
                    pending_texts = []
            flush(
                pending_claims,
                pending_texts,
                vector_handle,
                index_handle,
            )
            for handle in (vector_handle, index_handle):
                handle.flush()
                os.fsync(handle.fileno())
        if not count or dimensions is None:
            raise ValueError("cannot seal an empty embedding sidecar")
        artifacts = {}
        for path in (vectors_path, index_path):
            artifacts[path.name] = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        core = {
            "schema": EMBEDDING_SCHEMA,
            "model": encoder.model,
            "format": {
                "dtype": "float32",
                "endianness": "little",
                "dimensions": dimensions,
                "row_order": "index.jsonl",
            },
            "sources": source_receipts,
            "artifacts": artifacts,
            "counts": {"embeddings": count},
        }
        core["evidence_sha256"] = hashlib.sha256(canonical_json(core)).hexdigest()
        manifest = {"created_at": created_at.isoformat(), **core}
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("xb") as handle:
            handle.write(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        for path in (vectors_path, index_path, manifest_path):
            os.chmod(path, 0o444)
        os.chmod(temporary, 0o555)
        os.rename(temporary, output)
        return manifest
    except BaseException:
        try:
            os.chmod(temporary, 0o755)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "EMBEDDING_SCHEMA",
    "claim_embedding_text",
    "seal_embedding_sidecar",
]
