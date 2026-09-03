#!/usr/bin/env python3
"""Embed only claims explicitly admitted to the electronics sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from harness.electronics.admission import iter_claims
from harness.electronics.embeddings import seal_embedding_sidecar
from harness.electronics.models import ClaimAdmission
from harness.gci.encoder import EncoderResponse, PRODUCTION_ENCODER_URL, StrictEncoder
from harness.task.search import embed_texts, embedder_base_url


class SparkEncoder:
    """Read-only use of the configured free embed route; never semantic search."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed(self, texts: list[str]) -> EncoderResponse:
        started = time.perf_counter()
        vectors = embed_texts(
            texts,
            base_url=self.base_url,
            model=self.model,
        )
        return EncoderResponse(
            tuple(tuple(value for value in row) for row in vectors),
            (time.perf_counter() - started) * 1000,
        )


def _embedding_claim_ids(admission_root: Path) -> set[str]:
    output: set[str] = set()
    path = admission_root / "admissions.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                admission = ClaimAdmission.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"invalid admission at line {line_number}"
                ) from exc
            if "embeddings" in admission.dataset_purposes:
                output.add(admission.claim_id)
    if not output:
        raise ValueError("admission bundle contains no embedding claims")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-bundle", type=Path, required=True)
    parser.add_argument("--admission-bundle", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--encoder-contract",
        choices=("spark-private", "strict-loopback"),
        default="spark-private",
    )
    parser.add_argument("--endpoint")
    parser.add_argument("--model", default="bge-m3-cr-tapes-v1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    args = parser.parse_args()
    claim_bundle = args.claim_bundle.expanduser().resolve(strict=True)
    admission_bundle = args.admission_bundle.expanduser().resolve(strict=True)
    admission_manifest_path = admission_bundle / "manifest.json"
    admission_manifest = json.loads(
        admission_manifest_path.read_text(encoding="utf-8")
    )
    claim_ids = _embedding_claim_ids(admission_bundle)
    if args.encoder_contract == "strict-loopback":
        encoder = StrictEncoder(
            url=args.endpoint or PRODUCTION_ENCODER_URL,
            timeout=args.timeout_seconds,
            model=args.model,
        )
    else:
        encoder = SparkEncoder(
            args.endpoint or embedder_base_url(),
            args.model,
        )
    claims = (
        claim for claim in iter_claims(claim_bundle) if claim.claim_id in claim_ids
    )
    manifest = seal_embedding_sidecar(
        args.output_directory,
        claims,
        encoder=encoder,
        batch_size=args.batch_size,
        source_receipts={
            "claim_bundle": str(claim_bundle),
            "claim_bundle_evidence_sha256": admission_manifest["sources"][
                "claim_bundle"
            ]["evidence_sha256"],
            "admission_bundle": str(admission_bundle),
            "admission_manifest_sha256": hashlib.sha256(
                admission_manifest_path.read_bytes()
            ).hexdigest(),
            "admission_evidence_sha256": admission_manifest["evidence_sha256"],
        },
        created_at=datetime.now(timezone.utc),
    )
    if manifest["counts"]["embeddings"] != len(claim_ids):
        raise ValueError("not every admitted claim was embedded")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
