#!/usr/bin/env python3
"""Compare local BGE checkpoints on the owner-pinned contrastive holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_sample(path: Path, *, per_type: int, seed: int) -> list[tuple[Any, ...]]:
    by_type: dict[str, list[tuple[str, str, list[str]]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query = row.get("text") or row.get("query")
            positive = row.get("positive_label") or (row.get("pos") or [None])[0]
            negatives: list[str] = []
            if row.get("hard_negative_label"):
                negatives.append(str(row["hard_negative_label"]))
            negatives.extend((row.get("metadata") or {}).get("extra_negatives") or [])
            negatives.extend(row.get("neg") or [])
            negatives = [
                str(value)
                for value in negatives
                if value and str(value) != str(positive)
            ][:4]
            if query and positive and negatives:
                label_type = str(row.get("label_type") or row.get("type") or "?")
                by_type[label_type].append(
                    (str(query), str(positive), negatives)
                )
    rng = random.Random(seed)
    sample: list[tuple[Any, ...]] = []
    for label_type, rows in sorted(by_type.items()):
        rng.shuffle(rows)
        sample.extend((label_type, *row) for row in rows[:per_type])
    if not sample:
        raise ValueError("holdout produced no evaluable rows")
    return sample


def evaluate_model(
    path: Path,
    sample: list[tuple[Any, ...]],
    *,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from FlagEmbedding import BGEM3FlagModel

    started = time.monotonic()
    model = BGEM3FlagModel(
        str(path),
        use_fp16=device != "cpu",
        device=device,
    )
    texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for _label_type, query, positive, negatives in sample:
        start = len(texts)
        texts.extend([query, positive, *negatives])
        spans.append((start, len(negatives)))
    output = model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    embeddings = np.asarray(output["dense_vecs"], dtype=np.float32)
    embeddings /= np.clip(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None
    )
    hits: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for (label_type, *_), (start, negative_count) in zip(sample, spans):
        query_vector = embeddings[start]
        positive_score = float(query_vector @ embeddings[start + 1])
        negative_scores = [
            float(query_vector @ embeddings[start + 2 + index])
            for index in range(negative_count)
        ]
        counts[label_type] += 1
        hits[label_type] += int(positive_score > max(negative_scores))
    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    total = sum(counts.values())
    total_hits = sum(hits.values())
    return {
        "checkpoint": str(path),
        "seconds": round(time.monotonic() - started, 3),
        "overall": {
            "hits": total_hits,
            "samples": total,
            "positive_at_1": round(total_hits / total, 6),
        },
        "by_type": {
            label_type: {
                "hits": hits[label_type],
                "samples": count,
                "positive_at_1": round(hits[label_type] / count, 6),
            }
            for label_type, count in sorted(counts.items())
        },
    }


def comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    by_type = {
        label_type: round(
            candidate["by_type"][label_type]["positive_at_1"]
            - baseline["by_type"][label_type]["positive_at_1"],
            6,
        )
        for label_type in baseline["by_type"]
    }
    return {
        "overall_delta": round(
            candidate["overall"]["positive_at_1"]
            - baseline["overall"]["positive_at_1"],
            6,
        ),
        "by_type_delta": by_type,
        "candidate_improves_without_type_regression": (
            candidate["overall"]["positive_at_1"]
            > baseline["overall"]["positive_at_1"]
            and all(delta >= 0 for delta in by_type.values())
        ),
        "note": "Evidence only; owner open-set and task-specific gates still apply.",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--holdout-sha256", required=True)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-type", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if file_sha256(args.holdout) != args.holdout_sha256.lower():
        parser.error("holdout SHA-256 does not match the owner pin")
    for path in (args.baseline, args.candidate):
        if not path.is_dir():
            parser.error(f"checkpoint directory does not exist: {path}")
    if not 1 <= args.per_type <= 10_000 or not 1 <= args.batch_size <= 512:
        parser.error("evaluation limits are invalid")
    sample = load_sample(args.holdout, per_type=args.per_type, seed=args.seed)
    baseline = evaluate_model(
        args.baseline, sample, batch_size=args.batch_size, device=args.device
    )
    candidate = evaluate_model(
        args.candidate, sample, batch_size=args.batch_size, device=args.device
    )
    payload = {
        "schema": "harness.bge-holdout-evaluation.v1",
        "holdout": {
            "path": str(args.holdout),
            "sha256": args.holdout_sha256.lower(),
            "per_type": args.per_type,
            "seed": args.seed,
            "sample_identity_sha256": hashlib.sha256(
                "\n".join(
                    f"{label_type}\0{query}\0{positive}"
                    for label_type, query, positive, _negatives in sample
                ).encode()
            ).hexdigest(),
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison(baseline, candidate),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
