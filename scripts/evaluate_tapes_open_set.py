#!/usr/bin/env python3
"""Reproduce Tapes-owned open-set encoder evaluations without updating weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha16(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def require_pin(path: Path, expected: str) -> str:
    actual = sha16(path)
    if actual != expected.lower():
        raise ValueError(f"pin mismatch for {path}: expected {expected}, got {actual}")
    return actual


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def encode_batch(
    endpoint: str,
    texts: list[str],
    *,
    batch_size: int,
    timeout: float,
) -> list[list[float]]:
    output: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        request = urllib.request.Request(
            endpoint,
            data=json.dumps({"texts": chunk, "batch_size": batch_size}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
        vectors = payload.get("embeddings") or payload.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != len(chunk):
            raise RuntimeError(
                f"encoder returned {len(vectors) if isinstance(vectors, list) else 'no'} "
                f"vectors for batch of {len(chunk)}"
            )
        output.extend(vectors)
        print(
            json.dumps(
                {"encoded": min(start + len(chunk), len(texts)), "total": len(texts)}
            ),
            flush=True,
        )
    return output


def _normalized_tensor(vectors: list[list[float]]):
    import torch

    tensor = torch.tensor(vectors, dtype=torch.float32)
    return torch.nn.functional.normalize(tensor, p=2, dim=1)


def evaluate_kim_tag(
    endpoint: str,
    mask_path: Path,
    split_path: Path,
    *,
    batch_size: int,
    timeout: float,
) -> dict[str, Any]:
    import torch

    mask = json.loads(mask_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    prototypes = mask.get("rows")
    held_out = split.get("rows")
    if not isinstance(prototypes, list) or not isinstance(held_out, list):
        raise ValueError("kim-tag mask or split has no rows list")
    if len(prototypes) != split.get("meta", {}).get("n_prototypes"):
        raise ValueError("kim-tag prototype count does not match frozen split")
    slug_ids = [str(row["slug_id"]) for row in prototypes]
    if len(slug_ids) != len(set(slug_ids)):
        raise ValueError("kim-tag prototype slugs are not unique")
    unknown = sorted({str(row["slug_id"]) for row in held_out} - set(slug_ids))
    if unknown:
        raise ValueError(f"kim-tag split references unknown slugs: {unknown[:5]}")

    prototype_vectors = _normalized_tensor(
        encode_batch(
            endpoint,
            [str(row["slug_canonical_text"]) for row in prototypes],
            batch_size=batch_size,
            timeout=timeout,
        )
    )
    keyword_vectors = _normalized_tensor(
        encode_batch(
            endpoint,
            [str(row["keyword"]) for row in held_out],
            batch_size=batch_size,
            timeout=timeout,
        )
    )
    scores = keyword_vectors @ prototype_vectors.T
    top3 = torch.topk(scores, k=3, dim=1).indices.tolist()
    expected = [slug_ids.index(str(row["slug_id"])) for row in held_out]
    top1_hits = [indices[0] == target for indices, target in zip(top3, expected)]
    top3_hits = [target in indices for indices, target in zip(top3, expected)]
    return {
        "samples": len(held_out),
        "prototypes": len(prototypes),
        "top1_accuracy": round(sum(top1_hits) / len(top1_hits), 4),
        "top3_accuracy": round(sum(top3_hits) / len(top3_hits), 4),
        "decisions": [
            {
                "keyword": str(row["keyword"]),
                "slug_id": str(row["slug_id"]),
                "top1_pred": slug_ids[indices[0]],
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
            }
            for row, indices, top1_hit, top3_hit in zip(
                held_out, top3, top1_hits, top3_hits
            )
        ],
    }


def _load_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if "_meta" in payload:
                if meta is not None:
                    raise ValueError(f"multiple metadata rows in {path}")
                meta = payload["_meta"]
            else:
                rows.append(payload)
    if meta is None or not rows:
        raise ValueError(f"missing metadata or rows in {path}")
    return meta, rows


def evaluate_retrieval(
    endpoint: str,
    split_path: Path,
    *,
    batch_size: int,
    timeout: float,
) -> dict[str, Any]:
    import torch

    meta, queries = _load_jsonl(split_path)
    pool = sorted(
        {
            str(text)
            for row in queries
            for text in (row.get("positive_doc_texts") or ())
            if isinstance(text, str) and text
        }
    )
    pool_normalized = [normalize(text) for text in pool]
    pool_vectors = _normalized_tensor(
        encode_batch(
            endpoint,
            pool,
            batch_size=batch_size,
            timeout=timeout,
        )
    )
    query_vectors = _normalized_tensor(
        encode_batch(
            endpoint,
            [str(row["query"]) for row in queries],
            batch_size=batch_size,
            timeout=timeout,
        )
    )
    scores = query_vectors @ pool_vectors.T
    if bool(meta.get("eval_protocol", {}).get("skip_self_match", True)):
        for index, row in enumerate(queries):
            query_normalized = normalize(str(row["query"]))
            for candidate, candidate_normalized in enumerate(pool_normalized):
                if candidate_normalized == query_normalized:
                    scores[index, candidate] = -torch.inf
    top10 = torch.topk(scores, k=min(10, len(pool)), dim=1).indices.tolist()

    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    by_domain: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "hit_at_1": 0, "hit_at_3": 0, "hit_at_5": 0, "hit_at_10": 0}
    )
    decisions: list[dict[str, Any]] = []
    for row, ranked in zip(queries, top10):
        positives = {
            normalize(str(text)) for text in (row.get("positive_doc_texts") or ())
        }
        rank = next(
            (
                index
                for index, candidate in enumerate(ranked, start=1)
                if pool_normalized[candidate] in positives
            ),
            None,
        )
        domain = str(row.get("domain") or "unknown")
        by_domain[domain]["n"] += 1
        for cutoff in hits:
            hit = int(rank is not None and rank <= cutoff)
            hits[cutoff] += hit
            by_domain[domain][f"hit_at_{cutoff}"] += hit
        decisions.append(
            {
                "query": str(row["query"]),
                "domain": domain,
                **{
                    f"hit_at_{cutoff}": int(rank is not None and rank <= cutoff)
                    for cutoff in hits
                },
            }
        )

    count = len(queries)
    return {
        "samples": count,
        "candidates": len(pool),
        "recall_at_1": round(hits[1] / count, 4),
        "recall_at_3": round(hits[3] / count, 4),
        "recall_at_5": round(hits[5] / count, 4),
        "recall_at_10": round(hits[10] / count, 4),
        "by_domain": {
            domain: {
                "n": values["n"],
                **{
                    f"recall_at_{cutoff}": round(
                        values[f"hit_at_{cutoff}"] / values["n"], 4
                    )
                    for cutoff in hits
                },
            }
            for domain, values in sorted(by_domain.items())
        },
        "decisions": decisions,
    }


def _expected_metrics(kim_result: Path, retrieval_result: Path) -> dict[str, Any]:
    kim_payload = json.loads(kim_result.read_text(encoding="utf-8"))
    retrieval_payload = json.loads(retrieval_result.read_text(encoding="utf-8"))
    kim = kim_payload["report"]
    retrieval = retrieval_payload["_meta"]
    return {
        "kim_tag": {
            "top1_accuracy": kim["top1_accuracy"],
            "top3_accuracy": kim["top3_accuracy"],
            "decisions": [
                {
                    "keyword": row["keyword"],
                    "slug_id": row["slug_id"],
                    "top1_pred": row["top1_pred"],
                    "top1_hit": row["top1_hit"],
                    "top3_hit": row["top3_hit"],
                }
                for row in kim_payload["per_keyword"]
            ],
        },
        "retrieval": {
            **retrieval["headline"],
            **{
                f"category_alignment_recall_at_{cutoff}": retrieval["by_domain"][
                    "category_alignment"
                ][f"hit_at_{cutoff}"]
                for cutoff in (1, 3, 5, 10)
            },
            "decisions": [
                {
                    "query": row["query"],
                    "domain": row["domain"],
                    **{
                        f"hit_at_{cutoff}": row[f"hit_at_{cutoff}"]
                        for cutoff in (1, 3, 5, 10)
                    },
                }
                for row in retrieval_payload["per_query"]
            ],
        },
    }


def _decision_changes(
    observed: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    *,
    identity_fields: tuple[str, ...],
    decision_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    def index(rows: list[dict[str, Any]]) -> dict[tuple[str, ...], dict[str, Any]]:
        indexed: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in rows:
            identity = tuple(str(row[field]) for field in identity_fields)
            if identity in indexed:
                raise ValueError(f"duplicate baseline decision identity: {identity}")
            indexed[identity] = row
        return indexed

    observed_index = index(observed)
    expected_index = index(expected)
    if observed_index.keys() != expected_index.keys():
        raise ValueError("observed and expected decision identities differ")
    changes: list[dict[str, Any]] = []
    for identity, expected_row in expected_index.items():
        observed_row = observed_index[identity]
        changed = {
            field: {
                "observed": observed_row[field],
                "expected": expected_row[field],
            }
            for field in decision_fields
            if observed_row[field] != expected_row[field]
        }
        if changed:
            changes.append({"identity": identity, "changed": changed})
    return changes


def _comparison(
    observed: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "kim_tag_top1": (
            observed["kim_tag"]["top1_accuracy"],
            expected["kim_tag"]["top1_accuracy"],
        ),
        "kim_tag_top3": (
            observed["kim_tag"]["top3_accuracy"],
            expected["kim_tag"]["top3_accuracy"],
        ),
        "retrieval_r1": (
            observed["retrieval"]["recall_at_1"],
            expected["retrieval"]["recall_at_1"],
        ),
        "retrieval_r3": (
            observed["retrieval"]["recall_at_3"],
            expected["retrieval"]["recall_at_3"],
        ),
        "retrieval_r5": (
            observed["retrieval"]["recall_at_5"],
            expected["retrieval"]["recall_at_5"],
        ),
        "retrieval_r10": (
            observed["retrieval"]["recall_at_10"],
            expected["retrieval"]["recall_at_10"],
        ),
        "category_alignment_r1": (
            observed["retrieval"]["by_domain"]["category_alignment"]["recall_at_1"],
            expected["retrieval"]["category_alignment_recall_at_1"],
        ),
        "category_alignment_r3": (
            observed["retrieval"]["by_domain"]["category_alignment"]["recall_at_3"],
            expected["retrieval"]["category_alignment_recall_at_3"],
        ),
        "category_alignment_r5": (
            observed["retrieval"]["by_domain"]["category_alignment"]["recall_at_5"],
            expected["retrieval"]["category_alignment_recall_at_5"],
        ),
        "category_alignment_r10": (
            observed["retrieval"]["by_domain"]["category_alignment"]["recall_at_10"],
            expected["retrieval"]["category_alignment_recall_at_10"],
        ),
    }
    kim_changes = _decision_changes(
        observed["kim_tag"]["decisions"],
        expected["kim_tag"]["decisions"],
        identity_fields=("keyword", "slug_id"),
        decision_fields=("top1_pred", "top1_hit", "top3_hit"),
    )
    retrieval_changes = _decision_changes(
        observed["retrieval"]["decisions"],
        expected["retrieval"]["decisions"],
        identity_fields=("query", "domain"),
        decision_fields=("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10"),
    )
    category_changes = [
        change
        for change in retrieval_changes
        if change["identity"][1] == "category_alignment"
    ]
    exact_metrics = all(actual == wanted for actual, wanted in checks.values())
    owner_accepted = (
        len(kim_changes) <= 1
        and len(retrieval_changes) <= 1
        and not category_changes
    )
    return {
        "exact_reproduction": exact_metrics
        and not kim_changes
        and not retrieval_changes,
        "owner_tolerance_accepted": owner_accepted,
        "acceptance_contract": (
            "cursor-tapes: <=1 kim-tag decision, <=1 retrieval query, "
            "exact category-alignment"
        ),
        "decision_changes": {
            "kim_tag": {"count": len(kim_changes), "rows": kim_changes[:10]},
            "retrieval": {
                "count": len(retrieval_changes),
                "rows": retrieval_changes[:10],
            },
            "category_alignment": {
                "count": len(category_changes),
                "rows": category_changes[:10],
            },
        },
        "metrics": {
            name: {
                "observed": actual,
                "expected": wanted,
                "delta": round(actual - wanted, 4),
            }
            for name, (actual, wanted) in checks.items()
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--kim-split", required=True, type=Path)
    parser.add_argument("--kim-baseline", required=True, type=Path)
    parser.add_argument("--retrieval-split", required=True, type=Path)
    parser.add_argument("--retrieval-baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--kim-batch-size", type=int)
    parser.add_argument("--retrieval-batch-size", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--mask-sha16", default="040d362ec678aa13")
    parser.add_argument("--kim-split-sha16", default="cbd43e800c428e18")
    parser.add_argument("--kim-baseline-sha16", default="dce4f6e27c610c53")
    parser.add_argument("--retrieval-split-sha16", default="c95f2a59a3acfca8")
    parser.add_argument("--retrieval-baseline-sha16", default="a6cc8dd9d94f3765")
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or (args.kim_batch_size is not None and args.kim_batch_size < 1)
        or (
            args.retrieval_batch_size is not None
            and args.retrieval_batch_size < 1
        )
        or not 1 <= args.timeout <= 600
    ):
        parser.error("batch size and timeout must be positive and bounded")
    if not args.endpoint.startswith(("http://", "https://")):
        parser.error("endpoint must be HTTP(S)")
    return args


def _historical_main() -> int:
    args = parse_args()
    pins = {
        "mask": require_pin(args.mask, args.mask_sha16),
        "kim_split": require_pin(args.kim_split, args.kim_split_sha16),
        "kim_baseline": require_pin(args.kim_baseline, args.kim_baseline_sha16),
        "retrieval_split": require_pin(
            args.retrieval_split, args.retrieval_split_sha16
        ),
        "retrieval_baseline": require_pin(
            args.retrieval_baseline, args.retrieval_baseline_sha16
        ),
    }
    observed = {
        "kim_tag": evaluate_kim_tag(
            args.endpoint,
            args.mask,
            args.kim_split,
            batch_size=args.kim_batch_size or args.batch_size,
            timeout=args.timeout,
        ),
        "retrieval": evaluate_retrieval(
            args.endpoint,
            args.retrieval_split,
            batch_size=args.retrieval_batch_size or args.batch_size,
            timeout=args.timeout,
        ),
    }
    expected = _expected_metrics(args.kim_baseline, args.retrieval_baseline)
    payload = {
        "endpoint": args.endpoint,
        "pins": pins,
        "observed": observed,
        "expected": expected,
        "comparison": _comparison(observed, expected),
    }
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pins": payload["pins"],
                "comparison": payload["comparison"],
                "observed": {
                    "kim_tag": {
                        key: value
                        for key, value in payload["observed"]["kim_tag"].items()
                        if key != "decisions"
                    },
                    "retrieval": {
                        key: value
                        for key, value in payload["observed"]["retrieval"].items()
                        if key != "decisions"
                    },
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["comparison"]["owner_tolerance_accepted"] else 1


def main() -> int:
    print(
        "CategoryRank/Tapes processing is suspended pending new owner guidance.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
