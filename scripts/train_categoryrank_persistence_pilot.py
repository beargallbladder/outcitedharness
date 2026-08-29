#!/usr/bin/env python3
"""Train a de-identified temporal persistence pilot from CategoryRank retrieval data."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any


FEATURE_NAMES = (
    "log_mentions",
    "strength",
    "inverse_rank",
    "strength_missing",
    "rank_missing",
)


def _week_start(value: str) -> date:
    year, week = value.split("-W", 1)
    return date.fromisocalendar(int(year), int(week), 1)


def _features(row: dict[str, Any]) -> tuple[float, ...]:
    strength = row.get("avg_strength")
    rank = row.get("avg_rank")
    return (
        math.log1p(max(0, int(row["n_mentions"]))) / 5.0,
        float(strength) / 100.0 if strength is not None else 0.0,
        1.0 / max(1.0, float(rank)) if rank is not None else 0.0,
        1.0 if strength is None else 0.0,
        1.0 if rank is None else 0.0,
    )


def transition_examples(
    source: Path,
    *,
    through_week: str,
) -> Iterator[tuple[str, tuple[float, ...], int]]:
    """Yield week/features/next-week-presence without exposing identity values."""

    through = _week_start(through_week)
    previous_key: tuple[str, str, str] | None = None
    previous_week = ""
    previous_features: tuple[float, ...] | None = None
    last_sort_key: tuple[str, str, str, str] | None = None

    def emit(next_row_week: str | None) -> tuple[str, tuple[float, ...], int] | None:
        if previous_features is None or _week_start(previous_week) >= through:
            return None
        persists = int(
            next_row_week is not None
            and (_week_start(next_row_week) - _week_start(previous_week)).days == 7
        )
        return previous_week, previous_features, persists

    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row["brand_domain"]),
                str(row["kim_slug"]),
                str(row["model_id"]),
            )
            week = str(row["time_window"])
            sort_key = (*key, week)
            if last_sort_key is not None and sort_key <= last_sort_key:
                raise ValueError(
                    f"CategoryRank input is not strictly canonical-sorted at line {line_number}"
                )
            if previous_key is not None:
                example = emit(week if key == previous_key else None)
                if example is not None:
                    yield example
            previous_key = key
            previous_week = week
            previous_features = _features(row)
            last_sort_key = sort_key

    example = emit(None)
    if example is not None:
        yield example


def _batches(
    source: Path,
    *,
    through_week: str,
    test_feature_week: str,
    include_test: bool,
    batch_size: int,
) -> Iterator[tuple[list[tuple[float, ...]], list[float]]]:
    features: list[tuple[float, ...]] = []
    labels: list[float] = []
    for week, values, label in transition_examples(source, through_week=through_week):
        if (week == test_feature_week) is not include_test:
            continue
        features.append(values)
        labels.append(float(label))
        if len(features) >= batch_size:
            yield features, labels
            features, labels = [], []
    if features:
        yield features, labels


def _auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ranked[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.manual_seed(args.seed)
    model = torch.nn.Linear(len(FEATURE_NAMES), 1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    train_examples = 0
    losses: list[float] = []
    for epoch in range(args.epochs):
        for features, labels in _batches(
            args.source,
            through_week=args.through_week,
            test_feature_week=args.test_feature_week,
            include_test=False,
            batch_size=args.batch_size,
        ):
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(feature_tensor).squeeze(1)
            loss = loss_function(logits, label_tensor)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if epoch == 0:
                train_examples += len(labels)

    labels: list[int] = []
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for features, raw_labels in _batches(
            args.source,
            through_week=args.through_week,
            test_feature_week=args.test_feature_week,
            include_test=True,
            batch_size=args.batch_size,
        ):
            probabilities = torch.sigmoid(
                model(torch.tensor(features, dtype=torch.float32)).squeeze(1)
            )
            labels.extend(int(value) for value in raw_labels)
            scores.extend(float(value) for value in probabilities)
    if not labels:
        raise ValueError("no held-out CategoryRank transitions were found")
    predicted = [int(score >= 0.5) for score in scores]
    true_positive = sum(predict == label == 1 for predict, label in zip(predicted, labels))
    false_positive = sum(predict == 1 and label == 0 for predict, label in zip(predicted, labels))
    false_negative = sum(predict == 0 and label == 1 for predict, label in zip(predicted, labels))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    accuracy = sum(predict == label for predict, label in zip(predicted, labels)) / len(labels)
    positive_rate = sum(labels) / len(labels)
    summary = {
        "schema": "harness.categoryrank.persistence-pilot.v1",
        "identity_features": False,
        "source_data_use": "retrieval_only",
        "derived_model_data_use": "training",
        "through_week": args.through_week,
        "test_feature_week": args.test_feature_week,
        "train_examples": train_examples,
        "test_examples": len(labels),
        "epochs": args.epochs,
        "mean_train_loss": sum(losses) / len(losses),
        "test_positive_rate": positive_rate,
        "baseline_majority_accuracy": max(positive_rate, 1.0 - positive_rate),
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_auc": _auc(labels, scores),
        "feature_names": FEATURE_NAMES,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": summary["schema"],
            "feature_names": FEATURE_NAMES,
            "state_dict": model.state_dict(),
        },
        args.output,
    )
    _write_json(args.output.with_suffix(args.output.suffix + ".metrics.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--through-week", required=True)
    parser.add_argument("--test-feature-week", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("training parameters must be positive")
    if _week_start(args.test_feature_week) >= _week_start(args.through_week):
        parser.error("test feature week must precede through week")
    return args


def main() -> int:
    args = parse_args()
    summary = train(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
