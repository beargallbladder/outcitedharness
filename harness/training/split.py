from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

from harness.training.models import FactValue


T = TypeVar("T")
Key = str | Callable[[T], Any]


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value < 0 for value in values):
            raise ValueError("split ratios cannot be negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to one")


def _value(item: T, key: Key[T]) -> Any:
    if callable(key):
        return key(item)
    if isinstance(item, Mapping):
        try:
            return item[key]
        except KeyError as exc:
            raise ValueError(f"record is missing split key {key!r}") from exc
    try:
        return getattr(item, key)
    except AttributeError as exc:
        raise ValueError(f"record is missing split key {key!r}") from exc


def _bucket(value: str, seed: str, ratios: SplitRatios) -> Split:
    raw = hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest()
    point = int.from_bytes(raw[:8], "big") / 2**64
    if point < ratios.train:
        return Split.TRAIN
    if point < ratios.train + ratios.validation:
        return Split.VALIDATION
    return Split.TEST


def grouped_lineage_split(
    records: Iterable[T],
    *,
    lineage_key: Key[T] = "lineage_id",
    seed: str = "harness-training-v1",
    ratios: SplitRatios = SplitRatios(),
) -> dict[Split, list[T]]:
    """Hash whole lineages, never individual examples, into stable splits."""

    output = {split: [] for split in Split}
    for record in records:
        lineage = str(_value(record, lineage_key)).strip()
        if not lineage:
            raise ValueError("lineage_id cannot be empty")
        output[_bucket(lineage, seed, ratios)].append(record)
    return output


def grouped_temporal_split(
    records: Iterable[T],
    *,
    lineage_key: Key[T] = "lineage_id",
    time_key: Key[T] = "observed_at",
    ratios: SplitRatios = SplitRatios(),
) -> dict[Split, list[T]]:
    """Split oldest-to-newest while keeping every lineage in one partition.

    A lineage is ordered by its newest observation. This intentionally moves an
    entire lineage later when it spans time, preventing train/test leakage.
    """

    groups: dict[str, list[tuple[datetime, T]]] = defaultdict(list)
    for record in records:
        lineage = str(_value(record, lineage_key)).strip()
        observed = _value(record, time_key)
        if not lineage:
            raise ValueError("lineage_id cannot be empty")
        if not isinstance(observed, datetime):
            raise ValueError("temporal split values must be datetime objects")
        if observed.tzinfo is None or observed.utcoffset() is None:
            raise ValueError("temporal split datetimes must include a timezone")
        groups[lineage].append((observed, record))

    ordered = sorted(
        groups.items(),
        key=lambda row: (
            max(observed for observed, _record in row[1]),
            hashlib.sha256(row[0].encode("utf-8")).hexdigest(),
        ),
    )
    total = sum(len(rows) for _lineage, rows in ordered)
    train_target = total * ratios.train
    validation_target = total * (ratios.train + ratios.validation)
    assigned = 0
    output = {split: [] for split in Split}
    for _lineage, rows in ordered:
        midpoint = assigned + len(rows) / 2
        if midpoint <= train_target:
            split = Split.TRAIN
        elif midpoint <= validation_target:
            split = Split.VALIDATION
        else:
            split = Split.TEST
        output[split].extend(record for _observed, record in rows)
        assigned += len(rows)
    return output


def assert_no_lineage_leakage(
    partitions: Mapping[Split | str, Iterable[T]],
    *,
    lineage_key: Key[T] = "lineage_id",
) -> None:
    owners: dict[str, str] = {}
    for split, records in partitions.items():
        split_name = split.value if isinstance(split, Split) else str(split)
        for record in records:
            raw_lineage = str(_value(record, lineage_key))
            lineage = raw_lineage.strip()
            if not lineage or lineage != raw_lineage:
                raise ValueError("lineage_id must be non-empty and canonical")
            previous = owners.setdefault(lineage, split_name)
            if previous != split_name:
                raise ValueError(
                    f"lineage {lineage!r} leaks across {previous} and {split_name}"
                )


def known_labels(
    records: Iterable[T],
    *,
    label_key: Key[T] = "label",
) -> list[T]:
    """Return only explicit labels; UNKNOWN is never interpreted as negative."""

    output: list[T] = []
    for record in records:
        raw = _value(record, label_key)
        label = raw if isinstance(raw, FactValue) else FactValue(str(raw))
        if label is not FactValue.UNKNOWN:
            output.append(record)
    return output
