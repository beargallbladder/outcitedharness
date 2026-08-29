from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from harness.training.hygiene import (
    SecretDetectedError,
    assert_no_secrets,
    content_fingerprint,
    deduplicate,
    redact_text,
)
from harness.training.models import FactValue
from harness.training.split import (
    Split,
    SplitRatios,
    assert_no_lineage_leakage,
    grouped_lineage_split,
    grouped_temporal_split,
    known_labels,
)


def test_lineage_split_is_deterministic_and_has_no_leakage():
    rows = [
        {"id": index, "lineage_id": f"family-{index // 2}"}
        for index in range(40)
    ]
    first = grouped_lineage_split(rows, seed="fixed")
    second = grouped_lineage_split(reversed(rows), seed="fixed")
    first_assignment = {
        row["id"]: split for split, records in first.items() for row in records
    }
    second_assignment = {
        row["id"]: split for split, records in second.items() for row in records
    }
    assert first_assignment == second_assignment
    assert_no_lineage_leakage(first)


def test_temporal_split_keeps_lineages_together_and_time_ordered():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "lineage_id": f"family-{index}",
            "observed_at": start + timedelta(days=index),
        }
        for index in range(10)
    ]
    split = grouped_temporal_split(
        rows,
        ratios=SplitRatios(train=0.6, validation=0.2, test=0.2),
    )
    assert_no_lineage_leakage(split)
    assert max(row["observed_at"] for row in split[Split.TRAIN]) < min(
        row["observed_at"] for row in split[Split.TEST]
    )
    assert len(split[Split.TRAIN]) == 6
    assert len(split[Split.VALIDATION]) == 2
    assert len(split[Split.TEST]) == 2


def test_unknown_is_excluded_not_coerced_to_negative():
    rows = [
        {"id": 1, "label": FactValue.POSITIVE},
        {"id": 2, "label": FactValue.NEGATIVE},
        {"id": 3, "label": FactValue.UNKNOWN},
    ]
    assert [row["id"] for row in known_labels(rows)] == [1, 2]


def test_redaction_dedupe_and_secret_rejection():
    assert redact_text("mail me at person@example.com") == "mail me at [REDACTED_EMAIL]"
    assert content_fingerprint(" Café\n") == content_fingerprint("café")
    assert deduplicate(["Alpha", " alpha  ", "Beta"], key=lambda value: value) == [
        "Alpha",
        "Beta",
    ]
    with pytest.raises(SecretDetectedError, match="aws_access_key"):
        assert_no_secrets("AWS key AKIA1234567890ABCDEF")
    with pytest.raises(SecretDetectedError, match="private_key"):
        assert_no_secrets("-----BEGIN PRIVATE KEY-----\nabc")


def test_bad_ratios_and_naive_times_are_rejected():
    with pytest.raises(ValueError, match="sum"):
        SplitRatios(train=0.5, validation=0.2, test=0.2)
    with pytest.raises(ValueError, match="timezone"):
        grouped_temporal_split(
            [{"lineage_id": "a", "observed_at": datetime(2026, 1, 1)}]
        )
