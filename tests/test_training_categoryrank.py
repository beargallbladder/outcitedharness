from __future__ import annotations

import pytest
from pydantic import ValidationError

from harness.training.categoryrank import (
    CategoryAlias,
    CategoryMentionV2,
    CategoryRankSource,
    CategorySuccessor,
    filter_category_mentions,
    resolve_category_successor,
)
from harness.training.models import FactValue, SourceProvenance


def _provenance() -> SourceProvenance:
    return SourceProvenance(
        source_kind="categoryrank",
        source_uri="snapshot://categoryrank/category_mentions_v2/2026-W34",
        source_record_id="snapshot-2026-W34",
        collected_at="2026-08-29T12:00:00Z",
        content_sha256="d" * 64,
        lineage_id="categoryrank-weekly",
        license="internal-retrieval",
        mutable_facts=True,
        data_use="retrieval_only",
    )


def _mention(category: str, **overrides) -> CategoryMentionV2:
    values = {
        "brand_id": 7,
        "model": "lane-a",
        "category": category,
        "kim_category_id": None,
        "time_window": "2026-W34",
    }
    values.update(overrides)
    return CategoryMentionV2(**values)


def test_sentinels_are_explicitly_filtered_and_stay_unknown():
    mentions = [
        _mention("__unknown__"),
        _mention(" n "),
        _mention("-UNKNOWN-"),
        _mention("unknown"),
        _mention("Laptops"),
    ]
    assert [row.category for row in filter_category_mentions(mentions)] == ["Laptops"]
    assert all(row.fact is FactValue.UNKNOWN for row in mentions[:4])
    with pytest.raises(ValidationError, match="must stay UNKNOWN"):
        _mention("__unknown__", fact="negative")


def test_categoryrank_source_is_retrieval_only_and_schema_bound():
    source = CategoryRankSource(
        provenance=_provenance(),
        mentions=(_mention("Laptops"),),
        aliases=(CategoryAlias(alias="notebooks", category_id="laptops"),),
        successors=(
            CategorySuccessor(
                predecessor_category_id="portable-computers",
                successor_category_id="laptops",
                effective_from="2026-W20",
            ),
        ),
    )
    assert source.schema_name == "category_mentions_v2"
    assert source.real_mentions()[0].category == "Laptops"

    raw = _provenance().model_dump()
    raw["mutable_facts"] = False
    with pytest.raises(ValidationError, match="marked mutable"):
        CategoryRankSource(provenance=raw, mentions=())


def test_alias_ranges_and_successor_graph_are_validated():
    with pytest.raises(ValidationError, match="precedes"):
        CategoryAlias(
            alias="old",
            category_id="new",
            valid_from="2026-W20",
            valid_through="2026-W10",
        )
    with pytest.raises(ValidationError, match="cycle"):
        CategoryRankSource(
            provenance=_provenance(),
            mentions=(),
            successors=(
                CategorySuccessor(
                    predecessor_category_id="a",
                    successor_category_id="b",
                    effective_from="2026-W10",
                ),
                CategorySuccessor(
                    predecessor_category_id="b",
                    successor_category_id="a",
                    effective_from="2026-W11",
                ),
            ),
        )


def test_successor_resolution_is_as_of():
    relations = [
        CategorySuccessor(
            predecessor_category_id="a",
            successor_category_id="b",
            effective_from="2026-W10",
        ),
        CategorySuccessor(
            predecessor_category_id="b",
            successor_category_id="c",
            effective_from="2026-W20",
        ),
    ]
    assert resolve_category_successor("a", relations, as_of="2026-W05") == "a"
    assert resolve_category_successor("a", relations, as_of="2026-W15") == "b"
    assert resolve_category_successor("a", relations, as_of="2026-W25") == "c"
