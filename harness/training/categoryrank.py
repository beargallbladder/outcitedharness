from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from harness.training.models import (
    DataUse,
    FactValue,
    NonEmpty,
    SourceKind,
    SourceProvenance,
    StrictModel,
)


CATEGORY_MENTIONS_SCHEMA = "category_mentions_v2"
CATEGORY_SENTINELS = frozenset({"__unknown__", "n", "-unknown-", "unknown"})
_ISO_WEEK = re.compile(r"^\d{4}-W(?:0[1-9]|[1-4]\d|5[0-3])$")


class CategoryMentionV2(StrictModel):
    """Retrieval-only representation of a raw category_mentions_v2 row."""

    brand_id: int | NonEmpty
    model: NonEmpty
    category: NonEmpty
    kim_category_id: int | str | None = None
    time_window: Annotated[str, Field(pattern=_ISO_WEEK.pattern)]
    strength: float | None = None
    rank: Annotated[int, Field(ge=1)] | None = None
    fact: FactValue = FactValue.UNKNOWN
    mutable_facts: Literal[True] = True
    data_use: Literal[DataUse.RETRIEVAL_ONLY] = DataUse.RETRIEVAL_ONLY

    @property
    def is_sentinel(self) -> bool:
        return self.category.strip().casefold() in CATEGORY_SENTINELS

    @model_validator(mode="after")
    def sentinel_stays_unknown(self) -> CategoryMentionV2:
        if self.is_sentinel:
            if self.fact is not FactValue.UNKNOWN:
                raise ValueError("coverage sentinel must stay UNKNOWN")
            if self.kim_category_id is not None:
                raise ValueError("coverage sentinel cannot have kim_category_id")
        return self


class CategoryAlias(StrictModel):
    alias: NonEmpty
    category_id: int | NonEmpty
    valid_from: Annotated[str, Field(pattern=_ISO_WEEK.pattern)] | None = None
    valid_through: Annotated[str, Field(pattern=_ISO_WEEK.pattern)] | None = None

    @model_validator(mode="after")
    def valid_range(self) -> CategoryAlias:
        if (
            self.valid_from is not None
            and self.valid_through is not None
            and self.valid_through < self.valid_from
        ):
            raise ValueError("alias valid_through precedes valid_from")
        return self


class CategorySuccessor(StrictModel):
    predecessor_category_id: int | NonEmpty
    successor_category_id: int | NonEmpty
    effective_from: Annotated[str, Field(pattern=_ISO_WEEK.pattern)]

    @model_validator(mode="after")
    def no_self_successor(self) -> CategorySuccessor:
        if str(self.predecessor_category_id) == str(self.successor_category_id):
            raise ValueError("category cannot succeed itself")
        return self


class CategoryRankSource(StrictModel):
    """Offline source envelope. It deliberately has no database connector."""

    schema_name: Literal["category_mentions_v2"] = CATEGORY_MENTIONS_SCHEMA
    provenance: SourceProvenance
    mentions: tuple[CategoryMentionV2, ...]
    aliases: tuple[CategoryAlias, ...] = ()
    successors: tuple[CategorySuccessor, ...] = ()

    @field_validator("provenance")
    @classmethod
    def source_is_retrieval_only(
        cls, value: SourceProvenance
    ) -> SourceProvenance:
        if value.source_kind is not SourceKind.CATEGORYRANK:
            raise ValueError("CategoryRank source_kind is required")
        if not value.mutable_facts:
            raise ValueError("CategoryRank raw facts must be marked mutable")
        if value.data_use is not DataUse.RETRIEVAL_ONLY:
            raise ValueError("CategoryRank raw facts are retrieval-only")
        return value

    @model_validator(mode="after")
    def relation_graph_is_valid(self) -> CategoryRankSource:
        alias_keys: set[tuple[str, str | None, str | None]] = set()
        for alias in self.aliases:
            key = (alias.alias.casefold(), alias.valid_from, alias.valid_through)
            if key in alias_keys:
                raise ValueError(f"duplicate alias interval for {alias.alias!r}")
            alias_keys.add(key)

        edges = {
            str(edge.predecessor_category_id): str(edge.successor_category_id)
            for edge in self.successors
        }
        if len(edges) != len(self.successors):
            raise ValueError("a category has multiple successor relations")
        for start in edges:
            seen: set[str] = set()
            current = start
            while current in edges:
                if current in seen:
                    raise ValueError("category successor relations contain a cycle")
                seen.add(current)
                current = edges[current]
        return self

    def real_mentions(self) -> tuple[CategoryMentionV2, ...]:
        return tuple(filter_category_mentions(self.mentions))


def is_category_sentinel(category: str) -> bool:
    return category.strip().casefold() in CATEGORY_SENTINELS


def filter_category_mentions(
    mentions: Iterable[CategoryMentionV2],
) -> list[CategoryMentionV2]:
    """Exclude both current and legacy coverage sentinels explicitly."""

    return [mention for mention in mentions if not mention.is_sentinel]


def resolve_category_successor(
    category_id: int | str,
    relations: Iterable[CategorySuccessor],
    *,
    as_of: str,
) -> str:
    if not _ISO_WEEK.fullmatch(as_of):
        raise ValueError("as_of must be an ISO week")
    current = str(category_id)
    applicable = sorted(
        (edge for edge in relations if edge.effective_from <= as_of),
        key=lambda edge: edge.effective_from,
    )
    seen: set[str] = set()
    while True:
        if current in seen:
            raise ValueError("category successor relations contain a cycle")
        seen.add(current)
        edge = next(
            (
                candidate
                for candidate in reversed(applicable)
                if str(candidate.predecessor_category_id) == current
            ),
            None,
        )
        if edge is None:
            return current
        current = str(edge.successor_category_id)
