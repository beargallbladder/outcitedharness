from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


CATEGORIES = (
    "coding",
    "bugfix",
    "repo_question",
    "pdf_extraction",
    "vision",
    "structured_extraction",
    "engineering_reasoning",
    "classification",
    "data_cleanup",
    "other",
)

EvalType = Literal[
    "exact_text",
    "exact_json",
    "numeric_fields",
    "required_fields",
    "json_fields",
    "keyword_rubric",
    "regex",
    "command",
    "human",
]


class Historical(BaseModel):
    local_failed: bool | None = None
    frontier_succeeded: bool | None = None
    notes: str | None = None


class EvaluationSpec(BaseModel):
    type: EvalType
    file: str | None = None
    pattern: str | None = None
    flags: str | None = None
    ignore_order: bool = True
    normalize_whitespace: bool = True
    case_insensitive: bool = False
    fields: dict[str, Any] = Field(default_factory=dict)
    command: list[str] | str | None = None
    timeout_s: float = 30
    extract_json: bool = True
    tolerance: float | None = None
    groups: list[list[str]] = Field(default_factory=list)


class ReferenceAnswer(BaseModel):
    file: str | None = None
    text: str | None = None
    data: Any = None


class Case(BaseModel):
    id: str
    title: str
    category: str = "other"
    tags: list[str] = Field(default_factory=list)
    path: Path
    prompt: str
    notes: str = ""
    input_files: list[str] = Field(default_factory=list)
    reference_answer: ReferenceAnswer | None = None
    evaluation: EvaluationSpec
    historical: Historical = Field(default_factory=Historical)
    system_prompt: str | None = None

    @property
    def inputs_dir(self) -> Path:
        return self.path / "inputs"

    @property
    def expected_dir(self) -> Path:
        return self.path / "expected"
