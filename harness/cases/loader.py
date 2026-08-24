from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from harness.cases.schema import Case, EvaluationSpec, ReferenceAnswer


TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".py",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".rs",
    ".go",
    ".js",
    ".ts",
    ".toml",
    ".ini",
    ".csv",
    ".log",
    ".xml",
    ".patch",
    ".diff",
}


def load_case(path: Path) -> Case:
    case_dir = path.resolve()
    spec_path = case_dir / "case.yaml"
    if not spec_path.exists():
        raise FileNotFoundError(f"No case.yaml in {case_dir}")

    raw = yaml.safe_load(spec_path.read_text()) or {}
    prompt_path = case_dir / "prompt.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"No prompt.md in {case_dir}")

    notes_path = case_dir / "notes.md"
    notes = notes_path.read_text() if notes_path.exists() else ""

    evaluation = EvaluationSpec(**(raw.get("evaluation") or {}))
    reference = None
    if raw.get("reference_answer"):
        ref_raw = dict(raw["reference_answer"])
        if "json" in ref_raw and "data" not in ref_raw:
            ref_raw["data"] = ref_raw.pop("json")
        reference = ReferenceAnswer(**ref_raw)

    return Case(
        id=raw["id"],
        title=raw.get("title") or raw["id"],
        category=raw.get("category") or "other",
        tags=list(raw.get("tags") or []),
        path=case_dir,
        prompt=prompt_path.read_text(),
        notes=notes,
        input_files=list(raw.get("input_files") or []),
        reference_answer=reference,
        evaluation=evaluation,
        historical=raw.get("historical") or {},
        system_prompt=raw.get("system_prompt"),
    )


def discover_cases(path: Path) -> list[Case]:
    target = path.resolve()
    if (target / "case.yaml").exists():
        return [load_case(target)]
    if not target.exists():
        raise FileNotFoundError(target)

    cases = []
    for spec in sorted(target.glob("*/case.yaml")):
        cases.append(load_case(spec.parent))
    if not cases:
        raise FileNotFoundError(f"No cases found under {target}")
    return cases


def load_reference_text(case: Case) -> str | None:
    if not case.reference_answer:
        return None
    if case.reference_answer.text is not None:
        return case.reference_answer.text
    if case.reference_answer.file:
        return (case.path / case.reference_answer.file).read_text()
    if case.reference_answer.data is not None:
        import json

        return json.dumps(case.reference_answer.data, indent=2)
    return None


def load_reference_json(case: Case) -> Any:
    if case.reference_answer and case.reference_answer.data is not None:
        return case.reference_answer.data
    text = load_reference_text(case)
    if text is None:
        return None
    import json

    return json.loads(text)


def collect_text_evidence(case: Case) -> list[tuple[str, str]]:
    evidence: list[tuple[str, str]] = []
    for rel in case.input_files:
        file_path = case.path / rel
        if not file_path.exists():
            evidence.append((rel, f"[missing file: {rel}]"))
            continue
        if file_path.suffix.lower() in TEXT_SUFFIXES:
            evidence.append((rel, file_path.read_text()))
    return evidence


def collect_binary_inputs(case: Case) -> list[Path]:
    files = []
    for rel in case.input_files:
        file_path = case.path / rel
        if file_path.exists() and file_path.suffix.lower() not in TEXT_SUFFIXES:
            files.append(file_path)
    return files
