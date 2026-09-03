from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_pinout_vision_source.py"
SPEC = importlib.util.spec_from_file_location("pinout_source_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _corpus(tmp_path: Path) -> dict[str, Path]:
    validated = tmp_path / "validated"
    ground_truth = tmp_path / "ground-truth"
    pdf = tmp_path / "pdf"
    (validated / "vendor").mkdir(parents=True)
    ground_truth.mkdir()
    pdf.mkdir()
    pins = [
        {"pin_no": index, "name": f"P{index}", "type": "gpio"}
        for index in range(1, 9)
    ]
    payload = {
        "pinout": {
            "pin_functions_summary": pins,
            "packages": ["LQFP8"],
        },
        "_meta": {
            "extracted_by": "Claude Sonnet 5",
            "batch_id": "batch-1",
            "custom_id": "part-1",
        },
    }
    ground_truth_payload = json.loads(json.dumps(payload))
    published_payload = json.loads(json.dumps(payload))
    published_payload["_meta"]["provenance"] = {
        "ground_truth_source": audit.GROUND_TRUTH_SOURCE,
        "validation": audit.VALIDATION,
        "published_at": "2026-09-01T00:00:00Z",
    }
    (ground_truth / "part-1.json").write_text(json.dumps(ground_truth_payload))
    (validated / "vendor" / "part-1.json").write_text(
        json.dumps(published_payload)
    )
    (pdf / "part-1.pdf").write_bytes(b"%PDF-1.7\nfixture")
    quality = tmp_path / "quality.json"
    quality.write_text(json.dumps({"part-1": {"ok": True}}))
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"suspect": []}))
    return {
        "validated": validated,
        "ground_truth": ground_truth,
        "pdf": pdf,
        "quality": quality,
        "provenance": provenance,
    }


def _run(paths: dict[str, Path]) -> dict:
    return audit.audit_source(
        validated_root=paths["validated"],
        ground_truth_root=paths["ground_truth"],
        pdf_root=paths["pdf"],
        quality_audit=paths["quality"],
        provenance_audit=paths["provenance"],
        minimum_records=1,
    )


def test_audit_seals_frontier_validated_record(tmp_path: Path) -> None:
    paths = _corpus(tmp_path)

    result = _run(paths)

    assert result["policy"]["training_authorized"] is False
    assert result["counts"]["prealignment_candidates"] == 1
    assert result["counts"]["unique_pdf_sha256"] == 1
    candidate = result["candidates"][0]
    assert candidate["package_candidates"] == ["LQFP8"]
    assert candidate["pin_rows"] == 8
    core = {key: value for key, value in result.items() if key != "created_at"}
    expected = core.pop("evidence_sha256")
    assert hashlib.sha256(audit._canonical(core)).hexdigest() == expected


def test_audit_rejects_nonvalidated_and_duplicate_pin_rows(tmp_path: Path) -> None:
    paths = _corpus(tmp_path)
    published = paths["validated"] / "vendor" / "part-1.json"
    value = json.loads(published.read_text())
    value["_meta"]["provenance"]["validation"] = "UNVALIDATED"
    published.write_text(json.dumps(value))

    result = _run(paths)

    assert result["counts"]["prealignment_candidates"] == 0
    assert result["counts"]["rejection_reasons"] == {
        "validation_not_eligible:UNVALIDATED": 1
    }

    value["_meta"]["provenance"]["validation"] = audit.VALIDATION
    value["pinout"]["pin_functions_summary"][-1] = value["pinout"][
        "pin_functions_summary"
    ][0]
    published.write_text(json.dumps(value))
    ground_truth = paths["ground_truth"] / "part-1.json"
    ground_value = json.loads(ground_truth.read_text())
    ground_value["pinout"]["pin_functions_summary"] = value["pinout"][
        "pin_functions_summary"
    ]
    ground_truth.write_text(json.dumps(ground_value))

    result = _run(paths)

    assert result["counts"]["prealignment_candidates"] == 0
    assert result["counts"]["rejection_reasons"] == {
        "source_invalid:pin rows contain a duplicate physical identity": 1
    }


def test_audit_rejects_provenance_suspect(tmp_path: Path) -> None:
    paths = _corpus(tmp_path)
    paths["provenance"].write_text(
        json.dumps({"suspect": [{"part": "part-1"}]})
    )

    result = _run(paths)

    assert result["counts"]["prealignment_candidates"] == 0
    assert result["counts"]["rejection_reasons"] == {
        "provenance_audit_suspect": 1
    }


def test_write_new_is_immutable(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    audit.write_new(output, {"ok": True})

    assert json.loads(output.read_text()) == {"ok": True}
    assert output.stat().st_mode & 0o777 == 0o444
    with pytest.raises(ValueError, match="already exists"):
        audit.write_new(output, {"ok": False})
