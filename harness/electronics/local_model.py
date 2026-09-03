"""Local text/vision extraction client and strict response contracts."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Mapping

import httpx

from harness.electronics.claims import canonical_json
from harness.electronics.table_extractors import project_pin_table


LOCAL_RESULT_SCHEMA = "harness.electronics-local-model-result.v1"
RESPONSE_SCHEMAS: dict[str, dict[str, Any]] = {
    "pin_or_ball": {
        "type": "object",
        "required": ["package", "pins"],
        "properties": {
            "package": {"type": ["string", "null"]},
            "pins": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["pin_no", "name"],
                    "properties": {
                        "pin_no": {"type": ["string", "integer"]},
                        "name": {"type": "string"},
                    },
                },
            },
        },
    },
    "pin_semantics": {
        "type": "object",
        "required": ["pins"],
        "properties": {
            "pins": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "pin_no",
                        "name",
                        "type",
                        "dir",
                        "supply_domain",
                        "functions",
                    ],
                    "properties": {
                        "pin_no": {"type": ["string", "integer"]},
                        "name": {"type": "string"},
                        "type": {"type": ["string", "null"]},
                        "dir": {"type": ["string", "null"]},
                        "supply_domain": {"type": ["string", "null"]},
                        "functions": {"type": "array"},
                    },
                },
            },
        },
    },
    "parametrics": {
        "type": "object",
        "required": ["facts"],
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "field",
                        "value",
                        "value_role",
                        "unit",
                        "conditions",
                    ],
                    "properties": {
                        "field": {"type": "string"},
                        "value": {
                            "type": ["string", "number", "integer"],
                        },
                        "value_role": {"type": ["string", "null"]},
                        "unit": {"type": ["string", "null"]},
                        "conditions": {"type": "object"},
                    },
                },
            },
        },
    },
    "series_summary": {
        "type": "object",
        "required": ["summary", "characteristics", "applications"],
        "properties": {
            "summary": {"type": "string"},
            "characteristics": {
                "type": "array",
                "items": {"type": "string"},
            },
            "applications": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
    "opn_decoder": {
        "type": "object",
        "required": ["series", "base_part", "package_code", "suffixes"],
        "properties": {
            "series": {"type": ["string", "null"]},
            "base_part": {"type": ["string", "null"]},
            "package_code": {"type": ["string", "null"]},
            "suffixes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["code", "meaning"],
                    "properties": {
                        "code": {"type": "string"},
                        "meaning": {"type": ["string", "null"]},
                    },
                },
            },
        },
    },
}


def validate_local_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "http" or parsed.username or parsed.password:
        raise ValueError("local model URL must use unauthenticated HTTP")
    if parsed.path.rstrip("/") not in {"", "/v1"} or parsed.query or parsed.fragment:
        raise ValueError("local model URL must end at /v1")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("local model URL has no hostname")
    if hostname not in {"localhost"}:
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError as exc:
            raise ValueError(
                "local model URL must use localhost or a private IP"
            ) from exc
        if not (address.is_loopback or address.is_private):
            raise ValueError("local model URL must use a private address")
    if parsed.port is None:
        raise ValueError("local model URL requires an explicit port")
    return value.rstrip("/")


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
    stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    if start < 0:
        raise ValueError("model response has no JSON object")
    value, _ = json.JSONDecoder().raw_decode(stripped[start:])
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return value


def _type_matches(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, True)


def validate_response(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str = "$",
) -> None:
    expected = schema.get("type")
    if isinstance(expected, str) and not _type_matches(value, expected):
        raise ValueError(f"{path}: expected {expected}")
    if isinstance(expected, list) and not any(
        _type_matches(value, item) for item in expected
    ):
        raise ValueError(f"{path}: unexpected value type")
    if isinstance(value, dict):
        required = schema.get("required") or []
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        properties = schema.get("properties") or {}
        for key, child in value.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, Mapping):
                validate_response(child, child_schema, path=f"{path}.{key}")
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, child in enumerate(value):
            validate_response(
                child,
                schema["items"],
                path=f"{path}[{index}]",
            )


def _overlaps(left: Any, right: Any) -> bool:
    if not (
        isinstance(left, (list, tuple))
        and isinstance(right, (list, tuple))
        and len(left) == len(right) == 4
    ):
        return False
    lx0, ly0, lx1, ly1 = (float(value) for value in left)
    rx0, ry0, rx1, ry1 = (float(value) for value in right)
    return lx1 > rx0 and rx1 > lx0 and ly1 > ry0 and ry1 > ly0


def focused_page_context(
    capability: str,
    page_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Project broad page evidence onto selected structural/package scope."""

    structural = page_evidence.get("structural_evidence")
    regions = (
        structural.get("regions") or []
        if isinstance(structural, Mapping)
        else []
    )
    if not regions:
        blocks = page_evidence.get("blocks") or []
        tables = page_evidence.get("tables") or []
    else:
        bboxes = [
            region.get("bbox")
            for region in regions
            if isinstance(region, Mapping)
        ]
        table_indexes = {
            int(region["table_index"])
            for region in regions
            if (
                isinstance(region, Mapping)
                and region.get("table_index") is not None
            )
        }
        blocks = [
            block
            for block in page_evidence.get("blocks") or []
            if isinstance(block, Mapping)
            and any(_overlaps(block.get("bbox"), bbox) for bbox in bboxes)
        ]
        tables = [
            table
            for fallback_index, table in enumerate(
                page_evidence.get("tables") or []
            )
            if isinstance(table, Mapping)
            and int(table.get("table_index", fallback_index))
            in table_indexes
        ]
    scope = (
        structural.get("package_scope")
        if isinstance(structural, Mapping)
        else None
    )
    selected_header = (
        str(scope.get("column_header") or "").strip().casefold()
        if isinstance(scope, Mapping)
        else ""
    )
    if capability in {"pin_or_ball", "pin_semantics"} and selected_header:
        tables = [
            project_pin_table(table, selected_header) for table in tables
        ]
    digital_text = page_evidence.get("digital_text")
    if isinstance(digital_text, Mapping):
        text = str(digital_text.get("text") or "")
        digital_text = {
            **digital_text,
            "text": text[:30_000],
            "truncated": len(text) > 30_000,
        }
    return {
        "document_sha256": page_evidence["document_sha256"],
        "page_1based": page_evidence["page_1based"],
        "blocks": blocks,
        "tables": tables,
        "digital_text": digital_text,
        "structural_evidence": structural,
    }


def local_prompt(
    capability: str,
    page_evidence: Mapping[str, Any],
    *,
    include_page_evidence: bool = True,
) -> str:
    schema = RESPONSE_SCHEMAS[capability]
    structural = page_evidence.get("structural_evidence")
    package_scope = (
        structural.get("package_scope")
        if isinstance(structural, Mapping)
        else None
    )
    encoded = ""
    if include_page_evidence:
        evidence = focused_page_context(capability, page_evidence)
        encoded = json.dumps(
            evidence,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(encoded) > 80_000:
            evidence["digital_text"] = {
                "available": False,
                "text": "",
                "reason": "omitted_to_preserve_complete_structural_json",
            }
            encoded = json.dumps(
                evidence,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        if len(encoded) > 80_000:
            raise ValueError("focused page evidence exceeds safe prompt bound")
    capability_rules = {
        "pin_or_ball": (
            "Return one row per visibly printed physical pin or ball. "
            "Never combine multiple identifiers into one row and never fill a "
            "sequence that is not printed. Keep package variants isolated. "
        ),
        "pin_semantics": (
            "Return one row per visibly printed physical pin or ball. "
            "Never combine multiple identifiers into one row, and omit rows "
            "whose selected-package pin or ball cell is blank, -, —, or N/A. "
            "Use dir only for a printed direction or pin-category value such "
            "as I, O, I/O, P, or S. Use type only for a printed electrical "
            "I/O structure or level such as FT_h or 5VT. Use supply_domain "
            "only for an explicitly headed supply-domain value. Put values "
            "from Description or Function columns in functions. Never copy a "
            "description or supply domain into type. Copy each value only "
            "from the matching identity row and matching semantic column. "
            "Represent blank and non-value scalar cells as JSON null. Keep "
            "package variants isolated. "
        ),
        "parametrics": (
            "Return one fact per actual data value. Skip blank cells and "
            "non-value markers such as -, —, or N/A. Copy field exactly from "
            "one printed Parameter or Symbol cell; never construct it by "
            "joining cells. Put size, test, and operating qualifiers in "
            "conditions. Copy value, unit, min/typ/max role, and conditions "
            "verbatim from the same visible row and its headers. Never "
            "paraphrase, expand an acronym, interpret a label, or combine "
            "qualifiers from separate cells; omit a fact that requires such "
            "a transformation. Use value_role for the printed "
            "min/typ/max/value header. Do not combine values, select a "
            "different condition, or convert units. "
        ),
        "series_summary": (
            "Copy each characteristic and application nearly verbatim from "
            "the page, then summarize only those copied facts. "
            "Do not invent positioning, recommendations, or competitors. "
        ),
        "opn_decoder": (
            "Decode only fields explicitly shown by the ordering diagram or "
            "table. Return suffixes as objects with code and nullable meaning. "
            "Use null for every unstated segment. "
        ),
    }[capability]
    scope_rule = ""
    if isinstance(package_scope, Mapping):
        scope_rule = (
            "Extract only package "
            f"{package_scope.get('package')!r} using table column "
            f"{package_scope.get('column_header')!r}. Ignore every other "
            "package column. "
        )
    source_instruction = (
        f"Extracted page evidence:\n{encoded}"
        if include_page_evidence
        else (
            "Read only the supplied rendered datasheet page image. No "
            "PDF-extracted table text is provided or authoritative."
        )
    )
    return (
        f"Extract capability={capability} from the supplied datasheet page. "
        "Use only printed evidence. Preserve units and package identity. "
        "Use null for absent scalar values. Return JSON only.\n\n"
        f"Capability rules: {scope_rule}{capability_rules}\n\n"
        f"Response schema:\n{json.dumps(schema, sort_keys=True)}\n\n"
        f"{source_instruction}"
    )


class LocalExtractionClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_s: float = 900,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = validate_local_url(base_url)
        if not model.strip():
            raise ValueError("local model cannot be blank")
        self.model = model
        self.timeout_s = timeout_s
        self.transport = transport

    def extract(
        self,
        *,
        capability: str,
        page_evidence: Mapping[str, Any],
        image_path: Path | None = None,
    ) -> dict[str, Any]:
        if capability not in RESPONSE_SCHEMAS:
            raise ValueError(f"unsupported local capability: {capability}")
        prompt = local_prompt(
            capability,
            page_evidence,
            include_page_evidence=image_path is None,
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_sha256 = None
        if image_path is not None:
            path = image_path.expanduser().resolve(strict=True)
            payload = path.read_bytes()
            image_sha256 = hashlib.sha256(payload).hexdigest()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            "data:image/png;base64,"
                            + base64.b64encode(payload).decode("ascii")
                        )
                    },
                }
            )
        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a local electronics extraction model. "
                        "Do not infer facts absent from source evidence."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": 4096,
            "enable_thinking": False,
        }
        request_sha = hashlib.sha256(canonical_json(request)).hexdigest()
        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                json=request,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        body = response.json()
        try:
            text = str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("local model response has no content") from exc
        parsed = parse_json_response(text)
        validate_response(parsed, RESPONSE_SCHEMAS[capability])
        return {
            "schema": LOCAL_RESULT_SCHEMA,
            "provider": "local",
            "model": self.model,
            "capability": capability,
            "request_sha256": request_sha,
            "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "image_sha256": image_sha256,
            "latency_ms": latency_ms,
            "usage": body.get("usage"),
            "result": parsed,
        }


__all__ = [
    "LOCAL_RESULT_SCHEMA",
    "LocalExtractionClient",
    "RESPONSE_SCHEMAS",
    "focused_page_context",
    "local_prompt",
    "parse_json_response",
    "validate_local_url",
    "validate_response",
]
