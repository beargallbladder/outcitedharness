from __future__ import annotations

import json
import re
from typing import Any


FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for match in FENCE_RE.finditer(stripped):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue

    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("Could not parse JSON from model output")


def values_equal(
    left: Any,
    right: Any,
    ignore_list_order: bool = False,
    tolerance: float | None = None,
) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right or left == right
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(
            values_equal(left[k], right[k], ignore_list_order, tolerance) for k in left
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        if ignore_list_order:
            unused = list(right)
            for item in left:
                found = None
                for idx, candidate in enumerate(unused):
                    if values_equal(item, candidate, ignore_list_order, tolerance):
                        found = idx
                        break
                if found is None:
                    return False
                unused.pop(found)
            return True
        return all(
            values_equal(a, b, ignore_list_order, tolerance)
            for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if tolerance is not None:
            return abs(float(left) - float(right)) <= tolerance
        return float(left) == float(right)
    return left == right
