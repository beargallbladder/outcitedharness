#!/usr/bin/env python3
"""Create the immutable v1 repository-repair coding holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "harness.coding-repair-fixture.v1"


CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "merge-touching-ranges",
        "task_class": "algorithm_repair",
        "prompt": (
            "Fix merge_ranges. Inputs can be reversed, touching integer ranges "
            "must merge, output is sorted, and the input must not be mutated."
        ),
        "files": {
            "solution.py": (
                "def merge_ranges(ranges):\n"
                "    output = []\n"
                "    for start, end in sorted(ranges):\n"
                "        if output and start <= output[-1][1]:\n"
                "            output[-1] = (output[-1][0], max(output[-1][1], end))\n"
                "        else:\n"
                "            output.append((start, end))\n"
                "    return output\n"
            ),
            "test_solution.py": (
                "from solution import merge_ranges\n\n"
                "def test_overlap():\n"
                "    assert merge_ranges([(1, 3), (2, 5)]) == [(1, 5)]\n"
            ),
        },
        "hidden_test": (
            "from solution import merge_ranges\n\n"
            "def test_edges_and_immutability():\n"
            "    rows = [[5, 3], [1, 2], [8, 8], [7, 6]]\n"
            "    original = [row[:] for row in rows]\n"
            "    assert merge_ranges(rows) == [(1, 8)]\n"
            "    assert rows == original\n"
        ),
    },
    {
        "case_id": "strict-json-pointer",
        "task_class": "parser_repair",
        "prompt": (
            "Repair json_pointer_get to implement RFC 6901 reads. Reject malformed "
            "tilde escapes and non-canonical list indexes; preserve normal lookup "
            "exceptions."
        ),
        "files": {
            "solution.py": (
                "def json_pointer_get(document, pointer):\n"
                "    if not pointer:\n"
                "        return document\n"
                "    current = document\n"
                "    for raw in pointer.lstrip('/').split('/'):\n"
                "        key = raw.replace('~1', '/').replace('~0', '~')\n"
                "        current = current[int(key)] if isinstance(current, list) else current[key]\n"
                "    return current\n"
            ),
            "test_solution.py": (
                "from solution import json_pointer_get\n\n"
                "def test_basic_pointer():\n"
                "    assert json_pointer_get({'a': {'b': 2}}, '/a/b') == 2\n"
            ),
        },
        "hidden_test": (
            "import pytest\n"
            "from solution import json_pointer_get\n\n"
            "def test_strict_pointer_rules():\n"
            "    doc = {'a/b': {'~x': [3]}}\n"
            "    assert json_pointer_get(doc, '/a~1b/~0x/0') == 3\n"
            "    for pointer in ('a', '/a~2b', '/a~', '/items/01', '/items/-1'):\n"
            "        with pytest.raises((ValueError, KeyError, TypeError)):\n"
            "            json_pointer_get({'items': [1, 2]}, pointer)\n"
        ),
    },
    {
        "case_id": "strict-event-revisions",
        "task_class": "state_machine_repair",
        "prompt": (
            "Fix reduce_events so each entity accepts only strictly increasing "
            "integer revisions, ignores stale or duplicate revisions, and does not "
            "mutate the input."
        ),
        "files": {
            "solution.py": (
                "def reduce_events(events):\n"
                "    revisions = {}\n"
                "    states = {}\n"
                "    for event in events:\n"
                "        if event['revision'] >= revisions.get(event['id'], -1):\n"
                "            revisions[event['id']] = event['revision']\n"
                "            states[event['id']] = event['state']\n"
                "    return states\n"
            ),
            "test_solution.py": (
                "from solution import reduce_events\n\n"
                "def test_newer_wins():\n"
                "    assert reduce_events([{'id':'a','revision':1,'state':'x'},"
                "{'id':'a','revision':2,'state':'y'}]) == {'a':'y'}\n"
            ),
        },
        "hidden_test": (
            "from solution import reduce_events\n\n"
            "def test_duplicate_negative_and_immutable():\n"
            "    rows = [{'id':'a','revision':-2,'state':'old'},"
            "{'id':'a','revision':-1,'state':'new'},"
            "{'id':'a','revision':-1,'state':'duplicate'}]\n"
            "    copy = [dict(row) for row in rows]\n"
            "    assert reduce_events(rows) == {'a':'new'}\n"
            "    assert rows == copy\n"
        ),
    },
    {
        "case_id": "deterministic-topological-batches",
        "task_class": "graph_repair",
        "prompt": (
            "Repair topological_batches. Include dependencies absent as keys, emit "
            "all currently ready nodes in sorted batches, reject cycles, and do not "
            "mutate the graph."
        ),
        "files": {
            "solution.py": (
                "def topological_batches(graph):\n"
                "    pending = {key: set(value) for key, value in graph.items()}\n"
                "    output = []\n"
                "    while pending:\n"
                "        ready = sorted(key for key, deps in pending.items() if not deps)\n"
                "        if not ready:\n"
                "            break\n"
                "        output.append(ready)\n"
                "        for key in ready:\n"
                "            pending.pop(key)\n"
                "        for deps in pending.values():\n"
                "            deps.difference_update(ready)\n"
                "    return output\n"
            ),
            "test_solution.py": (
                "from solution import topological_batches\n\n"
                "def test_simple_chain():\n"
                "    assert topological_batches({'b':['a'],'a':[]}) == [['a'],['b']]\n"
            ),
        },
        "hidden_test": (
            "import pytest\n"
            "from solution import topological_batches\n\n"
            "def test_missing_nodes_cycle_and_immutable():\n"
            "    graph = {'build': {'lint', 'test'}, 'lint': {'parse'}, 'test': {'parse'}}\n"
            "    original = {key: set(value) for key, value in graph.items()}\n"
            "    assert topological_batches(graph) == [['parse'], ['lint', 'test'], ['build']]\n"
            "    assert graph == original\n"
            "    with pytest.raises(ValueError):\n"
            "        topological_batches({'a':['b'],'b':['a']})\n"
        ),
    },
    {
        "case_id": "canonical-query-builder",
        "task_class": "api_contract_repair",
        "prompt": (
            "Fix build_query. Sort keys, repeat sequence values, encode with RFC "
            "3986 percent escapes, omit None values, and render booleans lowercase."
        ),
        "files": {
            "solution.py": (
                "from urllib.parse import urlencode\n\n"
                "def build_query(values):\n"
                "    return urlencode(values)\n"
            ),
            "test_solution.py": (
                "from solution import build_query\n\n"
                "def test_basic_query():\n"
                "    assert build_query({'a':'x'}) == 'a=x'\n"
            ),
        },
        "hidden_test": (
            "from solution import build_query\n\n"
            "def test_canonical_encoding():\n"
            "    assert build_query({'z':None,'b':['x y','/'],'a':True}) == "
            "'a=true&b=x%20y&b=%2F'\n"
        ),
    },
    {
        "case_id": "bounded-chunks",
        "task_class": "boundary_repair",
        "prompt": (
            "Repair chunks so size must be a positive integer, every item appears "
            "exactly once, the final chunk may be short, and input is not mutated."
        ),
        "files": {
            "solution.py": (
                "def chunks(items, size):\n"
                "    return [items[index:index + size] for index in range(0, len(items) - 1, size)]\n"
            ),
            "test_solution.py": (
                "from solution import chunks\n\n"
                "def test_even_chunks():\n"
                "    assert chunks([1,2,3,4], 2) == [[1,2],[3,4]]\n"
            ),
        },
        "hidden_test": (
            "import pytest\n"
            "from solution import chunks\n\n"
            "def test_boundaries():\n"
            "    values = [1,2,3,4,5]\n"
            "    assert chunks(values, 2) == [[1,2],[3,4],[5]]\n"
            "    assert values == [1,2,3,4,5]\n"
            "    assert chunks([], 1) == []\n"
            "    for bad in (0, -1, 1.5, True):\n"
            "        with pytest.raises((TypeError, ValueError)):\n"
            "            chunks(values, bad)\n"
        ),
    },
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def build(destination: Path) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise ValueError("coding fixture destination already exists")
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "version": "v1",
        "cases": list(CASES),
    }
    core["core_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(core, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        os.chmod(destination, 0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    arguments = parser.parse_args()
    value = build(arguments.destination)
    print(
        json.dumps(
            {"cases": len(value["cases"]), "core_sha256": value["core_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
