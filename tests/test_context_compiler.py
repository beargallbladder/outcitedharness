from __future__ import annotations

import copy
import re

from harness.context_compiler import (
    CHARS_PER_TOKEN,
    compile_coder_context,
    compile_context,
)
from harness.orch_loop import LoopState, WorkingFile


def _file(content: str) -> WorkingFile:
    return WorkingFile(content=content, content_hash=f"hash-{len(content)}")


def _bloated_state() -> LoopState:
    state = LoopState(
        intent="Fix sparse cohort greeting behavior",
        iteration=3,
        last_cmd="pytest tests/test_greeting.py -q",
        last_exit=1,
        stdout_tail=("noise\n" * 200) + "CURRENT_FAILURE test_empty expected Welcome\n",
    )
    state.working_set.objective = state.intent
    state.working_set.acceptance_commands = [
        "pytest tests/test_greeting.py -q",
        "ruff check src",
    ]
    state.working_set.files_changed = ["src/greeting.py", "src/stale.py"]
    state.working_set.stale_files = ["src/stale.py"]
    state.working_set.current_diff = (
        "diff --git a/src/greeting.py b/src/greeting.py\n"
        + "CURRENT_DIFF_MARKER\n"
        + ("+changed\n" * 120)
    )
    state.working_set.files_read = {
        "src/greeting.py": _file(
            "def greet(name):\n"
            "    if not name:\n"
            "        return 'Welcome'\n"
            "    return f'Welcome, {name}!'\n"
            + ("# current implementation\n" * 40)
        ),
        "src/stale.py": _file("STALE_COPY_MUST_NOT_APPEAR\n" * 100),
        "tests/test_greeting.py": _file(
            "def test_empty():\n"
            "    assert greet('') == 'Welcome!'\n"
            "DIRECT_TEST_MARKER = True\n"
        ),
        "services/cache/redis.py": _file(
            "TRACEBACK_EVIDENCE_MARKER = 'causal'\n"
            "def fetch():\n"
            "    return WidgetFactory.build()\n"
        ),
        "services/semantic_guess.py": _file(
            "UNRELATED_SEMANTIC_MARKER = True\n" + ("guess = 1\n" * 700)
        ),
        "src/unrelated.py": _file(
            "UNRELATED_OLD_EVIDENCE = True\n" + ("old = 1\n" * 700)
        ),
    }
    state.expansion_paths = [
        "services/cache/redis.py",
        "services/semantic_guess.py",
    ]
    state.semantic_expansion_paths = ["services/semantic_guess.py"]
    state.attempt_summaries = [
        {
            "iteration": 1,
            "command": "pytest tests/test_greeting.py -q",
            "exit_code": 1,
            "failure": "Added punctuation; failed empty-name assertion.",
            "changed_files": ["src/greeting.py"],
            "diff_hash": "first",
        },
        {
            "iteration": 2,
            "command": "pytest tests/test_greeting.py -q",
            "exit_code": 1,
            "failure": "Added empty branch; failed CURRENT_FAILURE.",
            "changed_files": ["src/greeting.py"],
            "diff_hash": "second",
        },
        {
            "iteration": 3,
            "command": "pytest tests/test_greeting.py -q",
            "exit_code": 1,
            "failure": "current failure is represented concretely above",
            "changed_files": ["src/greeting.py"],
            "diff_hash": "third",
        },
    ]
    return state


def test_compiler_keeps_current_evidence_under_budget_deterministically():
    state = _bloated_state()
    before = copy.deepcopy(state.to_dict())

    first = compile_context(state, budget_tokens=2_048)
    second = compile_context(state, budget_tokens=2_048)

    assert first == second
    assert state.to_dict() == before
    assert len(first.text) <= 2_048 * CHARS_PER_TOKEN
    assert first.estimated_tokens <= 2_048
    assert '<CODER_CONTEXT phase="repair"' in first.text
    assert 'budget_tokens="2048"' in first.text
    assert "used_chars=" in first.text
    used = re.search(r'used_chars="(\d+)"', first.text)
    assert used and int(used.group(1)) == len(first.text)
    assert "<OBJECTIVE>" in first.text
    assert "<ACCEPTANCE>" in first.text
    assert "pytest tests/test_greeting.py -q" in first.text
    assert "CURRENT_FAILURE" in first.text
    assert 'command="pytest tests/test_greeting.py -q"' in first.text
    assert 'exit_code="1"' in first.text
    assert '<CURRENT_FILE path="src/greeting.py"' in first.text
    assert "CURRENT_DIFF_MARKER" in first.text
    assert '<RELEVANT_TEST path="tests/test_greeting.py"' in first.text
    assert "DIRECT_TEST_MARKER" in first.text
    assert "TRACEBACK_EVIDENCE_MARKER" in first.text
    assert first.text.index("TRACEBACK_EVIDENCE_MARKER") < first.text.index(
        "DIRECT_TEST_MARKER"
    )
    assert "STALE_COPY_MUST_NOT_APPEAR" not in first.text
    assert 'path="src/stale.py" status="refresh_pending"' in first.text
    if "UNRELATED_SEMANTIC_MARKER" in first.text:
        assert first.text.index("TRACEBACK_EVIDENCE_MARKER") < first.text.index(
            "UNRELATED_SEMANTIC_MARKER"
        )


def test_compiler_uses_current_source_and_structured_prior_attempts():
    state = _bloated_state()
    compiled = compile_context(state, budget_tokens=6_000)

    assert "return f'Welcome, {name}!'" in compiled.text
    assert "<PREVIOUS_ATTEMPTS>" in compiled.text
    assert "1. command=pytest tests/test_greeting.py -q" in compiled.text
    assert "Added punctuation; failed empty-name assertion." in compiled.text
    assert "2. command=pytest tests/test_greeting.py -q" in compiled.text
    assert "current failure is represented concretely above" not in compiled.text
    assert compiled.text.count("CURRENT_FAILURE test_empty") == 1
    assert "UNRELATED_OLD_EVIDENCE" not in compiled.text


def test_compiler_drops_peripheral_evidence_before_required_state():
    state = _bloated_state()
    for index in range(20):
        state.working_set.files_read[f"src/peripheral_{index}.py"] = _file(
            f"PERIPHERAL_{index}\n" + ("x = 1\n" * 2_000)
        )

    compiled = compile_context(state, budget_tokens=1_024)

    assert len(compiled.text) <= 1_024 * CHARS_PER_TOKEN
    assert '<CURRENT_FILE path="src/greeting.py"' in compiled.text
    assert "CURRENT_FAILURE" in compiled.text
    assert "CURRENT_DIFF" in compiled.text
    assert compiled.omitted_paths
    assert any(path.startswith("src/peripheral_") for path in compiled.omitted_paths)


def test_authoritative_working_set_overrides_contradictory_legacy_thread():
    state = LoopState(intent="fix greeting")
    state.working_set.objective = state.intent
    state.working_set.files_changed = ["greet.py"]
    state.working_set.files_read["greet.py"] = _file(
        "def greet(name):\n    return f'Welcome, {name}!'\n"
    )

    compiled = compile_coder_context(
        state,
        phase="repair",
        budget_tokens=1_024,
        legacy_thread=(
            "OLD THREAD COPY\n"
            "greet.py:\n"
            "def greet(name):\n    return f'Hello, {name}.'\n"
        ),
    )

    assert "return f'Welcome, {name}!'" in compiled.text
    assert "OLD THREAD COPY" not in compiled.text
    assert "return f'Hello, {name}.'" not in compiled.text
    assert "<LEGACY_RECOVERY>" not in compiled.text
