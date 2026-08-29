from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from harness.greenfield.models import GreenfieldManifest
from harness.greenfield.workspace import full_tree_state_hash
from harness.repo_contract import RepoContract, build_repo_contract


class StackAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerificationResult:
    command: str
    exit_code: int
    output: str


class StackAdapter(ABC):
    stack: str

    @abstractmethod
    def bootstrap(self, root: Path, manifest: GreenfieldManifest) -> RepoContract:
        raise NotImplementedError

    @staticmethod
    def write_file(root: Path, relative: str, content: str) -> None:
        path = root / relative
        resolved = path.resolve()
        if root.resolve() not in resolved.parents:
            raise StackAdapterError(f"bootstrap path escapes repository: {relative}")
        if path.exists():
            if path.read_text() != content:
                raise StackAdapterError(
                    f"bootstrap resume found divergent file: {relative}"
                )
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    @staticmethod
    def contract(root: Path) -> RepoContract:
        contract = build_repo_contract(root)
        if contract is None or not contract.commands:
            raise StackAdapterError("bootstrap did not produce a verification contract")
        return contract

    @staticmethod
    def verify(contract: RepoContract) -> list[VerificationResult]:
        results = []
        expected_state = full_tree_state_hash(Path(contract.repo_root))
        for command in contract.commands:
            proc = subprocess.run(
                list(command.argv),
                cwd=contract.repo_root,
                capture_output=True,
                text=True,
                timeout=command.timeout_s,
                check=False,
            )
            output = ((proc.stdout or "") + (proc.stderr or ""))[-4_000:]
            results.append(
                VerificationResult(
                    command=command.command,
                    exit_code=proc.returncode,
                    output=output,
                )
            )
            if proc.returncode != 0:
                raise StackAdapterError(
                    f"bootstrap verification failed: {command.command}\n{output}"
                )
            actual_state = full_tree_state_hash(Path(contract.repo_root))
            if actual_state != expected_state:
                raise StackAdapterError(
                    f"verification command mutated repository state: {command.command}"
                )
        return results


def adapter_for(stack: str) -> StackAdapter:
    if stack == "python":
        from harness.greenfield.stacks.python import PythonAdapter

        return PythonAdapter()
    if stack == "node-typescript":
        from harness.greenfield.stacks.node_typescript import NodeTypeScriptAdapter

        return NodeTypeScriptAdapter()
    raise StackAdapterError(f"unsupported stack: {stack}")
