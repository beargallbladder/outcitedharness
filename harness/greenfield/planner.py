from __future__ import annotations

import re

from harness.greenfield.models import (
    GreenfieldDiscovery,
    MilestonePlan,
    MilestoneSpec,
    ProductSpec,
)
from harness.greenfield.manifest import safe_project_name


SUPPORTED_STACKS = {"python", "node-typescript"}
_DEPENDENCY = re.compile(
    r"^(?:@[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*(?:\[[a-z0-9,._-]+\])?"
    r"(?:[<>=!~^][A-Za-z0-9.*+!<>=~^_-]+)?$",
    re.IGNORECASE,
)


def validate_dependency(value: str, stack: str | None = None) -> str:
    dependency = value.strip()
    if (
        not dependency
        or len(dependency) > 120
        or not _DEPENDENCY.fullmatch(dependency)
        or dependency.startswith(("-", ".", "/"))
        or "://" in dependency
    ):
        raise ValueError(f"unsafe dependency specification: {value!r}")
    if stack == "python" and ("^" in dependency or dependency.startswith("@")):
        raise ValueError(f"invalid Python dependency specification: {value!r}")
    if stack == "node-typescript" and "[" in dependency:
        raise ValueError(f"invalid Node dependency specification: {value!r}")
    return dependency


def _inferred_dependencies(intent: str, stack: str) -> list[str]:
    lowered = intent.lower()
    if stack == "python":
        candidates = [
            ("fastapi", "fastapi"),
            ("fastapi", "uvicorn"),
            ("pydantic", "pydantic"),
            ("typer", "typer"),
            ("sqlalchemy", "sqlalchemy"),
        ]
    else:
        candidates = [
            ("express", "express"),
            ("react", "react"),
            ("zod", "zod"),
        ]
    return [package for keyword, package in candidates if keyword in lowered]


def _interfaces(intent: str) -> tuple[str, ...]:
    lowered = intent.lower()
    out = []
    if any(word in lowered for word in ("api", "service", "fastapi", "http")):
        out.append("HTTP API")
    if any(word in lowered for word in ("cli", "command line", "terminal")):
        out.append("command-line interface")
    if any(word in lowered for word in ("web", "react", "frontend", "dashboard")):
        out.append("web user interface")
    return tuple(out or ["documented library interface"])


def build_product_spec(
    *,
    name: str,
    stack: str,
    intent: str,
    discovery: GreenfieldDiscovery,
    dependencies: tuple[str, ...] = (),
) -> ProductSpec:
    project_name = safe_project_name(name)
    if stack not in SUPPORTED_STACKS:
        raise ValueError(f"unsupported greenfield stack: {stack}")
    purpose = intent.strip()
    if len(purpose) < 12:
        raise ValueError("greenfield intent must describe concrete product behavior")
    approved = []
    for dependency in (*dependencies, *_inferred_dependencies(purpose, stack)):
        normalized = validate_dependency(dependency, stack)
        if normalized not in approved:
            approved.append(normalized)
    commands = (
        (".venv/bin/pytest -q", ".venv/bin/ruff check .")
        if stack == "python"
        else ("npm run test", "npm run lint", "npm run typecheck")
    )
    return ProductSpec(
        project_name=project_name,
        purpose=purpose,
        target_user="the operator described by the approved intent",
        stack=stack,
        runtime="python>=3.11" if stack == "python" else "node>=20",
        package_manager="uv" if stack == "python" else "npm",
        approved_dependencies=tuple(approved),
        functional_requirements=(purpose,),
        nonfunctional_requirements=(
            "deterministic local setup",
            "mechanically verified behavior",
            "clear failure responses and no silent data loss",
        ),
        public_interfaces=_interfaces(purpose),
        persistence=(
            "SQLite"
            if any(word in purpose.lower() for word in ("sqlite", "database", "persist"))
            else "none unless required by the functional behavior"
        ),
        security_constraints=(
            "no embedded credentials",
            "validate untrusted input at public boundaries",
            "no runtime dependency on a GCI source repository",
        ),
        exclusions=(
            "automatic cross-repository mutation",
            "unapproved package installation",
            "production credential provisioning",
        ),
        definition_of_done=(
            "all approved functional requirements have tests",
            "all repository verification commands pass against one file state",
            "repository is documented and contains no conflict markers",
        ),
        final_acceptance=commands,
        discovery_patterns=tuple(discovery.selected_patterns),
    )


def build_milestone_plan(spec: ProductSpec) -> MilestonePlan:
    commands = spec.final_acceptance
    milestones = (
        MilestoneSpec(
            milestone_id="m0",
            title="Bootstrap verified repository",
            objective=(
                f"Create the deterministic {spec.stack} project skeleton, verification "
                "contract, smoke test, and package metadata."
            ),
            dependencies=(),
            expected_components=(
                "package/runtime metadata",
                "source and test layout",
                ".harness.toml",
                "README",
            ),
            acceptance_commands=commands,
            verification_class="bootstrap",
        ),
        MilestoneSpec(
            milestone_id="m1",
            title="Implement approved product behavior",
            objective=(
                f"Implement the domain behavior in the approved specification: {spec.purpose}"
            ),
            dependencies=("m0",),
            expected_components=(
                "domain implementation",
                "input validation",
                "focused automated tests",
            ),
            acceptance_commands=commands,
            verification_class="repository",
        ),
        MilestoneSpec(
            milestone_id="m2",
            title="Complete interfaces and integration",
            objective=(
                "Complete and test the approved public interfaces "
                f"({', '.join(spec.public_interfaces)}), documentation, and integration behavior."
            ),
            dependencies=("m1",),
            expected_components=(
                "public interfaces",
                "integration tests",
                "operator documentation",
            ),
            acceptance_commands=commands,
            verification_class="integration",
        ),
    )
    validate_plan(milestones)
    return MilestonePlan(
        milestones=milestones,
        final_acceptance_commands=commands,
    )


def validate_plan(milestones: tuple[MilestoneSpec, ...]) -> None:
    if not milestones or len(milestones) > 8:
        raise ValueError("greenfield plans require between one and eight milestones")
    seen: set[str] = set()
    for index, milestone in enumerate(milestones):
        if milestone.milestone_id in seen:
            raise ValueError("duplicate milestone id")
        if not milestone.objective.strip() or not milestone.acceptance_commands:
            raise ValueError("every milestone needs an objective and acceptance")
        missing = set(milestone.dependencies) - seen
        if missing:
            raise ValueError(
                f"milestone {milestone.milestone_id} has unresolved dependencies: {sorted(missing)}"
            )
        if index == 0 and milestone.verification_class != "bootstrap":
            raise ValueError("the first milestone must be deterministic bootstrap")
        seen.add(milestone.milestone_id)
