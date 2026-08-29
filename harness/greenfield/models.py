from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DiscoveryQuery:
    category: str
    query: str
    mode: str
    limit: int


@dataclass(frozen=True)
class DiscoveryHit:
    category: str
    repo_id: str
    source_host: str
    repo_root: str
    revision: str
    state_hash: str
    path: str
    symbol: str | None
    symbol_type: str | None
    start_line: int
    end_line: int
    score: float
    match_type: str
    excerpt: str

    @property
    def provenance(self) -> str:
        return (
            f"gci://{self.source_host}/{self.repo_id}/{self.path}"
            f"#{self.start_line}-{self.end_line}"
        )


@dataclass(frozen=True)
class DiscoveryPattern:
    category: str
    summary: str
    provenance: str
    reason: str


@dataclass
class GreenfieldDiscovery:
    queries: list[DiscoveryQuery] = field(default_factory=list)
    repo_hits: list[DiscoveryHit] = field(default_factory=list)
    selected_patterns: list[DiscoveryPattern] = field(default_factory=list)
    rejected_patterns: list[DiscoveryPattern] = field(default_factory=list)
    compiled_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> GreenfieldDiscovery:
        data = raw or {}
        return cls(
            queries=[DiscoveryQuery(**row) for row in data.get("queries", [])],
            repo_hits=[DiscoveryHit(**row) for row in data.get("repo_hits", [])],
            selected_patterns=[
                DiscoveryPattern(**row) for row in data.get("selected_patterns", [])
            ],
            rejected_patterns=[
                DiscoveryPattern(**row) for row in data.get("rejected_patterns", [])
            ],
            compiled_at=str(data.get("compiled_at") or ""),
        )


@dataclass(frozen=True)
class ProductSpec:
    project_name: str
    purpose: str
    target_user: str
    stack: str
    runtime: str
    package_manager: str
    approved_dependencies: tuple[str, ...]
    functional_requirements: tuple[str, ...]
    nonfunctional_requirements: tuple[str, ...]
    public_interfaces: tuple[str, ...]
    persistence: str
    security_constraints: tuple[str, ...]
    exclusions: tuple[str, ...]
    definition_of_done: tuple[str, ...]
    final_acceptance: tuple[str, ...]
    discovery_patterns: tuple[DiscoveryPattern, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ProductSpec:
        data = dict(raw)
        for name in (
            "approved_dependencies",
            "functional_requirements",
            "nonfunctional_requirements",
            "public_interfaces",
            "security_constraints",
            "exclusions",
            "definition_of_done",
            "final_acceptance",
        ):
            data[name] = tuple(data.get(name) or ())
        data["discovery_patterns"] = tuple(
            DiscoveryPattern(**row) for row in data.get("discovery_patterns", ())
        )
        return cls(**data)


@dataclass(frozen=True)
class MilestoneSpec:
    milestone_id: str
    title: str
    objective: str
    dependencies: tuple[str, ...]
    expected_components: tuple[str, ...]
    acceptance_commands: tuple[str, ...]
    verification_class: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MilestoneSpec:
        data = dict(raw)
        for name in ("dependencies", "expected_components", "acceptance_commands"):
            data[name] = tuple(data.get(name) or ())
        return cls(**data)


@dataclass(frozen=True)
class MilestonePlan:
    milestones: tuple[MilestoneSpec, ...]
    final_acceptance_commands: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MilestonePlan:
        return cls(
            milestones=tuple(
                MilestoneSpec.from_dict(row) for row in raw.get("milestones", ())
            ),
            final_acceptance_commands=tuple(raw.get("final_acceptance_commands") or ()),
        )


@dataclass(frozen=True)
class GreenfieldManifest:
    run_id: str
    project_name: str
    stack: str
    runtime: str
    package_manager: str
    approved_dependencies: tuple[str, ...]
    destination: str
    destination_fingerprint: str
    spec_hash: str
    plan_hash: str
    discovery_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GreenfieldManifest:
        data = dict(raw)
        data["approved_dependencies"] = tuple(data.get("approved_dependencies") or ())
        return cls(**data)


@dataclass
class MilestoneState:
    ordinal: int
    milestone: MilestoneSpec
    state: str = "pending"
    task_id: str | None = None
    starting_commit: str | None = None
    verified_state_hash: str | None = None
    commit_sha: str | None = None
    attempts: int = 0
    error: str | None = None


@dataclass
class GreenfieldRun:
    run_id: str
    intent: str
    project_name: str
    stack: str
    destination: str
    destination_fingerprint: str
    status: str
    discovery: GreenfieldDiscovery
    spec: ProductSpec
    plan: MilestonePlan
    spec_hash: str
    plan_hash: str
    workspace_root: str | None = None
    manifest: GreenfieldManifest | None = None
    manifest_hash: str | None = None
    approved_at: str | None = None
    current_milestone: int = 0
    final_state_hash: str | None = None
    published_path: str | None = None
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""
    milestones: list[MilestoneState] = field(default_factory=list)
