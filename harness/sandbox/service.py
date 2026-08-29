from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Callable

from .backend import BackendContainer, BackendError, DockerCLIBackend
from .models import (
    SandboxManifest,
    SandboxRecord,
    SandboxSpec,
    SandboxState,
    SandboxStatus,
    utc_now,
)
from .registry import JsonSandboxRegistry


class LifecycleError(RuntimeError):
    pass


class SandboxService:
    """Coordinates durable state transitions and manifest-bound cleanup."""

    def __init__(
        self,
        backend: DockerCLIBackend,
        registry: JsonSandboxRegistry,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.clock = clock

    def create(self, spec: SandboxSpec, *, start: bool = True) -> SandboxStatus:
        now = self._now()
        record = SandboxRecord(
            sandbox_id=spec.sandbox_id,
            state=SandboxState.CREATING,
            state_hash=spec.state_hash.lower(),
            spec_hash=spec.spec_hash,
            created_at=now,
            expires_at=now + timedelta(seconds=spec.ttl_seconds),
            updated_at=now,
        )
        self.registry.add(record)
        try:
            manifest = self.backend.create(spec)
            record = replace(
                record,
                state=SandboxState.CREATED,
                manifest=manifest,
                updated_at=self._now(),
            )
            self.registry.put(record)
            if start:
                container = self.backend.start(manifest)
                record = replace(
                    record,
                    state=self._state_for(container),
                    updated_at=self._now(),
                )
                self.registry.put(record)
        except Exception as exc:
            detail = self._safe_detail(exc)
            failed = replace(
                record,
                state=SandboxState.ERROR,
                updated_at=self._now(),
                detail=detail,
            )
            self.registry.put(failed)
            raise LifecycleError(
                f"could not create sandbox {spec.sandbox_id}: {detail}"
            ) from exc
        return record.status()

    provision = create

    def start(self, sandbox_id: str) -> SandboxStatus:
        record = self.registry.require(sandbox_id)
        manifest = self._require_manifest(record)
        now = self._now()
        if record.expires_at <= now:
            self._mark(record, SandboxState.EXPIRED, "sandbox TTL has elapsed")
            raise LifecycleError(f"sandbox {sandbox_id} has expired")
        if record.state is SandboxState.REMOVED:
            raise LifecycleError(f"sandbox {sandbox_id} has been removed")
        try:
            container = self.backend.start(manifest)
        except Exception as exc:
            self._mark(record, SandboxState.ERROR, self._safe_detail(exc))
            raise
        return self._mark(
            record,
            SandboxState.RUNNING if container.running else SandboxState.STOPPED,
        ).status()

    def stop(self, sandbox_id: str, *, grace_seconds: int = 10) -> SandboxStatus:
        record = self.registry.require(sandbox_id)
        manifest = self._require_manifest(record)
        if record.state is SandboxState.REMOVED:
            raise LifecycleError(f"sandbox {sandbox_id} has been removed")
        try:
            container = self.backend.stop(manifest, grace_seconds=grace_seconds)
        except Exception as exc:
            self._mark(record, SandboxState.ERROR, self._safe_detail(exc))
            raise
        state = SandboxState.RUNNING if container.running else SandboxState.STOPPED
        return self._mark(record, state).status()

    def destroy(self, sandbox_id: str) -> SandboxStatus:
        record = self.registry.require(sandbox_id)
        if record.state is SandboxState.REMOVED:
            return record.status()
        if record.manifest is None:
            return self._mark(record, SandboxState.REMOVED).status()
        try:
            # The backend re-inspects all ownership labels before destructive work.
            self.backend.remove_owned(record.manifest)
        except Exception as exc:
            self._mark(record, SandboxState.ERROR, self._safe_detail(exc))
            raise
        return self._mark(record, SandboxState.REMOVED).status()

    def status(self, sandbox_id: str, *, refresh: bool = False) -> SandboxStatus:
        record = self.registry.require(sandbox_id)
        if not refresh or record.manifest is None or record.state is SandboxState.REMOVED:
            return record.status()
        try:
            container = self.backend.inspect_owned(record.manifest)
        except BackendError as exc:
            return self._mark(
                record, SandboxState.MISSING, self._safe_detail(exc)
            ).status()
        return self._mark(record, self._state_for(container)).status()

    def list(self) -> tuple[SandboxStatus, ...]:
        return tuple(record.status() for record in self.registry.list())

    def reap_expired(self) -> tuple[SandboxStatus, ...]:
        now = self._now()
        reaped: list[SandboxStatus] = []
        for record in self.registry.list():
            if (
                record.state is SandboxState.REMOVED
                or record.expires_at > now
            ):
                continue
            expired = self._mark(
                record, SandboxState.EXPIRED, "sandbox TTL has elapsed"
            )
            if expired.manifest is not None:
                try:
                    self.backend.remove_owned(expired.manifest)
                except BackendError as exc:
                    reaped.append(
                        self._mark(
                            expired, SandboxState.ERROR, self._safe_detail(exc)
                        ).status()
                    )
                    continue
            reaped.append(self._mark(expired, SandboxState.REMOVED).status())
        return tuple(reaped)

    def recover(self, *, reap_expired: bool = True) -> tuple[SandboxStatus, ...]:
        """Reconcile persisted records with labels on managed containers."""

        discovered = self.backend.discover_managed()
        by_manifest = {item.manifest_id: item for item in discovered}
        recovered: list[SandboxStatus] = []
        for record in self.registry.list():
            if record.state is SandboxState.REMOVED:
                recovered.append(record.status())
                continue
            container: BackendContainer | None = None
            manifest = record.manifest
            if manifest is not None:
                candidate = by_manifest.get(manifest.manifest_id)
                if candidate and self._matches(record, candidate):
                    container = candidate
                elif candidate:
                    recovered.append(
                        self._mark(
                            record,
                            SandboxState.ERROR,
                            "managed container ownership labels do not match registry",
                        ).status()
                    )
                    continue
            elif record.state is SandboxState.CREATING:
                matches = [
                    item
                    for item in discovered
                    if self._matches(record, item)
                    and item.sandbox_id == record.sandbox_id
                ]
                if len(matches) == 1:
                    container = matches[0]
                    manifest = SandboxManifest(
                        manifest_id=container.manifest_id,
                        sandbox_id=container.sandbox_id,
                        container_id=container.container_id,
                        container_name=container.name,
                        state_hash=container.state_hash,
                        spec_hash=container.spec_hash,
                        image=container.image,
                    )
                    record = replace(record, manifest=manifest)
                elif len(matches) > 1:
                    recovered.append(
                        self._mark(
                            record,
                            SandboxState.ERROR,
                            "multiple managed containers match an incomplete record",
                        ).status()
                    )
                    continue

            if container is None:
                recovered.append(
                    self._mark(
                        record,
                        SandboxState.MISSING,
                        "manifest-owned container was not found",
                    ).status()
                )
            else:
                recovered.append(
                    self._mark(record, self._state_for(container)).status()
                )

        if reap_expired:
            reaped_ids = {
                status.sandbox_id: status for status in self.reap_expired()
            }
            recovered = [
                reaped_ids.get(status.sandbox_id, status) for status in recovered
            ]
        return tuple(recovered)

    def _mark(
        self,
        record: SandboxRecord,
        state: SandboxState,
        detail: str | None = None,
    ) -> SandboxRecord:
        updated = replace(
            record,
            state=state,
            updated_at=self._now(),
            detail=detail,
        )
        self.registry.put(updated)
        return updated

    @staticmethod
    def _matches(record: SandboxRecord, container: BackendContainer) -> bool:
        return (
            container.sandbox_id == record.sandbox_id
            and container.state_hash == record.state_hash
            and container.spec_hash == record.spec_hash
        )

    @staticmethod
    def _state_for(container: BackendContainer) -> SandboxState:
        if container.running:
            return SandboxState.RUNNING
        if container.status == "created":
            return SandboxState.CREATED
        return SandboxState.STOPPED

    @staticmethod
    def _require_manifest(record: SandboxRecord) -> SandboxManifest:
        if record.manifest is None:
            raise LifecycleError(
                f"sandbox {record.sandbox_id} has no resource manifest"
            )
        return record.manifest

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            raise LifecycleError("sandbox clock must return timezone-aware timestamps")
        return value

    @staticmethod
    def _safe_detail(exc: Exception) -> str:
        return str(exc).replace("\n", " ")[:1200] or type(exc).__name__
