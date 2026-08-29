from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Callable

from .backend import BackendContainer, BackendError, DockerCLIBackend
from .events import SandboxEvent, SandboxEventStore
from .models import (
    SandboxManifest,
    SandboxRecord,
    SandboxSpec,
    SandboxState,
    SandboxStatus,
    utc_now,
)
from .preview import PreviewRoute, TailscalePreviewPublisher
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
        preview_publisher: TailscalePreviewPublisher | None = None,
        max_active_sandboxes: int = 8,
        event_store: SandboxEventStore | None = None,
    ) -> None:
        if not 1 <= max_active_sandboxes <= 64:
            raise ValueError("max_active_sandboxes must be between 1 and 64")
        self.backend = backend
        self.registry = registry
        self.clock = clock
        self.preview_publisher = preview_publisher
        self.max_active_sandboxes = max_active_sandboxes
        self.event_store = event_store

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
        self.registry.add(record, max_active=self.max_active_sandboxes)
        self._event("create_requested", record)
        try:
            manifest = self.backend.create(spec)
            record = replace(
                record,
                state=SandboxState.CREATED,
                manifest=manifest,
                updated_at=self._now(),
            )
            self.registry.put(record)
            self._event("container_created", record)
            if start:
                container = self.backend.start(manifest)
                record = replace(
                    record,
                    state=self._state_for(container),
                    updated_at=self._now(),
                )
                self.registry.put(record)
                self._event("container_started", record)
        except Exception as exc:
            detail = self._safe_detail(exc)
            failed = replace(
                record,
                state=SandboxState.ERROR,
                updated_at=self._now(),
                detail=detail,
            )
            self.registry.put(failed)
            self._event("create_failed", failed)
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
            record = self._remove_preview(record)
            # The backend re-inspects all ownership labels before destructive work.
            self.backend.remove_owned(record.manifest)
        except Exception as exc:
            self._mark(record, SandboxState.ERROR, self._safe_detail(exc))
            raise
        return self._mark(record, SandboxState.REMOVED).status()

    def publish_preview(
        self,
        sandbox_id: str,
        *,
        https_port: int | None = None,
    ) -> SandboxStatus:
        record = self.registry.require(sandbox_id)
        manifest = self._require_manifest(record)
        if record.state is not SandboxState.RUNNING:
            raise LifecycleError("preview publishing requires a running sandbox")
        if record.preview_url is not None:
            raise LifecycleError("sandbox preview is already published")
        if self.preview_publisher is None:
            raise LifecycleError("preview publisher is not configured")
        if len(manifest.proxies) != 1:
            raise LifecycleError("preview publishing requires exactly one TCP proxy")
        proxy = manifest.proxies[0]
        route = self.preview_publisher.publish(
            proxy.host_port,
            https_port=https_port,
        )
        updated = replace(
            record,
            preview_url=route.url,
            preview_https_port=route.https_port,
            updated_at=self._now(),
        )
        self.registry.put(updated)
        self._event("preview_published", updated)
        return updated.status()

    def remove_preview(self, sandbox_id: str) -> SandboxStatus:
        record = self.registry.require(sandbox_id)
        return self._remove_preview(record).status()

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

    def logs(self, sandbox_id: str, *, tail: int = 200) -> str:
        record = self.registry.require(sandbox_id)
        manifest = self._require_manifest(record)
        return self.backend.logs_owned(manifest, tail=tail)

    def list(self) -> tuple[SandboxStatus, ...]:
        return tuple(record.status() for record in self.registry.list())

    def events(
        self,
        *,
        sandbox_id: str | None = None,
        limit: int = 200,
    ) -> tuple[SandboxEvent, ...]:
        if self.event_store is None:
            raise LifecycleError("sandbox event store is not configured")
        return self.event_store.list(sandbox_id=sandbox_id, limit=limit)

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
                    expired = self._remove_preview(expired)
                    self.backend.remove_owned(expired.manifest)
                except Exception as exc:
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

    def _remove_preview(self, record: SandboxRecord) -> SandboxRecord:
        if record.preview_url is None and record.preview_https_port is None:
            return record
        if (
            self.preview_publisher is None
            or record.preview_url is None
            or record.preview_https_port is None
            or record.manifest is None
            or len(record.manifest.proxies) != 1
        ):
            raise LifecycleError("cannot safely remove persisted preview route")
        proxy = record.manifest.proxies[0]
        self.preview_publisher.remove(
            PreviewRoute(
                host_port=proxy.host_port,
                https_port=record.preview_https_port,
                url=record.preview_url,
            )
        )
        updated = replace(
            record,
            preview_url=None,
            preview_https_port=None,
            updated_at=self._now(),
        )
        self.registry.put(updated)
        self._event("preview_removed", updated)
        return updated

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
        if updated.state is not record.state or updated.detail != record.detail:
            self._event("state_changed", updated)
        return updated

    def _event(self, kind: str, record: SandboxRecord) -> None:
        if self.event_store is not None:
            self.event_store.append(kind, record)

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
