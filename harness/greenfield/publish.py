from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from harness.greenfield.manifest import assert_destination_unchanged
from harness.greenfield.models import GreenfieldRun
from harness.greenfield.workspace import IGNORED_STATE_DIRS, full_tree_state_hash


class PublishError(RuntimeError):
    pass


def publish_verified_run(run: GreenfieldRun) -> Path:
    if run.status != "complete" or not run.final_state_hash:
        raise PublishError("only a complete verified greenfield run may be published")
    if run.manifest is None or not run.workspace_root:
        raise PublishError("greenfield run has no approved manifest or workspace")
    assert_destination_unchanged(run.manifest)
    source = Path(run.workspace_root).resolve()
    destination = Path(run.destination).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise PublishError("reserved destination is no longer empty")
    temporary = destination.parent / f".{destination.name}.harness-{uuid.uuid4().hex}"
    if temporary.exists():
        raise PublishError("publication staging path already exists")

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in (IGNORED_STATE_DIRS - {".git"})
        }

    try:
        shutil.copytree(source, temporary, symlinks=False, ignore=ignore)
        if full_tree_state_hash(temporary) != run.final_state_hash:
            raise PublishError("published copy does not match verified repository state")
        os.rename(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination
