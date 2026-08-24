Production checkout (isolated). House rule 10: sha-check, never presence-check.

`scripts/audit/ingest-core-ai-index-and-below-floor-artifacts.ts` skips the
Full Substrate upload when kind+week is already in cr_artifacts
("not re-ingesting"). That keeps a stale blob after an upstream re-ship.

Fix it: compare on-disk sha256 to the stored row; re-upload when they differ.
Do not leave a presence-only skip. `run` until PASS, then finish.
