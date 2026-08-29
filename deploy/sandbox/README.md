# M5 deployment sandboxes

Harness previews run in the dedicated `colima-harness-sandbox` Docker context.
Runtime state and staged build contexts live under
`/Volumes/M5_4TB/harness-sandboxes`; no Docker socket or host secrets are
mounted into application containers.

## Deploy

The source directory must contain an ARM64-compatible `Dockerfile`. Build
network access is disabled, so dependencies must already exist in the base
image or build context.

```bash
harness sandbox up ./app \
  --id example-preview \
  --container-port 8000 \
  --ttl-seconds 3600 \
  --expected READY
```

The command:

1. copies a content-addressed source snapshot to the M5 sandbox root;
2. rejects symlinks, secret-bearing files, remote `ADD`, secret/SSH mounts,
   and non-ARM64 platform overrides;
3. builds with `--network none` for `linux/arm64`;
4. starts an unprivileged, capability-free, read-only container with CPU,
   memory, PID, and tmpfs limits;
5. keeps application egress denied while a Caddy sidecar exposes only a
   loopback host port; and
6. publishes that loopback port at `/` on a dedicated, tailnet-only Tailscale
   HTTPS port.

Each lifecycle record persists the container ownership manifest, expiration,
state hash, and preview URL. Destructive operations revalidate all ownership
labels before acting.

## Operate

```bash
harness sandbox list
harness sandbox status example-preview --refresh
harness sandbox logs example-preview --tail 200
harness sandbox unpublish example-preview
harness sandbox down example-preview
harness sandbox gc
```

`down` removes the Tailscale route before removing containers. `gc` reaps
expired previews and records managed orphan containers without adopting
unlabelled Docker resources.

For autonomous builds, `harness build preview RUN_ID` applies the same flow
only after the run is complete, final verification passed, the workspace still
matches its recorded state hash, and the repository provides a `Dockerfile`
preview contract.
