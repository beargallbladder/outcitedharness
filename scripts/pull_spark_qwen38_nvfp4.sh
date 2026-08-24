#!/bin/zsh
# Disk-only pull of Qwen3.8-27B-NVFP4 onto the Spark.
# Does NOT restart :8900 / :8800 / systemd. Safe to run while 3.6 is serving.
set -euo pipefail

DEST="${DEST:-/data/models/Qwen3.8-27B-NVFP4}"
REPO="${REPO:-unsloth/Qwen3.8-27B-NVFP4}"

mkdir -p "$DEST"
export HF_HUB_DISABLE_XET=1
echo "snapshot_download $REPO → $DEST"
python3 - <<PY
import os
from pathlib import Path
from huggingface_hub import snapshot_download

dest = Path(os.environ.get("DEST", "/data/models/Qwen3.8-27B-NVFP4"))
token = os.environ.get("HF_TOKEN") or None
path = snapshot_download(
    repo_id=os.environ.get("REPO", "unsloth/Qwen3.8-27B-NVFP4"),
    local_dir=str(dest),
    token=token,
)
print("done", path)
print("has_config", (dest / "config.json").exists())
PY
du -sh "$DEST"
