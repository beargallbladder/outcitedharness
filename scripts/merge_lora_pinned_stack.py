"""Merge a LoRA adapter into base shards, one shard at a time, fail-closed."""
import json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

BASE = "/training/models/Qwen3-VL-30B-A3B-Instruct-BF16"
ADAPTER = "/training/checkpoints/electronics-30b-pin-gate-v1-20260904-sft"
OUT = "/training/models/Qwen3-VL-30B-PinGate-SFT-457-Merged"

cfg = json.load(open(f"{ADAPTER}/adapter_config.json"))
scaling = cfg["lora_alpha"] / cfg["r"]
adapter = load_file(f"{ADAPTER}/adapter_model.safetensors")

# module path -> (A, B)
pairs = {}
for key, tensor in adapter.items():
    if key.endswith(".lora_A.weight"):
        mod, which = key[:-len(".lora_A.weight")], "A"
    elif key.endswith(".lora_B.weight"):
        mod, which = key[:-len(".lora_B.weight")], "B"
    else:
        raise SystemExit(f"unexpected adapter key: {key}")
    mod = mod.removeprefix("base_model.model.")
    pairs.setdefault(mod, {})[which] = tensor
for mod, ab in pairs.items():
    if set(ab) != {"A", "B"}:
        raise SystemExit(f"incomplete LoRA pair for {mod}")

index = json.load(open(f"{BASE}/model.safetensors.index.json"))
weight_map = index["weight_map"]
targets = {f"{mod}.weight": mod for mod in pairs}
missing = [t for t in targets if t not in weight_map]
if missing:
    raise SystemExit(f"{len(missing)} adapter targets not in base: {missing[:5]}")

os.makedirs(OUT, exist_ok=True)
applied = 0
for shard in sorted(set(weight_map.values())):
    tensors = load_file(f"{BASE}/{shard}")
    changed = False
    for name in list(tensors):
        if name in targets:
            ab = pairs[targets[name]]
            delta = (ab["B"].float() @ ab["A"].float()) * scaling
            if delta.shape != tensors[name].shape:
                raise SystemExit(f"shape mismatch {name}: {delta.shape} vs {tensors[name].shape}")
            tensors[name] = (tensors[name].float() + delta).to(torch.bfloat16)
            applied += 1
            changed = True
    save_file(tensors, f"{OUT}/{shard}", metadata={"format": "pt"})
    print(f"{shard}: {'merged' if changed else 'copied'}", flush=True)

if applied != len(targets):
    raise SystemExit(f"applied {applied} of {len(targets)} deltas")
for f in os.listdir(BASE):
    if not f.endswith(".safetensors"):
        shutil.copy2(f"{BASE}/{f}", f"{OUT}/{f}")
print(f"MERGE457_OK applied={applied} scaling={scaling}")
