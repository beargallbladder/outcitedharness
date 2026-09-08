# Handoff — 2026-09-07 12:15 PT (session restart)

Read this first in the new chat. Written because the previous session's shell tool
wedged (every command returned "no exit status"); all box work in the last hour ran
through subagents. A fresh session should have a working shell.

## Current state of the Qwen3.8-Flash-Next cluster (asus2 head / asus4 worker)

- Endpoint: `http://100.68.133.1:8888/v1` (asus2 Tailscale IP; works from both Macs).
  Served model name `qwen38-flash-next-nvfp4-sglang`. API key: `~/.secrets/qwen38-key`
  on m5max-ai (copied from `API_KEY=` in `~/qwen38-sglang-recipe/.env` on asus2).
- Sam is actively using it from the M4 (`macbook-pro`, Tailscale 100.102.121.37) via
  opencode.ai. **Do not send test inference to :8888 while he is working** — it queues
  behind his turns and he noticed. GET /v1/models is fine.
- Recipe: `~/qwen38-sglang-recipe/` on asus2 (start.sh / stop.sh / .env). Image is a
  locally built derivative `local/qwen38-flashnext-sglang:sm121-344f9d`, rebuilt
  automatically by `start.sh` when the patch context changes.
- Patches now baked into the image build (in `start.sh`'s Dockerfile heredoc):
  - sglang#36806 + #36845 (recipe originals: SM121 QSA fallback, no TRT-LLM sparse decode)
  - **sglang#37110** (added this morning): QSA draft-extend row sizing — fixed the
    NaN/device-assert crash that killed both ranks under concurrent long prompts.
  - **sglang#35821** (added ~11:40): MambaRadixCache ghost-node fix + spec_utils clamp
    at BOTH sites. Prerequisite for radix cache; without it, spec acceptance decays to
    ~0 over ~24h (issue #37326).
- `.env` config now: `SPECULATIVE=1` (NEXTN/EAGLE on, eager), `ENABLE_DECODE_GRAPHS=0`
  and `--disable-cuda-graph` (graphs permanently off — silent corruption, issue #37111),
  radix/prefix cache **ON** (`--disable-radix-cache` removed), `--max-mamba-cache-size 97`,
  `MAX_RUNNING_REQUESTS=6`, `extra_buffer` mamba strategy, 262k context.
- Qualified 11:44: 6 gates passed incl. 4x22k concurrent, prefix reuse cold 15.1s ->
  warm 0.7s (22.7x, cached_tokens=25088), 60-tiny-request ghost-node smoke, zero
  asserts/NaN on head or worker. Health check at 12:06: both up, 0 errors, accept
  rate 0.68-0.78, decode 19-25 tok/s single-stream, RoCE link clean.
- Backups on asus2: `start.sh.bak-pre37110`, `start.sh.bak-pre35821`,
  `.env.bak-pre-radix-20260907`. Revert = copy back + `./stop.sh && ./start.sh` (~5 min).
- Qualifier scripts (local, not in repo): `/tmp/qualify_qwen38.py`, `/tmp/qualify_radix.py`
  (also copied to asus2:/tmp). Need `QWEN_API_KEY` env.

### Open watch items
1. **Prefix cache hits on Sam's real traffic**: at 12:06 the only cache hits were from
   my qualifier; his turns (up to 148k tokens) had 0 cached tokens. Likely just first
   turns after the restart. If his turns still take minutes tomorrow, capture two
   consecutive opencode request bodies and diff the front of the prompt.
2. **Ghost-node decay** is slow (24h). Re-check `accept rate` in head logs and warm-turn
   latency after a day of real use.
3. opencode installed on m5max-ai (`brew install anomalyco/tap/opencode`, v1.18.29);
   config at `~/.config/opencode/opencode.json` points at the endpoint above.

## Cleanup completed this session (Sam approved: "1. delete 2. kill 3. GO")
- Cursor shadow lane KILLED: launchd services `com.harness.cursor-shadow`,
  `com.harness.cursor-shadow-processor`, `com.samkim.litellm`, `com.samkim.litellm-travel`,
  `com.samkim.harness-orch`, `com.samkim.harness-smoke` unloaded; plists moved to
  `~/Library/LaunchAgents/disabled-20260907/`. Only `com.samkim.harness-gci-refresh` remains.
- `.cursor/hooks.json` -> `.cursor/hooks.json.disabled-shadow` (repo, uncommitted).
- `~/.harness/shadow/` replays + work (23 GB) deleted; spool DB, learning DB, logs
  archived at `/Volumes/M5_4TB/archives/cursor-shadow-lane-20260907/` (131 MB).
- Why: 8 days of shadowing produced 2 training pairs; hooks dropped most events
  (75k JSON errors); processor crash-looping since Sep 5; LiteLLM health checks were
  the only real traffic hitting Qwen (4 req / 30s, 24/7). Sam: not training to beat
  Cursor; just wants Qwen for opencode. Electronics/datasheet lane untouched and independent.

## "fix #2" disk cleanup — DONE 12:50 PT (authorized scope only)
- **asus2: 95% -> 74% (233 GB free).** Deleted `~/models/DeepSeek-V4-Flash-DSpark` (159 GB),
  7 exited containers (1.3 GB), 24 GB of dangling images (nvcr pytorch layers + one stale
  untagged `lmsysorg/sglang@sha256:14ed…` digest; the `lmsysorg/sglang:qwen38flashnext` tag
  was already absent before, so `start.sh` rebuild behaviour is unchanged). Head container
  and `local/qwen38-flashnext-sglang:sm121-344f9d` verified intact; :8888 /v1/models OK.
  Container configs snapshotted to `asus2:~/archives/docker-containers-inspect-20260907.json`.
- **asus3: 98% -> 98% (26 GB free).** Only ~2 GB reclaimed: 7 exited containers pruned, two
  `<none>` images removed but their layers are shared with `harness/qwen38-native-tp`, so
  nothing came back. Snapshot at `asus3:~/archives/docker-containers-inspect-20260907.json`.
  v4 vision :8912 still 200. Left untouched: untagged `lmsysorg/sglang` (28.4 GB, pinned by
  digest in `~/nemotron_sglang.sh`).
- **asus3 round 2 (Sam approved 12:54): 98% -> 71%, 255 GB free.** Deleted
  `~/models/DeepSeek-V4-Flash-DSpark` (159 GB), `~/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
  (21 GB), untagged `lmsysorg/sglang` image (28.4 GB; `~/nemotron_sglang.sh` will now need a
  re-pull to run), `harness/qwen38-native-tp:tf5.16.1-fla0.5.2` (24.6 GB). `~/models` is now
  empty; only image left is `nvcr.io/nvidia/vllm:26.05.post1-py3` (serving v4). :8912 still 200.
  - Under `~/harness-training/models` (543 GB, hands off by rule, listed for awareness):
    `Qwen3.8-Flash-Next-BF16` 336 GB, `Qwen3-VL-30B-A3B-Instruct-BF16` 61 GB,
    `PinGate-V4-Merged` 58 GB (serving), `PinGate-SFT-457-Merged` 58 GB, `Instruct-FP8` 31 GB.
- NEVER touch: `~/.cache/qwen38-sglang-hf`, `~/qwen38-sglang-recipe`, `~/harness-training`
  or any `/training/` path (BF16 base + checkpoints for the vision training rounds),
  running containers/images, `local/qwen38-flashnext-sglang:*`, `lmsysorg/sglang*`.
- Note: on asus2 `~/models/RadixArk/Qwen3.8-Flash-Next-NVFP4` and the HF cache snapshot in
  `~/.cache/qwen38-sglang-hf` are **hardlinks to the same 126 GB** (verified same inodes;
  `du` shows the cache as 72K only because it deduplicates). Deleting RadixArk wouldn't free
  space and the cache would keep working, but treat `~/models/RadixArk` as never-touch too.

## INCIDENT 13:55 PT — spark kernel panic caused by me (recovered, ~2.5 min outage)
- While investigating why spark's ConnectX-7 was absent from `lspci`, I ran
  `echo 1 > /sys/bus/pci/rescan`. The kernel panicked and auto-rebooted; spark was
  unreachable 13:55-13:57:45. All services came back on their own: `bge-m3-embed` (:8800
  `/embed` and `/v1/embeddings` -> 200), `harness-gci` (:8810 on Tailscale -> 401 = alive),
  `cloudflared`, `tailscaled`. CR was not notified yet — Sam to decide.
- **Root cause of the "missing" CX7 (by design, not a fault):** DGX Spark ships a platform
  driver `cx7-pcie-hotplug` (ACPI `MTKP0001:00`) that powers the CX7 off and removes it from
  PCI ~15 s after boot when no QSFP cable is present (`Cable removal` in dmesg). spark has no
  DAC to the CRS812, so its CX7 is always powered down. dgx2 shows the same message at boot
  but is cabled, so its card is hot-added. Plugging a cable in triggers hot-add; **never**
  `pci/rescan` a Spark with the CX7 powered down — that is what panicked it.
- To put spark on the fabric: all four CRS812 QSFP ports are used (see network section), so
  use a QSA56 adapter (e.g. MAM1Q00A-QSA56) in one CX7 port + SFP56 DAC into a free SFP56
  cage = 50G link. CX7 supports 50GbE single-lane.

## Network topology (mapped 13:45 via LLDP; switch = MikroTik CRS812-8DS-2DQ-2DDQ-RM)
- 200G fabric `10.77.0.0/24` (+ mirror `10.77.1.0/24` on 2nd port function), all via
  Tensor Juice QSFP-DD breakout DACs: `qsfp56-1` asus1 (.2), `qsfp56-2` dgx2 (.1, =spark-49af,
  ssh alias `dgx2`, idle, 2.3 TB free), `qsfp56-dd-1` dgx3 (.3) + asus3 (.4),
  `qsfp56-dd-2` asus2 (.5) + asus4 (.6). All QSFP ports consumed. spark not on fabric.
- Management: every box's 10G RJ45 negotiated **1 Gb/s**; LLDP shows they all sit on a dumb
  1G switch/eero LAN that uplinks to CRS812 `ether2`. Free on the CRS812: 8x SFP56, `ether1`.
- Recommendation given to Sam: shared NFS model store over the 200G fabric (dgx2 or dgx3 as
  server), cheap 10GBASE-T switch for management, SFP56 ports for a future storage box at
  50G; skip per-box USB drives.

### NFS phase 0 — DONE 14:50 PT, and it exposed a fabric problem
- Server dgx2: `nfs-kernel-server` installed, `/srv/models` (owner 1000:1000) exported to
  `10.77.0.0/24` + `10.77.1.0/24` (`rw,async,all_squash,anonuid=1000`), 32 nfsd threads,
  TCP :2049 + RDMA :20049 (`/etc/nfs.conf.d/harness.conf`). Bench files `.bench-20g.bin`
  (fallocate) and `.bench-8g-real.bin` (urandom) left in `/srv/models` for re-testing.
- Client asus3: mounted read-only at `/mnt/models` (currently `proto=rdma`). Not mounted on
  asus2 — no point until the fabric is fixed, and a hard NFS mount on the Qwen head is a risk.
- Results from asus3: TCP nconnect=16 1.6-1.7 GB/s, RDMA 1.4-1.5 GB/s, parallel fio same.
  Local NVMe on asus3 reads 4.8 GB/s. **Every 200G link is capped at ~13.3 Gb/s per PCIe
  half** (should be ~100): `ib_write_bw -q 4` for every pair among asus1/asus3/dgx2/dgx3,
  both directions, = 13.2-13.3 Gb/s; 8 QPs bidirectional = 23.9 total; iperf3 8-stream 12.4.
  asus2/asus4 not measured (Sam using Qwen) but almost certainly the same => **the Qwen TP
  link runs at ~13 Gb/s.**
- **Switch is NOT the cause** (checked via REST API with the label password, stored at
  `~/.secrets/crs812-admin`, 0600, never in repo): all 6 qsfp bridge ports `hw-offload=true`,
  no bandwidth/queue/QoS limits, `only-hardware-queue`, l2mtu 9216, CPU port moved 67 MB rx /
  340 MB tx in 6 days (I pushed >60 GB), CPU load 0, RDMA latency asus3->dgx2 2.5 us.
  Minor: `qsfp56-dd-2-1` (asus2) 11 and `qsfp56-dd-1-1` (dgx3) 6 uncorrected RS-FEC
  codewords; leg-2 ports have 0. Watch, not urgent.
- **Working diagnosis: CX7 firmware power-throttle state on every node.** Published DGX Spark
  cluster reports describe exactly this — "12.74 Gb/s RDMA on a link that looked healthy in
  every status tool", fixed by OS/firmware update + reboot or a **full power drain**. All six
  nodes log `mlx5_pcie_event: Detected insufficient power on the PCIe slot (27W)` at every
  CX7 (re)init (boot, and each cable event via `cx7-pcie-hotplug`); MPEIN now reads
  `pwr_status=2` (sufficient) so the throttle is not visible in status. All on fw 28.45.4028 =
  the bundled version (fw manager says "already on same version", nothing newer). Kernel
  6.17.0-1032 is available (all nodes on -1031); 165 packages upgradable on dgx2.
  `pci=pcie_bus_safe` / MRRS 512 is NVIDIA's default on all Sparks incl. healthy ones —
  not the cause. `setpci` is blocked by kernel lockdown anyway.
  Note the fabric doc's Aug 31 NCCL numbers (~23 Gb/s bus bw, same on the pre-switch
  direct cable) already showed this cap — it predates the switch.
- **CONFIRMED 15:07 by power-draining dgx2** (Sam unplugged/replugged): dgx2 sending went
  13.3 -> **111.4 Gb/s** on half 1 and **111.7 Gb/s** on half 2; dgx2 -> asus3 111.6.
  asus1 -> dgx2 still 13.3 (asus1 not yet drained) => the throttle is on the *sending* NIC and
  a full power drain clears it. The boot after the drain still logs the `insufficient power
  (27W)` line, so that message alone does not indicate the throttled state.
- NFS from asus3 after the fix: RDMA single dd 3.4 GB/s, fio 8 jobs 11.4 GB/s (RDMA) /
  13.4 GB/s (TCP nconnect=16). Local NVMe on asus3 = 4.8 GB/s. asus3 left mounted
  read-only at `/mnt/models` (rdma).
- **ROLLOUT DONE 15:38 PT — all six fabric nodes drained and verified at 111.6-111.7 Gb/s
  per PCIe half** (8.4x): dgx2 15:07; asus1/asus3/asus2/asus4/dgx3 powered off 15:10
  (Qwen `stop.sh` first), unplugged/replugged by Sam ~15:20. GX10s auto-power-on when
  plugged in; the NVIDIA Sparks (dgx3) need the button. Qwen pair asus2<->asus4 111.6 both
  ways. Vision v4 :8912 came back by itself on asus1/asus3/dgx3 (`restart=unless-stopped`).
  Qwen restarted via `start.sh` (log `~/qwen38-boot-postdrain-20260907.log`), /v1/models 200
  at 15:37, smoke `DRAIN_OK`, 0 assert/NaN. Full re-qualification (`/tmp/qualify_qwen38.py`)
  not yet re-run post-drain. Watch whether the throttle returns after future cable events /
  reboots — re-measure with `ib_write_bw` if Qwen prefill or NFS gets slow again.
  spark (DGX1): not on the fabric; untouched.
  Do NOT try `mstfwreset`/PCI rescan remotely (see spark incident).
- LaCie Rugged Mini SSD 500 GB (exFAT, ~17 GB of personal "Charley" files) is mounted
  read-only at `asus3:/mnt/lacie`; leave or unmount per Sam.

## Other findings to act on / tell Sam
- **spark is NOT idle (earlier claim was wrong — services run outside Docker)**:
  `bge-m3-embed.service` :8800 (CR's FAE v4 embedder, 3.3 GB GPU, "DO NOT restart/
  repurpose" per config/workers.yaml), `harness-gci.service` :8810 (bound to Tailscale
  100.81.201.24 only), `cloudflared`. GPU ~110 GB free. Disk 2.4 TB free.
- **DONE 17:00 PT — Qwen3-Coder-Next lane live on spark** (Sam: "yeah do it"). Box roles
  per Sam: asus1/asus3/dgx2/dgx3 = training/extraction pool; asus2/asus4 = Flash-Next coder
  pair; spark = embedder + this lane. **Do not put lanes on dgx2.**
  - Container `qwen3-coder-next` on spark, `nvcr.io/nvidia/vllm:26.05.post1-py3`,
    `restart=unless-stopped`, host :8900, weights `~/models/qwen3-coder-next-nvfp4-gb10` (ro).
    Same flags as the Aug 29 asus2 run (kv fp8, flashinfer, chunked prefill, tool parser
    `qwen3_coder`) but `--gpu-memory-utilization 0.55 --max-num-seqs 3`. No DFlash draft on
    spark, so no spec decode yet: 62.5 tok/s single stream. Cold start ~8 min.
  - API key: `~/.secrets/coder-next-key` (m5max-ai) = `spark:~/.secrets/coder-next-key`.
    Endpoint `http://100.81.201.24:8900/v1`, model `qwen3-coder-next`. 401 without key.
  - Smoke: chat OK, tool call parsed OK. Embedder :8800 p50 15 ms idle -> 37 ms with 3
    concurrent coder streams. GPU: coder 63.6 GB, embedder 1.3 GB.
  - opencode on m5max-ai (`~/.config/opencode/opencode.json`): provider `coder-spark`,
    `small_model` + `agent.explore.model` + `agent.general.model` -> `coder-spark/qwen3-coder-next`;
    `build` stays on `qwen-dgx`. `opencode models` lists both.
  - **M4 done 17:50 PT.** SSH is `samsonkim@100.102.121.37` (Tailscale, key auth; user is
    `samsonkim`, not `samkim`). Sam runs **OpenCode.app** (desktop, 1.18.29) there, config is
    `~/.config/opencode/opencode.jsonc` (existing provider `asus2` -> Flash-Next pair). Added
    `coder-spark` provider + `small_model` + `agent.explore/general` -> `coder-spark/qwen3-coder-next`;
    key at `~/.secrets/coder-next-key` (0600); backup `opencode.jsonc.bak-20260907`. curl from the
    M4 to spark :8900 with the key works. The app was running during the edit — restart it to pick
    up the change. Ignore `~/.opencode/bin/opencode` on the M4: stale CLI 0.3.61 from Jul 2025,
    doesn't read this config.
  - CR heads-up drafted at `/tmp/cr-heads-up-coder-next.md`, NOT sent — Sam to approve.
  - Rollback: `ssh spark docker stop qwen3-coder-next`.
  - Follow-ups: download `qwen3-coder-next-dflash` draft + build the sm121 sglang image if
    62 tok/s is too slow; `config/models.yaml` still lists five dead `:8900` endpoints.
- `config/models.yaml` in the harness lists five dead `qwen3-coder-next :8900` endpoints.
- CR question 17:50Z (apps drops extraction status) ANSWERED 19:12Z via the protocol CLI
  (`npx tsx scripts/orchestration/send-message.ts` in
  `/Volumes/M5_4TB/repos/outcited-ai-sandbox-20260822`; body was /tmp/cr-reply-apps-drops.md).
  Told them: all three drops run+delivered Sep 6 (`m5_gd_apps_verbatim_v1_20260906.jsonl`,
  `m5_apps_verbatim_v2_20260906.jsonl`), missed because our two delivery mails predated
  the frontmatter protocol and used invalid type `drop`. Offered to re-stamp; asked for
  their 15 node_ids to diff against the 210 apps rows. Watch for their reply.
- Hourly health loop (23 ticks) was running in the old session's terminal; it dies with
  the session. Vision v4 endpoints dgx3/asus1/asus3 :8912 were all 200 at 12:08.

## Production vision model (unchanged today)
- v4 (round 4, from pristine base) promoted to dgx3/asus1/asus3 :8912 alias
  `qwen3-vl-30b-pin-gate-v4` (+ legacy 457 alias). CR holdout scoreboard: type
  regression was an instrument artifact, candidate promote-leaning -> promoted.

## 2026-09-08 09:15 PT — CR knives PRD: step 1 (census) DONE, Sam green-lit the lane
- CR mail `prd-document-derived-knives-v0-20260908` (in `agent-inbox/m5-cursor/`; PRD at
  `exports/cr_requests/prd-document-derived-knives-20260908/`, sha verified). Discussed with
  Sam; agreed: target is the document ABOVE the OPN (family/series/group), two outputs
  (shared facts for content + per-variant matrix for knives), doc is primary and the 125k
  vendor knife rows are a second source not ground truth, bogey/denominator first, bogey goes
  to the verifier never the prompt. Sam: "GO FORTH". No mail sent to CR yet — Sam sees the
  numbers first.
- Built `harness/electronics/family_census.py` + `scripts/census_family_documents.py` +
  tests (13); moved `read_scope` into `harness/electronics/document_scope.py` (script is a
  thin wrapper now). Full corpus run: `results/family-census-20260908/` (gitignored), copy at
  `/Volumes/M5_4TB/exports/family-census-20260908/` (census.jsonl sha
  `185c98e5…a17d2`). Numbers in `DATASHEET_FACTORY.md` §Above-OPN family census.
- Next (step 2, not started): deterministic device-table read for ST (254 matrix docs, best
  feed overlap) emitting the CR conventions schema with `_meta` receipts; agreement report vs
  CR knife rows per attribute; text-strategy table finder for Renesas/new-ST borderless tables.
- New agent `m5-opencode` is live (sent CR a question overnight, got an answer). Coordinate:
  this lane is mine unless Sam reassigns.

## 2026-09-08 12:10 PT — knives step 2 DONE: reader, agreement, promotion (Sam: "yes promote")
- Reader `harness/electronics/family_device_matrix.py` + `scripts/extract_family_device_matrix.py`
  (deterministic PyMuPDF only; parts_as_columns and parts_as_rows; header→column binding by
  printed token / code list / stem+codes / device-summary join / parts-named join / word
  geometry for rotated two-level headers / `PARTYxxxx` package-letter headers; merged-cell
  fill-down flagged; composite rows `SPI/I2S`, `FLASH / SRAM (KB)`; tri-state `-`=0; ST
  ordering-code flash self-check). Unknown on any ambiguity, never a guess. 42 tests.
- Census v2 (`results/family-census-20260908b/`, `--page-index` REQUIRED or the lane pages are
  lost): part-density page signal + product-list keywords + caption-row strip. family_matrix
  720→834 docs, strictly monotone (no downgrades). ST 276, TI 168, power 144, Microchip 92,
  GD 58, SiLabs 42, Renesas 13.
- Yield (all vendors, `results/family-matrix-<vendor>-20260908/`, copy at
  `/Volumes/M5_4TB/exports/family-matrix-20260908/` with SHA256SUMS): 3,790 distinct parts
  bound. Unknown rate: ST 8.6 %, GD 4.3 %, SiLabs 4.9 %, Renesas 7.9 %, Microchip 16 %, TI 21 %
  (TI residue is MSP430 "Timer A: 3, 2, 2, 2" instance lists and nominal-only voltages).
- Agreement vs CR referee (`scripts/knife_agreement_report.py`, ST only, 1,188 parts,
  `results/family-matrix-st-20260908/agreement/`): ≥95 % on code_flash 99.0, sram 95.1,
  freq 99.5, can 98.6, usb 99.4, pin_count 99.4. Below: spi/usart/i2s/i2c (semantics — CR
  counts SPI-capable I2S etc.; contract proposed and confirmed by CR), operating_voltage
  (doc prints 1.8–3.6 with "down to 1.65 at power-down", feed prints 1.65), temp_range
  (grain: doc prints grade set). gpio_count WITHDRAWN by CR (their field is "up to N").
- Promotion `scripts/promote_document_knives.py`, two-source rule (Sam asked what a second
  source is when the doc is the reference manual — answer in mail
  `knives-st-v0-promoted-drop-20260908`): Tier A doc+feed, Tier B doc+vendor ordering-code
  decode, Tier C single statement stays content-only. Drop at
  `/Volumes/M5_4TB/exports/family-matrix-st-20260908/promoted/document_derived_knives_st_v0.json`
  (1,100 parts, five approved attributes, per-value `_provenance`), `held_single_source.jsonl`.
- sram_kb added to the drop 11:53 PT (Sam: "add it"); six attributes, 1,100 parts, drop sha
  `f01ee6f0…`. Open: TI/Microchip agreement not run (no CR
  referee for them yet). CR's 637-doc census on the reissued privacy-gated inventory pending.

## 2026-09-08 12:40 PT — other brands: ordering-code decoders + second-document corroboration
- `harness/electronics/ordering_codes.py`: part-number decoders for STM32, GD32 (same letter
  tables), SiLabs EFM32/EFR32/EFM8, AVR Dx / tinyAVR / megaAVR-0, PIC32, PIC24FJ, Renesas RA.
  `scripts/validate_ordering_codes.py` checks each against the documents: every scheme 100 %
  (n 9…1262). ST pin letter is nominal (WLCSP12 under "D"=14) → corroborates only, never holds.
  MSP430 / PIC16 / PIC18 deliberately undecodable (return {}), not guessed.
- `scripts/promote_document_knives.py` rewritten: referee optional; Tier B = ordering code OR a
  second document (different sha) printing the same value; ordering-code contradiction holds;
  set-valued pins per package; `held.jsonl` with reasons.
- ST drop refreshed (`…/family-matrix-st-20260908/promoted/`, sha `e25d0194…`): 1,166 parts,
  3,770 Tier A + 251 Tier B values. Sent to CR? NOT YET for this refresh.
- Other vendors, Tier B only, `approved_by` = PENDING Samson (six attributes approved for ST
  only): GD 384 parts, SiLabs 349, Microchip 149, Renesas 129, TI 35. Files at
  `/Volumes/M5_4TB/exports/family-matrix-20260908/<vendor>/promoted/`. TI needs CR's referee:
  no decodable scheme and few duplicate documents → 741 flash values single-source.
- Next: Sam's word on the non-ST drops; CR referee for TI/Microchip/GD/Renesas/SiLabs (asked);
  adjudication queue for the frontier teacher.
