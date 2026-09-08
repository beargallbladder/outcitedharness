# Agent onboarding — take over Harnessv1 from the 2026-09-07 agent

Written 2026-09-07 19:30 PT by the outgoing agent, for the agent replacing it.
Goal: after working through this you should be able to operate the fleet, the
datasheet factory, and Sam's local coder lane without asking him anything he
has already answered.

## 1. Repo and access

| Item | Value |
|---|---|
| GitHub | https://github.com/beargallbladder/outcitedharness (**public**, branch `main`) |
| Cursor origin | `https://origin.cursor.com/samkim2dgx/harnessv1.git` (same content, pushed in lockstep) |
| HEAD at handoff | `0d982c16` "handoff 2026-09-07: CX7 power-throttle fix, NFS phase 0 on dgx2, coder-next lane on spark, asus2/3 cleanup" |
| Working copy | `/Users/samkim/Harnessv1` on `m5max-ai` (Sam's M5 Max, the ops box). Tree is clean; both remotes at parity. |
| Python | `uv run --python 3.11 ...` for everything. `uv run --python 3.11 python -m pytest -q` = 697 passed in 16 s at handoff. |
| Author | All 66 commits are authored as Sam Kim; agents commit under his identity. Keep doing that. |

Because the repo is public: **no secrets, keys, passwords, or customer data ever
go in**. Tailscale IPs and hostnames are already in the docs and are fine.
`results/` is gitignored on purpose (hash-sealed sqlite/evidence state lives
only on the M5 disk and `/Volumes/M5_4TB`).

Things you need from Sam that are not in the repo:

- SSH: your public key in `~/.ssh/authorized_keys` on each box (Sam adds it).
  Usernames differ per box (table below). Test with `ssh <alias> hostname`.
- Tailscale membership, or be on the 192.168.4.0/22 LAN.
- `~/.secrets/qwen38-key` (Flash-Next :8888), `~/.secrets/coder-next-key`
  (Coder-Next :8900), `~/.secrets/crs812-admin` (switch RouterOS password).
  All 0600, all on m5max-ai; the coder key also on spark and the M4.
- Anthropic key in env for Message Batch teacher runs (see `.env.example`).
- Push rights on GitHub if you are not running as Sam's `gh` login.

## 2. Reading order

1. `HANDOFF-20260907-qwen-cluster.md` — what happened today: Flash-Next
   recipe/patches, shadow-lane kill, spark kernel-panic incident, fabric
   throttle root cause and fix, NFS phase 0, Coder-Next lane. This is the
   authoritative *current state*; it overrides older docs where they disagree.
2. `NETWORK_FABRIC.md` — the 200G MikroTik fabric, port map, qualification,
   and the CX7 throttle known-issue block.
3. `DATASHEET_FACTORY.md` — the product. Corpus, teacher/verification
   contracts, the 30B learning cycle history, runbook. Read "Hard boundaries"
   and "PDF extraction contract" twice.
4. `LEARNING_FACTORY.md` — the data/training governance: immutable capture,
   never-train holdouts, capability ladder, frozen evaluation, promotion.
5. `QUALIFICATION.md` — what has been qualified and what is fail-closed.
6. `ARCHITECTURE.md` — runtime path, gateway, config ownership. **Its "Live
   allocation" section is stale** (written before 09-03); see §5 here.
7. `README.md` — CLI surface (`harness ...`), case format, repo verification
   contract.
8. Code: `harness/electronics/` (the factory: `extraction.py`, `claims.py`,
   `holdout.py`, `factory_control.py`, `frontier_batch.py`, `qualification.py`),
   then `scripts/run_datasheet_factory_supervisor.py`,
   `scripts/datasheet_frontier_batch.py`, `scripts/run_*_fleet.sh`,
   `scripts/harvest_teacher_batches.sh`. `git log --since=2026-09-03` is a
   good map of what each script exists for.

## 3. What this project is (mental model)

Two independent lanes share the hardware.

**Lane A — datasheet extraction factory (the product, for CategoryRank / "CR").**
Electronics datasheet PDFs (8,870-file sealed corpus) go through a
fail-closed pipeline: text-first extraction (PyMuPDF, `pdftotext -layout`),
geometry/locator, local Qwen3-VL-30B vision only for focused table crops, then
Anthropic Message Batch as *teacher* with independent source verification of
every fact. Only verified facts become SFT/DPO pairs and embedded claims.
Everything is receipted, hash-sealed, and cost-capped. Output to CR is
per-node deliverables (pin tables, parametric values, series summaries,
apps-verbatim) plus the trained local model that replaces paid teacher calls.
Recent threads (09-03 -> 09-07): MCU pin v3/v4 fleet runs, TI power pin lane
(borderless tables), apps-verbatim v1, document scope reader v2, pin contract
`package_as_read`, CR 40-doc pin holdout quarantined as never-train
(`configs/holdout-never-train-shas-20260906.json`), round-4 dataset recipe.

**Lane B — Sam's local coder for opencode.** Qwen3.8-Flash-Next NVFP4 on
asus2+asus4 (SGLang TP=2, :8888) is his daily driver from the M4 via
OpenCode.app. Qwen3-Coder-Next NVFP4 on spark (vLLM, :8900) serves
`small_model` and explore/general subagents. Sam is explicit: he is not
training to beat Cursor; he just wants Qwen for opencode. The Cursor shadow
lane that tried to learn from his edits was killed today.

## 4. Fleet

SSH aliases are in `~/.ssh/config` on m5max-ai. LAN addresses are DHCP; prefer
aliases or Tailscale IPs.

| Alias | Hostname | Tailscale | LAN | User | Role (as of 09-07 19:00) |
|---|---|---|---|---|---|
| `spark` (DGX1) | spark-e10b | 100.81.201.24 | 192.168.4.38 | samkim | **Protected**: CR bge-m3 FAE v4 embedder `:8800` (uvicorn, `/data/categoryrank/cr-finetune`), GCI `:8810`. Plus Coder-Next vLLM `:8900` (new today). Not on the 200G fabric. |
| `dgx2` | spark-49af | 100.116.221.82 | 192.168.4.45 | samkim2 | Training pool; immutable datasets/checkpoints owner; **NFS server** `/srv/models` (TCP :2049 + RDMA :20049, exported to 10.77.0.0/24 and 10.77.1.0/24). **No inference lanes here.** 2.3 TB free. |
| `dgx3` | spark-69c8 | 100.73.119.63 | 192.168.4.49 | samkim3 | Training pool; vision v4 vLLM `:8912` (`qwen3-vl-30b-pin-gate-v4-vllm`). |
| `asus1` | gx10-fc2e | 100.124.181.13 | 192.168.4.58 | samkimasus1 | Training pool; vision v4 `:8912`. |
| `asus3` | gx10-0309 | 100.89.118.36 | .local | samkimasus3 | Training pool; vision v4 `:8912`; NFS client `/mnt/models` (ro, rdma, hand-mounted). LaCie external at `/mnt/lacie` (ro). |
| `asus2` | gx10-26b6 | 100.68.133.1 | .local | samkimasus2 | Flash-Next **head** `:8888` (`~/qwen38-sglang-recipe/`). |
| `asus4` | gx10-33af | 100.100.116.82 | 192.168.4.48/.56 | samkimasus4 | Flash-Next **worker**. |
| — | m5max-ai | 100.106.13.24 | — | samkim | This repo, ops, `harness`, datasheet supervisor process. |
| — | macbook-pro (M4) | 100.102.121.37 | — | **samsonkim** | Sam's laptop; OpenCode.app; config `~/.config/opencode/opencode.jsonc`. |
| — | samsons-mac-mini | 100.80.101.116 | — | ? | Has a 16 TB external drive Sam offered for agent work. Not yet used; no SSH verified. |

Sam's stated box roles: asus1/asus3/dgx2/dgx3 = training/extraction pool;
asus2/asus4 = coder pair; spark = embedder (+ the coder lane he approved
today). **Do not put lanes on dgx2.**

Fabric: MikroTik CRS812 (`192.168.4.50`, DHCP), 200G RoCE `10.77.0.0/24`
(asus1 .2, dgx2 .1, dgx3 .3, asus3 .4, asus2 .5, asus4 .6) plus mirror
`10.77.1.0/24` on the second port function. All four QSFP cages used; 8x SFP56
free. Management NICs negotiate 1G on a dumb LAN switch.

## 5. Current state (2026-09-07 19:00 PT) — supersedes ARCHITECTURE.md §Live allocation

- Nothing is training. All GPUs 0%. Vision v4 servers on asus1/asus3/dgx3 up
  ~3 h (post power-drain reboot), zero requests in the last 30 min.
- Datasheet supervisor: `scripts/run_datasheet_factory_supervisor.py --config
  deploy/datasheet_factory_image_only_v1.json` running on m5max-ai (PID
  ~98778, 5 days). Both frontier runs finalized (`cr-mcu-opn-vision-v3`,
  `cr-mcu-parametric-vision-v2`), 0 leases, 57 SFT / 57 DPO pairs prepared,
  `ready_to_stage: false`. It is idle-polling, not driving work. State in
  `results/datasheet-image-only-supervisor-v1-20260902/`.
- Flash-Next :8888 up with today's patch set (sglang #36806/#36845/#37110/
  #35821), radix cache on, spec decode on, CUDA graphs off. Qualified 11:44.
  Watch items: prefix-cache hits on Sam's real traffic; ghost-node accept-rate
  decay over ~24 h.
- Coder-Next :8900 on spark up ~2 h, 62.5 tok/s single stream, no spec decode.
  Embedder p50 15 ms idle -> 37 ms with 3 coder streams.
- Fabric: all six nodes power-drained today; ~111 Gb/s per PCIe half verified.
- Deleted today (Sam approved): DeepSeek V4 weights on asus2 and asus3,
  Nemotron 3.5 Lightning weights + container on asus3, stale sglang images.
  Nemotron on asus3 :8900 no longer exists despite what ARCHITECTURE.md says.
- Stale in `ARCHITECTURE.md` §Live allocation: "ASUS2+ASUS4 stopped" (they
  are serving), "ASUS3 :8900 Nemotron" (deleted), "DGX3 :8902" (vision is now
  :8912 on three boxes). `config/models.yaml` also lists five dead `:8900`
  endpoints. Fixing these docs is a good first PR.

## 6. Hard rules (Sam's, and learned the hard way)

1. **Never train on holdout shas** in `configs/holdout-never-train-shas-20260906.json`
   and the frozen cohorts. Promotion is fail-closed; do not loosen a gate to
   make a candidate pass.
2. **spark :8800 embedder is CR's production.** Never restart, flip its
   symlink, repurpose, or use it as a Tapes v1 baseline. spark rebooted once
   today from an agent's PCI rescan (see incident); do not repeat.
3. **Never `echo 1 > /sys/bus/pci/rescan` or `mstfwreset` on a Spark/GX10
   remotely.** The `cx7-pcie-hotplug` driver powers the CX7 off the bus when
   uncabled; rescanning it kernel-panics the box.
4. **Do not send test inference to :8888 while Sam is working.** It queues
   behind his turns and he notices. `GET /v1/models` is fine. Ask, or test
   against :8900 instead.
5. **Do not delete model weights, rollback launchers, or `.bak-*` files
   during cleanup** without a written proposal Sam approves. Today's deletes
   were proposed as a list first; he replied "approved".
6. **No secrets in repo or shell history.** Use `~/.secrets/<name>` (0600) and
   `{file:~/.secrets/...}` / `$(cat ...)`.
7. **Frontier spend is capped per batch** (`--spend-cap-usd`, explicit). Never
   submit a teacher batch without a cap and without Sam knowing the number.
8. **Every state transition in the factory is an immutable, receipted step**
   (prepare / submit / status / retrieve / reconcile / finalize). Do not write
   to the sqlite by hand.
9. Keep the git tree clean and both remotes pushed at end of session. Sam
   checks. Update the dated HANDOFF file as you go, not at the end.

## 7. How work actually gets done

- Fleet runs: `scripts/run_*_fleet.sh` fan a work queue across the vision v4
  boxes; `scripts/harvest_teacher_batches.sh` retrieves/reconciles/verifies/
  finalizes Anthropic batches when they complete (designed to run overnight).
- Frontier batch lifecycle: `scripts/datasheet_frontier_batch.py {prepare,
  submit,status,retrieve,reconcile,finalize}`.
- Qualification: `scripts/qualify_datasheet_factory.py` with
  `deploy/training/datasheet_factory_qualification.yaml`; vision compare via
  `scripts/compare_datasheet_vision_qualification.py`.
- Training: `deploy/training/*.Dockerfile` (LlamaFactory GB10, SpecForge,
  qwen38-lora), six-rank NCCL over the fabric, BF16 pinned Qwen3-VL-30B
  checkpoint (FP8 path rejected). Handoff prep in
  `scripts/prepare_electronics_30b_training_handoff.py` and
  `scripts/training_launch_electronics_30b_handoff.sh`.
- Flash-Next ops: on asus2 `cd ~/qwen38-sglang-recipe && ./stop.sh && ./start.sh`
  (~5 min), qualify with `/tmp/qualify_qwen38.py` (not in repo; recreate from
  the handoff's gate list if `/tmp` was cleared).
- Coder-Next ops: `ssh spark docker {logs,restart,stop} qwen3-coder-next`.
- Fabric check: `ib_write_bw -d rocep1s0f1 -R -F -q 4 -s 65536 -D 5
  --report_gbits <peer 10.77.0.x>` with the box under test as client. Expect
  ~111 Gb/s; ~13 Gb/s means the throttle is back -> physical power drain.
- Switch: RouterOS REST at `https://192.168.4.50/rest/...` with the admin
  password from `~/.secrets/crs812-admin`. Read-only unless Sam approves.

## 8. Open items and skeletons

1. CR heads-up about the Coder-Next container sharing spark's GPU with their
   embedder: drafted at `/tmp/cr-heads-up-coder-next.md` on m5max-ai, **not
   sent**, and `/tmp` does not survive reboot. Sam has to approve the send.
2. No monitor for CX7 throttle recurrence. A cron `ib_write_bw` probe would
   catch a silent regression to 13 Gb/s.
3. NFS is phase 0: no `fstab` on any client, asus3 mount is by hand, asus2 not
   mounted. Phase 1 = decide what moves to `/srv/models`, fstab with
   `soft,ro` on clients, then cut over vision/coder weight paths.
4. Coder-Next has no DFlash draft model, so no spec decode (62 tok/s). The
   draft (`qwen3-coder-next-dflash`) plus an sm121 sglang build would roughly
   double it if Sam finds it slow.
5. Flash-Next ghost-node accept-rate decay: re-check head logs after 24 h of
   real use; if accept rate falls toward 0, sglang #35821 did not fully fix it.
6. Datasheet supervisor is finalized but still running; the next factory
   round (round-4 cumulative dataset, corrected pin cohort `v6` base-vs-
   candidate eval) has not been started. `DATASHEET_FACTORY.md` §Current 30B
   learning cycle ends on "a fresh base-versus-candidate run is required".
7. `ARCHITECTURE.md` and `config/models.yaml` drift (see §5).
8. Mac mini 16 TB drive offered but unused; no SSH from m5max-ai verified.
9. The M4's `~/.opencode/bin/opencode` is a July-2025 CLI (0.3.61). Sam uses
   OpenCode.app 1.18.29. Scripts against the CLI there see no custom providers.
10. Spark :8800 blipped ~2.5 min at 13:55 PT today (our fault). Unreported to CR.

## 9. Working with Sam

- Terse. "approved", "yeah do it", "done", "GO". Match it: lead with the
  answer, one screen, no preamble.
- Wants a numbered proposal before anything destructive, then a one-word
  approval. Then execute all of it, not part of it.
- Wants to know about anything that could bite him: hidden state, unsent
  messages, stale docs, unpushed commits. Ask yourself "where are the
  skeletons?" before saying you are done.
- He notices latency on his coder endpoint. Don't touch :8888 during his day.
- He will hand you physical tasks (unplug/replug) and say "done"; verify
  immediately and report the number.
- Cluster roles are his call. Today he corrected an agent that put a lane on
  the embedder box without spelling out the role tradeoff first. State the
  tradeoff, then let him decide.

## 10. First-day checklist to become useful

1. `git clone`, `uv run --python 3.11 python -m pytest -q` (expect 697 pass).
2. `for h in spark dgx2 dgx3 asus1 asus2 asus3 asus4; do ssh $h hostname; done`.
3. `ssh asus2 docker logs --tail 50 qwen38-flash-next-head` — find accept rate.
4. `curl -s http://100.81.201.24:8900/v1/models -H "Authorization: Bearer $(cat ~/.secrets/coder-next-key)"`.
5. `python3 -c 'import json;print(json.load(open("results/datasheet-image-only-supervisor-v1-20260902/supervisor-status.json"))["runs"][0]["result"]["counts"])'`.
6. Read `harness/electronics/holdout.py` and `configs/holdout-never-train-shas-20260906.json` until you can explain the never-train contract.
7. Read `git show 60f60e8c 678bed70 bad72968` (the last three factory commits) to see the current pin-contract work.
8. Run one `ib_write_bw` probe from asus3 to dgx2 to see 111 Gb/s with your own eyes.
9. Fix the `ARCHITECTURE.md` §Live allocation drift as your first commit.
