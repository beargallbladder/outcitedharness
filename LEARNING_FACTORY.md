# Dedicated learning factory

The factory converts owned, mechanically verifiable failures and successful
repairs into immutable candidates for local-model improvement. It is not
allowed to train continuously merely to keep GPUs busy. When no eligible,
verified work exists, workers idle rather than consuming untrusted data.

## Fixed topology

- **DGX2** owns the authoritative ledger, datasets, checkpoints, manifests,
  evaluation records, and queue.
- **ASUS1** is disposable compute and scratch. It may read a staged snapshot
  and return hashed artifacts; it is never the authoritative writer.
- The two nodes communicate over the qualified dual-rail 10.77.0.0/24 and
  10.77.1.0/24 ConnectX-7 link.
- **M5** captures operational events and may prepare sealed quarantine
  bundles. A bundle is not eligible training data until it is imported,
  verified, split, and registered on DGX2.

## Hard data boundary

CategoryRank and Tapes are excluded from capture, backfill, connector scopes,
dataset registration, and training. Their earlier experiments remain frozen
historical evidence only. Do not process them until the owner provides new,
explicit guidance for cleanliness, lineage, and approved use.

Tokens, passwords, private keys, and credentials never enter SQLite, job
configuration, manifests, or command lines. Sensitive request and response
content is redacted and stored in the private content-addressed artifact vault;
SQLite stores only hashes and pointers.

## Datasheet teacher-data mandate

The electronics lane is specified in `DATASHEET_FACTORY.md`. Anthropic Message
Batch spend in that lane is an investment in owned local training pairs, not a
permanent production dependency. Requests are eligible only after deterministic
and local-model attempts, carry an explicit batch spend cap, and remain
unadmitted until claim-level verification. Promotion requires measured local
capability gain and at least 70% paid-call replacement on frozen family- and
document-safe holdouts.

## Immutable capture and verification

The ledger is opt-in. `learning_capture_enabled` remains `false` until the
operator intentionally enables gateway capture. Append-only database triggers
reject updates and deletes for events, artifacts, verifications, dataset
versions, members, and evaluation records.

Verify all event records and external object hashes:

```shell
python scripts/verify_learning_ledger.py \
  --database /home/samkim2/harness-training/ledger/learning.db \
  --artifact-root /home/samkim2/harness-training/ledger/artifacts \
  --output /home/samkim2/harness-training/manifests/ledger-verification.json
```

A usable learning event needs a source URI and immutable revision, an
authorization scope, a lineage ID, the problem artifact, the attempted
solution, the verified result, and a proof artifact. Missing provenance or
proof remains quarantined.

## Backfill

`scripts/backfill_learning_factory.py` supports:

- audited DesignWins canonical JSONL, quarantined by default;
- single-parent commits only from `settings.code_index_repos`;
- complete Harness PASS answer artifacts and verified greenfield commits;
- revision-, ownership-, and proof-digest-bound Cursor envelopes;
- digest-bound CI failure-to-green envelopes; and
- per-record fail-closed inventory of incomplete legacy Harness rows.

`--admit-audited-designwins` is a separate, explicit action. Do not use it for
backlog construction. Raw Cursor chat is not a learning envelope and remains
rejected unless it carries a repository revision, self-ownership assertion,
prompt/response pair, and matching proof-output digest. Imported Cursor and CI
proof claims remain `unknown` until independently replayed.

Git history is captured as quarantine candidates, not automatic training
examples. Replaying the parent state and reproducing fail-before/pass-after is
required before promotion to `verified`. Merge commits and root commits are
rejected by the current importer.

Every backfill run writes a report with captured, duplicate, and rejected
counts plus exact rejection reasons. Seal candidate bundles with
`scripts/training_manifest.py` before transfer to DGX2.

The 2026-08-31 M5 quarantine snapshot contains 1,298 events and 3,828 artifact
pointers: 1,101 audited DesignWins train records, 128 complete Harness PASS
candidates, 66 owned Git-history candidates, and 3 greenfield commits. Every
event is quarantined and there are zero eligible admissions. All 1,298 events
and artifact hashes passed verification; the sealed evidence is under
`results/m5-quarantine-backfill-20260831/`. The raw Cursor history produced no
eligible envelope, and no CI source is configured; both facts are recorded as
rejections rather than implied backlog.

## Dataset versions and splits

`DatasetVersionRegistry` writes an immutable manifest over every member and
enforces no overlap by both lineage ID and source-document hash. DesignWins v3
uses its audited preassigned train, validation, and test splits:

- train: 1,101 examples;
- validation: 127 examples;
- test: 141 examples.

Register it with:

```shell
python scripts/register_designwins_dataset_version.py \
  --database /home/samkim2/harness-training/ledger/learning.db \
  --dataset-root /home/samkim2/harness-training/datasets/designwins-v3-20260829/canonical/text \
  --source-revision COMMIT_SHA
```

Never random-split examples after generation. Repository, component family,
datasheet, lineage, and time boundaries belong in the immutable split policy.
Every key named by `split_policy.leakage_keys` is required on every member and
may belong to only one split.

## Durable queue and workers

The queue orders eligible jobs by class before numeric priority:

1. production-failure replay;
2. frozen evaluation;
3. main-model LoRA/QLoRA;
4. electronics text or vision;
5. ablation and hyperparameter sweeps;
6. SpecForge draft optimization.

Within a class, priority is:

```text
(frequency × frontier_cost × local_failure_rate × verification_strength × diversity)
÷ expected_gpu_hours
```

Claims use expiring leases. Workers renew leases while a direct, allowlisted
executable runs. Expired work returns to `eligible` until `max_attempts`, then
becomes `rejected`. Worker handlers cannot be shell snippets or job-supplied
commands. The queue persists each handler PID/process group. An expired job
with an attached process is not reassigned; a restarted worker on the owning
node first verifies and terminates that process group, then releases the job.

Pre-eligible states live in the immutable event/admission ledger. Queue jobs
are born only in `eligible`; database triggers reject forged initial states and
job deletion. The queue also hashes dataset + job kind + declared config and
rejects an identical experiment forever. A purposeful ablation must change
the immutable dataset version or its explicit experiment config.

Inspect and administer jobs with `scripts/training_queue_admin.py`. Run a
worker only after replacing the paths in
`deploy/training/worker-handlers.example.yaml` with installed, reviewed,
executable launchers:

```shell
python scripts/training_queue_worker.py \
  --database /home/samkim2/harness-training/ledger/learning.db \
  --handlers /home/samkim2/harness-training/configs/worker-handlers.yaml \
  --node dgx2 \
  --log-root /home/samkim2/harness-training/runs/worker
```

An empty queue is a valid safe state. Build backlog by verification, not by
weakening admission gates.

## Capability ladder

`deploy/training/capability-ladder.yaml` encodes the non-negotiable gates.

- The Qwen3-8B electronics adapter must pass load, optimizer step,
  adapter-only save, resume, frozen holdout, family non-regression, and
  deterministic reproduction.
- A 30–35B coding adapter requires at least 500 mechanically verified repairs,
  then load/step/save/resume, frozen regression, shadow, and canary gates.
- Qwen3-Coder-Next 80B is conditional. The NVFP4 serving artifact is not a
  trainable checkpoint. Two-node FSDP-QLoRA must independently pass per-rank
  load, optimizer step, adapter-only save, and cross-node resume before a full
  job is eligible.
- Qwen3.8-Flash-Next is a separate four-node target and must use its BF16
  checkpoint. Stock ZeRO-3 is prohibited: Transformers concatenates the 128
  checkpoint pieces into one `320001536 x 160` BF16 PLE embedding (95.37 GiB),
  which ZeRO gathers on each rank for lookup. Training is eligible only after
  native tensor parallelism or a file-backed implementation proves that PLE
  remains sharded, completes a short-context optimizer step with finite LoRA
  gradients, and saves an adapter-only artifact. Long-context training remains
  blocked until QSA has a sparse backward-capable implementation.
- The native-TP mechanism passed those initial gates on 2026-09-01 at
  sequence length 8: 228 LoRA modules, 456 adapter tensors, 13,052,928
  trainable elements, identical post-step state on all four ranks, and a
  25 MiB adapter-only artifact. This is not yet a capability result or
  authorization for a full curriculum; the lowest post-step free-memory
  reading was 0.908 GiB. Evidence is sealed in
  `results/qwen38-native-tp-load-smoke-20260901.json` and
  `results/qwen38-native-tp-lora-smoke-20260901.json`.
- SpecForge improves draft speed after a quality winner exists. It cannot
  establish a capability gain.

Use `scripts/training_capability_doctor.py` with hash-bearing evidence. A
missing gate is a rejection, never an implied pass.

## Frozen evaluation and promotion

Evaluation inputs and scoring code are immutable for a candidate run.
`scripts/record_model_evaluation.py` records generic offline results; electronics
must use the sealed DesignWins recorder. A job can have only one immutable
result per stage, so a rejection cannot be replaced by a later pass.
The policy rejects critical regressions, excess latency or cost, insufficient
electronics F1 gain, invalid JSON regression, family regression, and
non-reproducible results.

Passing offline evaluation authorizes shadowing. Passing shadow authorizes a
small canary. Production promotion still requires the canary and an explicit
route change; training completion alone never changes a live endpoint.
Shadow and canary metrics must be derived with
`scripts/record_dual_run_evaluation.py` from paired JSONL evidence. The
collector hashes the evidence, enforces unique request IDs and minimum sample
counts, and rejects hand-authored stage metrics.

The same-day DesignWins proof uses all 141 held-out examples for the frozen
base, the candidate, and an independent repeat. The strict gate requires at
least +0.05 mean leaf F1, no valid-JSON regression, no eligible-family
regression beyond 0.02, no generation-limit hit, and identical metric
fingerprints across repeats.

The 2026-08-30 Qwen3-8B LoRA run proved a real training signal but failed the
production gate. Across all 141 held-out records and 394,894 response tokens,
teacher-forced mean token NLL improved from 0.364230 to 0.245597 (32.57%),
token accuracy improved from 91.78% to 93.10%, all 141 records improved, and
the independent cross-node repeat was bit-for-bit identical. The corrected
eight-record autoregressive rejection canary then regressed mean leaf F1 from
0.273016 to 0.044693, regressed valid JSON from 62.5% to 12.5%, hit the
generation limit on seven records, and did not reproduce exactly. The
checkpoint was therefore recorded as rejected and was never eligible for
shadow or production. Teacher-forced evidence can justify another training
iteration; it cannot substitute for the frozen autoregressive promotion gate.
An equal-budget diagnostic did not rescue the result: candidate mean leaf F1
was 0.404926 versus the base model's 0.401704, only +0.003222 against the
required +0.05, valid JSON regressed from 100% to 87.5%, one generation still
hit its limit, and the independent candidate fingerprint differed.
The postmortem found that the 4,096-token training cutoff truncated 1,058 of
1,101 training records (96.1%); the median combined sequence is 8,680 tokens
and the maximum is 23,216. Future DesignWins jobs must run
`scripts/audit_designwins_sequence_lengths.py --fail-on-truncation` before GPU
assignment. The next dataset version must chunk source and response records so
the complete JSON and EOS token fit the declared cutoff; raising the cutoff
without first proving GB10 memory headroom is not an acceptable workaround.

## Replacement economics

Paid-spend replacement is counted only when a local result has mechanical
proof, no critical regression, and no frontier escalation. Every claimed
verified replacement must reference an admitted learning event whose bound
verification passed. Record observations with
`scripts/record_replacement_observation.py`, then write the report:

```shell
python scripts/learning_factory_report.py \
  --database /home/samkim2/harness-training/ledger/learning.db \
  --output /home/samkim2/harness-training/runs/learning-factory-metrics.json
```

The report separates verified success rate, paid replacement rate by task
class, frontier escalation rate, mean time to green, known paid spend avoided,
and spend avoided per known GPU-hour. Unknown paid cost does not inflate
savings.

## External read-only connectors

`config/learning-connectors.example.yaml` is disabled by default. Each
connector requires HTTPS, a known provider host, an allowlisted path, GET-only
access, and a token supplied through a named environment variable. Never put a
token in YAML, a URL, chat, email, or a command argument.

Copy the example to a private operator file, narrow paths to owned resources,
set only the required environment variable, and then use
`scripts/capture_learning_connector.py`. Responses still enter quarantine and
must pass provenance, ownership, secret scanning, and mechanical verification.

The CR request contract is
`deploy/training/cr-learning-data-request.yaml`. It requests verified code
repairs and electronics pinout resolutions, explicitly excludes CategoryRank
and Tapes, and requires encrypted transfer plus a SHA-256 manifest.
