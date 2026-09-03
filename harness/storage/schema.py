SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    case_count INTEGER NOT NULL DEFAULT 0,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS case_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    minimum_model_that_solved TEXT,
    successful_tier INTEGER,
    total_escalation_latency_ms REAL,
    total_escalation_cost REAL,
    wasted_latency_before_success_ms REAL,
    wasted_cost_before_success REAL,
    failed_tiers INTEGER,
    started_at TEXT,
    finished_at TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS model_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    model_key TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    tier INTEGER,
    started_at TEXT,
    latency_ms REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost REAL,
    answer_path TEXT,
    raw_path TEXT,
    error TEXT,
    verdict TEXT,
    evaluator TEXT,
    evaluation_detail TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_model_results_run ON model_results(run_id);
CREATE INDEX IF NOT EXISTS idx_model_results_case ON model_results(case_id);
CREATE INDEX IF NOT EXISTS idx_case_runs_run ON case_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_case_runs_case ON case_runs(case_id);

CREATE TABLE IF NOT EXISTS gateway_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT,
    started_at TEXT NOT NULL,
    alias TEXT,
    model_key TEXT,
    upstream_model TEXT,
    stream INTEGER,
    status INTEGER,
    latency_ms REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost REAL,
    error TEXT,
    message_count INTEGER,
    has_tools INTEGER,
    prompt_chars INTEGER,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);
CREATE INDEX IF NOT EXISTS idx_gateway_turns_started ON gateway_turns(started_at);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    intent TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    plan TEXT,
    hypothesis TEXT,
    intervened INTEGER NOT NULL DEFAULT 0,
    frontier_required INTEGER NOT NULL DEFAULT 0,
    stage TEXT NOT NULL DEFAULT 'new',
    frontier_calls INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT,
    final_outcome TEXT
);

CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    worker TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    result TEXT,
    files_changed TEXT,
    commands TEXT,
    tests_passed INTEGER,
    tests_failed INTEGER,
    ttft_ms REAL,
    tokens_per_sec REAL,
    tool_calls INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost REAL,
    UNIQUE(task_id, attempt),
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    attempt INTEGER,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    text TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE INDEX IF NOT EXISTS idx_attempts_task ON attempts(task_id);
CREATE INDEX IF NOT EXISTS idx_evidence_task ON evidence(task_id);
CREATE INDEX IF NOT EXISTS idx_decisions_task ON decisions(task_id);

CREATE TABLE IF NOT EXISTS greenfield_runs (
    run_id TEXT PRIMARY KEY,
    intent TEXT NOT NULL,
    project_name TEXT NOT NULL,
    stack TEXT NOT NULL,
    destination TEXT NOT NULL,
    destination_fingerprint TEXT NOT NULL,
    workspace_root TEXT,
    status TEXT NOT NULL,
    discovery_json TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    manifest_json TEXT,
    spec_hash TEXT NOT NULL,
    plan_hash TEXT NOT NULL,
    manifest_hash TEXT,
    approved_at TEXT,
    current_milestone INTEGER NOT NULL DEFAULT 0,
    final_state_hash TEXT,
    published_path TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS greenfield_milestones (
    run_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    milestone_id TEXT NOT NULL,
    title TEXT NOT NULL,
    objective TEXT NOT NULL,
    acceptance_json TEXT NOT NULL,
    state TEXT NOT NULL,
    task_id TEXT,
    starting_commit TEXT,
    verified_state_hash TEXT,
    commit_sha TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, ordinal),
    UNIQUE (run_id, milestone_id),
    FOREIGN KEY (run_id) REFERENCES greenfield_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE TABLE IF NOT EXISTS greenfield_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES greenfield_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_greenfield_runs_status
    ON greenfield_runs(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_greenfield_milestones_run
    ON greenfield_milestones(run_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_greenfield_events_run
    ON greenfield_events(run_id, id);

CREATE TABLE IF NOT EXISTS learning_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    source_revision TEXT,
    task_id TEXT,
    lineage_id TEXT NOT NULL,
    authorization_scope TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'captured',
    estimated_cost REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    event_sha256 TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE TABLE IF NOT EXISTS learning_artifacts (
    artifact_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    uri TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    redacted INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (event_id, kind, sha256),
    FOREIGN KEY (event_id) REFERENCES learning_events(event_id)
);

CREATE TABLE IF NOT EXISTS learning_verifications (
    verification_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    verifier TEXT NOT NULL,
    command TEXT,
    output_artifact_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES learning_events(event_id),
    FOREIGN KEY (output_artifact_id) REFERENCES learning_artifacts(artifact_id)
);

CREATE TABLE IF NOT EXISTS learning_admissions (
    admission_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    verification_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    reason TEXT NOT NULL,
    source_revision TEXT NOT NULL,
    admission_sha256 TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES learning_events(event_id),
    FOREIGN KEY (verification_id) REFERENCES learning_verifications(verification_id)
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    dataset_version_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL UNIQUE,
    split_policy_json TEXT NOT NULL,
    source_revision TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS dataset_members (
    dataset_version_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    split TEXT NOT NULL,
    lineage_id TEXT NOT NULL,
    source_document_sha256 TEXT,
    repository_id TEXT,
    component_family TEXT,
    temporal_bucket TEXT,
    PRIMARY KEY (dataset_version_id, event_id, artifact_id),
    FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(dataset_version_id),
    FOREIGN KEY (event_id) REFERENCES learning_events(event_id),
    FOREIGN KEY (artifact_id) REFERENCES learning_artifacts(artifact_id)
);

CREATE TABLE IF NOT EXISTS training_jobs (
    job_id TEXT PRIMARY KEY,
    job_kind TEXT NOT NULL,
    dataset_version_id TEXT,
    state TEXT NOT NULL,
    priority REAL NOT NULL,
    observed_frequency REAL NOT NULL DEFAULT 0,
    frontier_cost REAL NOT NULL DEFAULT 0,
    local_failure_rate REAL NOT NULL DEFAULT 0,
    verification_strength REAL NOT NULL DEFAULT 0,
    diversity REAL NOT NULL DEFAULT 0,
    expected_gpu_hours REAL NOT NULL,
    assigned_node TEXT,
    attempt INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    lease_expires_at TEXT,
    lease_token TEXT,
    checkpoint_uri TEXT,
    checkpoint_sha256 TEXT,
    error TEXT,
    config_json TEXT NOT NULL DEFAULT '{}',
    experiment_sha256 TEXT,
    handler_pid INTEGER,
    handler_pgid INTEGER,
    handler_started_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(dataset_version_id)
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    evaluation_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    dataset_version_id TEXT NOT NULL,
    candidate_sha256 TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    decision TEXT NOT NULL,
    result_sha256 TEXT NOT NULL UNIQUE,
    gpu_hours REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES training_jobs(job_id),
    FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(dataset_version_id)
);

CREATE TABLE IF NOT EXISTS replacement_observations (
    observation_id TEXT PRIMARY KEY,
    event_id TEXT,
    evaluation_id TEXT,
    task_class TEXT NOT NULL,
    route TEXT NOT NULL,
    verified_success INTEGER NOT NULL,
    first_pass INTEGER NOT NULL,
    time_to_green_ms REAL,
    repair_cycles INTEGER NOT NULL,
    pinout_exact REAL,
    pinout_leaf_f1 REAL,
    critical_regression INTEGER NOT NULL,
    frontier_escalated INTEGER NOT NULL,
    actual_cost REAL,
    direct_frontier_cost REAL,
    gpu_hours REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES learning_events(event_id),
    FOREIGN KEY (evaluation_id) REFERENCES evaluation_results(evaluation_id)
);

CREATE INDEX IF NOT EXISTS idx_learning_events_state
    ON learning_events(state, created_at);
CREATE INDEX IF NOT EXISTS idx_learning_events_task
    ON learning_events(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_learning_artifacts_event
    ON learning_artifacts(event_id, kind);
CREATE INDEX IF NOT EXISTS idx_learning_verifications_event
    ON learning_verifications(event_id, status);
CREATE INDEX IF NOT EXISTS idx_learning_admissions_decision
    ON learning_admissions(decision, created_at);
CREATE INDEX IF NOT EXISTS idx_dataset_members_split
    ON dataset_members(dataset_version_id, split, lineage_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_queue
    ON training_jobs(state, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_job
    ON evaluation_results(job_id, created_at);
CREATE INDEX IF NOT EXISTS idx_replacement_observations_class
    ON replacement_observations(task_class, created_at);

CREATE TRIGGER IF NOT EXISTS learning_events_no_replace
BEFORE INSERT ON learning_events
WHEN EXISTS (
    SELECT 1 FROM learning_events
    WHERE event_id = NEW.event_id OR event_sha256 = NEW.event_sha256
)
BEGIN
    SELECT RAISE(ABORT, 'learning_events are immutable');
END;

CREATE TRIGGER IF NOT EXISTS learning_events_no_update
BEFORE UPDATE ON learning_events
BEGIN
    SELECT RAISE(ABORT, 'learning_events are immutable');
END;

CREATE TRIGGER IF NOT EXISTS learning_events_no_delete
BEFORE DELETE ON learning_events
BEGIN
    SELECT RAISE(ABORT, 'learning_events are immutable');
END;

CREATE TRIGGER IF NOT EXISTS learning_artifacts_no_update
BEFORE UPDATE ON learning_artifacts
BEGIN
    SELECT RAISE(ABORT, 'learning_artifacts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS learning_artifacts_no_replace
BEFORE INSERT ON learning_artifacts
WHEN EXISTS (
    SELECT 1 FROM learning_artifacts
    WHERE artifact_id = NEW.artifact_id
       OR (
           event_id = NEW.event_id
           AND kind = NEW.kind
           AND sha256 = NEW.sha256
       )
)
BEGIN
    SELECT RAISE(ABORT, 'learning_artifacts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS learning_artifacts_no_delete
BEFORE DELETE ON learning_artifacts
BEGIN
    SELECT RAISE(ABORT, 'learning_artifacts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS learning_verifications_no_update
BEFORE UPDATE ON learning_verifications
BEGIN
    SELECT RAISE(ABORT, 'learning_verifications are immutable');
END;

CREATE TRIGGER IF NOT EXISTS learning_verifications_no_replace
BEFORE INSERT ON learning_verifications
WHEN EXISTS (
    SELECT 1 FROM learning_verifications
    WHERE verification_id = NEW.verification_id
)
BEGIN
    SELECT RAISE(ABORT, 'learning_verifications are immutable');
END;

CREATE TRIGGER IF NOT EXISTS learning_verifications_no_delete
BEFORE DELETE ON learning_verifications
BEGIN
    SELECT RAISE(ABORT, 'learning_verifications are immutable');
END;

CREATE TRIGGER IF NOT EXISTS learning_admissions_no_update
BEFORE UPDATE ON learning_admissions
BEGIN
    SELECT RAISE(ABORT, 'learning_admissions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS learning_admissions_no_replace
BEFORE INSERT ON learning_admissions
WHEN EXISTS (
    SELECT 1 FROM learning_admissions
    WHERE admission_id = NEW.admission_id
       OR event_id = NEW.event_id
       OR admission_sha256 = NEW.admission_sha256
)
BEGIN
    SELECT RAISE(ABORT, 'learning_admissions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS learning_admissions_no_delete
BEFORE DELETE ON learning_admissions
BEGIN
    SELECT RAISE(ABORT, 'learning_admissions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS dataset_versions_no_update
BEFORE UPDATE ON dataset_versions
BEGIN
    SELECT RAISE(ABORT, 'dataset_versions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS dataset_versions_no_replace
BEFORE INSERT ON dataset_versions
WHEN EXISTS (
    SELECT 1 FROM dataset_versions
    WHERE dataset_version_id = NEW.dataset_version_id
       OR manifest_sha256 = NEW.manifest_sha256
       OR (name = NEW.name AND version = NEW.version)
)
BEGIN
    SELECT RAISE(ABORT, 'dataset_versions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS dataset_versions_no_delete
BEFORE DELETE ON dataset_versions
BEGIN
    SELECT RAISE(ABORT, 'dataset_versions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS dataset_members_no_update
BEFORE UPDATE ON dataset_members
BEGIN
    SELECT RAISE(ABORT, 'dataset_members are immutable');
END;

CREATE TRIGGER IF NOT EXISTS dataset_members_no_replace
BEFORE INSERT ON dataset_members
WHEN EXISTS (
    SELECT 1 FROM dataset_members
    WHERE dataset_version_id = NEW.dataset_version_id
      AND event_id = NEW.event_id
      AND artifact_id = NEW.artifact_id
)
BEGIN
    SELECT RAISE(ABORT, 'dataset_members are immutable');
END;

CREATE TRIGGER IF NOT EXISTS dataset_members_no_delete
BEFORE DELETE ON dataset_members
BEGIN
    SELECT RAISE(ABORT, 'dataset_members are immutable');
END;

CREATE TRIGGER IF NOT EXISTS evaluation_results_no_update
BEFORE UPDATE ON evaluation_results
BEGIN
    SELECT RAISE(ABORT, 'evaluation_results are immutable');
END;

CREATE TRIGGER IF NOT EXISTS evaluation_results_no_replace
BEFORE INSERT ON evaluation_results
WHEN EXISTS (
    SELECT 1 FROM evaluation_results
    WHERE evaluation_id = NEW.evaluation_id
       OR result_sha256 = NEW.result_sha256
)
BEGIN
    SELECT RAISE(ABORT, 'evaluation_results are immutable');
END;

CREATE TRIGGER IF NOT EXISTS evaluation_results_no_delete
BEFORE DELETE ON evaluation_results
BEGIN
    SELECT RAISE(ABORT, 'evaluation_results are immutable');
END;

DROP TRIGGER IF EXISTS evaluation_results_stage_once;
CREATE TRIGGER evaluation_results_stage_once
BEFORE INSERT ON evaluation_results
WHEN EXISTS (
    SELECT 1 FROM evaluation_results
    WHERE job_id = NEW.job_id
      AND json_extract(metrics_json, '$.stage')
          IS json_extract(NEW.metrics_json, '$.stage')
)
BEGIN
    SELECT RAISE(ABORT, 'evaluation job stage is immutable');
END;

CREATE TRIGGER IF NOT EXISTS evaluation_results_binding_guard
BEFORE INSERT ON evaluation_results
WHEN NEW.decision NOT IN ('shadow', 'canary', 'promote', 'reject')
  OR json_valid(NEW.metrics_json) != 1
  OR json_extract(NEW.metrics_json, '$.decision.action') IS NOT NEW.decision
  OR (
    json_extract(NEW.metrics_json, '$.stage') IN ('shadow', 'canary')
    AND (
      json_extract(
        NEW.metrics_json,
        '$.evaluation.metadata.dual_run_derived'
      ) IS NOT 1
      OR json_extract(
        NEW.metrics_json,
        '$.evaluation.metadata.dual_run_schema'
      ) IS NOT 'harness.dual-run.v1'
      OR length(json_extract(
        NEW.metrics_json,
        '$.evaluation.metadata.dual_run_log_sha256'
      )) != 64
    )
  )
  OR NOT EXISTS (
    SELECT 1 FROM training_jobs
    WHERE job_id = NEW.job_id
      AND dataset_version_id = NEW.dataset_version_id
      AND checkpoint_sha256 = NEW.candidate_sha256
      AND state = CASE json_extract(NEW.metrics_json, '$.stage')
        WHEN 'offline' THEN 'trained'
        WHEN 'shadow' THEN 'shadow'
        WHEN 'canary' THEN 'canary'
        ELSE '__invalid__'
      END
  )
BEGIN
    SELECT RAISE(ABORT, 'evaluation result is not bound to its job stage');
END;

CREATE TRIGGER IF NOT EXISTS training_jobs_identity_immutable
BEFORE UPDATE OF job_kind, dataset_version_id, priority, observed_frequency,
    frontier_cost, local_failure_rate, verification_strength, diversity,
    expected_gpu_hours, config_json, max_attempts, created_at ON training_jobs
WHEN NEW.job_kind IS NOT OLD.job_kind
  OR NEW.dataset_version_id IS NOT OLD.dataset_version_id
  OR NEW.priority IS NOT OLD.priority
  OR NEW.observed_frequency IS NOT OLD.observed_frequency
  OR NEW.frontier_cost IS NOT OLD.frontier_cost
  OR NEW.local_failure_rate IS NOT OLD.local_failure_rate
  OR NEW.verification_strength IS NOT OLD.verification_strength
  OR NEW.diversity IS NOT OLD.diversity
  OR NEW.expected_gpu_hours IS NOT OLD.expected_gpu_hours
  OR NEW.config_json IS NOT OLD.config_json
  OR NEW.max_attempts IS NOT OLD.max_attempts
  OR NEW.created_at IS NOT OLD.created_at
BEGIN
    SELECT RAISE(ABORT, 'training job identity is immutable');
END;

CREATE TRIGGER IF NOT EXISTS training_jobs_state_machine
BEFORE UPDATE OF state ON training_jobs
WHEN NEW.state != OLD.state
  AND NOT (
    (OLD.state = 'eligible' AND NEW.state IN ('assigned', 'rejected'))
    OR (OLD.state = 'assigned' AND NEW.state IN ('eligible', 'trained', 'rejected'))
    OR (OLD.state = 'trained' AND NEW.state IN ('evaluated', 'rejected'))
    OR (OLD.state = 'evaluated' AND NEW.state IN ('shadow', 'rejected'))
    OR (OLD.state = 'shadow' AND NEW.state IN ('canary', 'rejected'))
    OR (OLD.state = 'canary' AND NEW.state IN ('promoted', 'rejected'))
  )
BEGIN
    SELECT RAISE(ABORT, 'invalid training job state transition');
END;

CREATE TRIGGER IF NOT EXISTS training_jobs_attempt_guard
BEFORE UPDATE OF attempt ON training_jobs
WHEN NOT (
    (OLD.state = 'eligible' AND NEW.state = 'assigned'
     AND NEW.attempt = OLD.attempt + 1)
    OR NEW.attempt = OLD.attempt
)
BEGIN
    SELECT RAISE(ABORT, 'invalid training job attempt mutation');
END;

CREATE TRIGGER IF NOT EXISTS training_jobs_assignment_guard
BEFORE UPDATE ON training_jobs
WHEN (
    NEW.state = 'assigned'
    AND (
      NEW.assigned_node IS NULL
      OR NEW.lease_expires_at IS NULL
      OR NEW.lease_token IS NULL
    )
) OR (
    NEW.state != 'assigned'
    AND (
      NEW.assigned_node IS NOT NULL
      OR NEW.lease_expires_at IS NOT NULL
      OR NEW.lease_token IS NOT NULL
    )
)
BEGIN
    SELECT RAISE(ABORT, 'training job lease fields do not match state');
END;

CREATE TRIGGER IF NOT EXISTS training_jobs_checkpoint_guard
BEFORE UPDATE ON training_jobs
WHEN (
    OLD.checkpoint_sha256 IS NOT NULL
    AND (
      NEW.checkpoint_sha256 IS NOT OLD.checkpoint_sha256
      OR NEW.checkpoint_uri IS NOT OLD.checkpoint_uri
    )
) OR (
    NEW.state = 'trained'
    AND (NEW.checkpoint_uri IS NULL OR NEW.checkpoint_sha256 IS NULL)
)
BEGIN
    SELECT RAISE(ABORT, 'training job checkpoint evidence is invalid');
END;

DROP TRIGGER IF EXISTS training_jobs_evaluation_guard;
CREATE TRIGGER training_jobs_evaluation_guard
BEFORE UPDATE OF state ON training_jobs
WHEN NEW.state = 'evaluated'
  AND (
    SELECT decision FROM evaluation_results
    WHERE job_id = NEW.job_id
      AND json_extract(metrics_json, '$.stage') = 'offline'
  ) IS NOT 'shadow'
BEGIN
    SELECT RAISE(ABORT, 'evaluated state requires passing offline evidence');
END;

DROP TRIGGER IF EXISTS training_jobs_shadow_guard;
CREATE TRIGGER training_jobs_shadow_guard
BEFORE UPDATE OF state ON training_jobs
WHEN NEW.state = 'shadow'
  AND (
    SELECT decision FROM evaluation_results
    WHERE job_id = NEW.job_id
      AND json_extract(metrics_json, '$.stage') = 'offline'
  ) IS NOT 'shadow'
BEGIN
    SELECT RAISE(ABORT, 'shadow state requires passing offline evidence');
END;

DROP TRIGGER IF EXISTS training_jobs_canary_guard;
CREATE TRIGGER training_jobs_canary_guard
BEFORE UPDATE OF state ON training_jobs
WHEN NEW.state = 'canary'
  AND (
    SELECT decision FROM evaluation_results
    WHERE job_id = NEW.job_id
      AND json_extract(metrics_json, '$.stage') = 'shadow'
  ) IS NOT 'canary'
BEGIN
    SELECT RAISE(ABORT, 'canary state requires passing shadow evidence');
END;

DROP TRIGGER IF EXISTS training_jobs_promotion_guard;
CREATE TRIGGER training_jobs_promotion_guard
BEFORE UPDATE OF state ON training_jobs
WHEN NEW.state = 'promoted'
  AND (
    SELECT decision FROM evaluation_results
    WHERE job_id = NEW.job_id
      AND json_extract(metrics_json, '$.stage') = 'canary'
  ) IS NOT 'promote'
BEGIN
    SELECT RAISE(ABORT, 'promotion requires passing canary evidence');
END;

CREATE TRIGGER IF NOT EXISTS replacement_observations_no_update
BEFORE UPDATE ON replacement_observations
BEGIN
    SELECT RAISE(ABORT, 'replacement_observations are immutable');
END;

CREATE TRIGGER IF NOT EXISTS replacement_observations_no_replace
BEFORE INSERT ON replacement_observations
WHEN EXISTS (
    SELECT 1 FROM replacement_observations
    WHERE observation_id = NEW.observation_id
)
BEGIN
    SELECT RAISE(ABORT, 'replacement_observations are immutable');
END;

CREATE TRIGGER IF NOT EXISTS replacement_observations_no_delete
BEFORE DELETE ON replacement_observations
BEGIN
    SELECT RAISE(ABORT, 'replacement_observations are immutable');
END;

DROP TRIGGER IF EXISTS replacement_observations_verified_guard;
CREATE TRIGGER replacement_observations_verified_guard
BEFORE INSERT ON replacement_observations
WHEN (
    NEW.first_pass = 1 AND NEW.verified_success != 1
) OR (
    NEW.verified_success = 1
    AND (
      NEW.event_id IS NULL
      OR NOT EXISTS (
        SELECT 1
        FROM learning_admissions AS admission
        JOIN learning_verifications AS verification
          ON verification.verification_id = admission.verification_id
        WHERE admission.event_id = NEW.event_id
          AND admission.decision = 'eligible'
          AND verification.event_id = admission.event_id
          AND verification.status = 'pass'
      )
    )
)
BEGIN
    SELECT RAISE(
      ABORT,
      'verified replacement requires admitted mechanical proof'
    );
END;
"""
