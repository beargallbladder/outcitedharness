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

CREATE TABLE IF NOT EXISTS cline_turns (
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
CREATE INDEX IF NOT EXISTS idx_cline_turns_started ON cline_turns(started_at);

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
"""
