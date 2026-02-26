-- migration: 1

CREATE TABLE IF NOT EXISTS workflow_journal (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL,
    seq_num INTEGER NOT NULL,
    task_key VARCHAR(512) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workflow_journal_run_id ON workflow_journal (run_id);

CREATE TABLE IF NOT EXISTS run_lifecycle (
    run_id VARCHAR(255) PRIMARY KEY,
    state VARCHAR(20) NOT NULL DEFAULT 'active',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS executor_leases (
    slot_id TEXT PRIMARY KEY,
    holder TEXT,
    heartbeat_at TIMESTAMPTZ,
    acquired_at TIMESTAMPTZ
);
