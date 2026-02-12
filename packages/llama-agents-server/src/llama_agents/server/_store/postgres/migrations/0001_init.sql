-- migration: 1

CREATE TABLE IF NOT EXISTS wf_handlers (
    handler_id VARCHAR(255) PRIMARY KEY,
    workflow_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    run_id VARCHAR(255),
    error TEXT,
    result TEXT,
    started_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    idle_since TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS wf_events (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL,
    sequence INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    event_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_wf_events_run_id ON wf_events (run_id);
CREATE INDEX IF NOT EXISTS idx_wf_handlers_run_id ON wf_handlers (run_id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'uq_wf_events_run_id_sequence'
    ) THEN
        ALTER TABLE wf_events
            ADD CONSTRAINT uq_wf_events_run_id_sequence UNIQUE (run_id, sequence);
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS workflow_state (
    run_id VARCHAR(255) PRIMARY KEY,
    state_json TEXT NOT NULL,
    state_type VARCHAR(255),
    state_module VARCHAR(255),
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS workflow_journal (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL,
    seq_num INTEGER NOT NULL,
    task_key VARCHAR(512) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workflow_journal_run_id ON workflow_journal (run_id);
