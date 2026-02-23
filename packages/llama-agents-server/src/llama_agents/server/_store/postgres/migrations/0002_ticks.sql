-- migration: 2

CREATE TABLE IF NOT EXISTS wf_ticks (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL,
    sequence INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    tick_data JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_wf_ticks_run_id ON wf_ticks (run_id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'uq_wf_ticks_run_id_sequence'
    ) THEN
        ALTER TABLE wf_ticks
            ADD CONSTRAINT uq_wf_ticks_run_id_sequence UNIQUE (run_id, sequence);
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS run_lifecycle (
    run_id VARCHAR(255) PRIMARY KEY,
    state VARCHAR(20) NOT NULL DEFAULT 'active',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
