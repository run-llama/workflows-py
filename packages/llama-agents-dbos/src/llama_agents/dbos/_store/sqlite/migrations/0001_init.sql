-- migration: 1

CREATE TABLE IF NOT EXISTS workflow_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    seq_num INTEGER NOT NULL,
    task_key TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workflow_journal_run_id ON workflow_journal (run_id);

CREATE TABLE IF NOT EXISTS run_lifecycle (
    run_id TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'active',
    updated_at TEXT NOT NULL
);
