PRAGMA user_version = 5;

CREATE TABLE IF NOT EXISTS run_lifecycle (
    run_id TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'active',
    updated_at TEXT NOT NULL
);
