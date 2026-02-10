PRAGMA user_version = 4;

CREATE TABLE IF NOT EXISTS ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    tick_data TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ticks_run_id ON ticks (run_id);
CREATE INDEX IF NOT EXISTS idx_ticks_run_id_sequence ON ticks (run_id, sequence);

CREATE TABLE IF NOT EXISTS state (
    run_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL DEFAULT '{}',
    state_type TEXT NOT NULL DEFAULT 'DictState',
    state_module TEXT NOT NULL DEFAULT 'workflows.context.state_store',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    event_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run_id_sequence ON events (run_id, sequence);

CREATE INDEX IF NOT EXISTS idx_handlers_run_id ON handlers (run_id);
