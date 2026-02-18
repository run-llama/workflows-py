PRAGMA user_version=1;

-- Initial table creation matching the original minimal schema
CREATE TABLE IF NOT EXISTS handlers (
    handler_id TEXT PRIMARY KEY,
    workflow_name TEXT,
    status TEXT,
    ctx TEXT
);
