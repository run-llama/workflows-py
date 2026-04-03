PRAGMA user_version=1;

-- Initial schema for llamactl config database

CREATE TABLE IF NOT EXISTS profiles (
    name TEXT NOT NULL,
    api_url TEXT NOT NULL,
    project_id TEXT NOT NULL,
    api_key TEXT,
    PRIMARY KEY (name, api_url)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS environments (
    api_url TEXT PRIMARY KEY,
    requires_auth INTEGER NOT NULL,
    min_llamactl_version TEXT
);

-- Seed defaults (idempotent)
-- 1) Ensure current environment setting exists (do not overwrite if already set)
INSERT OR IGNORE INTO settings (key, value)
VALUES ('current_environment_api_url', 'https://api.cloud.llamaindex.ai');

-- 2) Backfill environments from any existing profiles (avoid duplicates)
INSERT OR IGNORE INTO environments (api_url, requires_auth)
SELECT DISTINCT api_url, 0 FROM profiles;

-- 3) Ensure the default cloud environment exists with auth required
INSERT OR IGNORE INTO environments (api_url, requires_auth, min_llamactl_version)
VALUES ('https://api.cloud.llamaindex.ai', 1, NULL);
