PRAGMA user_version=2;

-- Add new columns for extended handler persistence
ALTER TABLE handlers ADD COLUMN run_id TEXT;
ALTER TABLE handlers ADD COLUMN error TEXT;
ALTER TABLE handlers ADD COLUMN result TEXT;
ALTER TABLE handlers ADD COLUMN started_at TEXT;
ALTER TABLE handlers ADD COLUMN updated_at TEXT;
ALTER TABLE handlers ADD COLUMN completed_at TEXT;
