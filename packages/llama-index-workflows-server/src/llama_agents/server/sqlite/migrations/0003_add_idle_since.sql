PRAGMA user_version=3;

-- Add idle_since column for tracking when a workflow became idle
ALTER TABLE handlers ADD COLUMN idle_since TEXT;
