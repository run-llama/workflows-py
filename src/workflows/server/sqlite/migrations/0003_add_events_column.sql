PRAGMA user_version=3;

-- Add events column to store event history as JSON
ALTER TABLE handlers ADD COLUMN events TEXT DEFAULT '[]';
