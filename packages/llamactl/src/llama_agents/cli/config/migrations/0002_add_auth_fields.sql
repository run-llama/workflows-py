PRAGMA user_version=2;

-- Add new fields to profiles: api_key_id and device_oidc (stored as JSON string)
ALTER TABLE profiles ADD COLUMN api_key_id TEXT;
ALTER TABLE profiles ADD COLUMN device_oidc TEXT;

-- Add synthetic identifier for profiles
ALTER TABLE profiles ADD COLUMN id TEXT;

-- Populate existing rows with random UUIDv4 values
UPDATE profiles
SET id = lower(
    hex(randomblob(4)) || '-' ||
    hex(randomblob(2)) || '-' ||
    '4' || substr(hex(randomblob(2)), 2) || '-' ||
    substr('89ab', 1 + (abs(random()) % 4), 1) || substr(hex(randomblob(2)), 2) || '-' ||
    hex(randomblob(6))
)
WHERE id IS NULL;

-- Ensure id values are unique
CREATE UNIQUE INDEX IF NOT EXISTS idx_profiles_id ON profiles(id);
