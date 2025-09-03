-- Add new columns to users table
ALTER TABLE users
ADD COLUMN last_login DATETIME NULL,
ADD COLUMN last_password_change DATETIME NULL,
ADD COLUMN failed_login_attempts INT NOT NULL DEFAULT 0,
ADD COLUMN is_locked BOOLEAN NOT NULL DEFAULT FALSE,
ADD COLUMN lockout_until DATETIME NULL,
ADD COLUMN mfa_secret VARCHAR(255) NULL,
ADD COLUMN mfa_enabled BOOLEAN NOT NULL DEFAULT FALSE;

-- Create password_history table
CREATE TABLE IF NOT EXISTS password_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create password_reset_tokens table
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    token VARCHAR(255) NOT NULL,
    expires_at DATETIME NOT NULL,
    is_used BOOLEAN NOT NULL DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY (token)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Update the alembic_version table to mark this migration as applied
-- This is a workaround since we're not using alembic for this migration
-- Replace '1234abcd5678' with the actual revision ID you want to use
-- If the table doesn't exist, this will be skipped
INSERT IGNORE INTO alembic_version (version_num) VALUES ('1234abcd5678');
