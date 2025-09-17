-- Create database if not exists
CREATE DATABASE IF NOT EXISTS beer_game CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Create user if not exists and grant privileges
CREATE USER IF NOT EXISTS 'beer_user'@'%' IDENTIFIED BY 'Daybreak@2025';
GRANT ALL PRIVILEGES ON beer_game.* TO 'beer_user'@'%' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON *.* TO 'beer_user'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;

-- Use the database
USE beer_game;

-- Create users table if not exists
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NULL,
    email VARCHAR(100) NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(100) NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
    roles JSON NULL,
    last_login DATETIME NULL,
    last_password_change DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    failed_login_attempts INT NOT NULL DEFAULT 0,
    locked_until DATETIME NULL,
    mfa_secret VARCHAR(100) NULL,
    mfa_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_users_username (username),
    UNIQUE KEY uq_users_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create groups table
CREATE TABLE IF NOT EXISTS groups (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT NULL,
    logo VARCHAR(255) NULL,
    admin_id INT NOT NULL,
    UNIQUE KEY uq_group_admin (admin_id),
    CONSTRAINT fk_group_admin FOREIGN KEY (admin_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS group_id INT NULL,
    ADD CONSTRAINT fk_user_group FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE;

-- Insert default users if they don't exist
-- Password for all users is 'Daybreak@2025' (hashed with bcrypt)
INSERT IGNORE INTO users (username, email, hashed_password, full_name, is_superuser, is_active) VALUES
('systemadmin', 'systemadmin@daybreak.ai', '$2b$12$/FAxQ94QmW1WFdMZd5nKzegYJZkZSi.JUSX/4IvImY3cE2vtleAu6', 'System Admin', TRUE, TRUE)
ON DUPLICATE KEY UPDATE
    email = VALUES(email),
    full_name = VALUES(full_name),
    is_superuser = VALUES(is_superuser),
    is_active = VALUES(is_active),
    updated_at = CURRENT_TIMESTAMP;

-- Verify users were created
SELECT id, username, email, is_superuser, is_active FROM users;
