-- Minimal bootstrap for MariaDB: create database and application user.
CREATE DATABASE IF NOT EXISTS beer_game CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'beer_user'@'%' IDENTIFIED BY 'Daybreak@2025';
GRANT ALL PRIVILEGES ON beer_game.* TO 'beer_user'@'%';
GRANT ALL PRIVILEGES ON *.* TO 'beer_user'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;
