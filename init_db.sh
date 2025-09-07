#!/bin/bash

#!/bin/bash

# Get database credentials from environment variables or use defaults
DB_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD:-Daybreak@2025}
DB_NAME=${MYSQL_DATABASE:-beer_game}
DB_USER=${MYSQL_USER:-beer_user}
DB_PASSWORD=${MYSQL_PASSWORD:-Daybreak@2025}

# Log the database initialization
echo "Initializing database: $DB_NAME with user: $DB_USER"

# Connect to MySQL and create the database and user
mysql -h db -u root -p"$DB_ROOT_PASSWORD" <<EOF
-- Create database if not exists
CREATE DATABASE IF NOT EXISTS \`$DB_NAME\`;

-- Create user if not exists and grant privileges
CREATE USER IF NOT EXISTS '$DB_USER'@'%' IDENTIFIED BY '$DB_PASSWORD';
GRANT ALL PRIVILEGES ON \`$DB_NAME\`.* TO '$DB_USER'@'%';
FLUSH PRIVILEGES;

-- Use the database
USE \`$DB_NAME\`;

-- Create users table if not exists
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(100) NOT NULL,
    full_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Insert default users if they don't exist
INSERT IGNORE INTO users (username, email, hashed_password, full_name, is_superuser, is_active) VALUES
('admin', 'admin@daybreak.ai', '\$2b\$12\$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', 'System Administrator', TRUE, TRUE),
('retailer', 'retailer@daybreak.ai', '\$2b\$12\$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', 'Retailer User', FALSE, TRUE),
('distributor', 'distributor@daybreak.ai', '\$2b\$12\$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', 'Distributor User', FALSE, TRUE),
('manufacturer', 'manufacturer@daybreak.ai', '\$2b\$12\$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', 'Manufacturer User', FALSE, TRUE),
('wholesaler', 'wholesaler@daybreak.ai', '\$2b\$12\$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', 'Wholesaler User', FALSE, TRUE);

-- Verify users were created
SELECT id, username, email, is_superuser, is_active FROM users;
EOF
