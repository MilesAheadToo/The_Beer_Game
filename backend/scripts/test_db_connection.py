#!/usr/bin/env python3
"""Test database connection with localhost."""

import os
import urllib.parse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Database connection parameters
DB_USER = "beer_user"
DB_PASSWORD = "beergame123"  # Updated to match docker-compose
DB_HOST = "localhost"  # Changed from 'db' to 'localhost'
DB_PORT = "3307"       # Default is 3306, but we're mapping to 3307 in docker-compose
DB_NAME = "beer_game"

# URL encode the password to handle special characters
encoded_password = urllib.parse.quote_plus(DB_PASSWORD)

# Create the database URL with URL-encoded password
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

print(f"Connecting to database with URL: mysql+pymysql://{DB_USER}:*****@{DB_HOST}:{DB_PORT}/{DB_NAME}")

try:
    # Create an engine to connect to the database
    engine = create_engine(DATABASE_URL)
    
    # Test the connection
    with engine.connect() as connection:
        print("Successfully connected to the database!")
        # Try to execute a simple query using text()
        from sqlalchemy import text
        result = connection.execute(text("SELECT 1"))
        print("Test query result:", result.fetchone())
        
        # List all tables
        result = connection.execute(text("SHOW TABLES"))
        print("\nTables in the database:")
        for row in result:
            print(f"- {row[0]}")
            
except Exception as e:
    print(f"Error connecting to the database: {e}")
    raise
