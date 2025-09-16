#!/usr/bin/env python3
"""Test database connection with direct MySQL connection."""

import os
import pymysql
from pymysql.constants import CLIENT

def test_connection():
    """Test database connection with provided credentials."""
    try:
        # Get connection parameters from environment variables
        db_config = {
            'host': os.getenv("MYSQL_HOST", "localhost"),
            'port': int(os.getenv("MYSQL_PORT", "3307")),
            'user': os.getenv("MYSQL_USER", "beer_user"),
            'password': os.getenv("MYSQL_PASSWORD", "beergame123"),
            'database': os.getenv("MYSQL_DATABASE", "beer_game"),
            'client_flag': CLIENT.MULTI_STATEMENTS,
            'connect_timeout': 10
        }
        
        print("Testing database connection with the following parameters:")
        print(f"Host: {db_config['host']}")
        print(f"Port: {db_config['port']}")
        print(f"User: {db_config['user']}")
        print(f"Database: {db_config['database']}")
        
        # Try to connect to the database
        connection = pymysql.connect(**db_config)
        
        # Test the connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            print(f"\nConnection successful! Result: {result}")
            
            # List all tables in the database
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print("\nTables in the database:")
            for table in tables:
                print(f"- {table[0]}")
        
        connection.close()
        return True
        
    except Exception as e:
        print(f"\nError connecting to the database: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection()
