import os
import sqlite3
from pathlib import Path

def check_sqlite_db():
    db_path = Path("beer_game.db")
    
    # Check if file exists
    if not db_path.exists():
        print(f"Error: Database file {db_path} does not exist.")
        print("Current working directory:", os.getcwd())
        print("Files in current directory:", os.listdir('.'))
        return
    
    # Check file size
    file_size = db_path.stat().st_size
    print(f"Database file size: {file_size} bytes")
    
    try:
        # Try to connect to the database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in the database.")
        else:
            print("\nTables in the database:")
            for table in tables:
                print(f"- {table[0]}")
                
                # Show table structure
                cursor.execute(f"PRAGMA table_info({table[0]});")
                columns = cursor.fetchall()
                print(f"  Columns: {[col[1] for col in columns]}")
        
        # Check for alembic_version table
        cursor.execute("SELECT * FROM sqlite_master WHERE type='table' AND name='alembic_version';")
        if cursor.fetchone():
            cursor.execute("SELECT * FROM alembic_version;")
            version = cursor.fetchone()
            print(f"\nAlembic version: {version[0] if version else 'Unknown'}")
        else:
            print("\nAlembic version table not found.")
            
    except sqlite3.Error as e:
        print(f"\nSQLite error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_sqlite_db()
