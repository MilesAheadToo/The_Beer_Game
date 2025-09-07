import mysql.connector
import os
from dotenv import load_dotenv

def run_migration():
    # Load environment variables
    load_dotenv()
    
    # Get database connection details
    db_config = {
        'host': os.getenv('MYSQL_HOST', 'localhost'),
        'user': os.getenv('MYSQL_USER'),
        'password': os.getenv('MYSQL_PASSWORD'),
        'database': os.getenv('MYSQL_DATABASE'),
        'port': int(os.getenv('MYSQL_PORT', 3306))
    }
    
    # Read the SQL file
    with open('scripts/apply_migration.sql', 'r') as file:
        sql_script = file.read()
    
    # Execute the SQL script
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        # Split the SQL script into individual statements and execute them
        for statement in sql_script.split(';'):
            if statement.strip():
                cursor.execute(statement + ';')
        
        connection.commit()
        print("Database migration completed successfully!")
        
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    run_migration()
