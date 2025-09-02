import os
import pymysql
from pymysql.constants import CLIENT

try:
    # Get database connection parameters from environment variables
    db_host = os.getenv("MYSQL_SERVER", "db")
    db_port = int(os.getenv("MYSQL_PORT", "3306"))
    db_user = os.getenv("MYSQL_USER", "beer_user")
    db_password = os.getenv("MYSQL_PASSWORD", "beer_password")
    db_name = os.getenv("MYSQL_DB", "beer_game")

    print(f"Attempting to connect to MySQL at {db_user}@{db_host}:{db_port}/{db_name}")
    
    # Try to connect to the database
    connection = pymysql.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database=db_name,
        client_flag=CLIENT.MULTI_STATEMENTS,
        connect_timeout=5
    )
    
    print("Successfully connected to MySQL!")
    
    # Try a simple query
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print(f"Query result: {result[0]}")
    
    connection.close()
    print("Connection closed.")
    
except Exception as e:
    print(f"Error: {str(e)}")
