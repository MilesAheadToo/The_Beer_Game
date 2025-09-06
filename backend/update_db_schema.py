from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, text
from sqlalchemy.orm import sessionmaker
import os

def update_database():
    # Database connection details from environment variables
    db_user = "beer_user"
    db_password = "Daybreak@2025"
    db_host = "db"
    db_name = "beer_game"
    
    # Create SQLAlchemy engine
    DATABASE_URL = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    engine = create_engine(DATABASE_URL)
    
    # Create a session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Add missing columns to users table
        alter_queries = [
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login DATETIME DEFAULT NULL",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_password_change DATETIME DEFAULT CURRENT_TIMESTAMP",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS failed_login_attempts INT DEFAULT 0",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS locked_until DATETIME DEFAULT NULL",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS mfa_secret VARCHAR(255) DEFAULT NULL",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS mfa_enabled BOOLEAN DEFAULT FALSE"
        ]
        
        for query in alter_queries:
            try:
                session.execute(text(query))
                session.commit()
                print(f"Successfully executed: {query}")
            except Exception as e:
                session.rollback()
                print(f"Error executing {query}: {e}")
        
        print("Database schema update completed successfully!")
        
    except Exception as e:
        session.rollback()
        print(f"Error updating database schema: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    update_database()
