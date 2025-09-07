from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

def apply_migration():
    # Load environment variables
    load_dotenv()
    
    # Get database URL from environment
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    # Create engine
    engine = create_engine(db_url)
    
    # SQL statements to execute
    migration_sql = """
    -- Add round_time_limit column to games table
    ALTER TABLE games 
    ADD COLUMN IF NOT EXISTS round_time_limit INT NOT NULL DEFAULT 60;
    
    -- Add current_round_ends_at column to games table
    ALTER TABLE games 
    ADD COLUMN IF NOT EXISTS current_round_ends_at DATETIME NULL;
    
    -- Add is_processed column to game_rounds table
    ALTER TABLE game_rounds 
    ADD COLUMN IF NOT EXISTS is_processed BOOLEAN NOT NULL DEFAULT FALSE;
    
    -- Add is_completed and completed_at columns if they don't exist
    ALTER TABLE game_rounds 
    ADD COLUMN IF NOT EXISTS is_completed BOOLEAN NOT NULL DEFAULT FALSE;
    
    ALTER TABLE game_rounds 
    ADD COLUMN IF NOT EXISTS completed_at DATETIME NULL;
    """
    
    # Execute the migration
    with engine.connect() as connection:
        with connection.begin():
            # Split the SQL into individual statements and execute them
            for statement in migration_sql.split(';'):
                if statement.strip():
                    connection.execute(text(statement + ';'))
    
    print("Database migration completed successfully!")

if __name__ == "__main__":
    apply_migration()
