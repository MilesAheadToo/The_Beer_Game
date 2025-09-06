from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from passlib.hash import bcrypt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection details
DB_USER = os.getenv("MYSQL_USER", "beer_user")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "Daybreak@2025")
DB_HOST = os.getenv("MYSQL_SERVER", "db")
DB_NAME = os.getenv("MYSQL_DB", "beer_game")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

def update_all_passwords():
    try:
        # Generate bcrypt hash for the password
        password_hash = bcrypt.using(rounds=12).hash("Daybreak@2025")
        
        # Update all users' passwords
        result = db.execute(
            text("UPDATE users SET hashed_password = :password"),
            {"password": password_hash}
        )
        db.commit()
        
        print(f"Successfully updated passwords for {result.rowcount} users")
        
    except Exception as e:
        print(f"Error updating passwords: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    update_all_passwords()
