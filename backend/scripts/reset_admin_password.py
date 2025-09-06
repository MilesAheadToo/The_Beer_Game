from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.core.security import get_password_hash

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

def reset_admin_password():
    try:
        # Get admin user
        admin = db.execute(text("SELECT * FROM users WHERE username = 'admin'")).first()
        if not admin:
            print("Admin user not found")
            return
            
        # New password
        new_password = "Daybreak@2025"
        hashed_password = get_password_hash(new_password)
        
        # Update password
        db.execute(
            text("UPDATE users SET hashed_password = :password WHERE username = 'admin'"),
            {"password": hashed_password}
        )
        db.commit()
        print("Admin password has been reset successfully")
        
    except Exception as e:
        print(f"Error resetting password: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    reset_admin_password()
