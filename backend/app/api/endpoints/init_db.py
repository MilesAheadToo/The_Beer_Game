import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.models.user import User, Base
from app.core.security import get_password_hash

def init_db():
    # Create database engine
    SQLALCHEMY_DATABASE_URI = "sqlite:///./beer_game.db"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URI, connect_args={"check_same_thread": False}
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create a session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    # Create test user if it doesn't exist
    test_user = db.query(User).filter(User.email == "test@example.com").first()
    if not test_user:
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password=get_password_hash("testpassword"),
            full_name="Test User",
            is_active=True
        )
        db.add(user)
        db.commit()
        print("Test user created successfully!")
    else:
        print("Test user already exists!")
    
    db.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialization complete!")
