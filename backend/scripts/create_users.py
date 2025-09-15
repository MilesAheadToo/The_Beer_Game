import os
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.base import Base
from app.models.user import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def get_db_connection_string():
    server = os.getenv("MYSQL_SERVER", "db")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "beer_user")
    password = os.getenv("MYSQL_PASSWORD", "change-me-user")
    db_name = os.getenv("MYSQL_DB", "beer_game")
    from urllib.parse import quote_plus
    encoded_password = quote_plus(password)
    return f"mysql+pymysql://{user}:{encoded_password}@{server}:{port}/{db_name}?charset=utf8mb4"

def get_or_create_systemadmin(session):
    user = session.query(User).filter(User.email == "systemadmin@daybreak.ai").first()
    if user:
        print("System administrator user already exists")
        return user
    user = User(
        username="systemadmin",
        email="systemadmin@daybreak.ai",
        hashed_password=hash_password("Daybreak@2025"),
        full_name="System Admin",
        is_active=True,
        is_superuser=True,
        last_login=datetime.utcnow(),
    )
    session.add(user)
    return user

def main():
    SQLALCHEMY_DATABASE_URL = get_db_connection_string()
    print(f"Using MySQL database at: {SQLALCHEMY_DATABASE_URL.split('@')[-1]}")
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        systemadmin = get_or_create_systemadmin(db)
        db.commit()
        print(f"System administrator user ready: {systemadmin.email}")
    except Exception as e:
        print(f"Error creating system administrator: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()
