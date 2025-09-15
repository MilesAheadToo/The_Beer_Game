import sys
import os
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.base import Base
from app.models.user import User
from app.models.player import Player, PlayerRole, PlayerType, PlayerStrategy
from app.models.group import Group
from app.core.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def create_admin_user(session):
    # Check if admin already exists
    admin = session.query(User).filter(User.email == "admin@daybreak.ai").first()
    if admin:
        print("Admin user already exists")
        return admin
    
    # Create admin user
    admin = User(
        username="admin",
        email="admin@daybreak.ai",
        hashed_password=hash_password("Daybreak@2025"),
        full_name="System Administrator",
        is_active=True,
        is_superuser=True,
        last_login=datetime.utcnow()
    )
    session.add(admin)
    return admin

def create_role_user(session, role: str):
    username = role.lower()
    email = f"{username}@daybreak.ai"
    
    # Check if user already exists
    user = session.query(User).filter(User.email == email).first()
    if user:
        print(f"{role} user already exists")
        return user
    
    # Create role user
    user = User(
        username=username,
        email=email,
        hashed_password=hash_password("Daybreak@2025"),
        full_name=f"{role.capitalize()} User",
        is_active=True,
        is_superuser=False,
        last_login=datetime.utcnow()
    )
    session.add(user)
    return user

def get_or_create_default_group(session, admin_user):
    """Create the default 'Daybreak' group if it does not exist."""
    group = session.query(Group).filter(Group.name == "Daybreak").first()
    if group:
        return group

    group = Group(
        name="Daybreak",
        description="Default group",
        admin_id=admin_user.id,
    )
    session.add(group)
    session.flush()  # Ensure group ID is available
    
    # Skip supply_chain_configs relationship for now to avoid dependency issues
    if hasattr(Group, 'supply_chain_configs'):
        group.supply_chain_configs = []
    return group

def get_db_connection_string():
    """Construct the database connection string from environment variables."""
    server = os.getenv("MYSQL_SERVER", "db")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "beer_user")
    password = os.getenv("MYSQL_PASSWORD", "change-me-user")
    db_name = os.getenv("MYSQL_DB", "beer_game")
    
    # URL encode the password to handle special characters
    from urllib.parse import quote_plus
    encoded_password = quote_plus(password)
    
    return f"mysql+pymysql://{user}:{encoded_password}@{server}:{port}/{db_name}?charset=utf8mb4"

def main():
    # Database connection
    SQLALCHEMY_DATABASE_URL = get_db_connection_string()
    print(f"Using MySQL database at: {SQLALCHEMY_DATABASE_URL.split('@')[-1]}")
    
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Create a new session
    db = SessionLocal()
    
    try:
        # Create admin user
        admin = create_admin_user(db)
        print(f"Admin user created/updated: {admin.email}")

        # Create or get default group and assign admin
        group = get_or_create_default_group(db, admin)
        admin.group_id = group.id

        # Create role users
        roles = ["retailer", "distributor", "manufacturer", "wholesaler"]
        for role in roles:
            user = create_role_user(db, role)
            user.group_id = group.id
            print(f"{role.capitalize()} user created/updated: {user.email}")

        # Commit the changes
        db.commit()

        print(f"Assigned all users to default group: {group.name}")
        
    except Exception as e:
        print(f"Error creating users: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()
