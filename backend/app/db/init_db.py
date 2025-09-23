import os
import logging
import asyncio
from sqlalchemy import create_engine, exc, text, select, update
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all models to ensure they are registered with SQLAlchemy
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import models in the correct order to avoid circular imports
from app.models.base import Base
from app.models.user import User, RefreshToken, UserTypeEnum
from app.models.player import Player, PlayerRole, PlayerType, PlayerStrategy
from app.models.auth_models import PasswordHistory, PasswordResetToken
from app.models.session import TokenBlacklist, UserSession
from app.models.game import Game, GameStatus, Round, PlayerAction
from app.models.group import Group
try:
    from scripts.seed_core_config import seed_core_config
except ModuleNotFoundError:  # pragma: no cover - fallback when script package unavailable
    async def seed_core_config(*args, **kwargs):
        logger.warning(
            "seed_core_config module not found; continuing without core config seeding."
        )
from app.core.security import get_password_hash

# Ensure all models are imported and registered with SQLAlchemy
# This is necessary for proper relationship resolution
_models = [User, RefreshToken, Player, PasswordHistory, PasswordResetToken, 
           TokenBlacklist, UserSession, Game, Round, PlayerAction]

# Log model registration
logger.info(f"Registered models: {[model.__name__ for model in _models]}")

# Import settings
from app.core.config import settings

# Create async database engine
raw_uri = settings.SQLALCHEMY_DATABASE_URI
if raw_uri.startswith("mysql+pymysql://"):
    async_uri = raw_uri.replace("mysql+pymysql://", "mysql+aiomysql://", 1)
else:
    async_uri = raw_uri.replace("mysql://", "mysql+aiomysql://")

engine = create_async_engine(
    async_uri,
    echo=True
)

# Create async session factory
async_session_factory = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Get database connection details from environment variables
DB_USER = os.getenv("MARIADB_USER")
DB_PASSWORD = os.getenv("MARIADB_PASSWORD")
DB_ROOT_PASSWORD = os.getenv("MARIADB_ROOT_PASSWORD", "Daybreak2025")
DB_HOST = os.getenv("MARIADB_HOST", "db")
DB_NAME = os.getenv("MARIADB_DATABASE")

if not all([DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("Missing required database environment variables")

# Construct the database URIs
SQLALCHEMY_DATABASE_URI = f"mariadb+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}?charset=utf8mb4"
logger.info(f"Initializing MariaDB at: {DB_USER}@{DB_HOST}/{DB_NAME}")

# First, try to connect to the database as root to create the database if needed
root_engine = create_engine(
    f"mariadb+pymysql://root:{DB_ROOT_PASSWORD}@{DB_HOST}/mysql?connect_timeout=10&charset=utf8mb4",
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={
        'connect_timeout': 10,
        'read_timeout': 30,
        'write_timeout': 30,
        'ssl': False
    }
)

try:
    with root_engine.connect() as conn:
        result = conn.execute(text(f"SHOW DATABASES LIKE '{DB_NAME}'")).fetchone()
        if not result:
            logger.info(f"Creating database {DB_NAME} with UTF8MB4 character set")
            conn.execute(text(f"CREATE DATABASE `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
            conn.execute(text(f"CREATE USER IF NOT EXISTS '{DB_USER}'@'%' IDENTIFIED BY '{DB_PASSWORD}'"))
            conn.execute(text(f"GRANT ALL PRIVILEGES ON `{DB_NAME}`.* TO '{DB_USER}'@'%'"))
            conn.execute(text("FLUSH PRIVILEGES"))
            logger.info(f"Database {DB_NAME} created and privileges granted to {DB_USER}")
        else:
            logger.info(f"Database {DB_NAME} already exists")
except Exception as e:  # pragma: no cover - depends on db credentials
    logger.warning("Skipping root bootstrap step due to error: %s", e)
finally:
    root_engine.dispose()

async def init_db():
    """
    Initialize the database with required tables and initial data.
    """
    # Create all tables
    logger.info("Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created successfully")
    
    # Create a session to add initial data
    async with async_session_factory() as db:
        try:
            # Add any initial data here if needed
            logger.info("Adding initial data...")
            
            # Check if system administrator user already exists
            systemadmin_email = (
                os.getenv("SYSTEMADMIN_EMAIL")
                or os.getenv("SUPERADMIN_EMAIL")
                or "systemadmin@daybreak.ai"
            )
            systemadmin_password = (
                os.getenv("SYSTEMADMIN_PASSWORD")
                or os.getenv("SUPERADMIN_PASSWORD")
                or "Daybreak@2025"
            )

            result = await db.execute(
                select(User).where(User.email == systemadmin_email)
            )
            systemadmin = result.scalars().first()

            if not systemadmin:
                logger.info("Creating system administrator user...")
                systemadmin = User(
                    username="systemadmin",
                    email=systemadmin_email,
                    hashed_password=get_password_hash(systemadmin_password),
                    is_superuser=True,
                    is_active=True,
                    user_type=UserTypeEnum.SYSTEM_ADMIN,
                )
                db.add(systemadmin)
                await db.commit()
                await db.refresh(systemadmin)
                logger.info("System administrator user created successfully")

            # Ensure default Daybreak group exists
            result = await db.execute(select(Group).where(Group.name == "Daybreak"))
            group = result.scalars().first()
            if not group:
                group = Group(
                    name="Daybreak", description="Default group", admin_id=systemadmin.id
                )
                db.add(group)
                await db.flush()

            # Assign group to system administrator and any users missing a group
            systemadmin.group_id = group.id
            await db.execute(
                update(User).where(User.group_id.is_(None)).values(group_id=group.id)
            )

            # Seed core supply chain configuration
            await seed_core_config(db)

            # Add other initial data as needed

            await db.commit()
            logger.info("Initial data added successfully")
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error initializing database: {e}")
            raise

async def async_main():
    await init_db()
    await engine.dispose()

if __name__ == "__main__":
    print("Initializing database...")
    asyncio.run(async_main())
    print("Database initialized successfully!")
