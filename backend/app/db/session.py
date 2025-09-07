import logging
import os
from typing import AsyncGenerator
from urllib.parse import quote_plus

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from app.core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database connection details from environment variables
DB_USER = os.getenv("MYSQL_USER", "beer_user")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "Daybreak@2025")
DB_HOST = os.getenv("MYSQL_SERVER", "db")
DB_NAME = os.getenv("MYSQL_DB", "beer_game")

# Log the database connection details for debugging
logger.info(f"Database connection details - User: {DB_USER}, Host: {DB_HOST}, DB: {DB_NAME}")

if not all([DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("Missing required database environment variables")

# Construct the database URI with async MySQL dialect
encoded_password = quote_plus(DB_PASSWORD)
SQLALCHEMY_DATABASE_URI = f"mysql+aiomysql://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}?charset=utf8mb4"
logger.info(f"Connecting to MySQL: {DB_USER}@{DB_HOST}/{DB_NAME}")

# Create async engine with connection pooling
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URI,
    echo=True,
    future=True,
    pool_pre_ping=True,
    pool_recycle=3600,
    poolclass=NullPool,  # Using NullPool which doesn't support pool_size/max_overflow
    connect_args={
        'connect_timeout': 10,
        'ssl': False
    }
)

# Create async session factory
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

# Base class for models
Base = declarative_base()

# Dependency to get DB session
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function that yields db sessions
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            await session.close()
