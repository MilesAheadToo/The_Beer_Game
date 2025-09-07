import logging
import os
from typing import AsyncGenerator
from urllib.parse import quote_plus

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy import create_engine

from app.core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database connection details from environment variables
DB_USER = os.getenv("MYSQL_USER", "beer_user")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "Daybreak@2025")
# Use MYSQL_HOST if available, otherwise fall back to MYSQL_SERVER, then default to 'db' for Docker
DB_HOST = os.getenv("MYSQL_HOST") or os.getenv("MYSQL_SERVER", "db")
DB_NAME = os.getenv("MYSQL_DATABASE") or os.getenv("MYSQL_DB", "beer_game")
DB_PORT = os.getenv("MYSQL_PORT", "3306")

# Log the database connection details for debugging
logger.info(f"Database connection details - User: {DB_USER}, Host: {DB_HOST}, DB: {DB_NAME}")

if not all([DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("Missing required database environment variables")

# Construct the database URI with async MariaDB dialect
encoded_password = quote_plus(DB_PASSWORD)
# Use aiomysql as the async driver for MariaDB
SQLALCHEMY_DATABASE_URI = f"mysql+aiomysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
logger.info(f"Connecting to MariaDB with aiomysql: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Create async engine with connection pooling
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URI,
    echo=True,
    future=True,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=5,
    max_overflow=10,
    connect_args={
        'connect_timeout': 10,
        'ssl': False
    }
)

# Create a synchronous engine for migrations and table operations
SYNC_SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("mysql+aiomysql", "mysql+pymysql")
sync_engine = create_engine(
    SYNC_SQLALCHEMY_DATABASE_URI,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=5,
    max_overflow=10
)

# Create async session factory
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

# For backward compatibility with code that expects a synchronous session
SessionLocal = async_session_factory

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
