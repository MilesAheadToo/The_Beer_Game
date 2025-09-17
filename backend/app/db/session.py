import logging
import os
from typing import AsyncGenerator
from urllib.parse import quote_plus

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy import create_engine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database URL from settings
from app.core.config import settings

# Log the database connection details for debugging
logger.info(f"Using database URL from settings: {settings.SQLALCHEMY_DATABASE_URI}")

# Ensure we're using the async driver (aiomysql) for MariaDB
SQLALCHEMY_DATABASE_URI = settings.SQLALCHEMY_DATABASE_URI.replace(
    "mysql+pymysql://", "mysql+aiomysql://", 1
)

logger.info(f"Connecting to database with URL: {SQLALCHEMY_DATABASE_URI}")

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
