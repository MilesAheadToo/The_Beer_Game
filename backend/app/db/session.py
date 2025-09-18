import logging
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database URL from settings
from app.core.config import settings

# Log the database connection details for debugging
raw_database_uri = settings.SQLALCHEMY_DATABASE_URI
logger.info("Using database URL from settings: %s", raw_database_uri)
db_url = make_url(raw_database_uri)
logger.info("Connecting to database with URL: %s", raw_database_uri)

is_sqlite = db_url.get_backend_name().startswith("sqlite")

aiosqlite_available = True
if is_sqlite:
    try:  # pragma: no cover - optional dependency check
        import aiosqlite  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        aiosqlite_available = False
        logger.warning(
            "aiosqlite is not installed; async DB support will be disabled for the SQLite fallback."
        )

if is_sqlite and db_url.database:
    sqlite_path = Path(db_url.database)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

if is_sqlite:
    async_database_uri = raw_database_uri.replace("sqlite://", "sqlite+aiosqlite://", 1)
    sync_database_uri = raw_database_uri
    async_connect_args = {"check_same_thread": False}
    sync_connect_args = {"check_same_thread": False}
    engine_kwargs = {}
else:
    async_database_uri = raw_database_uri.replace("mysql+pymysql://", "mysql+aiomysql://", 1)
    sync_database_uri = raw_database_uri
    async_connect_args = {"connect_timeout": 10, "ssl": False}
    sync_connect_args = {"connect_timeout": 10, "ssl": False}
    engine_kwargs = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "pool_size": 5,
        "max_overflow": 10,
    }

engine = None
async_session_factory = None

if (not is_sqlite) or aiosqlite_available:
    engine = create_async_engine(
        async_database_uri,
        echo=True,
        future=True,
        connect_args=async_connect_args,
        **engine_kwargs,
    )

    async_session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )

sync_engine = create_engine(
    sync_database_uri,
    connect_args=sync_connect_args,
    **engine_kwargs,
)

# For backward compatibility with code that expects a synchronous session
SessionLocal = async_session_factory

from app.models.base import Base

if is_sqlite:
    Base.metadata.create_all(bind=sync_engine)

# Dependency to get DB session
if async_session_factory is None:

    async def get_db() -> AsyncGenerator[AsyncSession, None]:
        """Raise a helpful error when async DB access is unavailable."""
        raise RuntimeError(
            "Async database support requires the 'aiosqlite' package when using the SQLite fallback."
        )

else:

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
