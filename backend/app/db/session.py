import logging
from sqlalchemy import create_engine, exc, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from app.core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure we're using the correct database URI
SQLALCHEMY_DATABASE_URI = settings.SQLALCHEMY_DATABASE_URI
logger.info(f"Connecting to database: {SQLALCHEMY_DATABASE_URI.split('@')[-1]}")

# Create engine with connection pooling and error handling
engine = create_engine(
    SQLALCHEMY_DATABASE_URI,
    poolclass=QueuePool,
    pool_pre_ping=True,
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_size=5,        # Number of connections to keep open
    max_overflow=10,    # Max number of connections to create beyond pool_size
    connect_args={
        'connect_timeout': 10,  # 10 second timeout for initial connection
    }
)

# Test the connection
conn = None
try:
    conn = engine.connect()
    conn.execute(text("SELECT 1"))
    logger.info("Successfully connected to the database")
except Exception as e:
    logger.error(f"Failed to connect to the database: {e}")
    raise
finally:
    if conn:
        conn.close()

# Create session factory with scoped sessions for thread safety
SessionLocal = scoped_session(
    sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )
)

# Create base class for models
Base = declarative_base()
Base.query = SessionLocal.query_property()

def get_db():
    """Dependency for getting DB session"""
    db = SessionLocal()
    try:
        yield db
    except exc.SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()
