import logging
import os
from sqlalchemy import create_engine, exc, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from app.core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database connection details from environment variables
DB_USER = os.getenv("MYSQL_USER")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD")
DB_HOST = os.getenv("MYSQL_HOST", "db")
DB_NAME = os.getenv("MYSQL_DATABASE") or os.getenv("MYSQL_DB")

# Log the database connection details for debugging
logger.info(f"Database connection details - User: {DB_USER}, Host: {DB_HOST}, DB: {DB_NAME}")

if not all([DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("Missing required database environment variables")

# Construct the database URI
SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
logger.info(f"Connecting to database: {DB_USER}@{DB_HOST}/{DB_NAME}")

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
