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
DB_HOST = os.getenv("MYSQL_SERVER", "db")
DB_NAME = os.getenv("MYSQL_DB")

# Log the database connection details for debugging
logger.info(f"Database connection details - User: {DB_USER}, Host: {DB_HOST}, DB: {DB_NAME}")

if not all([DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("Missing required database environment variables")

# URL encode the password to handle special characters
from urllib.parse import quote_plus

# Construct the database URI with MariaDB dialect
encoded_password = quote_plus(DB_PASSWORD)
SQLALCHEMY_DATABASE_URI = f"mariadb+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}?charset=utf8mb4"
logger.info(f"Connecting to MariaDB: {DB_USER}@{DB_HOST}/{DB_NAME}")
logger.debug(f"Database URI: mariadb+pymysql://{DB_USER}:***@{DB_HOST}/{DB_NAME}?charset=utf8mb4")

# Create engine with connection pooling and error handling
engine = create_engine(
    SQLALCHEMY_DATABASE_URI,
    poolclass=QueuePool,
    pool_pre_ping=True,  # Enable connection health checks
    pool_recycle=3600,   # Recycle connections after 1 hour
    pool_size=5,         # Number of connections to keep open
    max_overflow=10,     # Max number of connections to create beyond pool_size
    connect_args={
        'connect_timeout': 10,    # 10 second timeout for initial connection
        'read_timeout': 30,       # 30 second timeout for read operations
        'write_timeout': 30,      # 30 second timeout for write operations
        'ssl': False              # Disable SSL for now, configure as needed
    },
    echo_pool=True,      # Log connection pool events
    echo=False           # Set to True for SQL query logging
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
