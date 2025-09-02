# Import from session.py to maintain a single source of truth for database configuration
from .session import Base, engine, SessionLocal, get_db

# This file is kept for backward compatibility
# All database configuration should be in session.py
