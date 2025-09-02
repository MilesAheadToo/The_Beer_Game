import os
import logging
from sqlalchemy import create_engine, exc
from sqlalchemy.orm import sessionmaker
from ..core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all models to ensure they are registered with SQLAlchemy
from ..models import Base

# Create engine with MySQL
SQLALCHEMY_DATABASE_URI = settings.SQLALCHEMY_DATABASE_URI
logger.info(f"Initializing database at: {SQLALCHEMY_DATABASE_URI}")

try:
    engine = create_engine(
        SQLALCHEMY_DATABASE_URI,
        pool_pre_ping=True,
        pool_recycle=3600,  # Recycle connections after 1 hour
        pool_size=5,
        max_overflow=10,
        connect_timeout=10
    )
    
    # Test the connection
    with engine.connect() as conn:
        conn.execute("SELECT 1")
    
    logger.info("Successfully connected to the database")
    
    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
except exc.SQLAlchemyError as e:
    logger.error(f"Failed to connect to the database: {str(e)}")
    raise

def init_db():
    """
    Initialize the database with required tables and initial data.
    """
    print(f"Initializing database at: {SQLALCHEMY_DATABASE_URI}")
    
    # Create all tables
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    # Create a session
    db = SessionLocal()
    
    try:
        # Check if we already have any users (to prevent re-adding test data)
        from ..models import User
        if not db.query(User).first():
            print("Adding test user...")
            # Add a test user
            from ..core.security import get_password_hash
            test_user = User(
                username="testuser",
                email="test@example.com",
                hashed_password=get_password_hash("testpassword"),
                full_name="Test User",
                is_active=True
            )
            db.add(test_user)
            db.commit()
            
            print("Adding test game...")
            # Add a test game
            test_game = Game(
                name="Test Game",
                status=GameStatus.CREATED,
                max_rounds=10,
                config={"test": True}
            )
            db.add(test_game)
            db.commit()
            
            # Add a test player
            test_player = Player(
                game_id=test_game.id,
                user_id=test_user.id,
                name="Test Player",
                role=PlayerRole.RETAILER,
                type=PlayerType.HUMAN,
                strategy=PlayerStrategy.MANUAL,
                inventory=10,
                backlog=0,
                cost=0
            )
            db.add(test_player)
            db.commit()
            
            print("Test data added successfully!")
        else:
            print("Database already contains data, skipping test data creation.")
            
    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialized successfully!")
