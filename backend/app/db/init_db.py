import os
import logging
from sqlalchemy import create_engine, exc, text
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
from app.models.user import User, RefreshToken
from app.models.player import Player, PlayerRole, PlayerType, PlayerStrategy
from app.models.auth_models import PasswordHistory, PasswordResetToken
from app.models.session import TokenBlacklist, UserSession
from app.models.game import Game, GameStatus, Round, PlayerAction

# Ensure all models are imported and registered with SQLAlchemy
# This is necessary for proper relationship resolution
_models = [User, RefreshToken, Player, PasswordHistory, PasswordResetToken, 
           TokenBlacklist, UserSession, Game, Round, PlayerAction]

# Log model registration
logger.info(f"Registered models: {[model.__name__ for model in _models]}")

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
        # Check if database exists
        result = conn.execute(text(f"SHOW DATABASES LIKE '{DB_NAME}'")).fetchone()
        if not result:
            logger.info(f"Creating database {DB_NAME} with UTF8MB4 character set")
            # Use CREATE DATABASE with explicit character set and collation
            conn.execute(text(f"CREATE DATABASE `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
            
            # Create user if not exists and grant privileges
            conn.execute(text(f"CREATE USER IF NOT EXISTS '{DB_USER}'@'%' IDENTIFIED BY '{DB_PASSWORD}'"))
            conn.execute(text(f"GRANT ALL PRIVILEGES ON `{DB_NAME}`.* TO '{DB_USER}'@'%'"))
            conn.execute(text("FLUSH PRIVILEGES"))
            logger.info(f"Database {DB_NAME} created and privileges granted to {DB_USER}")
        else:
            logger.info(f"Database {DB_NAME} already exists")

except Exception as e:
    logger.error(f"Error initializing database: {e}")
    raise

finally:
    root_engine.dispose()

# Now create tables in the target database
try:
    # Create engine with MariaDB specific settings
    engine = create_engine(
        SQLALCHEMY_DATABASE_URI,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={
            'connect_timeout': 10,
            'read_timeout': 30,
            'write_timeout': 30,
            'ssl': False
        }
    )
    
    # Create all tables with MariaDB specific settings
    logger.info("Creating database tables...")
    with engine.begin() as connection:
        # Set session variables for table creation
        connection.execute(text("SET SESSION sql_require_primary_key=0"))  # Temporarily allow tables without PK
        Base.metadata.create_all(bind=connection)
        
    logger.info("Database tables created successfully")

    # Create a session to add initial data
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Add any initial data here if needed
        # Example: 
        # if not db.query(User).filter(User.username == "admin").first():
        #     admin_user = User(username="admin", email="admin@example.com", hashed_password=..., is_superuser=True)
        #     db.add(admin_user)
        #     db.commit()
        #     logger.info("Created initial admin user")
        pass
    except Exception as e:
        logger.error(f"Error adding initial data: {e}")
        db.rollback()
        raise
    finally:
        db.close()
        engine.dispose()

except Exception as e:
    logger.error(f"Error creating database tables: {e}")
    if 'engine' in locals():
        engine.dispose()
    raise

def init_db():
    """
    Initialize the database with required tables and initial data.
    """
    print(f"Initializing database at: {SQLALCHEMY_DATABASE_URI}")
    
    # Import all models to ensure they're registered with SQLAlchemy
    # Import in dependency order to avoid circular imports
    from app.models.base import Base
    from app.models.user import User, RefreshToken, user_games
    from app.models.player import Player
    from app.models.game import Game, Round, PlayerAction
    from app.models.auth_models import PasswordHistory, PasswordResetToken
    from app.models.session import TokenBlacklist, UserSession
    
    # Verify all models are registered with the metadata
    registered_tables = set(Base.metadata.tables.keys())
    expected_tables = {
        'users', 'refresh_tokens', 'players', 'password_history',
        'password_reset_tokens', 'token_blacklist', 'user_sessions',
        'games', 'rounds', 'player_actions', 'user_games'
    }
    
    missing_tables = expected_tables - registered_tables
    if missing_tables:
        logger.warning(f"Missing tables in metadata: {missing_tables}")
    else:
        logger.info("All expected tables are registered in metadata")
        
    # Log the registered tables for debugging
    logger.info(f"Registered tables: {registered_tables}")
    
    # Create a new engine with root credentials to drop and recreate the database
    root_engine = create_engine(
        f"mysql+pymysql://root:beer_password@db/",
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={"connect_timeout": 10}
    )
    
    # Drop and recreate the database
    with root_engine.connect() as conn:
        conn.execute(text("DROP DATABASE IF EXISTS beer_game"))
        conn.execute(text("CREATE DATABASE beer_game CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
    
    # Now create all tables
    logger.info("Creating all tables...")
    
    # Create a new engine for the beer_game database
    beer_engine = create_engine(
        SQLALCHEMY_DATABASE_URI,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={"connect_timeout": 10}
    )
    
    # Create all tables using create_all() to handle dependencies automatically
    with beer_engine.connect() as conn:
        # Disable foreign key checks
        conn.execute(text('SET FOREIGN_KEY_CHECKS=0;'))
        
        try:
            # Drop all existing tables
            Base.metadata.drop_all(bind=conn)
            logger.info("Dropped all existing tables")
            
            # Create all tables
            Base.metadata.create_all(bind=conn)
            logger.info("Created all tables")
            
            # Get the list of created tables
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result]
            logger.info(f"Tables created: {tables}")
            
            # Verify tables were created
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result]
            logger.info(f"Tables in database: {tables}")
            
            # Verify foreign key constraints
            result = conn.execute(text("""
                SELECT 
                    TABLE_NAME, 
                    COLUMN_NAME, 
                    CONSTRAINT_NAME, 
                    REFERENCED_TABLE_NAME, 
                    REFERENCED_COLUMN_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE REFERENCED_TABLE_SCHEMA = DATABASE()
                ORDER BY TABLE_NAME, CONSTRAINT_NAME
            """))
            
            # Convert Row objects to dictionaries manually
            fk_columns = ['TABLE_NAME', 'COLUMN_NAME', 'CONSTRAINT_NAME', 'REFERENCED_TABLE_NAME', 'REFERENCED_COLUMN_NAME']
            fks = [dict(zip(fk_columns, row)) for row in result]
            logger.info(f"Found {len(fks)} foreign key constraints")
            
            # Verify all expected tables were created
            expected_tables = set(Base.metadata.tables.keys())
            missing_tables = expected_tables - set(tables)
            if missing_tables:
                logger.warning(f"Missing tables: {missing_tables}")
            
            logger.info("All tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            raise
        finally:
            # Re-enable foreign key checks
            conn.execute(text('SET FOREIGN_KEY_CHECKS=1;'))
    
    logger.info("Database initialization completed successfully")
    
    # Create a session
    db = SessionLocal()
    
    try:
        # Create initial admin and test users if they don't exist
        logger.info("Checking for admin and test users...")
        from app.models.user import User
        if not db.query(User).first():
            from app.core.security import get_password_hash
            
            # Create admin user
            print("Adding admin user...")
            admin_user = User(
                username="admin",
                email="admin@daybreak.ai",
                hashed_password=get_password_hash("testpassword"),
                full_name="Admin User",
                is_active=True,
                is_superuser=True
            )
            db.add(admin_user)
            
            # Create test user
            print("Adding test user...")
            test_user = User(
                username="testuser",
                email="testuser@daybreak.ai",
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
