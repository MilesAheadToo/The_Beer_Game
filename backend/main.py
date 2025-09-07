from fastapi import FastAPI, Depends, WebSocket, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
from typing import List, Optional
import uvicorn

from app.core.config import settings
from app.api.api_v1.api import api_router
from app.websockets.endpoints import router as ws_router
from app.websockets import manager
from app.api.endpoints.agent_game import router as agent_game_router
from app.api.endpoints.mixed_game import router as mixed_game_router
from app.api.endpoints.health import router as health_router
from app.db.base import Base, engine
from app.db.session import SessionLocal
from app.models.user import User
from app.core.security import get_current_user

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Create database tables
def init_db(drop_tables: bool = False):
    """Initialize the database with tables and initial data.
    
    Args:
        drop_tables: If True, drops all tables before creating them. Use with caution in production.
    """
    from app.db.session import engine, Base
    from sqlalchemy import text, inspect
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing database...")
        
        # Create a connection to execute raw SQL
        with engine.connect() as conn:
            # Check if tables exist
            inspector = inspect(engine)
            existing_tables = set(inspector.get_table_names())
            
            if drop_tables:
                logger.warning("Dropping all tables (drop_tables=True)")
                # Disable foreign key checks
                conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
                Base.metadata.drop_all(bind=engine)
                # Re-enable foreign key checks
                conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
            
            # Create tables in the correct order
            logger.info("Creating tables in the correct order...")
            
            # Import all models to ensure they are registered with SQLAlchemy
            from app.models.user import User, RefreshToken
            from app.models.auth_models import PasswordHistory, PasswordResetToken
            from app.models.session import TokenBlacklist, UserSession
            from app.models.game import Game, Round, PlayerAction
            from app.models.player import Player
            
            # Define the order of table creation
            tables_in_order = [
                User.__table__,
                PasswordHistory.__table__,
                PasswordResetToken.__table__,
                RefreshToken.__table__,
                TokenBlacklist.__table__,
                UserSession.__table__,
                Game.__table__,
                Player.__table__,
                Round.__table__,
                PlayerAction.__table__
            ]
            
            # Create tables in the specified order
            for table in tables_in_order:
                table.create(bind=engine, checkfirst=True)
            
            # Commit the transaction
            conn.commit()
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include health check router
app.include_router(health_router, tags=["health"])

# Set up CORS
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    settings.FRONTEND_URL,
    "http://localhost:3001",  # Common frontend dev port
    "http://127.0.0.1:3001"   # Common frontend dev port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "*",
        "Authorization",
        "Content-Type",
        "Access-Control-Allow-Credentials",
        "Access-Control-Allow-Origin"
    ],
    expose_headers=[
        "Content-Disposition",
        "Content-Length",
        "Content-Type"
    ],
    max_age=3600  # Cache preflight requests for 1 hour
)

# Include API routers
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(agent_game_router, prefix=settings.API_V1_STR)
app.include_router(mixed_game_router, prefix=settings.API_V1_STR)

# Include WebSocket router
app.include_router(ws_router, prefix="/ws")

# Add startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application...")
    # Only create tables if they don't exist (don't drop existing tables)
    init_db(drop_tables=False)
    logger.info("Application startup complete")

@app.get("/")
async def root():
    return {
        "message": "Welcome to The Beer Game API",
        "docs": "/docs",
        "websocket": "/ws/games/{game_id}",
        "api_version": settings.VERSION,
        "environment": settings.ENVIRONMENT
    }

# Health check endpoint with database connectivity check
@app.get("/api/health")
async def health_check():
    try:
        # Check database connection
        db = SessionLocal()
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db.close()
        return {
            "status": "healthy",
            "version": settings.VERSION,
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )

# WebSocket test endpoint
@app.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")
            await websocket.send_json({
                "message": f"Echo: {data}",
                "timestamp": "2025-09-01T12:00:00Z"
            })
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning"
    )
