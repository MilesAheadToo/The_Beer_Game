from fastapi import FastAPI, Depends, WebSocket, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
from typing import List, Optional, Union, Callable, Awaitable
import uvicorn
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

# Import CSRF middleware and security utilities
from app.middleware.csrf import CSRFMiddleware
from app.core.config import settings

from app.core.config import settings
from app.api.api_v1.api import api_router
from app.websockets.endpoints import router as ws_router
from app.websockets import manager
from app.api.endpoints.agent_game import router as agent_game_router
from app.api.endpoints.mixed_game import router as mixed_game_router
from app.api.endpoints.health import router as health_router
from app.api.routes.agent import router as agent_router
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
async def init_db(drop_tables: bool = False):
    """Initialize the database with tables and initial data.
    
    Args:
        drop_tables: If True, drops all tables before creating them. Use with caution in production.
    """
    from app.db.session import engine, Base, async_session_factory, SQLALCHEMY_DATABASE_URI
    from sqlalchemy import text, inspect, create_engine
    from sqlalchemy.schema import CreateTable, DropTable
    from sqlalchemy.ext.asyncio import create_async_engine
    import asyncio
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing database...")
        
        # Import all models to ensure they are registered with SQLAlchemy
        from app.models.user import User, RefreshToken
        from app.models.auth_models import PasswordHistory, PasswordResetToken
        from app.models.session import TokenBlacklist, UserSession
        from app.models.game import Game, Round, PlayerAction
        from app.models.player import Player
        
        # Define the correct table creation order to handle foreign key dependencies
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
        
        # Import the sync_engine from session.py
        from app.db.session import sync_engine, engine as async_engine
        
        # Use a synchronous connection for table operations
        with sync_engine.connect() as conn:
            if drop_tables:
                logger.warning("Dropping all tables (drop_tables=True)")
                # Disable foreign key checks
                conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
                conn.commit()
                
                # Drop tables in reverse order to handle foreign key constraints
                for table in reversed(tables_in_order):
                    conn.execute(text(f"DROP TABLE IF EXISTS {table.name}"))
                    conn.commit()
                
                # Re-enable foreign key checks
                conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
                conn.commit()
            
            # Create tables one by one in the correct order
            logger.info("Creating tables in the correct order...")
            for table in tables_in_order:
                try:
                    table.create(bind=sync_engine, checkfirst=True)
                    logger.info(f"Created table: {table.name}")
                except Exception as e:
                    logger.error(f"Error creating table {table.name}: {e}")
                    continue  # Skip to the next table instead of failing
            
            logger.info("Database initialization completed successfully")
        
        # Close the sync engine
        sync_engine.dispose()
        
        # Verify tables with an async connection
        async with async_engine.connect() as conn:
            result = await conn.execute(text("SHOW TABLES"))
            tables = result.fetchall()
            logger.info(f"Found {len(tables)} tables in the database")
            
        # Close the async engine
        await async_engine.dispose()
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming requests and outgoing responses."""
    
    async def dispatch(self, request: Request, call_next):
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        logger.info(f"Response: {request.method} {request.url} - {response.status_code}")
        
        return response

# Configure CORS middleware with credentials support
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # List of allowed origins
    allow_credentials=True,               # Required for cookies
    allow_methods=["*"],                  # Allow all methods
    allow_headers=["*"],                  # Allow all headers
)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc"
)

# Include health check router
app.include_router(health_router, tags=["health"])

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Set-Cookie", "X-CSRF-Token"],
)

# Add CSRF middleware
try:
    app.add_middleware(CSRFMiddleware)
    logger.info("CSRF middleware enabled")
except Exception as e:
    logger.error(f"Failed to add CSRF middleware: {e}")
    raise

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Include API routers
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(agent_game_router, prefix=f"{settings.API_V1_STR}/agent-game")
app.include_router(mixed_game_router, prefix=f"{settings.API_V1_STR}/mixed-game")
app.include_router(health_router, prefix=f"{settings.API_V1_STR}/health")
app.include_router(agent_router, prefix=f"{settings.API_V1_STR}")

# Include WebSocket router
app.include_router(ws_router, prefix="/ws")

# Add startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application...")
    # Only create tables if they don't exist (don't drop existing tables)
    await init_db(drop_tables=False)
    logger.info("Application startup complete")

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint that provides basic API information and health status."""
    import os
    from datetime import datetime
    
    # Get environment information
    env_info = {
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "database_url": f"{settings.SQLALCHEMY_DATABASE_URI.split('@')[-1]}" if settings.SQLALCHEMY_DATABASE_URI else None,
        "redis_enabled": bool(settings.REDIS_URL),
    }
    
    # Get system information
    import psutil
    import platform
    
    system_info = {
        "python_version": platform.python_version(),
        "system": platform.system(),
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "process_id": os.getpid(),
        "started_at": datetime.now().isoformat(),
    }
    
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "documentation": {
            "swagger": f"{settings.API_V1_STR}/docs",
            "redoc": f"{settings.API_V1_STR}/redoc",
            "openapi_json": f"{settings.API_V1_STR}/openapi.json",
        },
        "environment": env_info,
        "system": system_info,
        "timestamp": datetime.utcnow().isoformat(),
    }

# Health check endpoint with database connectivity check
@app.get("/api/health")
async def health_check():
    try:
        # Create a new async session for the health check
        from sqlalchemy import text
        from app.db.session import async_session_factory, engine
        
        async with async_session_factory() as session:
            # Check database connection
            result = await session.execute(text("SELECT 1"))
            result.scalar_one()  # Execute the query and get the result
            
            return {
                "status": "healthy",
                "version": settings.VERSION,
                "database": "connected"
            }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {str(e)}"
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
