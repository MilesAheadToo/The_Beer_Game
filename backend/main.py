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
def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")

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
    settings.FRONTEND_URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
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
    init_db()
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

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.VERSION}

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
