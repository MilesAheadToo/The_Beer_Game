from fastapi import FastAPI, Depends, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.api_v1.api import api_router
from app.api.endpoints.websocket import router as ws_router, manager
from app.db.base import Base, engine

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(api_router, prefix=settings.API_V1_STR)

# Include WebSocket router
app.include_router(ws_router, prefix="")

@app.get("/")
async def root():
    return {
        "message": "Welcome to The Beer Game API",
        "docs": "/docs",
        "websocket": "/ws/games/{game_id}/players/{player_id}",
        "api_version": settings.VERSION
    }

# Add WebSocket endpoint for testing
@app.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message received: {data}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
