from fastapi import APIRouter

from app.api.endpoints import game, auth
from app.core.config import settings

api_router = APIRouter()

# Include API routes
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(game.router, prefix="/games", tags=["games"])
