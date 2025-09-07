from fastapi import APIRouter

from app.api.endpoints import (
    auth_router,
    users_router,
    game_router,
    model_router,
    dashboard_router
)
from app.core.config import settings

api_router = APIRouter()

# Include API routes
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(users_router, prefix="/users", tags=["users"])
api_router.include_router(game_router, prefix="/games", tags=["games"])
api_router.include_router(model_router, prefix="/model", tags=["model"])
api_router.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])
