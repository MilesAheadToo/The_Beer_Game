from .auth import router as auth_router
from .users import router as users_router
from .game import router as game_router
from .model import router as model_router
from .dashboard import dashboard_router
from .config import router as config_router

# Export all routers
__all__ = [
    'auth_router',
    'users_router',
    'game_router',
    'model_router',
    'dashboard_router',
    'config_router',
]
