"""Models package with all SQLAlchemy models."""
import logging
from typing import List, Type, Any

# Import base first
from .base import Base

# Import models in the correct order to avoid circular imports
from .user import User, RefreshToken
from .player import Player, PlayerRole, PlayerType, PlayerStrategy
from .game import Game, GameStatus, Round, PlayerAction

# Get logger
logger = logging.getLogger(__name__)

# Ensure all models are imported and registered with SQLAlchemy
# This is necessary for proper relationship resolution
_models: List[Type[Base]] = [
    User,
    RefreshToken,
    Player,
    Game,
    Round,
    PlayerAction,
]

# No need for separate relationship setup as we're using string-based relationships
logger.info("All models imported successfully")

# Set __all__ for explicit exports
__all__ = [
    'Base',
    'User',
    'RefreshToken',
    'Game',
    'GameStatus',
    'Round',
    'PlayerAction',
    'Player',
    'PlayerRole',
    'PlayerType',
    'PlayerStrategy',
]
