"""Models package with all SQLAlchemy models."""
import logging
from typing import List, Type, Any

# Configure logger
logger = logging.getLogger(__name__)

# Import base first - this will also import all models
from .base import Base

# Import all models here to ensure they are registered with SQLAlchemy
from sqlalchemy import inspect

# Import models in dependency order to avoid circular imports
# 1. Core models with no dependencies
from .user import RefreshToken  # Must be imported before User to avoid circular import
from .user import User, user_games

# 2. Models that depend on User
from .player import Player, PlayerRole, PlayerType, PlayerStrategy

# 3. Game-related models
from .supervisor_action import SupervisorAction
from .game import Game, GameStatus, Round, PlayerAction
from .agent_config import AgentConfig
from .auth_models import PasswordHistory, PasswordResetToken
from .session import TokenBlacklist, UserSession

# Verify all models are properly registered
registered_tables = set(Base.metadata.tables.keys())
expected_tables = {
    'users', 'refresh_tokens', 'players', 'password_history',
    'password_reset_tokens', 'token_blacklist', 'user_sessions',
    'games', 'rounds', 'player_actions', 'user_games'
}

missing_tables = expected_tables - registered_tables
if missing_tables:
    logger.warning(f"Missing tables in metadata: {missing_tables}")
    
logger.info(f"Registered tables in metadata: {registered_tables}")

# Explicitly import all models to ensure they are registered with SQLAlchemy metadata
# This helps SQLAlchemy discover all models and their relationships
__all__ = [
    'Base',
    'User',
    'RefreshToken',
    'Player',
    'Game',
    'AgentConfig',
    'PasswordHistory',
    'PasswordResetToken',
    'TokenBlacklist',
    'UserSession',
    'user_games',
    'PlayerRole',
    'PlayerType',
    'PlayerStrategy',
    'GameStatus',
    'Round',
    'PlayerAction'
]
