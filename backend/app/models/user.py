from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey, Table
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

# Import for type checking only to avoid circular imports
if TYPE_CHECKING:
    from .game import Game
    from .player import Player
    from .session import UserSession
    from .auth_models import PasswordHistory, PasswordResetToken
    from .user import RefreshToken

# Association table for many-to-many relationship between users and games
# Using older style Column syntax for Table definition
user_games = Table(
    'user_games',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('game_id', Integer, ForeignKey('games.id'), primary_key=True)
)

class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(100), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_password_change: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    mfa_secret: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    
    # Relationships
    games: Mapped[List["Game"]] = relationship(
        "Game", 
        secondary=user_games, 
        back_populates="users",
        lazy='selectin'
    )
    
    players: Mapped[List["Player"]] = relationship("Player", back_populates="user", lazy="selectin")
    sessions: Mapped[List["UserSession"]] = relationship(
        "UserSession", 
        back_populates="user", 
        cascade="all, delete-orphan"
    )
    password_history: Mapped[List["PasswordHistory"]] = relationship(
        "PasswordHistory", 
        back_populates="user", 
        cascade="all, delete-orphan"
    )
    password_reset_tokens: Mapped[List["PasswordResetToken"]] = relationship(
        "PasswordResetToken", 
        back_populates="user", 
        cascade="all, delete-orphan"
    )
    refresh_tokens: Mapped[List["RefreshToken"]] = relationship(
        "RefreshToken", 
        back_populates="user", 
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<User {self.username}>"

class RefreshToken(Base):
    """Refresh token model for JWT token refresh functionality."""
    __tablename__ = "refresh_tokens"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    token: Mapped[str] = mapped_column(String(500), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="refresh_tokens")
