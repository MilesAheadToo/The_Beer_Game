from datetime import datetime
from typing import Optional, List
from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship

from app.db.base import Base

# Association table for many-to-many relationship between users and games
# Defined here to avoid circular imports
user_games = Table(
    'user_games',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('game_id', Integer, ForeignKey('games.id'), primary_key=True)
)

class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(100), nullable=False)
    full_name = Column(String(100), nullable=True)
    is_active = Column(Boolean(), default=True)
    is_superuser = Column(Boolean(), default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    games = relationship("Game", 
                       secondary=user_games, 
                       back_populates="users",
                       lazy='selectin')
    
    players = relationship("Player", back_populates="user", lazy="selectin")
    
    def __repr__(self):
        return f"<User {self.username}>"

class RefreshToken(Base):
    """Refresh token model for JWT token refresh functionality."""
    __tablename__ = "refresh_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String(500), unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="refresh_tokens")

# Add the relationship to the User model after RefreshToken is defined
User.refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
