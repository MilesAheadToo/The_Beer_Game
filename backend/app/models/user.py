from datetime import datetime
from enum import Enum
from typing import Optional, TYPE_CHECKING, Any, Dict, List
from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey, Table, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Enum as SAEnum
from pydantic import BaseModel, EmailStr, Field

from .base import Base

# Import for type checking only to avoid circular imports
if TYPE_CHECKING:
    from .game import Game
    from .player import Player
    from .session import UserSession
    from .auth_models import PasswordHistory, PasswordResetToken
    from .user import RefreshToken
    from .group import Group

# Association table for many-to-many relationship between users and games
# Using older style Column syntax for Table definition
user_games = Table(
    'user_games',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('game_id', Integer, ForeignKey('games.id'), primary_key=True)
)

class UserTypeEnum(str, Enum):
    """Application-level user type classification."""
    SYSTEM_ADMIN = "SystemAdmin"
    GROUP_ADMIN = "GroupAdmin"
    PLAYER = "Player"


class UserBase(BaseModel):
    """Base Pydantic model for User data validation."""
    email: EmailStr
    username: Optional[str] = None
    full_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False
    group_id: Optional[int] = None
    user_type: UserTypeEnum = Field(default=UserTypeEnum.PLAYER)

    class Config:
        from_attributes = True  # Updated from orm_mode in Pydantic v2
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

class UserCreate(UserBase):
    """Model for creating a new user."""
    password: str = Field(..., min_length=8)

class UserUpdate(UserBase):
    """Model for updating an existing user."""
    email: Optional[EmailStr] = None
    password: Optional[str] = None

class UserPasswordChange(BaseModel):
    """Model for changing a user's password."""
    current_password: str
    new_password: str = Field(..., min_length=8)

class UserInDB(UserBase):
    """Model for user data in the database."""
    id: int
    hashed_password: str
    created_at: datetime
    updated_at: datetime

class UserPublic(UserBase):
    """Public user model (excludes sensitive data)."""
    id: int
    created_at: datetime
    updated_at: datetime
    is_superuser: bool = False
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }

class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    user_type: Mapped[UserTypeEnum] = mapped_column(
        SAEnum(UserTypeEnum, name="user_type_enum"),
        nullable=False,
        server_default=UserTypeEnum.PLAYER.value,
        default=UserTypeEnum.PLAYER,
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_password_change: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    mfa_secret: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    group_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("groups.id", ondelete="CASCADE"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    
    # Helper methods
    @property
    def is_authenticated(self) -> bool:
        """Check if the user is authenticated."""
        return self.is_active

    @property
    def is_admin(self) -> bool:
        """Check if the user is classified as a group administrator."""
        return self.user_type == UserTypeEnum.GROUP_ADMIN

    def has_role(self, role: str) -> bool:
        """Legacy role helper for compatibility with older checks."""
        normalized = (role or "").strip().lower()
        if normalized in {"systemadmin", "system_admin", "superadmin"}:
            return self.user_type == UserTypeEnum.SYSTEM_ADMIN
        if normalized in {"groupadmin", "group_admin", "admin"}:
            return self.user_type == UserTypeEnum.GROUP_ADMIN
        if normalized in {"player"}:
            return self.user_type == UserTypeEnum.PLAYER
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert user object to dictionary."""
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_superuser": self.is_superuser,
            "group_id": self.group_id,
            "user_type": self.user_type.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

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
    
    group: Mapped[Optional["Group"]] = relationship("Group", back_populates="users", foreign_keys=[group_id])
    admin_of_group: Mapped[Optional["Group"]] = relationship(
        "Group", back_populates="admin", uselist=False, foreign_keys="Group.admin_id"
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
