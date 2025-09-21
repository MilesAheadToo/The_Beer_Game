from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from sqlalchemy import Integer, String, DateTime, Enum as SQLEnum, JSON, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

# Import for type checking only to avoid circular imports
if TYPE_CHECKING:
    from .game import Game, PlayerAction
    from .user import User

class PlayerRole(str, Enum):
    RETAILER = "RETAILER"
    WHOLESALER = "WHOLESALER"
    DISTRIBUTOR = "DISTRIBUTOR"
    MANUFACTURER = "MANUFACTURER"

class PlayerType(str, Enum):
    HUMAN = "HUMAN"
    AI = "AI"

class PlayerStrategy(str, Enum):
    # Basic strategies
    MANUAL = "MANUAL"
    RANDOM = "RANDOM"
    FIXED = "FIXED"

    # Advanced strategies
    DEMAND_AVERAGE = "DEMAND_AVERAGE"
    TREND_FOLLOWER = "TREND_FOLLOWER"

    # LLM-based strategies
    LLM_BASIC = "LLM_BASIC"
    LLM_ADVANCED = "LLM_ADVANCED"
    LLM_REINFORCEMENT = "LLM_REINFORCEMENT"

class Player(Base):
    __tablename__ = "players"  # Explicitly set table name to match foreign key references

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    game_id: Mapped[int] = mapped_column(Integer, ForeignKey("games.id", ondelete="CASCADE"))
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[PlayerRole] = mapped_column(SQLEnum(PlayerRole), nullable=False)
    type: Mapped[PlayerType] = mapped_column(SQLEnum(PlayerType), default=PlayerType.HUMAN)
    strategy: Mapped[PlayerStrategy] = mapped_column(SQLEnum(PlayerStrategy), default=PlayerStrategy.MANUAL)
    is_ai: Mapped[bool] = mapped_column(Boolean, default=False)
    ai_strategy: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    can_see_demand: Mapped[bool] = mapped_column(Boolean, default=False)
    llm_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, default="gpt-4o-mini")
    
    # Inventory and order tracking
    inventory: Mapped[int] = mapped_column(Integer, default=0)
    backlog: Mapped[int] = mapped_column(Integer, default=0)
    cost: Mapped[int] = mapped_column(Integer, default=0)
    
    # Game state
    is_ready: Mapped[bool] = mapped_column(Boolean, default=False)
    last_order: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Define relationships directly with string-based references
    game: Mapped["Game"] = relationship("Game", back_populates="players", lazy="selectin")
    user: Mapped[Optional["User"]] = relationship("User", back_populates="players", lazy="selectin")
    actions: Mapped[List["PlayerAction"]] = relationship("PlayerAction", back_populates="player", lazy="selectin")
    
    def __repr__(self) -> str:
        return f"<Player {self.name} ({self.role})>"
