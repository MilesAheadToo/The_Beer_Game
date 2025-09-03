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
    RETAILER = "retailer"
    WHOLESALER = "wholesaler"
    DISTRIBUTOR = "distributor"
    MANUFACTURER = "manufacturer"

class PlayerType(str, Enum):
    HUMAN = "human"
    AI = "ai"

class PlayerStrategy(str, Enum):
    # Basic strategies
    MANUAL = "manual"
    RANDOM = "random"
    FIXED = "fixed"
    
    # Advanced strategies
    DEMAND_AVERAGE = "demand_average"
    TREND_FOLLOWER = "trend_follower"
    
    # LLM-based strategies
    LLM_BASIC = "llm_basic"
    LLM_ADVANCED = "llm_advanced"
    LLM_REINFORCEMENT = "llm_reinforcement"

class Player(Base):
    __tablename__ = "players"  # Explicitly set table name to match foreign key references

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    game_id: Mapped[int] = mapped_column(Integer, ForeignKey("games.id", ondelete="CASCADE"))
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[PlayerRole] = mapped_column(SQLEnum(PlayerRole), nullable=False)
    type: Mapped[PlayerType] = mapped_column(SQLEnum(PlayerType), default=PlayerType.HUMAN)
    strategy: Mapped[PlayerStrategy] = mapped_column(SQLEnum(PlayerStrategy), default=PlayerStrategy.MANUAL)
    
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
