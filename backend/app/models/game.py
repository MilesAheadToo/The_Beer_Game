from enum import Enum
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import Column, Integer, String, DateTime, Enum as SQLEnum, JSON, ForeignKey
from sqlalchemy.orm import relationship, Mapped, mapped_column

from ..db.base_class import Base

# Import for type checking only to avoid circular imports
if TYPE_CHECKING:
    from .player import Player
    from .round import Round
    from .game import PlayerAction  # Add this line to resolve circular import

class GameStatus(str, Enum):
    CREATED = "created"
    STARTED = "started"
    ROUND_IN_PROGRESS = "round_in_progress"
    ROUND_COMPLETED = "round_completed"
    FINISHED = "finished"

class Game(Base):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[GameStatus] = mapped_column(SQLEnum(GameStatus), default=GameStatus.CREATED)
    current_round: Mapped[int] = mapped_column(Integer, default=0)
    max_rounds: Mapped[int] = mapped_column(Integer, default=52)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)  # Store game configuration
    
    # Define relationships directly with string-based references
    players: Mapped[List["Player"]] = relationship("Player", back_populates="game", lazy="selectin")
    rounds: Mapped[List["Round"]] = relationship("Round", back_populates="game", lazy="selectin")
    users = relationship("User", secondary="user_games", back_populates="games", lazy="selectin")

class Round(Base):
    __tablename__ = "rounds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    game_id: Mapped[int] = mapped_column(Integer, ForeignKey("games.id"))
    round_number: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending, in_progress, completed
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)  # Round-specific configuration
    
    # Define relationships directly with string-based references
    game: Mapped["Game"] = relationship("Game", back_populates="rounds", lazy="selectin")
    player_actions: Mapped[List["PlayerAction"]] = relationship("PlayerAction", back_populates="round", lazy="selectin")

class PlayerAction(Base):
    __tablename__ = "player_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    game_id: Mapped[int] = mapped_column(Integer, ForeignKey("games.id"))
    round_id: Mapped[int] = mapped_column(Integer, ForeignKey("rounds.id"))
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"))
    action_type: Mapped[str] = mapped_column(String)
    quantity: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Define relationships directly with string-based references
    round: Mapped["Round"] = relationship("Round", back_populates="player_actions", lazy="selectin")
    player: Mapped["Player"] = relationship("Player", back_populates="actions", lazy="selectin")
    game: Mapped["Game"] = relationship("Game", lazy="selectin")
