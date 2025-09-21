from enum import Enum
from datetime import datetime
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from sqlalchemy import Integer, String, DateTime, Enum as SQLEnum, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship, Mapped, mapped_column, Session
# Using standard JSON for MySQL compatibility

from .base import Base

# Import for type checking only to avoid circular imports
if TYPE_CHECKING:
    from .group import Group
    from .player import Player
    from .user import User
    from .agent_config import AgentConfig

class GameStatus(str, Enum):
    CREATED = "CREATED"
    STARTED = "STARTED"
    ROUND_IN_PROGRESS = "ROUND_IN_PROGRESS"
    ROUND_COMPLETED = "ROUND_COMPLETED"
    FINISHED = "FINISHED"

class Game(Base):
    __tablename__ = "games"  # Explicitly set table name to match foreign key references

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[GameStatus] = mapped_column(SQLEnum(GameStatus), default=GameStatus.CREATED)
    current_round: Mapped[int] = mapped_column(Integer, default=0)
    max_rounds: Mapped[int] = mapped_column(Integer, default=52)
    # Optional metadata/ownership
    created_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    demand_pattern: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)  # Store game configuration
    group_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("groups.id", ondelete="CASCADE"), nullable=True)
    
    # Role assignments: {role: {'is_ai': bool, 'agent_config_id': Optional[int], 'user_id': Optional[int]}}
    role_assignments: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Define relationships directly with string-based references
    players: Mapped[List["Player"]] = relationship("Player", back_populates="game", lazy="selectin")
    rounds: Mapped[List["Round"]] = relationship("Round", back_populates="game", lazy="selectin")
    users = relationship("User", secondary="user_games", back_populates="games", lazy="selectin")
    supervisor_actions = relationship("SupervisorAction", back_populates="game", lazy="selectin")
    agent_configs = relationship("AgentConfig", back_populates="game", lazy="selectin")
    group: Mapped[Optional["Group"]] = relationship("Group", back_populates="games")
    
    def get_role_assignment(self, role: str) -> Dict[str, Any]:
        """Get the assignment for a specific role"""
        return self.role_assignments.get(role, {'is_ai': False, 'agent_config_id': None, 'user_id': None})
    
    def set_role_assignment(self, role: str, is_ai: bool, agent_config_id: Optional[int] = None, user_id: Optional[int] = None):
        """Set the assignment for a specific role"""
        if not hasattr(self, 'role_assignments') or not self.role_assignments:
            self.role_assignments = {}
        self.role_assignments[role] = {
            'is_ai': is_ai,
            'agent_config_id': agent_config_id,
            'user_id': user_id if not is_ai else None
        }
    
    def get_agent_config(self, role: str, db: Session) -> Optional['AgentConfig']:
        """Get the agent configuration for a role"""
        assignment = self.get_role_assignment(role)
        if not assignment or not assignment['is_ai'] or not assignment['agent_config_id']:
            return None
        return db.query(AgentConfig).filter(
            AgentConfig.id == assignment['agent_config_id'],
            AgentConfig.game_id == self.id
        ).first()

class Round(Base):
    __tablename__ = "rounds"  # Explicitly set table name to match foreign key references

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    game_id: Mapped[int] = mapped_column(Integer, ForeignKey("games.id", ondelete="CASCADE"))
    round_number: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, in_progress, completed
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)  # Round-specific configuration
    
    # Define relationships directly with string-based references
    game: Mapped["Game"] = relationship("Game", back_populates="rounds", lazy="selectin")
    player_actions: Mapped[List["PlayerAction"]] = relationship("PlayerAction", back_populates="round", lazy="selectin")

class PlayerAction(Base):
    __tablename__ = "player_actions"  # Explicitly set table name to match foreign key references

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    game_id: Mapped[int] = mapped_column(Integer, ForeignKey("games.id", ondelete="CASCADE"))
    round_id: Mapped[int] = mapped_column(Integer, ForeignKey("rounds.id", ondelete="CASCADE"))
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id", ondelete="CASCADE"))
    action_type: Mapped[str] = mapped_column(String(50))
    quantity: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Define relationships directly with string-based references
    round: Mapped["Round"] = relationship("Round", back_populates="player_actions", lazy="selectin")
    player: Mapped["Player"] = relationship("Player", back_populates="actions", lazy="selectin")
    game: Mapped["Game"] = relationship("Game", lazy="selectin")
