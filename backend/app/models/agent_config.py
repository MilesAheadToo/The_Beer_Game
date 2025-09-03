from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base

class AgentConfig(Base):
    """Stores configuration for AI agents in the game"""
    __tablename__ = "agent_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)  # Which role this config is for
    agent_type = Column(String(50), nullable=False)  # e.g., 'base', 'reinforcement_learning'
    config = Column(JSON, default=dict)  # Agent-specific configuration
    
    # Relationships
    game = relationship("Game", back_populates="agent_configs")
    
    def __repr__(self):
        return f"<AgentConfig(id={self.id}, game_id={self.game_id}, role={self.role}, type={self.agent_type})>"
