from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .user import User
    from .supply_chain_config import SupplyChainConfig
    from .game import Game

class Group(Base):
    """Organization grouping admins, configs and games."""
    __tablename__ = "groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    logo: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    admin_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)

    admin: Mapped["User"] = relationship("User", back_populates="admin_of_group", foreign_keys=[admin_id])
    users: Mapped[List["User"]] = relationship("User", back_populates="group", foreign_keys="User.group_id", cascade="all, delete-orphan")
    
    # Make supply_chain_configs relationship optional
    if TYPE_CHECKING:
        supply_chain_configs: Mapped[List["SupplyChainConfig"]]
    else:
        supply_chain_configs = relationship(
            "SupplyChainConfig", 
            back_populates="group", 
            cascade="all, delete-orphan",
            lazy='dynamic'
        )
        
    games: Mapped[List["Game"]] = relationship("Game", back_populates="group", cascade="all, delete-orphan")
