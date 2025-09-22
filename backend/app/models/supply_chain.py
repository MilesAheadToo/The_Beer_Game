"""Supplemental supply-chain tables that hang off the primary Game / Player models."""

from __future__ import annotations

import datetime
from typing import Dict, Any

from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship

from .base import Base


class PlayerInventory(Base):
    __tablename__ = "player_inventory"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id", ondelete="CASCADE"), nullable=False)
    current_stock = Column(Integer, default=12)
    incoming_shipments = Column(JSON, default=list)
    backorders = Column(Integer, default=0)
    cost = Column(Float, default=0.0)

    player = relationship("Player", back_populates="inventory")


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id", ondelete="CASCADE"), nullable=False)
    round_number = Column(Integer, nullable=False)
    quantity = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    player = relationship("Player", back_populates="orders")
    game = relationship("Game")


class GameRound(Base):
    __tablename__ = "game_rounds"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    round_number = Column(Integer, nullable=False)
    customer_demand = Column(Integer, nullable=False)
    is_completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    game = relationship("Game", back_populates="rounds")
    player_rounds = relationship("PlayerRound", back_populates="game_round")


class PlayerRound(Base):
    __tablename__ = "player_rounds"

    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id", ondelete="CASCADE"), nullable=False)
    round_id = Column(Integer, ForeignKey("game_rounds.id", ondelete="CASCADE"), nullable=False)
    order_placed = Column(Integer, nullable=False)
    order_received = Column(Integer, default=0)
    inventory_before = Column(Integer, nullable=False)
    inventory_after = Column(Integer, nullable=False)
    backorders_before = Column(Integer, default=0)
    backorders_after = Column(Integer, default=0)
    holding_cost = Column(Float, default=0.0)
    backorder_cost = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    comment = Column(JSON, default=dict)

    player = relationship("Player", back_populates="player_rounds")
    game_round = relationship("GameRound", back_populates="player_rounds")
