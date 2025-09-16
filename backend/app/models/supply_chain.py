from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON, Boolean, Enum, and_
from sqlalchemy.orm import relationship
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func, select
from enum import Enum as PyEnum
from typing import Optional, List, Dict, Any
from app.db.base import Base
import datetime
from app.db.session import get_db

class GameStatus(str, PyEnum):
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"

class PlayerRole(str, PyEnum):
    RETAILER = "retailer"
    WHOLESALER = "wholesaler"
    DISTRIBUTOR = "distributor"
    FACTORY = "factory"

class Game(Base):
    __tablename__ = "games"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    status = Column(Enum(GameStatus), default=GameStatus.CREATED)
    current_round = Column(Integer, default=0)
    max_rounds = Column(Integer, default=52)  # Default to 52 weeks (1 year)
    round_time_limit = Column(Integer, default=60)  # Default 60 seconds per round
    current_round_ends_at = Column(DateTime, nullable=True)  # When the current round will end
    demand_pattern = Column(JSON, default={
        "type": "classic",
        "params": {
            "initial_demand": 4,
            "change_week": 6,
            "final_demand": 8
        }
    })
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    players = relationship("Player", back_populates="game")
    rounds = relationship("GameRound", back_populates="game")
    users = relationship("User", secondary="user_games", back_populates="games")

class Player(Base):
    __tablename__ = "players"
    
    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Null for AI players
    role = Column(Enum(PlayerRole), nullable=False)
    name = Column(String(100), nullable=False)
    is_ai = Column(Boolean, default=False)
    user = relationship("User")
    
    game = relationship("Game", back_populates="players")
    inventory = relationship("PlayerInventory", back_populates="player")
    orders = relationship("Order", back_populates="player")
    player_rounds = relationship("PlayerRound", back_populates="player")
    
    @property
    async def upstream_player(self, db: Optional[AsyncSession] = None):
        """Get the player's upstream player in the supply chain."""
        if not db:
            async with get_db() as db_session:
                return await self.upstream_player(db_session)
                
        if self.role == PlayerRole.RETAILER:
            return None
            
        upstream_role = {
            PlayerRole.WHOLESALER: PlayerRole.RETAILER,
            PlayerRole.DISTRIBUTOR: PlayerRole.WHOLESALER,
            PlayerRole.FACTORY: PlayerRole.DISTRIBUTOR
        }.get(self.role)
        
        if not upstream_role:
            return None
            
        result = await db.execute(
            select(Player).where(
                Player.game_id == self.game_id,
                Player.role == upstream_role
            )
        )
        return result.scalars().first()

    async def downstream_player(self, db: Optional[AsyncSession] = None):
        """Get the player's downstream player in the supply chain."""
        if not db:
            async with get_db() as db_session:
                return await self.downstream_player(db_session)
                
        if self.role == PlayerRole.FACTORY:
            return None
            
        downstream_role = {
            PlayerRole.RETAILER: PlayerRole.WHOLESALER,
            PlayerRole.WHOLESALER: PlayerRole.DISTRIBUTOR,
            PlayerRole.DISTRIBUTOR: PlayerRole.FACTORY
        }.get(self.role)
        
        if not downstream_role:
            return None
            
        result = await db.execute(
            select(Player).where(
                Player.game_id == self.game_id,
                Player.role == downstream_role
            )
        )
        return result.scalars().first()

class PlayerInventory(Base):
    __tablename__ = "player_inventory"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    current_stock = Column(Integer, default=12)  # Starting inventory
    incoming_shipments = Column(JSON, default=[])  # List of shipments in transit with arrival round
    backorders = Column(Integer, default=0)  # Unfulfilled customer orders
    cost = Column(Float, default=0.0)  # Total cost incurred
    
    player = relationship("Player", back_populates="inventory")

class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    quantity = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    player = relationship("Player", back_populates="orders")
    game = relationship("Game")

class GameRound(Base):
    __tablename__ = "game_rounds"
    
    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    customer_demand = Column(Integer, nullable=False)  # Random demand for this round
    is_completed = Column(Boolean, default=False)  # Whether the round is completed
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)  # When the round was completed
    
    game = relationship("Game", back_populates="rounds")
    player_rounds = relationship("PlayerRound", back_populates="game_round")
    
    async def all_players_submitted(self, db: AsyncSession):
        """Check if all players have submitted their orders for this round."""
        from sqlalchemy import func
        
        # Count distinct players who have submitted for this round
        submitted_players_result = await db.execute(
            select(func.count(PlayerRound.player_id.distinct())).where(
                PlayerRound.round_id == self.id
            )
        )
        submitted_players = submitted_players_result.scalar()
        
        # Count total players in the game
        total_players_result = await db.execute(
            select(func.count(Player.id)).where(
                Player.game_id == self.game_id
            )
        )
        total_players = total_players_result.scalar()
        
        return submitted_players >= total_players

class PlayerRound(Base):
    __tablename__ = "player_rounds"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    round_id = Column(Integer, ForeignKey("game_rounds.id"), nullable=False)
    order_placed = Column(Integer, nullable=False)  # Order placed this round
    order_received = Column(Integer, nullable=False)  # Order received from downstream
    inventory_before = Column(Integer, nullable=False)
    inventory_after = Column(Integer, nullable=False)
    backorders_before = Column(Integer, default=0)
    backorders_after = Column(Integer, default=0)
    holding_cost = Column(Float, default=0.0)
    backorder_cost = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    comment = Column(String(255), nullable=True)
    
    player = relationship("Player", back_populates="player_rounds")
    game_round = relationship("GameRound", back_populates="player_rounds")

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(255))
    unit_cost = Column(Float, default=0.0)

class SimulationRun(Base):
    __tablename__ = "simulation_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(255))
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    parameters = Column(JSON)  # Store simulation parameters
    
    steps = relationship("SimulationStep", back_populates="simulation_run")

class SimulationStep(Base):
    __tablename__ = "simulation_steps"
    
    id = Column(Integer, primary_key=True, index=True)
    simulation_run_id = Column(Integer, ForeignKey("simulation_runs.id"), nullable=False)
    step_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    state = Column(JSON)  # Store the complete state of the simulation at this step
    
    simulation_run = relationship("SimulationRun", back_populates="steps")
