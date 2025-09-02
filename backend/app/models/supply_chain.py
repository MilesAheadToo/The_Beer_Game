from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON, Boolean, Enum, and_
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func
from enum import Enum as PyEnum
from typing import Optional, List, Dict, Any
from app.db.base import Base
import datetime

# Import database session
try:
    from app.db.session import SessionLocal
except ImportError:
    # Fallback for when running in a context where the app package isn't fully initialized
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import create_engine
    from app.core.config import settings
    
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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
    demand_pattern = Column(JSON, default={
        "type": "classic",
        "params": {
            "stable_period": 5,
            "step_increase": 4
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
    def upstream_player(self, db: Optional[Session] = None) -> Optional['Player']:
        """Get the player's upstream player in the supply chain."""
        if not db:
            db = SessionLocal()
            try:
                return self.upstream_player(db)
            finally:
                db.close()
                
        if self.role == PlayerRole.RETAILER:
            return None
            
        role_order = [PlayerRole.FACTORY, PlayerRole.DISTRIBUTOR, PlayerRole.WHOLESALER, PlayerRole.RETAILER]
        current_index = role_order.index(self.role)
        upstream_role = role_order[current_index + 1] if current_index + 1 < len(role_order) else None
        
        if upstream_role:
            return db.query(Player).filter(
                Player.game_id == self.game_id,
                Player.role == upstream_role
            ).first()
        return None

    def downstream_player(self, db: Optional[Session] = None) -> Optional['Player']:
        """Get the player's downstream player in the supply chain."""
        if not db:
            db = SessionLocal()
            try:
                return self.downstream_player(db)
            finally:
                db.close()
                
        if self.role == PlayerRole.FACTORY:
            return None
            
        role_order = [PlayerRole.FACTORY, PlayerRole.DISTRIBUTOR, PlayerRole.WHOLESALER, PlayerRole.RETAILER]
        current_index = role_order.index(self.role)
        downstream_role = role_order[current_index - 1] if current_index > 0 else None
        
        if downstream_role:
            return db.query(Player).filter(
                Player.game_id == self.game_id,
                Player.role == downstream_role
            ).first()
        return None

# ... rest of the code remains the same ...
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
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    game = relationship("Game", back_populates="rounds")
    player_rounds = relationship("PlayerRound", back_populates="game_round")

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
