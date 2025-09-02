from typing import Optional, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum

class PlayerRole(str, Enum):
    RETAILER = "retailer"
    WHOLESALER = "wholesaler"
    DISTRIBUTOR = "distributor"
    FACTORY = "factory"

class PlayerType(str, Enum):
    HUMAN = "human"
    AGENT = "agent"

class PlayerStrategy(str, Enum):
    # Basic strategies
    NAIVE = "naive"
    BULLWHIP = "bullwhip"
    CONSERVATIVE = "conservative"
    RANDOM = "random"
    # Advanced strategies
    DEMAND_DRIVEN = "demand_driven"
    COST_OPTIMIZATION = "cost_optimization"
    # LLM-based strategies
    LLM_CONSERVATIVE = "llm_conservative"
    LLM_BALANCED = "llm_balanced"
    LLM_AGGRESSIVE = "llm_aggressive"
    LLM_ADAPTIVE = "llm_adaptive"

class PlayerAssignment(BaseModel):
    """Schema for assigning a player to a game role."""
    role: PlayerRole
    player_type: PlayerType = PlayerType.HUMAN
    user_id: Optional[int] = None  # Required for human players
    strategy: Optional[PlayerStrategy] = PlayerStrategy.NAIVE  # For AI players
    can_see_demand: bool = False  # Whether this player can see customer demand

    @validator('user_id')
    def validate_user_id(cls, v, values):
        if values.get('player_type') == PlayerType.HUMAN and v is None:
            raise ValueError("user_id is required for human players")
        return v

    @validator('strategy')
    def validate_strategy(cls, v, values):
        if values.get('player_type') == PlayerType.AGENT and v is None:
            raise ValueError("strategy is required for AI players")
        return v

class PlayerResponse(BaseModel):
    """Schema for player response data."""
    id: int
    game_id: int
    user_id: Optional[int]
    role: PlayerRole
    player_type: PlayerType
    name: str
    strategy: Optional[PlayerStrategy]
    can_see_demand: bool
    is_ready: bool = False

    class Config:
        orm_mode = True

class PlayerUpdate(BaseModel):
    """Schema for updating player information."""
    strategy: Optional[PlayerStrategy] = None
    can_see_demand: Optional[bool] = None
    is_ready: Optional[bool] = None
