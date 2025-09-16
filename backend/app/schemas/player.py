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
    # Daybreak strategies
    DAYBREAK_DTCE = "daybreak_dtce"
    DAYBREAK_DTCE_CENTRAL = "daybreak_dtce_central"
    DAYBREAK_DTCE_GLOBAL = "daybreak_dtce_global"

class PlayerAssignment(BaseModel):
    """Schema for assigning a player to a game role."""
    role: PlayerRole
    player_type: PlayerType = PlayerType.HUMAN
    user_id: Optional[int] = None  # Required for human players
    strategy: Optional[PlayerStrategy] = PlayerStrategy.NAIVE  # For AI players
    can_see_demand: bool = False  # Whether this player can see customer demand
    llm_model: Optional[str] = Field(
        default="gpt-4o-mini", description="Selected LLM when using LLM strategies"
    )
    llm_config: Optional[dict] = None  # temperature, max_tokens, prompt
    basic_config: Optional[dict] = None  # heuristic params, e.g., base_stock_target, smoothing
    daybreak_override_pct: Optional[float] = Field(
        default=None,
        ge=0.05,
        le=0.5,
        description="Optional override percentage for centralized Daybreak coordination (0.05-0.5)",
    )

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
    llm_model: Optional[str] = None
    is_ready: bool = False

    class Config:
        orm_mode = True

class PlayerUpdate(BaseModel):
    """Schema for updating player information."""
    strategy: Optional[PlayerStrategy] = None
    can_see_demand: Optional[bool] = None
    llm_model: Optional[str] = None
    is_ready: Optional[bool] = None
