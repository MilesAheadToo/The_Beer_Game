from typing import List, Optional, Dict, Any, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, validator, root_validator
from datetime import datetime
from .player import PlayerAssignment, PlayerResponse, PlayerUpdate, PlayerRole

class DemandPatternType(str, Enum):
    CLASSIC = "classic"
    RANDOM = "random"
    SEASONAL = "seasonal"
    CONSTANT = "constant"

class DemandPattern(BaseModel):
    type: DemandPatternType = Field(..., description="Type of demand pattern")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the demand pattern"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "classic",
                "params": {
                    "initial_demand": 4,
                    "change_week": 6,
                    "final_demand": 8
                }
            }
        }
    )

class GameStatus(str, Enum):
    CREATED = "CREATED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    PAUSED = "PAUSED"

class PlayerRole(str, Enum):
    RETAILER = "RETAILER"
    WHOLESALER = "WHOLESALER"
    DISTRIBUTOR = "DISTRIBUTOR"
    MANUFACTURER = "MANUFACTURER"

class RolePricing(BaseModel):
    selling_price: float = Field(..., gt=0, description="Selling price per unit")
    standard_cost: float = Field(..., gt=0, description="Standard cost per unit")
    
    @root_validator(skip_on_failure=True)
    def validate_margin(cls, values):
        selling_price = values.get('selling_price')
        standard_cost = values.get('standard_cost')
        if selling_price is not None and standard_cost is not None and selling_price <= standard_cost:
            raise ValueError('Selling price must be greater than standard cost')
        return values

class PricingConfig(BaseModel):
    retailer: RolePricing = Field(
        default_factory=lambda: RolePricing(selling_price=100.0, standard_cost=80.0),
        description="Pricing configuration for the retailer role"
    )
    wholesaler: RolePricing = Field(
        default_factory=lambda: RolePricing(selling_price=75.0, standard_cost=60.0),
        description="Pricing configuration for the wholesaler role"
    )
    distributor: RolePricing = Field(
        default_factory=lambda: RolePricing(selling_price=60.0, standard_cost=45.0),
        description="Pricing configuration for the distributor role"
    )
    manufacturer: RolePricing = Field(
        default_factory=lambda: RolePricing(selling_price=45.0, standard_cost=30.0),
        description="Pricing configuration for the manufacturer role"
    )

class NodePolicy(BaseModel):
    info_delay: int = Field(ge=0, le=52, default=2)
    ship_delay: int = Field(ge=0, le=52, default=2)
    init_inventory: int = Field(ge=0, default=12)
    price: float = Field(ge=0, default=0)
    standard_cost: float = Field(ge=0, default=0)
    variable_cost: float = Field(ge=0, default=0)
    min_order_qty: int = Field(ge=0, default=0)


class DaybreakLLMToggles(BaseModel):
    customer_demand_history_sharing: bool = Field(default=False)
    volatility_signal_sharing: bool = Field(default=False)
    downstream_inventory_visibility: bool = Field(default=False)


class DaybreakLLMConfig(BaseModel):
    toggles: DaybreakLLMToggles = Field(default_factory=DaybreakLLMToggles)
    shared_history_weeks: Optional[int] = Field(default=None, ge=0)
    volatility_window: Optional[int] = Field(default=None, ge=0)

    model_config = ConfigDict(extra="allow")


class GameBase(BaseModel):
    name: str = Field(..., max_length=100)
    max_rounds: int = Field(default=52, ge=1, le=1000)
    description: Optional[str] = Field(None, max_length=500)
    is_public: bool = Field(default=True, description="Whether the game is visible to all users")
    progression_mode: Literal['supervised', 'unsupervised'] = Field(
        default='supervised',
        description="Controls whether the Group Admin advances rounds manually (supervised) or the game auto-progresses once all orders are in (unsupervised)",
    )
    supply_chain_config_id: Optional[int] = Field(
        default=None,
        description="Identifier of the linked supply chain configuration",
    )
    supply_chain_name: Optional[str] = Field(
        default=None,
        description="Friendly name of the linked supply chain configuration",
    )
    pricing_config: PricingConfig = Field(
        default_factory=PricingConfig,
        description="Pricing configuration for different roles in the supply chain"
    )
    # Optional: per-node policies and system-wide variable ranges
    node_policies: Optional[Dict[str, NodePolicy]] = Field(default=None)
    system_config: Optional[Dict[str, Any]] = Field(default=None)
    global_policy: Optional[Dict[str, Any]] = Field(default=None, description="Optional top-level policy values (lead times, inventory, costs, capacities)")
    daybreak_llm: Optional[DaybreakLLMConfig] = Field(
        default=None,
        description="Configuration block for the Daybreak Beer Game Strategist",
    )

class GameCreate(GameBase):
    player_assignments: List[PlayerAssignment] = Field(
        ...,
        min_items=1,
        max_items=4,
        description="List of player assignments for the game"
    )
    demand_pattern: DemandPattern = Field(
        default_factory=lambda: DemandPattern(
            type=DemandPatternType.CLASSIC,
            params={"initial_demand": 4, "change_week": 6, "final_demand": 8}
        ),
        description="Configuration for the demand pattern to use in the game"
    )

class GameUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    status: Optional[GameStatus] = None
    current_round: Optional[int] = Field(None, ge=0)
    max_rounds: Optional[int] = Field(None, ge=1, le=1000)
    description: Optional[str] = Field(None, max_length=500)
    is_public: Optional[bool] = None
    supply_chain_config_id: Optional[int] = None
    supply_chain_name: Optional[str] = None

    @validator('status')
    def validate_status_transition(cls, v, values, **kwargs):
        # Add any status transition validation here
        return v

class GameInDBBase(GameBase):
    id: int
    status: GameStatus
    current_round: int
    demand_pattern: DemandPattern = Field(
        default_factory=lambda: DemandPattern(
            type=DemandPatternType.CLASSIC,
            params={"initial_demand": 4, "change_week": 6, "final_demand": 8}
        ),
        description="Configuration for the demand pattern used in the game"
    )
    pricing_config: PricingConfig = Field(
        default_factory=PricingConfig,
        description="Pricing configuration for different roles in the supply chain"
    )
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[int] = Field(None, description="User ID of the game creator")
    group_id: Optional[int] = Field(None, description="Owning group ID for the game")
    config: Dict[str, Any] = Field(default_factory=dict, description="Raw configuration blob as stored in the database")
    players: List[PlayerResponse] = Field(default_factory=list)
    
    class Config:
        orm_mode = True

    class Config:
        from_attributes = True

class Game(GameInDBBase):
    pass

class GameInDB(GameInDBBase):
    """Game model with database-specific fields."""
    class Config:
        orm_mode = True

class PlayerBase(BaseModel):
    name: str = Field(..., max_length=100)
    role: PlayerRole
    is_ai: bool = False
    user_id: Optional[int] = None

class PlayerCreate(PlayerBase):
    pass

class PlayerUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    is_ai: Optional[bool] = None

class PlayerInDBBase(PlayerBase):
    id: int
    game_id: int

    class Config:
        from_attributes = True

class Player(PlayerInDBBase):
    pass

class PlayerRound(PlayerInDBBase):
    pass

class RoundBase(BaseModel):
    """Base model for a game round."""
    game_id: int
    round_number: int
    status: str = "pending"
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class RoundCreate(RoundBase):
    """Model for creating a new round."""
    pass

class RoundUpdate(BaseModel):
    """Model for updating a round."""
    status: Optional[str] = None
    completed_at: Optional[datetime] = None

class Round(RoundBase):
    """Complete round model with database-specific fields."""
    id: int

    class Config:
        orm_mode = True

class PlayerActionBase(BaseModel):
    """Base model for player actions."""
    game_id: int
    player_id: int
    round_id: int
    action_type: str
    quantity: int
    timestamp: datetime

class PlayerActionCreate(PlayerActionBase):
    """Model for creating a new player action."""
    pass

class PlayerActionUpdate(BaseModel):
    """Model for updating a player action."""
    action_type: Optional[str] = None
    quantity: Optional[int] = None

class PlayerAction(PlayerActionBase):
    """Complete player action model with database-specific fields."""
    id: int

    class Config:
        orm_mode = True

class PlayerState(BaseModel):
    id: int
    name: str
    role: PlayerRole
    is_ai: bool
    current_stock: int
    incoming_shipments: List[Dict[str, Any]] = []
    backorders: int = 0
    total_cost: float = 0.0

class GameState(GameInDBBase):
    """Extended game state with player states and current round information."""
    players: List[PlayerState] = Field(default_factory=list)
    current_demand: Optional[int] = Field(None, description="Current round's customer demand")
    round_started_at: Optional[datetime] = Field(None, description="When the current round started")
    round_ends_at: Optional[datetime] = Field(None, description="When the current round will end")
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "name": "Supply Chain Game",
                "status": "in_progress",
                "current_round": 5,
                "max_rounds": 20,
                "current_demand": 8,
                "players": [
                    {
                        "id": 1,
                        "name": "Retailer (AI)",
                        "role": "retailer",
                        "is_ai": True,
                        "current_stock": 12,
                        "incoming_shipments": [],
                        "backorders": 0,
                        "total_cost": 24.5
                    }
                ]
            }
        }

class OrderCreate(BaseModel):
    quantity: int = Field(..., ge=0, description="Number of units to order")
    comment: Optional[str] = Field(None, max_length=255, description="Reason for this order")

class OrderResponse(BaseModel):
    id: int
    game_id: int
    player_id: int
    round_number: int
    quantity: int
    created_at: datetime

class PlayerRoundBase(BaseModel):
    order_placed: int = Field(..., ge=0)
    order_received: int = Field(0, ge=0)
    inventory_before: int = Field(0, ge=0)
    inventory_after: int = Field(0, ge=0)
    backorders_before: int = Field(0, ge=0)
    backorders_after: int = Field(0, ge=0)
    holding_cost: float = Field(0.0, ge=0)
    backorder_cost: float = Field(0.0, ge=0)
    total_cost: float = Field(0.0, ge=0)
    comment: Optional[str] = Field(None, description="Player's comment for this round")

class PlayerRoundCreate(PlayerRoundBase):
    pass

class PlayerRound(PlayerRoundBase):
    id: int
    player_id: int
    round_id: int

    class Config:
        from_attributes = True

class GameRoundBase(BaseModel):
    round_number: int = Field(..., ge=1)
    customer_demand: int = Field(..., ge=0)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class GameRoundCreate(GameRoundBase):
    pass

class GameRound(GameRoundBase):
    id: int
    game_id: int
    created_at: datetime
    player_rounds: List[PlayerRound] = []

    class Config:
        from_attributes = True
