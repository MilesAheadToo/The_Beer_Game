from typing import List, Optional, Dict, Any, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, validator
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
                    "stable_period": 5,
                    "step_increase": 4
                }
            }
        }
    )

class GameStatus(str, Enum):
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"

class PlayerRole(str, Enum):
    RETAILER = "retailer"
    WHOLESALER = "wholesaler"
    DISTRIBUTOR = "distributor"
    FACTORY = "factory"

class GameBase(BaseModel):
    name: str = Field(..., max_length=100)
    max_rounds: int = Field(default=52, ge=1, le=1000)
    description: Optional[str] = Field(None, max_length=500)
    is_public: bool = Field(default=True, description="Whether the game is visible to all users")

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
            params={"stable_period": 5, "step_increase": 4}
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
                params={"stable_period": 5, "step_increase": 4}
            ),
            description="Configuration for the demand pattern used in the game"
        )
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[int] = Field(None, description="User ID of the game creator")
    players: List[PlayerResponse] = Field(default_factory=list)
    
    class Config:
        orm_mode = True

    class Config:
        from_attributes = True

class Game(GameInDBBase):
    pass

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
