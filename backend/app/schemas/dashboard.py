from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class TimeSeriesPoint(BaseModel):
    """Represents a single data point in the time series."""
    week: int = Field(..., description="The week number in the game")
    inventory: float = Field(0, description="Current inventory level")
    order: float = Field(0, description="Current order quantity")
    cost: float = Field(0, description="Accumulated cost for this week")
    backlog: float = Field(0, description="Current backlog amount")
    demand: Optional[float] = Field(None, description="Demand for this week (if applicable)")
    supply: Optional[float] = Field(None, description="Supply for this week (if applicable)")

class PlayerMetrics(BaseModel):
    """Key performance metrics for a player."""
    current_inventory: float = Field(..., description="Current inventory level")
    inventory_change: float = Field(0, description="Percentage change in inventory from last week")
    backlog: float = Field(0, description="Current backlog amount")
    total_cost: float = Field(0, description="Total accumulated cost")
    avg_weekly_cost: float = Field(0, description="Average cost per week")
    service_level: float = Field(1.0, description="Current service level (0-1)")
    service_level_change: float = Field(0, description="Change in service level from last week")

class DashboardResponse(BaseModel):
    """Dashboard data response model."""
    game_name: str = Field(..., description="Name of the current game")
    current_round: int = Field(..., description="Current round number in the game")
    player_role: str = Field(..., description="Player's role in the game")
    metrics: PlayerMetrics = Field(..., description="Player performance metrics")
    time_series: List[TimeSeriesPoint] = Field(..., description="Time series data for the player")
    last_updated: str = Field(..., description="ISO timestamp of when the data was last updated")
