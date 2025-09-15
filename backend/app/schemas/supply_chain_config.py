from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime

class NodeType(str, Enum):
    RETAILER = "retailer"
    DISTRIBUTOR = "distributor"
    MANUFACTURER = "manufacturer"
    SUPPLIER = "supplier"

class RangeConfig(BaseModel):
    min: float = Field(..., ge=0, description="Minimum value")
    max: float = Field(..., gt=0, description="Maximum value")
    
    @validator('max')
    def max_greater_than_min(cls, v, values):
        if 'min' in values and v <= values['min']:
            raise ValueError('max must be greater than min')
        return v

class DemandPattern(BaseModel):
    type: str = Field(..., description="Type of demand pattern (normal, uniform, seasonal, etc.)")
    mean: Optional[float] = Field(None, description="Mean for normal distribution")
    stddev: Optional[float] = Field(None, ge=0, description="Standard deviation for normal distribution")
    min: Optional[float] = Field(None, ge=0, description="Minimum for uniform distribution")
    max: Optional[float] = Field(None, gt=0, description="Maximum for uniform distribution")
    seasonality: Optional[Dict[str, Any]] = Field(None, description="Seasonality configuration")
    
    @validator('type')
    def validate_type(cls, v):
        valid_types = ["normal", "uniform", "seasonal", "constant"]
        if v not in valid_types:
            raise ValueError(f"Invalid demand pattern type. Must be one of {valid_types}")
        return v

# Base schemas
class SupplyChainConfigBase(BaseModel):
    name: str = Field(..., max_length=100, description="Name of the configuration")
    description: Optional[str] = Field(None, max_length=500, description="Description of the configuration")
    is_active: bool = Field(False, description="Whether this is the active configuration")
    group_id: Optional[int] = Field(
        None,
        description="ID of the group that owns this configuration"
    )

class ItemBase(BaseModel):
    name: str = Field(..., max_length=100, description="Name of the item")
    description: Optional[str] = Field(None, max_length=500, description="Description of the item")
    unit_cost_range: RangeConfig = Field(..., description="Range of unit costs for training")

class NodeBase(BaseModel):
    name: str = Field(..., max_length=100, description="Name of the node")
    type: NodeType = Field(..., description="Type of the node")

class LaneBase(BaseModel):
    upstream_node_id: int = Field(..., description="ID of the upstream node")
    downstream_node_id: int = Field(..., description="ID of the downstream node")
    capacity: int = Field(..., gt=0, description="Capacity in units per day")
    lead_time_days: RangeConfig = Field(..., description="Range of lead times in days")
    
    @validator('downstream_node_id')
    def nodes_must_differ(cls, v, values):
        if 'upstream_node_id' in values and v == values['upstream_node_id']:
            raise ValueError('Upstream and downstream nodes must be different')
        return v

class ItemNodeConfigBase(BaseModel):
    item_id: int = Field(..., description="ID of the item")
    node_id: int = Field(..., description="ID of the node")
    inventory_target_range: RangeConfig = Field(..., description="Range of inventory targets")
    initial_inventory_range: RangeConfig = Field(..., description="Range of initial inventory levels")
    holding_cost_range: RangeConfig = Field(..., description="Range of holding costs")
    backlog_cost_range: RangeConfig = Field(..., description="Range of backlog costs")
    selling_price_range: RangeConfig = Field(..., description="Range of selling prices")

class MarketDemandBase(BaseModel):
    item_id: int = Field(..., description="ID of the item")
    retailer_id: int = Field(..., description="ID of the retailer node")
    demand_pattern: DemandPattern = Field(..., description="Demand pattern configuration")

# Create schemas
class SupplyChainConfigCreate(SupplyChainConfigBase):
    pass

class ItemCreate(ItemBase):
    pass

class NodeCreate(NodeBase):
    pass

class LaneCreate(LaneBase):
    pass

class ItemNodeConfigCreate(ItemNodeConfigBase):
    pass

class MarketDemandCreate(MarketDemandBase):
    pass

# Update schemas
class SupplyChainConfigUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100, description="Name of the configuration")
    description: Optional[str] = Field(None, max_length=500, description="Description of the configuration")
    is_active: Optional[bool] = Field(None, description="Whether this is the active configuration")
    group_id: Optional[int] = Field(None, description="ID of the group that owns this configuration")

class ItemUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100, description="Name of the item")
    description: Optional[str] = Field(None, max_length=500, description="Description of the item")
    unit_cost_range: Optional[RangeConfig] = Field(None, description="Range of unit costs for training")

class NodeUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100, description="Name of the node")
    type: Optional[NodeType] = Field(None, description="Type of the node")

class LaneUpdate(BaseModel):
    capacity: Optional[int] = Field(None, gt=0, description="Capacity in units per day")
    lead_time_days: Optional[RangeConfig] = Field(None, description="Range of lead times in days")

class ItemNodeConfigUpdate(BaseModel):
    inventory_target_range: Optional[RangeConfig] = Field(None, description="Range of inventory targets")
    initial_inventory_range: Optional[RangeConfig] = Field(None, description="Range of initial inventory levels")
    holding_cost_range: Optional[RangeConfig] = Field(None, description="Range of holding costs")
    backlog_cost_range: Optional[RangeConfig] = Field(None, description="Range of backlog costs")
    selling_price_range: Optional[RangeConfig] = Field(None, description="Range of selling prices")

class MarketDemandUpdate(BaseModel):
    demand_pattern: Optional[DemandPattern] = Field(None, description="Demand pattern configuration")

# Response schemas
class ItemNodeConfig(ItemNodeConfigBase):
    id: int
    
    class Config:
        orm_mode = True

class MarketDemand(MarketDemandBase):
    id: int
    
    class Config:
        orm_mode = True

class Lane(LaneBase):
    id: int
    upstream_node: 'Node'
    downstream_node: 'Node'
    
    class Config:
        orm_mode = True

class Node(NodeBase):
    id: int
    item_configs: List[ItemNodeConfig] = []
    upstream_lanes: List[Lane] = []
    downstream_lanes: List[Lane] = []
    
    class Config:
        orm_mode = True

class Item(ItemBase):
    id: int
    node_configs: List[ItemNodeConfig] = []
    
    class Config:
        orm_mode = True

class SupplyChainConfig(SupplyChainConfigBase):
    id: int
    created_at: datetime
    updated_at: datetime
    items: List[Item] = []
    nodes: List[Node] = []
    lanes: List[Lane] = []
    market_demands: List[MarketDemand] = []
    
    class Config:
        orm_mode = True

# Update forward refs for proper type hints
Node.update_forward_refs()
Lane.update_forward_refs()
