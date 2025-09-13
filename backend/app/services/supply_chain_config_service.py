"""
Service for managing supply chain configurations and their integration with game initialization.
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from datetime import datetime
import json

from app.models.supply_chain_config import (
    SupplyChainConfig, Item, Node, Lane, ItemNodeConfig, MarketDemand, NodeType
)
from app.schemas.game import GameCreate, NodePolicy, DemandPattern
from app.schemas.supply_chain_config import (
    SupplyChainConfigCreate, 
    ItemCreate, 
    NodeCreate, 
    LaneCreate, 
    ItemNodeConfigCreate,
    MarketDemandCreate
)

class SupplyChainConfigService:
    """Service for managing supply chain configurations and game integration."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_game_from_config(self, config_id: int, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a game configuration based on a supply chain configuration.
        
        Args:
            config_id: ID of the supply chain configuration
            game_data: Base game data (name, description, etc.)
            
        Returns:
            Dict containing the game configuration
        """
        # Get the supply chain configuration
        config = self.db.query(SupplyChainConfig).filter(
            SupplyChainConfig.id == config_id
        ).first()
        
        if not config:
            raise ValueError(f"Supply chain configuration with ID {config_id} not found")
        
        # Get all related data
        items = self.db.query(Item).filter(Item.config_id == config_id).all()
        nodes = self.db.query(Node).filter(Node.config_id == config_id).all()
        lanes = self.db.query(Lane).filter(Lane.config_id == config_id).all()
        item_node_configs = self.db.query(ItemNodeConfig).filter(
            ItemNodeConfig.node_id.in_([n.id for n in nodes])
        ).all()
        market_demands = self.db.query(MarketDemand).filter(
            MarketDemand.config_id == config_id
        ).all()
        
        # Map node types to player roles
        role_mapping = {
            NodeType.RETAILER: "retailer",
            NodeType.DISTRIBUTOR: "distributor",
            NodeType.MANUFACTURER: "manufacturer",
            NodeType.SUPPLIER: "supplier"
        }
        
        # Create node policies
        node_policies = {}
        for node in nodes:
            # Find item configs for this node
            node_item_configs = [inc for inc in item_node_configs if inc.node_id == node.id]
            
            # Use first item's config or defaults
            if node_item_configs:
                config = node_item_configs[0]
                init_inventory = config.initial_inventory_range.get('min', 12)
                price = config.selling_price_range.get('min', 10.0)
                standard_cost = config.holding_cost_range.get('min', 2.0)
            else:
                init_inventory = 12
                price = 10.0
                standard_cost = 2.0
            
            node_policies[node.name.lower()] = {
                "info_delay": 2,  # Default value
                "ship_delay": 1,   # Default value
                "init_inventory": init_inventory,
                "price": price,
                "standard_cost": standard_cost,
                "variable_cost": 0.1,  # Default value
                "min_order_qty": 1      # Default value
            }
        
        # Create demand pattern from market demands
        # For now, use a simple constant demand based on the first market demand found
        demand_pattern = {
            "type": "constant",
            "params": {"value": 4}  # Default value
        }
        
        if market_demands:
            # Use the first market demand's configuration
            md = market_demands[0]
            demand_pattern = {
                "type": md.demand_pattern.get('type', 'constant'),
                "params": md.demand_pattern.get('params', {'value': 4})
            }
        
        # Create the game configuration
        game_config = {
            "name": game_data.get('name', f"Game - {config.name}"),
            "description": game_data.get('description', config.description or ""),
            "max_rounds": game_data.get('max_rounds', 52),
            "is_public": game_data.get('is_public', True),
            "node_policies": node_policies,
            "demand_pattern": demand_pattern,
            "supply_chain_config_id": config_id  # Store reference to the config
        }
        
        return game_config
    
    def create_config_from_game(self, game_id: int, config_name: str) -> SupplyChainConfig:
        """
        Create a supply chain configuration from an existing game.
        
        Args:
            game_id: ID of the game to create a configuration from
            config_name: Name for the new configuration
            
        Returns:
            The created SupplyChainConfig object
        """
        # This would query the game and create a configuration from its settings
        # Implementation would be similar to the reverse of create_game_from_config
        raise NotImplementedError("This feature is not yet implemented")
