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
            NodeType.WHOLESALER: "wholesaler",
            NodeType.DISTRIBUTOR: "distributor",
            NodeType.MANUFACTURER: "manufacturer"
        }
        
        # Create node policies
        node_policies: Dict[str, Dict[str, Any]] = {}
        node_types: Dict[str, str] = {}
        for node in nodes:
            # Find item configs for this node
            node_item_configs = [inc for inc in item_node_configs if inc.node_id == node.id]
            
            # Use first item's config or defaults
            if node_item_configs:
                node_item_config = node_item_configs[0]
                init_inventory = node_item_config.initial_inventory_range.get('min', 12)
                price = node_item_config.selling_price_range.get('min', 10.0)
                standard_cost = node_item_config.holding_cost_range.get('min', 2.0)
            else:
                init_inventory = 12
                price = 10.0
                standard_cost = 2.0
            
            node_key = node.name.lower()
            node_policy = {
                "info_delay": 1,
                "ship_delay": 1,
                "init_inventory": init_inventory,
                "price": price,
                "standard_cost": standard_cost,
                "variable_cost": 0.1,
                "min_order_qty": 1,
            }
            if node.type in {NodeType.MARKET_SUPPLY, NodeType.MARKET_DEMAND}:
                node_policy.update({
                    "info_delay": 0,
                    "ship_delay": 0,
                    "init_inventory": 0,
                    "price": 0.0,
                    "standard_cost": 0.0,
                    "variable_cost": 0.0,
                    "min_order_qty": 0,
                })
            node_policies[node_key] = node_policy
            node_types[node_key] = (
                node.type.value.lower()
                if hasattr(node.type, "value")
                else str(node.type).lower()
            )

        lane_payload: List[Dict[str, Any]] = []
        for lane in lanes:
            upstream = lane.upstream_node.name.lower() if lane.upstream_node else None
            downstream = lane.downstream_node.name.lower() if lane.downstream_node else None
            if not upstream or not downstream:
                continue
            lane_payload.append(
                {
                    "from": upstream,
                    "to": downstream,
                    "capacity": lane.capacity,
                    "lead_time_days": lane.lead_time_days,
                }
            )

        market_nodes: List[str] = []
        if market_demands:
            for md in market_demands:
                retailer = md.retailer.name.lower() if md.retailer else None
                if retailer and retailer not in market_nodes:
                    market_nodes.append(retailer)

        if not market_nodes:
            explicit_market = [name for name, ntype in node_types.items() if ntype == NodeType.MARKET_DEMAND.value.lower() or ntype == "market_demand"]
            if explicit_market:
                market_nodes = sorted(explicit_market)
            else:
                upstream_names = {entry["from"] for entry in lane_payload if entry.get("from")}
                downstream_only = {entry["to"] for entry in lane_payload if entry.get("to") and entry["to"] not in upstream_names}
                if downstream_only:
                    market_nodes = sorted(downstream_only)
                else:
                    fallback_retailers = [name for name, ntype in node_types.items() if ntype == NodeType.RETAILER.value.lower() or ntype == "retailer"]
                    if fallback_retailers:
                        market_nodes = fallback_retailers
                    elif node_policies:
                        market_nodes = [next(iter(node_policies.keys()))]

        # Create demand pattern from market demands
        # For now, use a simple constant demand based on the first market demand found
        demand_pattern = {
            "type": "classic",
            "params": {
                "initial_demand": 4,
                "change_week": 15,
                "final_demand": 12,
            },
        }

        if market_demands:
            md = market_demands[0]
            params = md.demand_pattern.get('params', {}) if isinstance(md.demand_pattern, dict) else {}
            demand_pattern = {
                "type": md.demand_pattern.get('type', 'classic') if isinstance(md.demand_pattern, dict) else 'classic',
                "params": {
                    "initial_demand": params.get('initial_demand', 4),
                    "change_week": params.get('change_week', 15),
                    "final_demand": params.get('final_demand', params.get('new_demand', 12)),
                },
            }

        # Create the game configuration
        game_config = {
            "name": game_data.get('name', f"Game - {config.name}"),
            "description": game_data.get('description', config.description or ""),
            "max_rounds": game_data.get('max_rounds', 40),
            "is_public": game_data.get('is_public', True),
            "node_policies": node_policies,
            "demand_pattern": demand_pattern,
            "supply_chain_config_id": config_id,
            "simulation_parameters": {
                "weeks": 40,
                "order_lead_time": 1,
                "shipping_lead_time": 1,
                "production_lead_time": 2,
                "initial_inventory": init_inventory,
                "holding_cost_per_unit": 0.5,
                "backorder_cost_per_unit": 5.0,
                "initial_demand": demand_pattern['params'].get('initial_demand', 4),
                "demand_change_week": demand_pattern['params'].get('change_week', 15),
                "new_demand": demand_pattern['params'].get('final_demand', 12),
                "historical_weeks": 30,
                "volatility_window": 14,
                "enable_information_sharing": True,
                "enable_demand_volatility_signals": True,
                "enable_pipeline_signals": True,
                "enable_downstream_visibility": True,
            },
            "info_sharing": {
                "enabled": True,
                "historical_weeks": 30,
            },
            "demand_volatility": {
                "enabled": True,
                "window": 14,
            },
            "pipeline_signals": {
                "enabled": True,
            },
            "downstream_visibility": {
                "enabled": True,
            },
            "global_policy": {
                "production_lead_time": 2,
            },
            "progression_mode": game_data.get('progression_mode', 'unsupervised' if game_data.get('name', '').lower().startswith('daybreak') else 'supervised'),
            "enable_information_sharing": True,
            "enable_demand_volatility_signals": True,
            "enable_pipeline_signals": True,
            "enable_downstream_visibility": True,
            "historical_weeks_to_share": 30,
            "volatility_analysis_window": 14,
            "lanes": lane_payload,
            "node_types": node_types,
            "market_demand_nodes": market_nodes,
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
