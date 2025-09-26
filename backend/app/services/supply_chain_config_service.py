"""
Service for managing supply chain configurations and their integration with game initialization.
"""
from typing import Dict, Any, List, Optional, Tuple
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

DEFAULT_ROLE_PRICING: Dict[str, Dict[str, float]] = {
    "retailer": {"selling_price": 100.0, "standard_cost": 80.0},
    "wholesaler": {"selling_price": 75.0, "standard_cost": 60.0},
    "distributor": {"selling_price": 60.0, "standard_cost": 45.0},
    "manufacturer": {"selling_price": 45.0, "standard_cost": 30.0},
}


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _range_bounds(range_value: Any) -> Tuple[Optional[float], Optional[float]]:
    if isinstance(range_value, dict):
        return _to_float(range_value.get("min")), _to_float(range_value.get("max"))
    value = _to_float(range_value)
    return value, value


def _range_midpoint(range_value: Any, fallback: float) -> float:
    low, high = _range_bounds(range_value)
    if low is None and high is None:
        return fallback
    if low is None:
        return high if high is not None else fallback
    if high is None:
        return low
    return (low + high) / 2.0


def _update_min(current: Optional[float], value: Optional[float]) -> Optional[float]:
    if value is None:
        return current
    if current is None or value < current:
        return value
    return current


def _update_max(current: Optional[float], value: Optional[float]) -> Optional[float]:
    if value is None:
        return current
    if current is None or value > current:
        return value
    return current


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
            NodeType.MANUFACTURER: "manufacturer",
        }

        lanes_by_downstream: Dict[int, List[Lane]] = {}
        for lane in lanes:
            lanes_by_downstream.setdefault(lane.downstream_node_id, []).append(lane)

        # Aggregate inventory and cost ranges
        min_init_inventory: Optional[float] = None
        max_init_inventory: Optional[float] = None
        min_holding_cost: Optional[float] = None
        max_holding_cost: Optional[float] = None
        min_backlog_cost: Optional[float] = None
        max_backlog_cost: Optional[float] = None
        init_midpoints: List[float] = []
        holding_midpoints: List[float] = []
        backlog_midpoints: List[float] = []

        for inc in item_node_configs:
            init_lo, init_hi = _range_bounds(inc.initial_inventory_range)
            min_init_inventory = _update_min(min_init_inventory, init_lo)
            max_init_inventory = _update_max(max_init_inventory, init_hi)
            init_midpoints.append(_range_midpoint(inc.initial_inventory_range, 12.0))

            hold_lo, hold_hi = _range_bounds(inc.holding_cost_range)
            min_holding_cost = _update_min(min_holding_cost, hold_lo)
            max_holding_cost = _update_max(max_holding_cost, hold_hi)
            holding_midpoints.append(_range_midpoint(inc.holding_cost_range, 0.5))

            back_lo, back_hi = _range_bounds(inc.backlog_cost_range)
            min_backlog_cost = _update_min(min_backlog_cost, back_lo)
            max_backlog_cost = _update_max(max_backlog_cost, back_hi)
            backlog_midpoints.append(_range_midpoint(inc.backlog_cost_range, 5.0))

        default_init_inventory = (
            sum(init_midpoints) / len(init_midpoints)
            if init_midpoints
            else 12.0
        )
        default_holding_cost = (
            sum(holding_midpoints) / len(holding_midpoints)
            if holding_midpoints
            else 0.5
        )
        default_backlog_cost = (
            sum(backlog_midpoints) / len(backlog_midpoints)
            if backlog_midpoints
            else 5.0
        )

        min_lead_time: Optional[float] = None
        max_lead_time: Optional[float] = None
        max_lane_capacity: Optional[float] = None
        lead_midpoints: List[float] = []
        for lane in lanes:
            capacity_val = _to_float(lane.capacity)
            max_lane_capacity = _update_max(max_lane_capacity, capacity_val)

            lead_lo, lead_hi = _range_bounds(lane.lead_time_days)
            min_lead_time = _update_min(min_lead_time, lead_lo)
            max_lead_time = _update_max(max_lead_time, lead_hi)
            lead_midpoints.append(_range_midpoint(lane.lead_time_days, 1.0))

        average_ship_delay = (
            sum(lead_midpoints) / len(lead_midpoints)
            if lead_midpoints
            else 1.0
        )

        if min_lead_time is None:
            min_lead_time = 1.0
        if max_lead_time is None:
            max_lead_time = max(min_lead_time, 1.0)

        shipping_lead_time = int(round(min_lead_time)) if min_lead_time is not None else 1
        shipping_lead_time = max(shipping_lead_time, 1)

        min_demand: Optional[float] = None
        max_demand: Optional[float] = None

        def record_demand(value: Any) -> None:
            nonlocal min_demand, max_demand
            numeric = _to_float(value)
            if numeric is None:
                return
            min_demand = _update_min(min_demand, numeric)
            max_demand = _update_max(max_demand, numeric)

        for md in market_demands:
            payload = {}
            if isinstance(md.demand_pattern, dict):
                payload = md.demand_pattern.get('params', {}) or {}
                for key in ("initial_demand", "final_demand", "mean", "min", "max", "demand"):
                    record_demand(payload.get(key))
            else:
                record_demand(md.demand_pattern)

        node_policies: Dict[str, Dict[str, Any]] = {}
        node_types: Dict[str, str] = {}
        pricing_config: Dict[str, Dict[str, float]] = {}

        for node in nodes:
            node_item_configs = [inc for inc in item_node_configs if inc.node_id == node.id]
            node_key = node.name.lower()
            node_type_value = (
                node.type.value.lower()
                if hasattr(node.type, "value")
                else str(node.type).lower()
            )
            node_types[node_key] = node_type_value

            role = role_mapping.get(node.type)
            defaults = DEFAULT_ROLE_PRICING.get(role, {"selling_price": 0.0, "standard_cost": 0.0})
            selling_price = defaults["selling_price"]
            if node_item_configs:
                selling_price = _range_midpoint(node_item_configs[0].selling_price_range, selling_price)
            standard_cost = defaults["standard_cost"]
            if selling_price and standard_cost == 0.0:
                standard_cost = max(selling_price * 0.8, 0.0)

            inbound_lanes = lanes_by_downstream.get(node.id, [])
            if inbound_lanes:
                inbound_midpoints = [
                    _range_midpoint(lane.lead_time_days, average_ship_delay)
                    for lane in inbound_lanes
                ]
                ship_delay = sum(inbound_midpoints) / len(inbound_midpoints)
            else:
                ship_delay = average_ship_delay

            init_inventory_value = default_init_inventory
            if node_item_configs:
                init_inventory_value = _range_midpoint(
                    node_item_configs[0].initial_inventory_range,
                    default_init_inventory,
                )

            is_market_node = node.type in {NodeType.MARKET_SUPPLY, NodeType.MARKET_DEMAND}

            node_policy = {
                "info_delay": 0 if is_market_node else 1,
                "ship_delay": 0 if is_market_node else int(round(ship_delay)),
                "init_inventory": 0 if is_market_node else int(round(init_inventory_value)),
                "min_order_qty": 0 if is_market_node else 0,
                "variable_cost": 0.0,
                "price": 0.0 if is_market_node else round(float(selling_price or 0.0), 2),
                "standard_cost": 0.0 if is_market_node else round(float(standard_cost or 0.0), 2),
            }
            node_policies[node_key] = node_policy

            if role:
                role_key = str(role).lower()
                if role_key not in node_policies:
                    node_policies[role_key] = dict(node_policy)

            if role and role not in pricing_config:
                pricing_config[role] = {
                    "selling_price": round(float(selling_price or 0.0), 2),
                    "standard_cost": round(float(standard_cost or 0.0), 2),
                }

        for role, defaults in DEFAULT_ROLE_PRICING.items():
            pricing_config.setdefault(role, {
                "selling_price": defaults["selling_price"],
                "standard_cost": defaults["standard_cost"],
            })

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
            explicit_market = [
                name
                for name, ntype in node_types.items()
                if ntype == NodeType.MARKET_DEMAND.value.lower() or ntype == "market_demand"
            ]
            if explicit_market:
                market_nodes = sorted(explicit_market)
            else:
                upstream_names = {entry["from"] for entry in lane_payload if entry.get("from")}
                downstream_only = {
                    entry["to"]
                    for entry in lane_payload
                    if entry.get("to") and entry["to"] not in upstream_names
                }
                if downstream_only:
                    market_nodes = sorted(downstream_only)
                else:
                    fallback_retailers = [
                        name
                        for name, ntype in node_types.items()
                        if ntype == NodeType.RETAILER.value.lower() or ntype == "retailer"
                    ]
                    if fallback_retailers:
                        market_nodes = fallback_retailers
                    elif node_policies:
                        market_nodes = [next(iter(node_policies.keys()))]

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
            record_demand(demand_pattern['params'].get('initial_demand'))
            record_demand(demand_pattern['params'].get('final_demand'))

        if min_demand is None:
            min_demand = _to_float(demand_pattern['params'].get('initial_demand')) or 0.0
        if max_demand is None:
            max_demand = _to_float(demand_pattern['params'].get('final_demand')) or min_demand or 0.0

        max_order_candidates = [100.0]
        if max_init_inventory is not None:
            max_order_candidates.append(max_init_inventory)
        if max_lane_capacity is not None:
            max_order_candidates.append(max_lane_capacity)
        max_order_quantity = int(round(max(max_order_candidates))) if max_order_candidates else 100

        order_lead_range = {"min": 0, "max": 3}
        supply_lead_range = {
            "min": int(round(min_lead_time)) if min_lead_time is not None else 1,
            "max": int(round(max_lead_time)) if max_lead_time is not None else shipping_lead_time,
        }

        system_config = {
            "min_order_quantity": 0,
            "max_order_quantity": max_order_quantity,
            "min_holding_cost": round(float(min_holding_cost) if min_holding_cost is not None else 0.0, 2),
            "max_holding_cost": round(float(max_holding_cost) if max_holding_cost is not None else 10.0, 2),
            "min_backlog_cost": round(float(min_backlog_cost) if min_backlog_cost is not None else 0.0, 2),
            "max_backlog_cost": round(float(max_backlog_cost) if max_backlog_cost is not None else 20.0, 2),
            "min_demand": int(round(min_demand)) if min_demand is not None else 0,
            "max_demand": int(round(max_demand)) if max_demand is not None else 100,
            "min_lead_time": supply_lead_range["min"],
            "max_lead_time": supply_lead_range["max"],
            "min_starting_inventory": int(round(min_init_inventory)) if min_init_inventory is not None else 0,
            "max_starting_inventory": int(round(max_init_inventory)) if max_init_inventory is not None else max_order_quantity,
        }

        system_config["order_leadtime"] = dict(order_lead_range)
        system_config["supply_leadtime"] = dict(supply_lead_range)
        system_config["ship_order_leadtimedelay"] = dict(supply_lead_range)

        simulation_parameters = {
            "weeks": 40,
            "order_lead_time": 0,
            "shipping_lead_time": shipping_lead_time,
            "production_lead_time": 2,
            "initial_inventory": int(round(default_init_inventory)),
            "holding_cost_per_unit": round(default_holding_cost, 2),
            "backorder_cost_per_unit": round(default_backlog_cost, 2),
            "initial_demand": demand_pattern['params'].get('initial_demand', 4),
            "demand_change_week": demand_pattern['params'].get('change_week', 15),
            "new_demand": demand_pattern['params'].get('final_demand', 12),
            "historical_weeks": 30,
            "volatility_window": 14,
            "enable_information_sharing": True,
            "enable_demand_volatility_signals": True,
            "enable_pipeline_signals": True,
            "enable_downstream_visibility": True,
        }

        global_policy = {
            "info_delay": simulation_parameters["order_lead_time"],
            "ship_delay": simulation_parameters["shipping_lead_time"],
            "init_inventory": simulation_parameters["initial_inventory"],
            "holding_cost": simulation_parameters["holding_cost_per_unit"],
            "backlog_cost": simulation_parameters["backorder_cost_per_unit"],
            "max_inbound_per_link": int(round(max_lane_capacity)) if max_lane_capacity is not None else 100,
            "max_order": system_config["max_order_quantity"],
            "production_lead_time": simulation_parameters["production_lead_time"],
        }

        game_config = {
            "name": game_data.get('name', f"Game - {config.name}"),
            "description": game_data.get('description', config.description or ""),
            "max_rounds": game_data.get('max_rounds', 40),
            "is_public": game_data.get('is_public', True),
            "node_policies": node_policies,
            "demand_pattern": demand_pattern,
            "supply_chain_config_id": config_id,
            "pricing_config": pricing_config,
            "system_config": system_config,
            "global_policy": global_policy,
            "simulation_parameters": simulation_parameters,
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
            "progression_mode": game_data.get(
                'progression_mode',
                'unsupervised' if game_data.get('name', '').lower().startswith('daybreak') else 'supervised',
            ),
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
