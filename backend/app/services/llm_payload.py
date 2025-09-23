"""Utilities for constructing structured payloads for the Beer Game Daybreak LLM agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models.game import Game
from app.models.player import Player
from app.models.supply_chain import PlayerInventory, PlayerRound, GameRound

# Mapping between backend role identifiers and the labels expected by the Daybreak LLM
ROLE_NAME_MAP = {
    "manufacturer": "factory",
}


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}


def _normalize_pipeline(raw: Any) -> List[int]:
    if not raw:
        return []
    if isinstance(raw, list):
        normalized: List[int] = []
        for item in raw:
            if isinstance(item, (int, float)):
                normalized.append(int(round(item)))
            elif isinstance(item, dict):
                qty = (
                    item.get("quantity")
                    or item.get("qty")
                    or item.get("amount")
                    or item.get("value")
                )
                if qty is not None:
                    try:
                        normalized.append(int(round(float(qty))))
                    except (TypeError, ValueError):
                        continue
        return normalized
    return []


def build_llm_decision_payload(
    db: Session,
    game: Game,
    *,
    round_number: int,
    action_role: str,
    history_window: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble the structured JSON payload expected by the Daybreak LLM agent."""

    config_raw = _coerce_dict(getattr(game, "config", {}))
    sim_params = _coerce_dict(config_raw.get("simulation_parameters", {}))

    demand_pattern_raw = game.demand_pattern or config_raw.get("demand_pattern", {})
    demand_pattern = _coerce_dict(demand_pattern_raw)
    demand_params = _coerce_dict(demand_pattern.get("params", {}))

    total_weeks = int(game.max_rounds or sim_params.get("weeks") or 40)
    order_lead = int(sim_params.get("order_lead_time", sim_params.get("info_delay", 2) or 2))
    ship_lead = int(sim_params.get("shipping_lead_time", sim_params.get("ship_delay", 2) or 2))
    prod_lead = int(sim_params.get("production_lead_time", sim_params.get("prod_delay", 4) or 4))

    holding_cost = float(sim_params.get("holding_cost_per_unit", sim_params.get("holding_cost", 0.5) or 0.5))
    backorder_cost = float(sim_params.get("backorder_cost_per_unit", sim_params.get("backorder_cost", 0.5) or 0.5))

    toggles = {
        "share_demand_history": bool(
            config_raw.get("enable_information_sharing")
            or sim_params.get("enable_information_sharing")
            or config_raw.get("historical_weeks_to_share")
        ),
        "share_volatility_signal": bool(
            config_raw.get("enable_demand_volatility_signals")
            or sim_params.get("enable_demand_volatility_signals")
            or config_raw.get("volatility_analysis_window")
        ),
        "share_downstream_inventory": bool(
            config_raw.get("enable_downstream_visibility")
            or sim_params.get("enable_downstream_visibility")
            or config_raw.get("pipeline_signals", {}).get("enabled")
        ),
    }

    visible_history_weeks = int(
        history_window
        or sim_params.get("historical_weeks")
        or config_raw.get("historical_weeks_to_share")
        or 30
    )

    volatility_window = int(
        sim_params.get("volatility_window")
        or config_raw.get("volatility_analysis_window")
        or 14
    )

    config_section = {
        "total_weeks": total_weeks,
        "lead_times": {
            "order": order_lead,
            "shipping": ship_lead,
            "production": prod_lead,
        },
        "costs": {
            "holding_cost_per_unit": holding_cost,
            "backorder_cost_per_unit": backorder_cost,
        },
        "initial_inventory": int(sim_params.get("initial_inventory", 12)),
        "demand_pattern": {
            "type": demand_pattern.get("type", "classic"),
            "change_week": int(demand_params.get("change_week", demand_params.get("demand_change_week", 20))),
            "initial_demand": int(demand_params.get("initial_demand", 4)),
            "new_demand": int(
                demand_params.get("new_demand", demand_params.get("final_demand", demand_params.get("target_demand", 4)))
            ),
        },
        "toggles": toggles,
        "visible_history_weeks": visible_history_weeks,
        "volatility_window": volatility_window,
    }

    engine_state = _coerce_dict(config_raw.get("engine_state", {}))

    player_round_rows = (
        db.query(PlayerRound, Player, GameRound)
        .join(Player, PlayerRound.player_id == Player.id)
        .join(GameRound, PlayerRound.round_id == GameRound.id)
        .filter(Player.game_id == game.id)
        .order_by(GameRound.round_number.asc())
        .all()
    )

    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    history_by_role: Dict[str, List[Dict[str, Any]]] = {}
    orders_by_role_round: Dict[str, Dict[int, int]] = {}

    for round_rec, player_obj, game_round in player_round_rows:
        role_name = str(player_obj.role.value if hasattr(player_obj.role, "value") else player_obj.role).lower()
        round_number = _safe_int(getattr(game_round, "round_number", 0))

        order_up = _safe_int(
            getattr(round_rec, "order_placed", getattr(round_rec, "order_quantity", 0))
        )
        inventory_before = _safe_int(
            getattr(round_rec, "inventory_before", getattr(round_rec, "inventory", 0))
        )
        backlog_before = _safe_int(
            getattr(round_rec, "backorders_before", getattr(round_rec, "backlog", 0))
        )

        orders_by_role_round.setdefault(role_name, {})[round_number] = order_up

        entry = {
            "round": round_number,
            "order_up": order_up,
            "inventory_before": inventory_before,
            "backlog_before": backlog_before,
            "customer_demand": None,
        }

        if role_name == "retailer":
            demand_value = _safe_int(getattr(game_round, "customer_demand", 0))
            entry["customer_demand"] = demand_value

        history_by_role.setdefault(role_name, []).append(entry)

    players_with_inventory = (
        db.query(Player, PlayerInventory)
        .outerjoin(PlayerInventory, PlayerInventory.player_id == Player.id)
        .filter(Player.game_id == game.id)
        .all()
    )

    downstream_role_map = {
        "retailer": None,
        "wholesaler": "retailer",
        "distributor": "wholesaler",
        "manufacturer": "distributor",
    }

    roles_section: Dict[str, Dict[str, Any]] = {}
    for player, inventory in players_with_inventory:
        if not player or not getattr(player, "role", None):
            continue

        role_name_raw = str(player.role.value if hasattr(player.role, "value") else player.role).lower()
        role_key = ROLE_NAME_MAP.get(role_name_raw, role_name_raw)

        inventory_obj = inventory
        current_stock = 0
        current_backlog = 0
        incoming_shipments_raw: Any = []
        if inventory_obj is not None:
            current_stock = int(
                getattr(inventory_obj, "current_stock", getattr(inventory_obj, "current_inventory", 0)) or 0
            )
            current_backlog = int(
                getattr(inventory_obj, "backorders", getattr(inventory_obj, "current_backlog", 0)) or 0
            )
            incoming_shipments_raw = getattr(inventory_obj, "incoming_shipments", [])

        pipeline = _normalize_pipeline(incoming_shipments_raw)

        incoming_order = 0
        engine_entry = engine_state.get(role_name_raw, {})
        if isinstance(engine_entry, dict):
            try:
                incoming_order = int(engine_entry.get("incoming_orders", 0))
            except (TypeError, ValueError):
                incoming_order = 0

        role_records = history_by_role.get(role_name_raw, [])
        if visible_history_weeks and visible_history_weeks > 0:
            role_records = role_records[-visible_history_weeks:]

        orders_history: List[int] = []
        shipments_history: List[int] = []
        demand_history: List[int] = []

        downstream_role = downstream_role_map.get(role_name_raw)
        downstream_orders_map = orders_by_role_round.get(downstream_role, {}) if downstream_role else None

        for record in role_records:
            round_id = record["round"]
            order_up_value = record["order_up"]
            on_hand_value = record["inventory_before"]
            backlog_value = record.get("backlog_before", 0)

            if role_name_raw == "retailer":
                order_qty = record["customer_demand"] or 0
                demand_history.append(order_qty)
            elif downstream_orders_map is not None:
                order_qty = downstream_orders_map.get(round_id, 0)
            else:
                order_qty = 0

            total_demand = order_qty + backlog_value
            shipped_qty = min(on_hand_value, total_demand)

            orders_history.append(order_up_value)
            shipments_history.append(_safe_int(shipped_qty))

        history_section: Dict[str, Any] = {
            "shipments_sent": shipments_history,
        }
        if role_name_raw == "manufacturer":
            history_section["production_orders"] = orders_history
        else:
            history_section["orders_placed"] = orders_history
        if role_name_raw == "retailer" and demand_history:
            history_section["demand"] = demand_history

        roles_section[role_key] = {
            "inventory": current_stock,
            "backlog": current_backlog,
            "pipeline": pipeline,
            "incoming_order": incoming_order,
            "history": history_section,
        }

    action_role_key = ROLE_NAME_MAP.get(action_role.lower(), action_role.lower())

    return {
        "week": int(round_number),
        "config": config_section,
        "roles": roles_section,
        "action_request": {
            "role": action_role_key,
            "trigger_decision": True,
        },
    }
