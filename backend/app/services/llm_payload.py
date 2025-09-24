"""Utilities for constructing structured payloads for the Beer Game Daybreak LLM agent."""

from __future__ import annotations

import json
import math
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


def _pad_sequence(values: List[int], length: int) -> List[int]:
    normalized = [int(x) for x in values[:length]]
    if len(normalized) < length:
        normalized.extend([0] * (length - len(normalized)))
    return normalized


def _compute_volatility_signal(history: List[int]) -> Dict[str, Any]:
    if not history:
        return {"sigma": 0.0, "trend": "flat"}

    if len(history) == 1:
        return {"sigma": 0.0, "trend": "flat"}

    mean = sum(history) / len(history)
    variance = sum((value - mean) ** 2 for value in history) / (len(history) - 1)
    sigma = math.sqrt(max(variance, 0.0))

    trend = "flat"
    if history[-1] > history[-2]:
        trend = "up"
    elif history[-1] < history[-2]:
        trend = "down"

    if len(history) >= 3:
        recent = history[-3:]
        if recent[0] <= recent[1] <= recent[2] and recent[2] > recent[1]:
            trend = "up"
        elif recent[0] >= recent[1] >= recent[2] and recent[2] < recent[1]:
            trend = "down"

    return {"sigma": round(sigma, 4), "trend": trend}


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

    daybreak_cfg = _coerce_dict(config_raw.get("daybreak_llm", {}))
    daybreak_toggles = _coerce_dict(daybreak_cfg.get("toggles", {}))

    toggles = {
        "customer_demand_history_sharing": bool(
            daybreak_toggles.get("customer_demand_history_sharing")
            or config_raw.get("enable_information_sharing")
            or sim_params.get("enable_information_sharing")
            or config_raw.get("historical_weeks_to_share")
        ),
        "volatility_signal_sharing": bool(
            daybreak_toggles.get("volatility_signal_sharing")
            or config_raw.get("enable_demand_volatility_signals")
            or sim_params.get("enable_demand_volatility_signals")
            or config_raw.get("volatility_analysis_window")
        ),
        "downstream_inventory_visibility": bool(
            daybreak_toggles.get("downstream_inventory_visibility")
            or config_raw.get("enable_downstream_visibility")
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
    role_key_to_raw: Dict[str, str] = {}
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
        role_key_to_raw[role_key] = role_name_raw

    action_role_key = ROLE_NAME_MAP.get(action_role.lower(), action_role.lower())

    role_state = roles_section.get(action_role_key, {})
    raw_role = role_key_to_raw.get(action_role_key, action_role.lower())
    engine_entry = _coerce_dict(engine_state.get(raw_role, {}))

    incoming_order = int(engine_entry.get("incoming_orders", role_state.get("incoming_order", 0)) or 0)
    on_hand = int(role_state.get("inventory", 0))
    backlog = int(role_state.get("backlog", 0))
    received_shipment = int(engine_entry.get("last_arrival", 0) or 0)

    pipeline_orders = engine_entry.get("info_queue")
    if isinstance(pipeline_orders, list):
        pipeline_orders = [int(x) for x in pipeline_orders]
    else:
        pipeline_orders = []
    pipeline_orders = _pad_sequence(pipeline_orders, order_lead)

    inbound_pipeline_raw = engine_entry.get("ship_queue")
    if isinstance(inbound_pipeline_raw, list):
        inbound_pipeline_values = _normalize_pipeline(inbound_pipeline_raw)
    else:
        inbound_pipeline_values = _normalize_pipeline(role_state.get("pipeline", []))
    inbound_pipeline = _pad_sequence(inbound_pipeline_values, ship_lead)

    optional_section: Dict[str, Any] = {}

    retailer_history_records = history_by_role.get("retailer", [])
    retailer_demand_history = [entry["customer_demand"] or 0 for entry in retailer_history_records if entry.get("customer_demand") is not None]
    if retailer_demand_history and visible_history_weeks:
        retailer_demand_history = retailer_demand_history[-visible_history_weeks:]

    if toggles["customer_demand_history_sharing"] and retailer_demand_history:
        optional_section["shared_demand_history"] = retailer_demand_history

    if toggles["volatility_signal_sharing"] and retailer_demand_history:
        window = retailer_demand_history[-volatility_window:] if volatility_window else retailer_demand_history
        optional_section["shared_volatility_signal"] = _compute_volatility_signal(window)

    if toggles["downstream_inventory_visibility"]:
        downstream_role = downstream_role_map.get(raw_role)
        if downstream_role:
            downstream_key = ROLE_NAME_MAP.get(downstream_role, downstream_role)
            downstream_state = roles_section.get(downstream_key)
            if downstream_state:
                optional_section["visible_downstream"] = {
                    "on_hand": int(downstream_state.get("inventory", 0)),
                    "backlog": int(downstream_state.get("backlog", 0)),
                }

    local_optional = role_state.get("history", {})
    if local_optional:
        optional_section.setdefault("local_history", local_optional)

    return {
        "role": action_role_key,
        "week": int(round_number),
        "toggles": toggles,
        "parameters": {
            "holding_cost": holding_cost,
            "backlog_cost": backorder_cost,
            "L_order": order_lead,
            "L_ship": ship_lead,
            "L_prod": prod_lead,
        },
        "local_state": {
            "on_hand": on_hand,
            "backlog": backlog,
            "incoming_orders_this_week": incoming_order,
            "received_shipment_this_week": received_shipment,
            "pipeline_orders_upstream": pipeline_orders,
            "pipeline_shipments_inbound": inbound_pipeline,
            "optional": optional_section,
        },
    }
