from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional

from ..models.game import Game, GameStatus
from ..models.player import Player
from ..models.supply_chain import PlayerRound, GameRound


def _active_statuses() -> List[GameStatus]:
    """Return the list of game statuses considered active."""

    candidates = []
    for status_name in ("IN_PROGRESS", "STARTED", "ROUND_IN_PROGRESS", "PAUSED"):
        if hasattr(GameStatus, status_name):
            candidates.append(getattr(GameStatus, status_name))
    return candidates or [GameStatus.IN_PROGRESS]


def get_active_game_for_user(db: Session, user_id: int) -> Optional[Game]:
    """Get the most recent active game for the supplied user."""

    return (
        db.query(Game)
        .join(Player, Player.game_id == Game.id)
        .filter(Player.user_id == user_id, Game.status.in_(_active_statuses()))
        .order_by(Game.created_at.desc())
        .first()
    )


def _fallback_numeric(value: Optional[float]) -> float:
    return float(value or 0)


def get_player_metrics(db: Session, player_id: int, game_id: int) -> Dict[str, Any]:
    """Calculate key metrics for a player in a specific game."""

    player = (
        db.query(Player)
        .filter(Player.id == player_id, Player.game_id == game_id)
        .first()
    )
    if not player:
        return {}

    player_rounds = (
        db.query(PlayerRound)
        .join(GameRound, PlayerRound.round_id == GameRound.id)
        .filter(PlayerRound.player_id == player_id, GameRound.game_id == game_id)
        .order_by(GameRound.round_number.asc())
        .all()
    )

    if not player_rounds:
        current_inventory = _fallback_numeric(getattr(player, "current_inventory", getattr(player, "inventory", 0)))
        backlog = _fallback_numeric(getattr(player, "current_backlog", getattr(player, "backlog", 0)))
        total_cost = _fallback_numeric(getattr(player, "total_cost", getattr(player, "cost", 0)))
        return {
            "current_inventory": current_inventory,
            "inventory_change": 0,
            "backlog": backlog,
            "total_cost": total_cost,
            "avg_weekly_cost": 0,
            "service_level": 1.0,
            "service_level_change": 0,
        }

    latest_round = player_rounds[-1]
    previous_round = player_rounds[-2] if len(player_rounds) > 1 else None

    current_inventory = _fallback_numeric(
        getattr(latest_round, "inventory_after", None)
        if getattr(latest_round, "inventory_after", None) is not None
        else getattr(latest_round, "inventory_before", None)
    )
    previous_inventory = _fallback_numeric(
        getattr(previous_round, "inventory_after", None)
        if previous_round and getattr(previous_round, "inventory_after", None) is not None
        else getattr(latest_round, "inventory_before", None)
    )
    inventory_change = 0.0
    if previous_inventory:
        inventory_change = ((current_inventory - previous_inventory) / previous_inventory) * 100

    backlog = _fallback_numeric(
        getattr(latest_round, "backorders_after", None)
        if getattr(latest_round, "backorders_after", None) is not None
        else getattr(latest_round, "backorders_before", None)
    )

    total_cost = sum(_fallback_numeric(pr.total_cost) for pr in player_rounds)
    avg_weekly_cost = total_cost / len(player_rounds) if player_rounds else 0

    fulfilled_rounds = [1 if _fallback_numeric(pr.backorders_after) == 0 else 0 for pr in player_rounds]
    service_level = sum(fulfilled_rounds) / len(player_rounds) if player_rounds else 1.0
    if len(player_rounds) > 1:
        previous_service_level = sum(fulfilled_rounds[:-1]) / (len(player_rounds) - 1)
    else:
        previous_service_level = service_level
    service_level_change = service_level - previous_service_level

    return {
        "current_inventory": current_inventory,
        "inventory_change": inventory_change,
        "backlog": backlog,
        "total_cost": total_cost,
        "avg_weekly_cost": avg_weekly_cost,
        "service_level": service_level,
        "service_level_change": service_level_change,
    }


def get_time_series_metrics(db: Session, player_id: int, game_id: int, role: str) -> List[Dict[str, Any]]:
    """Build a week-by-week time series for the requested player."""

    rounds = (
        db.query(GameRound)
        .filter(GameRound.game_id == game_id)
        .order_by(GameRound.round_number.asc())
        .all()
    )

    player_rounds = (
        db.query(PlayerRound)
        .join(GameRound, PlayerRound.round_id == GameRound.id)
        .filter(PlayerRound.player_id == player_id, GameRound.game_id == game_id)
        .all()
    )
    rounds_by_id = {pr.round_id: pr for pr in player_rounds}

    series: List[Dict[str, Any]] = []
    for round_ in rounds:
        player_round = rounds_by_id.get(round_.id)

        order = _fallback_numeric(getattr(player_round, "order_placed", None)) if player_round else 0
        inventory = _fallback_numeric(getattr(player_round, "inventory_after", None)) if player_round else 0
        backlog = _fallback_numeric(getattr(player_round, "backorders_after", None)) if player_round else 0
        cost = _fallback_numeric(getattr(player_round, "total_cost", None)) if player_round else 0
        supply = _fallback_numeric(getattr(player_round, "order_received", None)) if player_round else 0
        reason = getattr(player_round, "comment", None) if player_round else None

        entry = {
            "week": getattr(round_, "round_number", 0),
            "inventory": inventory,
            "order": order,
            "cost": cost,
            "backlog": backlog,
            "demand": getattr(round_, "customer_demand", None),
            "supply": supply if supply else None,
            "reason": reason,
        }

        # Limit demand visibility based on role, mirroring the previous behaviour
        if role not in ["RETAILER", "MANUFACTURER", "DISTRIBUTOR"]:
            entry["demand"] = None
        if role not in ["SUPPLIER", "MANUFACTURER", "DISTRIBUTOR"]:
            entry["supply"] = None

        series.append(entry)

    return series
