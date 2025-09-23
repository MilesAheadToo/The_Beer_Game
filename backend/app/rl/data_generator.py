import json
import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import os as _os
try:
    import simpy  # type: ignore
except Exception:
    simpy = None
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import (
    ACTION_LEVELS,
    BeerGameParams,
    NODE_INDEX,
    NODES,
    NODE_FEATURES,
    ORDER_EDGES,
    SHIPMENT_EDGES,
)
from app.services.agents import BeerGameAgent, AgentStrategy, AgentType

logger = logging.getLogger(__name__)

# ---------- Action indexing helpers ------------------------------------------

def order_units_to_action_idx(units: int) -> int:
    """
    Map a raw order quantity to the nearest discrete ACTION_LEVEL index.
    """
    units = max(0, units)
    diffs = [abs(u - units) for u in ACTION_LEVELS]
    return int(np.argmin(diffs))

def action_idx_to_order_units(idx: int) -> int:
    idx = int(np.clip(idx, 0, len(ACTION_LEVELS) - 1))
    return ACTION_LEVELS[idx]

# ---------- Feature wiring ----------------------------------------------------

def role_onehot(role: str) -> List[float]:
    oh = [0.0] * len(NODES)
    oh[NODE_INDEX[role]] = 1.0
    return oh

def assemble_node_features(
    role: str,
    inventory: int,
    backlog: int,
    incoming_orders: int,
    incoming_shipments: int,
    on_order: int,
    params: BeerGameParams,
) -> np.ndarray:
    return np.array(
        [
            float(inventory),
            float(backlog),
            float(incoming_orders),
            float(incoming_shipments),
            float(on_order),
            *role_onehot(role),
            float(params.info_delay),
            float(params.ship_delay),
        ],
        dtype=np.float32,
    )

# ---------- DB loader (exact lookup function) --------------------------------

@dataclass
class DbLookupConfig:
    """Configuration for database lookup of game state sequences."""
    database_url: str
    steps_table: str = "beer_game_steps"
    column_map: Dict[str, str] = None

    def __post_init__(self):
        if self.column_map is None:
            self.column_map = {
                "game_id": "game_id",
                "week": "week",
                "role": "role",
                "inventory": "inventory",
                "backlog": "backlog",
                "incoming_orders": "incoming_orders",
                "incoming_shipments": "incoming_shipments",
                "on_order": "on_order",
                "placed_order": "placed_order",
            }

def _build_select_sql(cfg: DbLookupConfig, game_ids: Optional[List[int]]) -> Tuple[str, Dict]:
    cols = cfg.column_map
    table = cfg.steps_table
    base = f"""
      SELECT
        {cols['game_id']}     AS game_id,
        {cols['week']}        AS week,
        {cols['role']}        AS role,
        {cols['inventory']}   AS inventory,
        {cols['backlog']}     AS backlog,
        {cols['incoming_orders']}    AS incoming_orders,
        {cols['incoming_shipments']} AS incoming_shipments,
        {cols['on_order']}    AS on_order,
        {cols['placed_order']} AS placed_order
      FROM {table}
    """
    params: Dict = {}
    if game_ids:
        ph = ", ".join([f":gid_{i}" for i in range(len(game_ids))])
        where = f" WHERE {cols['game_id']} IN ({ph})"
        params = {f"gid_{i}": gid for i, gid in enumerate(game_ids)}
        return base + where + " ORDER BY game_id, week, role", params
    else:
        return base + " ORDER BY game_id, week, role", params

def load_sequences_from_db(
    cfg: DbLookupConfig,
    params: BeerGameParams,
    game_ids: Optional[List[int]] = None,
    window: int = 12,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load sequences of game states from the database.
    
    Returns (X, A, P, Y):
      X: [num_windows, T=window, N=4, F] node features
      A: [2, N, N] adjacency matrices (0: shipments, 1: orders)
      P: [num_windows, C] global context (optional), here empty placeholder
      Y: [num_windows, T=horizon, N] action indices (discrete) to imitate
    """
    engine: Engine = create_engine(cfg.database_url)
    sql, bind = _build_select_sql(cfg, game_ids)

    with engine.connect() as conn:
        rows = conn.execute(text(sql), bind).mappings().all()

    # Bucket rows by (game_id, week) to form 4 nodes per week in fixed order
    by_gw: Dict[Tuple[int, int], Dict[str, dict]] = {}
    for r in rows:
        key = (int(r["game_id"]), int(r["week"]))
        by_gw.setdefault(key, {})[r["role"]] = dict(r)

    # Build ordered timelines per game_id
    by_game: Dict[int, List[Dict[str, dict]]] = {}
    for (gid, wk), role_map in by_gw.items():
        by_game.setdefault(gid, []).append({"week": wk, "roles": role_map})
    for gid in by_game:
        by_game[gid].sort(key=lambda e: e["week"])

    # Convert to arrays
    X_windows: List[np.ndarray] = []
    Y_windows: List[np.ndarray] = []
    P_windows: List[np.ndarray] = []

    A_ship = np.zeros((4, 4), dtype=np.float32)
    A_order = np.zeros((4, 4), dtype=np.float32)
    for u, v in SHIPMENT_EDGES:
        A_ship[u, v] = 1.0
    for u, v in ORDER_EDGES:
        A_order[u, v] = 1.0
    A = np.stack([A_ship, A_order], axis=0)  # [2, 4, 4]

    for gid, timeline in by_game.items():
        # Require full 4 roles per week
        weeks = [w for w in timeline if set(w["roles"].keys()) == set(NODES)]
        if len(weeks) < window + horizon:
            continue

        # Slide windows
        for start in range(0, len(weeks) - (window + horizon) + 1):
            obs_block = weeks[start : start + window]
            fut_block = weeks[start + window : start + window + horizon]

            # X[t, n, f]
            X_block = np.zeros((window, 4, len(NODE_FEATURES)), dtype=np.float32)
            for t, w in enumerate(obs_block):
                for role in NODES:
                    rec = w["roles"][role]
                    X_block[t, NODE_INDEX[role]] = assemble_node_features(
                        role=role,
                        inventory=int(rec["inventory"]),
                        backlog=int(rec["backlog"]),
                        incoming_orders=int(rec["incoming_orders"]),
                        incoming_shipments=int(rec["incoming_shipments"]),
                        on_order=int(rec["on_order"]),
                        params=params,
                    )

            # Y[t, n] (action indices) — imitate the placed orders in the future block
            Y_block = np.zeros((horizon, 4), dtype=np.int64)
            for t, w in enumerate(fut_block):
                for role in NODES:
                    rec = w["roles"][role]
                    Y_block[t, NODE_INDEX[role]] = order_units_to_action_idx(int(rec["placed_order"]))

            X_windows.append(X_block)
            Y_windows.append(Y_block)
            P_windows.append(np.zeros((0,), dtype=np.float32))  # no globals for now

    if not X_windows:
        raise RuntimeError(
            "No training windows built from DB — check steps_table name and column_map!"
        )

    X = np.stack(X_windows, axis=0)  # [B, T, N, F]
    Y = np.stack(Y_windows, axis=0)  # [B, H, N]
    P = np.stack(P_windows, axis=0)  # [B, 0]
    return X, A, P, Y


def _load_param_ranges_from_config(
    supply_chain_config_id: int,
    db_url: Optional[str] = None,
) -> Dict[str, Tuple[float, float]]:
    """Derive simulator parameter ranges from a stored supply chain config."""

    resolved_url = db_url or _os.getenv("DATABASE_URL")
    if not resolved_url:
        logger.warning(
            "No database URL available when loading config %s; using defaults.",
            supply_chain_config_id,
        )
        return {}

    def _parse_range(raw) -> Optional[Tuple[float, float]]:
        if raw is None:
            return None
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                return None
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON range: %s", raw)
                return None
        if isinstance(raw, dict):
            lo = raw.get("min")
            hi = raw.get("max", lo)
            if lo is None and hi is None:
                return None
            if lo is None:
                lo = hi
            if hi is None:
                hi = lo
            try:
                return float(lo), float(hi)
            except (TypeError, ValueError):
                return None
        if isinstance(raw, (list, tuple)) and raw:
            try:
                return float(raw[0]), float(raw[-1])
            except (TypeError, ValueError):
                return None
        if isinstance(raw, (int, float)):
            val = float(raw)
            return val, val
        return None

    def _merge_range(
        bucket: Dict[str, Tuple[float, float]],
        name: str,
        candidate: Optional[Tuple[float, float]],
        *,
        as_int: bool = False,
    ) -> None:
        if not candidate:
            return
        lo, hi = candidate
        if lo is None or hi is None:
            return
        if as_int:
            lo = int(math.floor(lo))
            hi = int(math.ceil(hi))
        existing = bucket.get(name)
        if existing:
            lo = min(lo, existing[0])
            hi = max(hi, existing[1])
        bucket[name] = (lo, hi)

    ranges: Dict[str, Tuple[float, float]] = {}

    engine: Optional[Engine] = None
    try:
        engine = create_engine(resolved_url)
        with engine.connect() as conn:
            node_rows = conn.execute(
                text(
                    """
                    SELECT
                        inc.initial_inventory_range AS initial_inventory_range,
                        inc.inventory_target_range AS inventory_target_range,
                        inc.holding_cost_range AS holding_cost_range,
                        inc.backlog_cost_range AS backlog_cost_range
                    FROM item_node_configs inc
                    JOIN nodes n ON inc.node_id = n.id
                    WHERE n.config_id = :cfg_id
                    """
                ),
                {"cfg_id": supply_chain_config_id},
            ).mappings().all()

            for row in node_rows:
                init_range = _parse_range(row.get("initial_inventory_range"))
                _merge_range(ranges, "init_inventory", init_range, as_int=True)

                target_range = _parse_range(row.get("inventory_target_range"))
                _merge_range(ranges, "max_order", target_range, as_int=True)

                holding_range = _parse_range(row.get("holding_cost_range"))
                _merge_range(ranges, "holding_cost", holding_range)

                backlog_range = _parse_range(row.get("backlog_cost_range"))
                _merge_range(ranges, "backlog_cost", backlog_range)

            lane_rows = conn.execute(
                text(
                    """
                    SELECT capacity, lead_time_days
                    FROM lanes
                    WHERE config_id = :cfg_id
                    """
                ),
                {"cfg_id": supply_chain_config_id},
            ).mappings().all()

            for row in lane_rows:
                capacity = row.get("capacity")
                if capacity is not None:
                    _merge_range(
                        ranges,
                        "max_inbound_per_link",
                        (float(capacity), float(capacity)),
                        as_int=True,
                    )

                lead_range = _parse_range(row.get("lead_time_days"))
                if lead_range:
                    week_lo = max(0.0, lead_range[0] / 7.0)
                    week_hi = max(week_lo, lead_range[1] / 7.0)
                    _merge_range(
                        ranges,
                        "ship_delay",
                        (week_lo, week_hi),
                        as_int=True,
                    )

    except Exception as exc:
        logger.warning(
            "Failed to derive parameter ranges for config %s: %s",
            supply_chain_config_id,
            exc,
        )
        return {}
    finally:
        if engine is not None:
            engine.dispose()

    return ranges

# ---------- Simulator (synthetic data) ----------------------------------------

@dataclass
class SimDemand:
    """Simple piecewise-constant demand with an optional step change."""
    base: int = 4
    step_to: int = 12
    step_week: int = 4

    def __call__(self, t: int) -> int:
        return self.base if t < self.step_week else self.step_to

def simulate_beer_game(
    T: int,
    params: BeerGameParams,
    demand_fn=SimDemand(),
) -> Dict[str, Dict[str, List[int]]]:
    """
    Simulate a Beer Game run with the given parameters.
    
    Returns a dict per role with time series:
        inventory, backlog, incoming_orders, incoming_shipments, on_order, placed_order
    """
    roles = NODES
    agents = {
        role: BeerGameAgent(
            agent_id=index,
            agent_type=AgentType[role.upper()],
            strategy=AgentStrategy.LLM,
            can_see_demand=True,
            initial_inventory=params.init_inventory,
            initial_orders=4,
            llm_model="gpt-4o-mini",
        )
        for index, role in enumerate(roles)
    }
    # state
    inv = {r: [params.init_inventory] for r in roles}
    back = {r: [0] for r in roles}
    in_ord = {r: [0] for r in roles}
    in_ship = {r: [0] for r in roles}
    on_ord = {r: [0] for r in roles}
    placed = {r: [] for r in roles}

    # pipelines for delays (FIFO queues)
    info_pipes = {r: [0] * params.info_delay for r in roles}
    ship_pipes = {r: [0] * params.ship_delay for r in roles}

    def clip_ship(x):  # shipping capacity
        return min(x, params.max_inbound_per_link)

    for t in range(T):
        # incoming orders at retailer = external demand
        in_ord["retailer"].append(demand_fn(t))

        # propagate orders upstream with info_delay
        for dn, up in ORDER_EDGES:  # (downstream -> upstream)
            src_role = NODES[dn]
            dst_role = NODES[up]
            # downstream placed orders enter info pipe towards upstream
            outgoing = placed[src_role][-1] if placed[src_role] else 0
            pipe = info_pipes[dst_role]
            arriving = pipe.pop(0) if pipe else 0
            pipe.append(outgoing)
            in_ord[dst_role].append(arriving)

        # shipments downstream with ship_delay
        for up, dn in SHIPMENT_EDGES:
            src_role = NODES[up]
            dst_role = NODES[dn]
            outgoing = 0
            # can only ship what you have
            demand_here = in_ord[dst_role][-1] + back[dst_role][-1]
            outgoing = clip_ship(min(inv[src_role][-1], demand_here))
            pipe = ship_pipes[dst_role]
            arriving = pipe.pop(0) if pipe else 0
            pipe.append(outgoing)
            in_ship[dst_role].append(arriving)

        # update inventories/backlogs after shipments arrive & demand realized
        for r in roles:
            inv_r = inv[r][-1]
            incoming = in_ship[r][-1]
            demand = in_ord[r][-1] + back[r][-1]
            shipped = min(inv_r + incoming, demand)
            new_inv = max(0, inv_r + incoming - shipped)
            new_back = max(0, demand - (inv_r + incoming))
            inv[r].append(new_inv)
            back[r].append(new_back)

        # simple heuristic ordering (base-stock vibe) for behavior traces
        previous_orders_by_role = {
            role: placed[role][-1] if placed[role] else 0 for role in roles
        }

        for r in roles:
            agent = agents[r]
            agent.inventory = inv[r][-1]
            agent.backlog = back[r][-1]
            agent.pipeline = list(ship_pipes[r])
            local_state = {
                "inventory": inv[r][-1],
                "backlog": back[r][-1],
                "incoming_shipments": ship_pipes[r],
                "pipeline": on_ord[r][-1],
            }
            upstream_data = {
                "previous_orders_by_role": previous_orders_by_role,
                "previous_orders": placed[r][-3:] if len(placed[r]) >= 3 else placed[r],
            }
            current_demand = in_ord[r][-1] if in_ord[r] else None
            try:
                order_units = agent.make_decision(
                    current_round=t,
                    current_demand=current_demand,
                    upstream_data=upstream_data,
                    local_state=local_state,
                )
            except Exception:
                target = params.init_inventory + params.ship_delay * 2
                desired = target + back[r][-1] - inv[r][-1] - on_ord[r][-1]
                order_units = int(np.clip(desired, 0, params.max_order))
            order_units = max(0, int(order_units))
            order_units = min(order_units, params.max_order)
            act_idx = order_units_to_action_idx(order_units)
            order_units = action_idx_to_order_units(act_idx)
            placed[r].append(order_units)
            on_ord[r].append(max(0, on_ord[r][-1] + order_units - in_ship[r][-1]))

    # trim first seed element to align lengths
    def trim(series: List[int]) -> List[int]:
        return series[1:] if len(series) and len(series[1:]) == T else series[:T]

    out = {}
    for r in roles:
        out[r] = {
            "inventory": trim(inv[r]),
            "backlog": trim(back[r]),
            "incoming_orders": trim(in_ord[r]),
            "incoming_shipments": trim(in_ship[r]),
            "on_order": trim(on_ord[r]),
            "placed_order": placed[r][:T],
        }
    return out


def simulate_beer_game_simpy(
    T: int,
    params: BeerGameParams,
    demand_fn=SimDemand(),
    alpha: float = 0.3,
    wip_k: float = 1.0,
) -> Dict[str, Dict[str, List[int]]]:
    """
    SimPy-backed week-stepped Beer Game with a smoother ordering policy to reduce bullwhip.
    Default demand pattern is classic Beer Game: flat base, then a step up that stays flat.
    """
    if simpy is None:
        # If SimPy isn't available, fall back to the discrete simulator
        return simulate_beer_game(T=T, params=params, demand_fn=demand_fn)

    env = simpy.Environment()

    roles = list(NODES)
    inv = {r: [params.init_inventory] for r in roles}
    back = {r: [0] for r in roles}
    in_ord = {r: [0] for r in roles}
    in_ship = {r: [0] for r in roles}
    on_ord = {r: [0] for r in roles}
    placed = {r: [] for r in roles}

    info_pipes = {r: [0] * params.info_delay for r in roles}
    ship_pipes = {r: [0] * params.ship_delay for r in roles}
    # Simple per-role demand/throughput forecast to dampen swings
    forecast = {r: float(params.init_inventory) / max(1, params.ship_delay + 1) for r in roles}

    def clip_ship(x: int) -> int:
        return int(min(max(0, x), params.max_inbound_per_link))

    def step():
        # External demand hits retailer
        in_ord["retailer"].append(demand_fn(len(placed["retailer"])) )

        # Orders propagate upstream (info delay)
        for dn, up in ORDER_EDGES:
            src_role = NODES[dn]
            dst_role = NODES[up]
            outgoing = placed[src_role][-1] if placed[src_role] else 0
            pipe = info_pipes[dst_role]
            arriving = pipe.pop(0) if pipe else 0
            pipe.append(outgoing)
            in_ord[dst_role].append(arriving)

        # Shipments propagate downstream (shipping delay)
        for up, dn in SHIPMENT_EDGES:
            src_role = NODES[up]
            dst_role = NODES[dn]
            demand_here = in_ord[dst_role][-1] + back[dst_role][-1]
            outgoing = clip_ship(min(inv[src_role][-1], demand_here))
            pipe = ship_pipes[dst_role]
            arriving = pipe.pop(0) if pipe else 0
            pipe.append(outgoing)
            in_ship[dst_role].append(arriving)

        # Update inventory/backlog after shipments
        for r in roles:
            inv_r = inv[r][-1]
            incoming = in_ship[r][-1]
            demand = in_ord[r][-1] + back[r][-1]
            shipped = min(inv_r + incoming, demand)
            new_inv = max(0, inv_r + incoming - shipped)
            new_back = max(0, demand - (inv_r + incoming))
            inv[r].append(new_inv)
            back[r].append(new_back)

        # Smoother order policy (order-up-to with forecast + WIP correction)
        for r in roles:
            obs = float(in_ord[r][-1])
            forecast[r] = alpha * obs + (1.0 - alpha) * forecast[r]
            target_inv = params.init_inventory + params.ship_delay * 2
            order_up_to = target_inv + forecast[r] * params.ship_delay
            current_wip = inv[r][-1] + on_ord[r][-1]
            desired = max(0.0, order_up_to - current_wip + back[r][-1])
            raw_order = int(np.clip(wip_k * desired, 0, params.max_order))
            act_idx = order_units_to_action_idx(raw_order)
            order_units = action_idx_to_order_units(act_idx)
            placed[r].append(order_units)
            on_ord[r].append(max(0, on_ord[r][-1] + order_units - in_ship[r][-1]))

    def weekly(env):
        for _ in range(T):
            step()
            yield env.timeout(1)

    env.process(weekly(env))
    env.run()

    def trim(series: List[int]) -> List[int]:
        return series[1:] if len(series) and len(series[1:]) == T else series[:T]

    out = {}
    for r in roles:
        out[r] = {
            "inventory": trim(inv[r]),
            "backlog": trim(back[r]),
            "incoming_orders": trim(in_ord[r]),
            "incoming_shipments": trim(in_ship[r]),
            "on_order": trim(on_ord[r]),
            "placed_order": placed[r][:T],
        }
    return out

def _sample_params_uniform(base: BeerGameParams, ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> BeerGameParams:
    """Sample BeerGameParams uniformly within provided ranges (inclusive)."""
    rng = ranges or {}
    def pick(name: str, cur):
        if name not in rng:
            return cur
        lo, hi = rng[name]
        # integers for discrete params
        if isinstance(cur, int):
            return int(np.random.uniform(lo, hi + 1))
        else:
            return float(np.random.uniform(lo, hi))

    return BeerGameParams(
        info_delay=pick("info_delay", base.info_delay),
        ship_delay=pick("ship_delay", base.ship_delay),
        init_inventory=pick("init_inventory", base.init_inventory),
        holding_cost=pick("holding_cost", base.holding_cost),
        backlog_cost=pick("backlog_cost", base.backlog_cost),
        max_inbound_per_link=pick("max_inbound_per_link", base.max_inbound_per_link),
        max_order=pick("max_order", base.max_order),
    )

def generate_sim_training_windows(
    num_runs: int,
    T: int,
    window: int = 12,
    horizon: int = 1,
    params: BeerGameParams = BeerGameParams(),
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    randomize: bool = True,
    supply_chain_config_id: Optional[int] = None,
    db_url: Optional[str] = None,
    use_simpy: Optional[bool] = None,
    sim_alpha: float = 0.3,
    sim_wip_k: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create imitation-learning windows from the simulator.

    If `randomize` is True, each run samples parameters uniformly within
    `param_ranges` (or sensible defaults if not provided).
    """
    Xs, Ys, Ps = [], [], []
    A_ship = np.zeros((4, 4), dtype=np.float32)
    A_order = np.zeros((4, 4), dtype=np.float32)
    for u, v in SHIPMENT_EDGES:
        A_ship[u, v] = 1.0
    for u, v in ORDER_EDGES:
        A_order[u, v] = 1.0
    A = np.stack([A_ship, A_order], axis=0)

    # Default ranges (broad but sane for Beer Game)
    default_ranges: Dict[str, Tuple[float, float]] = {
        "info_delay": (0, 6),
        "ship_delay": (0, 6),
        "init_inventory": (4, 60),
        "holding_cost": (0.1, 2.0),
        "backlog_cost": (0.2, 4.0),
        "max_inbound_per_link": (50, 300),
        "max_order": (50, 300),
    }

    cfg_ranges: Dict[str, Tuple[float, float]] = {}
    if supply_chain_config_id is not None:
        cfg_ranges = _load_param_ranges_from_config(
            supply_chain_config_id,
            db_url=db_url,
        )

    ranges = default_ranges.copy()
    ranges.update(cfg_ranges)
    if param_ranges:
        ranges.update(param_ranges)

    # Decide simulator backend; default to SimPy unless USE_SIMPY="0"
    if use_simpy is None:
        use_simpy = (_os.getenv("USE_SIMPY", "1") != "0")
    use_simpy = bool(use_simpy and simpy is not None)
    if use_simpy is False and simpy is None:
        logger.debug("SimPy not available; using discrete simulator")

    def _safe_lookup(role: str, key: str, index: int, default: int = 0) -> int:
        try:
            sequence = trace[role][key]
            if index < 0 or index >= len(sequence):
                return default
            return int(sequence[index])
        except (KeyError, TypeError, ValueError):
            return default

    for _ in range(num_runs):
        sim_params = _sample_params_uniform(params, ranges) if randomize else params
        demand = SimDemand()  # flat, then step up (classic Beer Game)
        if use_simpy:
            trace = simulate_beer_game_simpy(
                T=T,
                params=sim_params,
                demand_fn=demand,
                alpha=sim_alpha,
                wip_k=sim_wip_k,
            )
        else:
            trace = simulate_beer_game(T=T, params=sim_params, demand_fn=demand)
        # slide windows
        for start in range(0, T - (window + horizon) + 1):
            X = np.zeros((window, 4, len(NODE_FEATURES)), dtype=np.float32)
            Y = np.zeros((horizon, 4), dtype=np.int64)

            for t in range(window):
                for role in NODES:
                    X[t, NODE_INDEX[role]] = assemble_node_features(
                        role=role,
                        inventory=_safe_lookup(role, "inventory", start + t),
                        backlog=_safe_lookup(role, "backlog", start + t),
                        incoming_orders=_safe_lookup(role, "incoming_orders", start + t),
                        incoming_shipments=_safe_lookup(role, "incoming_shipments", start + t),
                        on_order=_safe_lookup(role, "on_order", start + t),
                        params=sim_params,
                    )

            for t in range(horizon):
                for role in NODES:
                    order_val = _safe_lookup(role, "placed_order", start + window + t)
                    Y[t, NODE_INDEX[role]] = order_units_to_action_idx(order_val)

            Xs.append(X)
            Ys.append(Y)
            Ps.append(np.zeros((0,), dtype=np.float32))

    return np.stack(Xs, axis=0), A, np.stack(Ps, axis=0), np.stack(Ys, axis=0)
