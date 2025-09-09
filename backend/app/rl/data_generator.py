import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
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
        for r in roles:
            target = params.init_inventory + params.ship_delay * 2
            # try to restore inventory + clear backlog + maintain pipeline
            desired = target + back[r][-1] - inv[r][-1] - on_ord[r][-1]
            raw_order = int(np.clip(desired, 0, params.max_order))
            # snap to discrete action
            act_idx = order_units_to_action_idx(raw_order)
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
    ranges = param_ranges or default_ranges

    for _ in range(num_runs):
        sim_params = _sample_params_uniform(params, ranges) if randomize else params
        trace = simulate_beer_game(T=T, params=sim_params)
        # slide windows
        for start in range(0, T - (window + horizon) + 1):
            X = np.zeros((window, 4, len(NODE_FEATURES)), dtype=np.float32)
            Y = np.zeros((horizon, 4), dtype=np.int64)

            for t in range(window):
                for role in NODES:
                    X[t, NODE_INDEX[role]] = assemble_node_features(
                        role=role,
                        inventory=int(trace[role]["inventory"][start + t]),
                        backlog=int(trace[role]["backlog"][start + t]),
                        incoming_orders=int(trace[role]["incoming_orders"][start + t]),
                        incoming_shipments=int(trace[role]["incoming_shipments"][start + t]),
                        on_order=int(trace[role]["on_order"][start + t]),
                        params=sim_params,
                    )

            for t in range(horizon):
                for role in NODES:
                    Y[t, NODE_INDEX[role]] = order_units_to_action_idx(
                        int(trace[role]["placed_order"][start + window + t])
                    )

            Xs.append(X)
            Ys.append(Y)
            Ps.append(np.zeros((0,), dtype=np.float32))

    return np.stack(Xs, axis=0), A, np.stack(Ps, axis=0), np.stack(Ys, axis=0)
