from typing import List, Dict, Optional, Any, Sequence, Set
from datetime import datetime, timedelta
from enum import Enum
import random
import json
from collections import defaultdict, deque

from types import SimpleNamespace
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from app.models.game import Game, GameStatus as GameStatusDB
from app.models.player import Player, PlayerRole
from app.models.supply_chain import PlayerInventory, Order, GameRound, PlayerRound
from app.models.user import User, UserTypeEnum
from app.schemas.game import (
    GameCreate,
    GameUpdate,
    GameState,
    PlayerState,
    GameStatus,
    GameInDBBase,
)
from app.schemas.player import PlayerAssignment, PlayerType, PlayerStrategy
from app.services.agents import AgentManager, AgentType, AgentStrategy as AgentStrategyEnum
from app.services.llm_payload import build_llm_decision_payload
from app.api.endpoints.config import _read_cfg as read_system_cfg
from app.core.demand_patterns import (
    normalize_demand_pattern,
    DEFAULT_DEMAND_PATTERN,
    DEFAULT_CLASSIC_PARAMS,
)

class MixedGameService:
    """Service for managing games with mixed human and AI players."""
    
    def __init__(self, db: Session):
        self.db = db
        self.agent_manager = AgentManager()
        self._game_columns_cache: Optional[Sequence[str]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_game_columns(self) -> Sequence[str]:
        if self._game_columns_cache is not None:
            return self._game_columns_cache

        try:
            inspector = inspect(self.db.bind)
            columns = inspector.get_columns(Game.__tablename__)
            self._game_columns_cache = [column['name'] for column in columns]
        except Exception:
            # Fallback to model metadata if inspection fails
            self._game_columns_cache = [column.name for column in Game.__table__.columns]
        return self._game_columns_cache

    @staticmethod
    def _coerce_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _normalise_key(value: Any) -> str:
        return str(value).strip().lower()

    @staticmethod
    def _validate_lanes(node_policies: Dict[str, Any], lanes: List[Dict[str, Any]]) -> None:
        if not lanes:
            return
        known_nodes = {MixedGameService._normalise_key(name) for name in node_policies.keys()}
        missing: List[str] = []
        for lane in lanes:
            upstream = MixedGameService._normalise_key(lane.get("from") or lane.get("upstream"))
            downstream = MixedGameService._normalise_key(lane.get("to") or lane.get("downstream"))
            if upstream not in known_nodes:
                missing.append(upstream)
            if downstream not in known_nodes:
                missing.append(downstream)
        if missing:
            unique_missing = sorted(set(missing))
            raise ValueError(
                "Lane configuration references unknown nodes: "
                + ", ".join(unique_missing)
            )

    @staticmethod
    def _topological_order(shipments_map: Dict[str, List[str]], nodes: Sequence[str]) -> List[str]:
        indegree: Dict[str, int] = {node: 0 for node in nodes}
        for upstream, downstreams in shipments_map.items():
            for downstream in downstreams:
                indegree.setdefault(downstream, 0)
                indegree[downstream] += 1
                indegree.setdefault(upstream, 0)
        queue: deque[str] = deque(sorted([node for node, deg in indegree.items() if deg == 0]))
        order: List[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbour in shipments_map.get(node, []):
                indegree[neighbour] -= 1
                if indegree[neighbour] == 0:
                    queue.append(neighbour)
        if len(order) != len(indegree):
            # Cycle detected – fall back to existing node order to avoid crashing.
            return list(nodes)
        return order

    @staticmethod
    def _build_lane_views(node_policies: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
        lanes = cfg.get("lanes") or []
        node_keys = [MixedGameService._normalise_key(k) for k in node_policies.keys()]
        raw_types = cfg.get("node_types") or {}
        node_types = {
            MixedGameService._normalise_key(name): str(node_type).lower()
            for name, node_type in raw_types.items()
        }
        if not lanes and len(node_keys) >= 2:
            # Fall back to a linear chain using node policy ordering
            fallback: List[Dict[str, Any]] = []
            for idx in range(len(node_keys) - 1, 0, -1):
                upstream = node_keys[idx]
                downstream = node_keys[idx - 1]
                fallback.append({"from": upstream, "to": downstream})
            lanes = fallback

        shipments_map: Dict[str, List[str]] = defaultdict(list)
        orders_map: Dict[str, List[str]] = defaultdict(list)
        lane_records: List[Dict[str, Any]] = []
        lanes_by_upstream: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        all_nodes = set(node_keys)
        for lane in lanes:
            upstream_raw = lane.get("from") or lane.get("upstream")
            downstream_raw = lane.get("to") or lane.get("downstream")
            if upstream_raw is None or downstream_raw is None:
                continue
            upstream = MixedGameService._normalise_key(upstream_raw)
            downstream = MixedGameService._normalise_key(downstream_raw)
            lane_record = {
                "from": upstream,
                "to": downstream,
                "capacity": lane.get("capacity"),
                "lead_time_days": lane.get("lead_time_days"),
            }
            shipments_map[upstream].append(downstream)
            orders_map[downstream].append(upstream)
            lane_records.append(lane_record)
            lanes_by_upstream[upstream].append(lane_record)
            all_nodes.add(upstream)
            all_nodes.add(downstream)

        market_nodes = [MixedGameService._normalise_key(n) for n in cfg.get("market_demand_nodes", []) if n]
        if not market_nodes and lane_records:
            downstream_only = {record["to"] for record in lane_records}
            upstream_nodes = {record["from"] for record in lane_records}
            inferred = sorted(downstream_only - upstream_nodes)
            if inferred:
                market_nodes = inferred

        node_sequence = MixedGameService._topological_order(shipments_map, sorted(all_nodes)) if all_nodes else []
        return {
            "lanes": lane_records,
            "shipments_map": shipments_map,
            "orders_map": orders_map,
            "market_nodes": market_nodes,
            "all_nodes": sorted(all_nodes),
            "node_sequence": node_sequence,
            "lanes_by_upstream": lanes_by_upstream,
            "node_types": node_types,
        }

    @staticmethod
    def _ensure_engine_node(engine: Dict[str, Dict[str, Any]], node_policies: Dict[str, Any], node: str) -> Dict[str, Any]:
        policy = node_policies.get(node, {})
        info_delay = max(0, int(policy.get("info_delay", 0)))
        ship_delay = max(0, int(policy.get("ship_delay", 0)))
        state = engine.setdefault(node, {})
        state.setdefault("inventory", int(policy.get("init_inventory", 12)))
        state.setdefault("backlog", 0)
        state.setdefault("on_order", 0)
        state.setdefault("info_queue", [0] * info_delay if info_delay > 0 else [])
        if info_delay > 0:
            detail_queue = state.setdefault(
                "info_detail_queue",
                [{} for _ in range(info_delay)],
            )
            if len(detail_queue) < info_delay:
                detail_queue.extend({} for _ in range(info_delay - len(detail_queue)))
            elif len(detail_queue) > info_delay:
                state["info_detail_queue"] = detail_queue[-info_delay:]
        else:
            state.setdefault("info_detail_queue", [])
        state.setdefault("ship_queue", [0] * ship_delay if ship_delay > 0 else [])
        state.setdefault("last_order", 0)
        state.setdefault("holding_cost", 0.0)
        state.setdefault("backorder_cost", 0.0)
        state.setdefault("total_cost", 0.0)
        state.setdefault("incoming_orders", 0)
        state.setdefault("incoming_shipments", 0)
        state.setdefault("last_shipment_planned", 0)
        state.setdefault("last_arrival", 0)
        state.setdefault("backlog_breakdown", {})
        return state

    @staticmethod
    def _resolve_user_type(user: Optional[User]) -> Optional[UserTypeEnum]:
        user_type = getattr(user, "user_type", None)
        if isinstance(user_type, UserTypeEnum):
            return user_type
        if isinstance(user_type, Enum):
            try:
                return UserTypeEnum(user_type.value)
            except ValueError:
                return None
        if isinstance(user_type, str):
            try:
                return UserTypeEnum(user_type)
            except ValueError:
                return None
        return None

    @staticmethod
    def _schema_status_to_db_values(status: GameStatus) -> List[str]:
        mapping = {
            GameStatus.CREATED: [GameStatusDB.CREATED.value],
            GameStatus.IN_PROGRESS: [
                GameStatusDB.STARTED.value,
                getattr(GameStatusDB, "IN_PROGRESS", GameStatusDB.STARTED).value
                if hasattr(GameStatusDB, "IN_PROGRESS")
                else GameStatusDB.STARTED.value,
                getattr(GameStatusDB, "ROUND_IN_PROGRESS", GameStatusDB.STARTED).value,
                getattr(GameStatusDB, "ROUND_COMPLETED", GameStatusDB.STARTED).value,
                GameStatus.IN_PROGRESS.value,
            ],
            GameStatus.COMPLETED: [
                getattr(GameStatusDB, "FINISHED", GameStatusDB.CREATED).value,
                GameStatus.COMPLETED.value,
            ],
            GameStatus.PAUSED: [GameStatus.PAUSED.value, "PAUSED", "paused"],
        }
        return mapping.get(status, [status.value])

    @staticmethod
    def _map_status_to_schema(status_value: Any) -> GameStatus:
        if isinstance(status_value, GameStatus):
            return status_value
        if isinstance(status_value, GameStatusDB):
            raw = status_value.value
        elif isinstance(status_value, Enum):
            raw = status_value.value
        else:
            raw = str(status_value or "")

        mapping = {
            GameStatusDB.CREATED.value: GameStatus.CREATED,
            "CREATED": GameStatus.CREATED,
            "created": GameStatus.CREATED,
            GameStatusDB.STARTED.value: GameStatus.IN_PROGRESS,
            getattr(GameStatusDB, "IN_PROGRESS", GameStatusDB.STARTED).value: GameStatus.IN_PROGRESS,
            getattr(GameStatusDB, "ROUND_IN_PROGRESS", GameStatusDB.STARTED).value: GameStatus.IN_PROGRESS,
            getattr(GameStatusDB, "ROUND_COMPLETED", GameStatusDB.STARTED).value: GameStatus.IN_PROGRESS,
            "started": GameStatus.IN_PROGRESS,
            "IN_PROGRESS": GameStatus.IN_PROGRESS,
            "in_progress": GameStatus.IN_PROGRESS,
            (GameStatusDB.FINISHED.value if hasattr(GameStatusDB, "FINISHED") else "FINISHED"): GameStatus.COMPLETED,
            "finished": GameStatus.COMPLETED,
            "completed": GameStatus.COMPLETED,
            "COMPLETED": GameStatus.COMPLETED,
            "PAUSED": GameStatus.PAUSED,
            "paused": GameStatus.PAUSED,
        }

        for token in {raw, raw.upper(), raw.lower()}:
            normalized = mapping.get(token)
            if normalized:
                return normalized

        if raw in GameStatus.__members__:
            return GameStatus[raw]

        if raw in GameStatus._value2member_map_:
            return GameStatus(raw)

        return GameStatus.CREATED

    @staticmethod
    def _compute_updated_at(game: Game) -> datetime:
        for attr in ("updated_at", "finished_at", "completed_at", "started_at", "created_at"):
            value = getattr(game, attr, None)
            if value:
                return value
        return datetime.utcnow()

    def _serialize_game(self, game: Any) -> GameInDBBase:
        config = self._coerce_dict(getattr(game, "config", {}) or {})
        demand_pattern_source = getattr(game, "demand_pattern", None) or config.get("demand_pattern") or DEFAULT_DEMAND_PATTERN
        try:
            demand_pattern = normalize_demand_pattern(demand_pattern_source)
        except Exception:
            demand_pattern = normalize_demand_pattern(DEFAULT_DEMAND_PATTERN)

        group_id = getattr(game, "group_id", None) or config.get("group_id")
        if group_id is not None and "group_id" not in config:
            config["group_id"] = group_id

        payload: Dict[str, Any] = {
            "id": game.id,
            "name": getattr(game, "name", f"Game {game.id}") or f"Game {game.id}",
            "status": self._map_status_to_schema(getattr(game, "status", None)),
            "current_round": getattr(game, "current_round", 0) or 0,
            "max_rounds": getattr(game, "max_rounds", 0) or 0,
            "demand_pattern": demand_pattern,
            "created_at": getattr(game, "created_at", datetime.utcnow()),
            "updated_at": self._compute_updated_at(game),
            "started_at": getattr(game, "started_at", None),
            "completed_at": getattr(game, "completed_at", None) or getattr(game, "finished_at", None),
            "created_by": getattr(game, "created_by", None),
            "group_id": group_id,
            "config": config,
            "players": [],
        }

        if config.get("pricing_config"):
            payload["pricing_config"] = config["pricing_config"]
        if config.get("node_policies"):
            payload["node_policies"] = config["node_policies"]
        if config.get("system_config"):
            payload["system_config"] = config["system_config"]
        if config.get("global_policy"):
            payload["global_policy"] = config["global_policy"]
        if config.get("daybreak_llm"):
            payload["daybreak_llm"] = config["daybreak_llm"]

        try:
            return GameInDBBase.model_validate(payload)
        except Exception:
            # Fallback to minimal payload if custom config fails validation
            for key in ("pricing_config", "node_policies", "system_config", "global_policy"):
                payload.pop(key, None)
            payload["demand_pattern"] = normalize_demand_pattern(DEFAULT_DEMAND_PATTERN)
            return GameInDBBase.model_validate(payload)
    
    def create_game(self, game_data: GameCreate, created_by: int = None) -> Game:
        """Create a new game with mixed human/agent players.

        Persists extended configuration into Game.config JSON to avoid schema changes.
        """
        # Create the game
        normalized_pattern = (
            normalize_demand_pattern(game_data.demand_pattern.dict())
            if game_data.demand_pattern
            else normalize_demand_pattern(DEFAULT_DEMAND_PATTERN)
        )

        config: Dict[str, Any] = {
            "demand_pattern": normalized_pattern,
            "pricing_config": game_data.pricing_config.dict() if hasattr(game_data, 'pricing_config') else {},
            "node_policies": (game_data.node_policies or {}),
            "system_config": (game_data.system_config or {}),
            "global_policy": (game_data.global_policy or {}),
        }
        if getattr(game_data, "daybreak_llm", None):
            config["daybreak_llm"] = game_data.daybreak_llm.model_dump()
        game = Game(
            name=game_data.name,
            max_rounds=game_data.max_rounds,
            status=GameStatus.CREATED,
            config=config,
        )
        self.db.add(game)
        self.db.flush()

        # Persist creator/metadata into columns if present in schema (fallback-safe via raw SQL)
        try:
            from sqlalchemy import text
            dp = json.dumps(normalized_pattern) if getattr(game_data, 'demand_pattern', None) else None
            desc = getattr(game_data, 'description', None)
            is_public = getattr(game_data, 'is_public', True)
            self.db.execute(
                text(
                    "UPDATE games SET description = :desc, is_public = :is_public, demand_pattern = :dp, created_by = :creator WHERE id = :id"
                ),
                {
                    "desc": desc,
                    "is_public": bool(is_public),
                    "dp": dp,
                    "creator": created_by if created_by is not None else None,
                    "id": game.id,
                },
            )
        except Exception as _e:
            # Non-fatal: older schemas may not have these columns
            pass
        
        # Create players based on assignments
        # Validate node policies against system ranges (if provided/persisted)
        sys_cfg = read_system_cfg()
        rng = sys_cfg.dict() if sys_cfg else {}
        def _check_range(key: str, val: float):
            r = rng.get(key)
            if not r:
                return
            lo, hi = r.get('min'), r.get('max')
            if lo is not None and val < lo: 
                raise ValueError(f"{key} below minimum {lo}")
            if hi is not None and val > hi:
                raise ValueError(f"{key} above maximum {hi}")
        for node, pol in (game_data.node_policies or {}).items():
            _check_range('info_delay', pol.info_delay)
            _check_range('ship_delay', pol.ship_delay)
            _check_range('init_inventory', pol.init_inventory)
            _check_range('price', pol.price)
            _check_range('standard_cost', pol.standard_cost)
            _check_range('variable_cost', pol.variable_cost)
            _check_range('min_order_qty', pol.min_order_qty)

        cfg = game.config if game.config else {}

        for i, assignment in enumerate(game_data.player_assignments):
            is_ai = assignment.player_type == PlayerType.AGENT
            player = Player(
                game_id=game.id,
                role=assignment.role,
                name=f"{assignment.role.capitalize()} ({'AI' if is_ai else 'Human'})",
                is_ai=is_ai,
                ai_strategy=(assignment.strategy.value if hasattr(assignment.strategy, 'value') else str(assignment.strategy)) if is_ai else None,
                can_see_demand=assignment.can_see_demand,
                llm_model=assignment.llm_model if is_ai else None,
                user_id=assignment.user_id if not is_ai else None
            )
            self.db.add(player)

            # Initialize inventory for the player
            inventory = PlayerInventory(
                player=player,
                current_stock=12,
                incoming_shipments=[],
                backorders=0
            )
            self.db.add(inventory)

            # Initialize AI agent if this is an AI player
            if is_ai:
                try:
                    agent_type = AgentType(assignment.role.lower())
                    strategy_value = (
                        assignment.strategy.value
                        if hasattr(assignment.strategy, "value")
                        else str(assignment.strategy)
                    )
                    try:
                        strategy = AgentStrategyEnum(strategy_value.lower())
                    except ValueError:
                        # Map any llm_* strategy to generic LLM
                        if strategy_value.lower().startswith("llm"):
                            strategy = AgentStrategyEnum.LLM
                        else:
                            raise
                    override_pct = None
                    if strategy == AgentStrategyEnum.DAYBREAK_DTCE_CENTRAL:
                        override_pct = assignment.daybreak_override_pct
                        if override_pct is not None:
                            overrides = cfg.setdefault("daybreak_overrides", {})
                            overrides[assignment.role.value] = override_pct
                    self.agent_manager.set_agent_strategy(
                        agent_type,
                        strategy,
                        llm_model=assignment.llm_model,
                        override_pct=override_pct,
                    )
                except Exception:
                    # Fallback: ignore if mapping not supported
                    pass

        game.config = cfg

        self.db.commit()
        self.db.refresh(game)
        return game

    def update_game_config(
        self,
        game_id: int,
        node_policies: Optional[Dict[str, Any]] = None,
        system_config: Optional[Dict[str, Any]] = None,
        pricing_config: Optional[Dict[str, Any]] = None,
        global_policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        game = self.db.query(Game).filter(Game.id == game_id).first()
        if not game:
            raise ValueError("Game not found")

        cfg: Dict[str, Any] = dict(game.config or {})
        # Validate ranges if provided
        sys_cfg = read_system_cfg()
        rng = (sys_cfg.dict() if sys_cfg else {})
        def _check_range(key: str, val: float):
            r = rng.get(key)
            if not r:
                return
            lo, hi = r.get('min'), r.get('max')
            if lo is not None and val < lo: 
                raise ValueError(f"{key} below minimum {lo}")
            if hi is not None and val > hi:
                raise ValueError(f"{key} above maximum {hi}")

        if node_policies:
            for _, pol in node_policies.items():
                for k in ['info_delay','ship_delay','init_inventory','price','standard_cost','variable_cost','min_order_qty']:
                    if k in pol and pol[k] is not None:
                        _check_range(k, float(pol[k]))
            cfg['node_policies'] = node_policies

        if system_config:
            cfg['system_config'] = system_config

        if pricing_config:
            cfg['pricing_config'] = pricing_config

        if global_policy:
            for k, v in global_policy.items():
                if v is not None:
                    _check_range(k, float(v))
            cfg['global_policy'] = global_policy

        game.config = cfg
        self.db.add(game)
        self.db.commit()
    
    def start_game(self, game_id: int) -> Game:
        """Start a game, initializing the first round."""
        game = self.db.query(Game).filter(Game.id == game_id).first()
        if not game:
            raise ValueError("Game not found")
            
        if game.status != GameStatus.CREATED:
            raise ValueError("Game has already started")
            
        # Update game status
        game.status = GameStatus.IN_PROGRESS
        game.current_round = 0  # Will be incremented in start_new_round
        game.started_at = datetime.utcnow()
        
        # Initialize simple engine state if not present
        cfg = game.config or {}
        raw_policies = cfg.get('node_policies', {})
        if not raw_policies:
            fallback_roles = ['retailer', 'wholesaler', 'distributor', 'manufacturer']
            raw_policies = {
                role: {"info_delay": 2, "ship_delay": 2, "init_inventory": 12, "min_order_qty": 0}
                for role in fallback_roles
            }
        node_policies = {
            self._normalise_key(name): dict(policy)
            for name, policy in raw_policies.items()
        }
        cfg['node_policies'] = node_policies

        raw_types = cfg.get('node_types') or {}
        node_types = {
            self._normalise_key(name): str(node_type).lower()
            for name, node_type in raw_types.items()
        }
        cfg['node_types'] = node_types

        lanes = cfg.get('lanes') or []
        self._validate_lanes(node_policies, lanes)

        engine: Dict[str, Dict[str, Any]] = {}
        lane_views = self._build_lane_views(node_policies, cfg)
        for node in lane_views['all_nodes']:
            self._ensure_engine_node(engine, node_policies, node)

        cfg['engine_state'] = engine
        game.config = cfg
        
        # Start the first round
        self.start_new_round(game)
        
        self.db.commit()
        self.db.refresh(game)
        return game
    
    def stop_game(self, game_id: int) -> Game:
        """Stop a game that is in progress."""
        game = self.db.query(Game).filter(Game.id == game_id).first()
        if not game:
            raise ValueError("Game not found")
            
        if game.status != GameStatus.IN_PROGRESS:
            raise ValueError("Game is not in progress")
            
        game.status = GameStatus.COMPLETED
        game.completed_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(game)
        return game

    def delete_game(self, game_id: int, current_user: User) -> Dict[str, Any]:
        """Delete a game if the requester is allowed to manage it."""
        game = self.db.query(Game).filter(Game.id == game_id).first()
        if not game:
            raise ValueError("Game not found")

        user_type = self._resolve_user_type(current_user)
        if not current_user.is_superuser and user_type != UserTypeEnum.SYSTEM_ADMIN:
            group_id = getattr(current_user, "group_id", None)
            owns_group = group_id and group_id == getattr(game, "group_id", group_id)
            config_group = None
            cfg = self._coerce_dict(getattr(game, "config", {}) or {})
            if cfg:
                config_group = cfg.get("group_id")
            if not owns_group and config_group not in (group_id, None):
                raise PermissionError("Not enough permissions to delete this game")

        self.db.delete(game)
        self.db.commit()
        return {"status": "deleted", "game_id": game_id}
    
    def start_new_round(self, game: Game) -> GameRound:
        """Start a new round of the game."""
        # Complete the current round if there is one
        current_round = self.get_current_round(game.id)
        if current_round:
            self.complete_round(current_round)
        
        # Increment round number
        game.current_round += 1
        
        # Check if game is over
        if game.current_round > game.max_rounds:
            game.status = GameStatus.COMPLETED
            game.completed_at = datetime.utcnow()
            self.db.commit()
            return None
        
        # Determine demand for the round (market nodes)
        demand_value = self.calculate_demand(game, game.current_round)

        # Create new round record
        round = GameRound(
            game_id=game.id,
            round_number=game.current_round,
            customer_demand=demand_value,
            started_at=datetime.utcnow(),
        )
        self.db.add(round)
        self.db.flush()

        cfg = game.config or {}
        engine = cfg.get('engine_state', {})
        node_policies = cfg.get('node_policies', {})
        global_policy = cfg.get('global_policy', {})
        lane_views = self._build_lane_views(node_policies, cfg)
        node_types_map = lane_views.get('node_types', {})
        all_nodes = lane_views['all_nodes'] or list(node_policies.keys())

        # Ensure engine entries exist for all nodes referenced by lanes
        for node in all_nodes:
            self._ensure_engine_node(engine, node_policies, node)

        market_demand_nodes_config = set(
            n for n in lane_views['market_nodes'] if n in engine
        )
        market_demand_nodes_types = {
            node for node, node_type in node_types_map.items() if node_type == 'market_demand'
        }
        market_demand_nodes = market_demand_nodes_config or market_demand_nodes_types
        if not market_demand_nodes and all_nodes:
            market_demand_nodes = {all_nodes[-1]}

        market_supply_nodes = {
            node for node, node_type in node_types_map.items() if node_type == 'market_supply'
        }

        hold_cost = float(global_policy.get('holding_cost', 0.5))
        back_cost = float(global_policy.get('backlog_cost', 1.0))

        demand_inputs: Dict[str, int] = defaultdict(int)
        demand_detail_inputs: Dict[str, Dict[str, int]] = {}

        for node in market_demand_nodes:
            if node in engine:
                demand_inputs[node] += int(demand_value)

        shipments_map = lane_views.get('shipments_map', {})
        orders_map = lane_views.get('orders_map', {})
        lanes_by_upstream = lane_views.get('lanes_by_upstream', {})
        node_sequence = lane_views['node_sequence'] or all_nodes

        processing_order: List[str] = []
        seen_nodes: Set[str] = set()
        for node in reversed(node_sequence):
            if node in engine and node not in seen_nodes:
                processing_order.append(node)
                seen_nodes.add(node)
        for node in reversed(all_nodes):
            if node in engine and node not in seen_nodes:
                processing_order.append(node)
                seen_nodes.add(node)

        arrivals_map: Dict[str, int] = {}
        for node in all_nodes:
            state = engine[node]
            policy = node_policies.get(node, {})
            info_delay = max(0, int(policy.get('info_delay', 0)))
            if info_delay <= 0:
                state['info_queue'] = []
                state['info_detail_queue'] = []
            else:
                queue = state.get('info_queue')
                if not isinstance(queue, list):
                    queue = []
                queue = [int(x) for x in queue]
                if len(queue) < info_delay:
                    queue = [0] * (info_delay - len(queue)) + queue
                elif len(queue) > info_delay:
                    queue = queue[-info_delay:]
                state['info_queue'] = queue

                detail_queue = state.get('info_detail_queue')
                cleaned_detail: List[Dict[str, int]] = []
                if isinstance(detail_queue, list):
                    for entry in detail_queue:
                        if isinstance(entry, dict):
                            cleaned_detail.append({str(k): int(v) for k, v in entry.items()})
                        else:
                            cleaned_detail.append({})
                while len(cleaned_detail) < info_delay:
                    cleaned_detail.insert(0, {})
                if len(cleaned_detail) > info_delay:
                    cleaned_detail = cleaned_detail[-info_delay:]
                state['info_detail_queue'] = cleaned_detail

            ship_delay = max(0, int(policy.get('ship_delay', 0)))
            ship_queue = state.get('ship_queue')
            if ship_delay <= 0:
                state['ship_queue'] = []
                arriving = 0
            else:
                if not isinstance(ship_queue, list):
                    ship_queue = [0] * ship_delay
                else:
                    ship_queue = [int(x) for x in ship_queue]
                if len(ship_queue) < ship_delay:
                    ship_queue = [0] * (ship_delay - len(ship_queue)) + ship_queue
                elif len(ship_queue) > ship_delay:
                    ship_queue = ship_queue[-ship_delay:]
                arriving = ship_queue.pop(0) if ship_queue else 0
                state['ship_queue'] = ship_queue

            arrivals_map[node] = int(arriving)
            state['incoming_shipments'] = int(arriving)
            state['last_arrival'] = int(arriving)

            backlog_breakdown = state.get('backlog_breakdown')
            if not isinstance(backlog_breakdown, dict):
                backlog_breakdown = {}
            else:
                backlog_breakdown = {
                    str(k): int(v)
                    for k, v in backlog_breakdown.items()
                    if int(v) > 0
                }
            state['backlog_breakdown'] = backlog_breakdown
            state['backlog'] = int(state.get('backlog', 0))
            state['inventory'] = int(state.get('inventory', 0))
            state['on_order'] = int(state.get('on_order', 0))

        shipments_inbound: Dict[str, int] = defaultdict(int)

        for node in processing_order:
            if node not in engine:
                continue

            state = engine[node]
            policy = node_policies.get(node, {})
            node_type = node_types_map.get(node, '')
            arriving = arrivals_map.get(node, 0)
            available_inventory = max(0, state.get('inventory', 0)) + arriving

            info_delay = max(0, int(policy.get('info_delay', 0)))
            incoming_raw = int(demand_inputs.get(node, 0))
            incoming_detail_raw = demand_detail_inputs.get(node, {})
            if info_delay > 0:
                queue = state.get('info_queue', [])
                queue.append(incoming_raw)
                incoming_visible = queue.pop(0) if queue else 0
                state['info_queue'] = queue

                detail_queue = state.get('info_detail_queue', [])
                detail_queue.append({str(k): int(v) for k, v in incoming_detail_raw.items() if int(v) > 0})
                matured_detail = detail_queue.pop(0) if detail_queue else {}
                state['info_detail_queue'] = detail_queue
            else:
                incoming_visible = incoming_raw
                matured_detail = {
                    str(k): int(v)
                    for k, v in incoming_detail_raw.items()
                    if int(v) > 0
                }
                state['info_queue'] = []
                state['info_detail_queue'] = []

            demand_inputs.pop(node, None)
            demand_detail_inputs.pop(node, None)

            incoming_visible = max(0, int(incoming_visible))
            state['incoming_orders'] = incoming_visible

            downstream_nodes = shipments_map.get(node, [])
            backlog_breakdown = state.get('backlog_breakdown', {})
            backlog_breakdown = {
                str(k): int(v)
                for k, v in backlog_breakdown.items()
                if int(v) > 0
            }
            backlog_value = int(state.get('backlog', 0))
            if sum(backlog_breakdown.values()) != backlog_value:
                if backlog_value > 0 and downstream_nodes:
                    equal_share_backlog = backlog_value // len(downstream_nodes)
                    backlog_remainder = backlog_value % len(downstream_nodes)
                    backlog_breakdown = {}
                    for idx, downstream in enumerate(downstream_nodes):
                        allocation = equal_share_backlog + (1 if idx < backlog_remainder else 0)
                        if allocation > 0:
                            backlog_breakdown[downstream] = allocation
                else:
                    backlog_breakdown = {}
            state['backlog_breakdown'] = backlog_breakdown
            backlog_total_prev = backlog_value
            total_demand_value = backlog_total_prev + incoming_visible

            demand_distribution: Dict[str, int] = {}
            for downstream, amount in backlog_breakdown.items():
                if amount > 0:
                    demand_distribution[downstream] = demand_distribution.get(downstream, 0) + amount
            for downstream, amount in matured_detail.items():
                qty = int(amount)
                if qty > 0:
                    demand_distribution[downstream] = demand_distribution.get(downstream, 0) + qty
            if downstream_nodes:
                for downstream in downstream_nodes:
                    demand_distribution.setdefault(downstream, 0)
            distribution_sum = sum(demand_distribution.values())
            difference = total_demand_value - distribution_sum
            if difference > 0 and downstream_nodes:
                equal_share = difference // len(downstream_nodes)
                remainder = difference % len(downstream_nodes)
                for idx, downstream in enumerate(downstream_nodes):
                    add_amount = equal_share + (1 if idx < remainder else 0)
                    if add_amount > 0:
                        demand_distribution[downstream] = demand_distribution.get(downstream, 0) + add_amount
                distribution_sum = sum(demand_distribution.values())

            shipments_per_downstream: Dict[str, int] = {}
            if node_type == 'market_demand' or not downstream_nodes:
                ship_total = min(available_inventory, total_demand_value)
                state['inventory'] = max(0, available_inventory - ship_total)
                state['backlog'] = max(0, total_demand_value - ship_total)
                state['backlog_breakdown'] = {}
            elif node_type == 'market_supply':
                for downstream in downstream_nodes:
                    shipments_per_downstream[downstream] = demand_distribution.get(downstream, 0)
                ship_total = sum(shipments_per_downstream.values())
                state['inventory'] = 0
                state['backlog'] = max(0, total_demand_value - ship_total)
                state['backlog_breakdown'] = {}
            else:
                remaining_inventory = available_inventory
                backlog_remaining: Dict[str, int] = {}
                for lane in lanes_by_upstream.get(node, []):
                    downstream = lane['to']
                    need = demand_distribution.get(downstream, 0) - shipments_per_downstream.get(downstream, 0)
                    if need <= 0:
                        continue
                    capacity = lane.get('capacity')
                    if capacity is not None:
                        try:
                            need = min(need, int(capacity))
                        except (TypeError, ValueError):
                            pass
                    ship_qty = min(remaining_inventory, max(0, need))
                    if ship_qty <= 0:
                        continue
                    shipments_per_downstream[downstream] = shipments_per_downstream.get(downstream, 0) + ship_qty
                    remaining_inventory -= ship_qty
                    if remaining_inventory <= 0:
                        break
                ship_total = sum(shipments_per_downstream.values())
                for downstream, demand_value in demand_distribution.items():
                    sent = shipments_per_downstream.get(downstream, 0)
                    remaining = max(0, demand_value - sent)
                    if remaining > 0:
                        backlog_remaining[downstream] = remaining
                state['inventory'] = max(0, remaining_inventory)
                state['backlog'] = sum(backlog_remaining.values())
                state['backlog_breakdown'] = backlog_remaining

            state['last_shipment_planned'] = int(ship_total)
            for downstream, qty in shipments_per_downstream.items():
                if qty > 0:
                    shipments_inbound[downstream] += int(qty)

            if node_type == 'market_supply':
                state['holding_cost'] = 0.0
                state['backorder_cost'] = 0.0
                state['total_cost'] = 0.0
            elif node_type == 'market_demand':
                state['holding_cost'] = 0.0
                state['backorder_cost'] = state['backlog'] * back_cost
                state['total_cost'] = state['backorder_cost']
            else:
                state['holding_cost'] += state['inventory'] * hold_cost
                state['backorder_cost'] += state['backlog'] * back_cost
                state['total_cost'] = state['holding_cost'] + state['backorder_cost']

            order_qty = 0
            if node_type not in {'market_demand', 'market_supply'}:
                target_inventory = int(policy.get('init_inventory', 12) + 2 * policy.get('ship_delay', 2))
                desired = target_inventory + state['backlog'] - state['inventory'] - state.get('on_order', 0)
                order_qty = max(0, int(desired))
                moq = int(policy.get('min_order_qty', 0) or 0)
                if moq:
                    order_qty = ((order_qty + moq - 1) // moq) * moq
                state['last_order'] = order_qty
                on_order = int(state.get('on_order', 0))
                on_order = on_order - arrivals_map.get(node, 0) + order_qty
                state['on_order'] = max(0, on_order)
            else:
                state['last_order'] = 0
                state['on_order'] = 0

            if order_qty > 0:
                upstream_nodes = orders_map.get(node, [])
                if upstream_nodes:
                    share = order_qty // len(upstream_nodes)
                    remainder = order_qty % len(upstream_nodes)
                    for idx, upstream in enumerate(upstream_nodes):
                        allocation = share + (1 if idx < remainder else 0)
                        if allocation <= 0:
                            continue
                        demand_inputs[upstream] += allocation
                        detail_map = demand_detail_inputs.setdefault(upstream, {})
                        detail_map[node] = detail_map.get(node, 0) + allocation

        for node in all_nodes:
            state = engine[node]
            policy = node_policies.get(node, {})
            ship_delay = max(0, int(policy.get('ship_delay', 0)))
            planned = int(shipments_inbound.get(node, 0))
            if ship_delay > 0:
                queue = state.get('ship_queue', [])
                if not isinstance(queue, list):
                    queue = []
                while len(queue) < ship_delay - 1:
                    queue.insert(0, 0)
                queue.append(planned)
                state['ship_queue'] = queue
            elif planned > 0:
                state['incoming_shipments'] = state.get('incoming_shipments', 0) + planned
                state['last_arrival'] = state['incoming_shipments']
                state['inventory'] = state.get('inventory', 0) + planned

        cfg['engine_state'] = engine
        game.config = cfg

        self.db.commit()
        return round
    
    def process_ai_players(self, game: Game, game_round: GameRound) -> None:
        """Process AI players' moves for the current round."""
        players = self.db.query(Player).filter(
            Player.game_id == game.id,
            Player.is_ai == True
        ).all()

        if not players:
            return

        cfg = game.config or {}
        node_policies = cfg.get("node_policies", {})
        lane_views = self._build_lane_views(node_policies, cfg)
        node_types_map = lane_views.get("node_types", {})
        shipments_map = lane_views.get("shipments_map", {})
        node_sequence = lane_views.get("node_sequence") or lane_views.get("all_nodes", [])

        # Pre-fetch previous round orders for context sharing
        previous_round = self.db.query(GameRound).filter(
            GameRound.game_id == game.id,
            GameRound.round_number == game_round.round_number - 1
        ).first()
        previous_orders = [
            pr.order_placed for pr in previous_round.player_rounds
        ] if previous_round else []

        # Group players by their associated node/role key
        node_to_players: Dict[str, List[Player]] = defaultdict(list)
        for player in players:
            role_token = MixedGameService._normalise_key(
                getattr(player.role, "value", player.role)
            )
            node_to_players[role_token].append(player)

        # Evaluate nodes from downstream to upstream so downstream orders are
        # available as demand input for upstream agents.
        processing_nodes: List[str] = []
        for node in reversed(node_sequence):
            if node in node_to_players and node not in processing_nodes:
                processing_nodes.append(node)
        for node in node_to_players.keys():
            if node not in processing_nodes:
                processing_nodes.append(node)

        node_orders: Dict[str, int] = {}
        orders_by_role: Dict[str, int] = {}

        for node_key in processing_nodes:
            node_players = node_to_players.get(node_key, [])
            if not node_players:
                continue

            node_type = node_types_map.get(node_key, "")
            if node_type == "market_demand":
                # Demand sinks do not generate orders themselves.
                continue

            downstream_nodes = shipments_map.get(node_key, [])
            downstream_orders: Dict[str, int] = {}
            for downstream in downstream_nodes:
                downstream_key = MixedGameService._normalise_key(downstream)
                downstream_orders[downstream_key] = node_orders.get(downstream_key, 0)

            demand_from_downstream = sum(downstream_orders.values())
            if node_type == "retailer":
                current_demand_value = int(game_round.customer_demand or 0)
            else:
                current_demand_value = int(demand_from_downstream)
            if current_demand_value < 0:
                current_demand_value = 0

            accumulated_node_order = 0

            for player in node_players:
                role_token = MixedGameService._normalise_key(
                    getattr(player.role, "value", player.role)
                )
                agent_type = AgentType(role_token)

                strategy_value = (player.ai_strategy or "naive").lower()
                try:
                    strategy_enum = AgentStrategyEnum(strategy_value)
                except ValueError:
                    if strategy_value.startswith("llm"):
                        strategy_enum = AgentStrategyEnum.LLM
                    else:
                        strategy_enum = AgentStrategyEnum.NAIVE

                override_pct = None
                if strategy_enum == AgentStrategyEnum.DAYBREAK_DTCE_CENTRAL:
                    overrides = (cfg or {}).get("daybreak_overrides", {})
                    override_pct = overrides.get(role_token)

                self.agent_manager.set_agent_strategy(
                    agent_type,
                    strategy_enum,
                    llm_model=player.llm_model,
                    override_pct=override_pct,
                )
                agent = self.agent_manager.get_agent(agent_type)

                inventory = self.db.query(PlayerInventory).filter(
                    PlayerInventory.player_id == player.id
                ).first()

                incoming_shipments = []
                if inventory and getattr(inventory, "incoming_shipments", None):
                    incoming_shipments = inventory.incoming_shipments
                    if not isinstance(incoming_shipments, list):
                        incoming_shipments = []

                local_state = {
                    "inventory": getattr(
                        inventory,
                        "current_stock",
                        getattr(inventory, "current_inventory", 0) if inventory else 0,
                    ),
                    "backlog": getattr(
                        inventory,
                        "backorders",
                        getattr(inventory, "current_backlog", 0) if inventory else 0,
                    ),
                    "incoming_shipments": incoming_shipments,
                }

                llm_payload = None
                if strategy_enum == AgentStrategyEnum.LLM:
                    llm_payload = build_llm_decision_payload(
                        self.db,
                        game,
                        round_number=game_round.round_number,
                        action_role=role_token,
                    )

                upstream_context = {
                    "previous_orders": previous_orders,
                    "previous_orders_by_role": dict(orders_by_role),
                    "downstream_orders": dict(downstream_orders),
                }
                if llm_payload is not None:
                    upstream_context["llm_payload"] = llm_payload

                order_quantity = agent.make_decision(
                    current_round=game_round.round_number,
                    current_demand=current_demand_value,
                    upstream_data=upstream_context,
                    local_state=local_state,
                )

                inventory_stock = getattr(inventory, "current_stock", 0) if inventory else 0
                inventory_backorders = getattr(inventory, "backorders", 0) if inventory else 0

                player_round = PlayerRound(
                    player_id=player.id,
                    round_id=game_round.id,
                    order_placed=order_quantity,
                    order_received=0,
                    inventory_before=inventory_stock,
                    inventory_after=inventory_stock,
                    backorders_before=inventory_backorders,
                    backorders_after=inventory_backorders,
                    comment=agent.get_last_explanation_comment(),
                )
                self.db.add(player_round)

                accumulated_node_order += order_quantity

            node_orders[node_key] = accumulated_node_order
            orders_by_role[node_key] = accumulated_node_order
    
    def complete_round(self, game_round: GameRound) -> None:
        """Complete the current round, updating player inventories and costs."""
        # Get all player rounds for this game round
        player_rounds = self.db.query(PlayerRound).filter(
            PlayerRound.round_id == game_round.id
        ).all()
        
        for pr in player_rounds:
            # Get player's inventory
            inventory = self.db.query(PlayerInventory).filter(
                PlayerInventory.player_id == pr.player_id
            ).first()
            
            # Update inventory based on orders received
            # (This is a simplified version - actual implementation would consider lead times)
            pr.inventory_after = inventory.current_stock - pr.order_placed
            if pr.inventory_after < 0:
                pr.backorders_after = abs(pr.inventory_after)
                pr.inventory_after = 0
            
            # Calculate costs (simplified)
            pr.holding_cost = pr.inventory_after * 0.5  # $0.5 per unit per round
            pr.backorder_cost = pr.backorders_after * 2  # $2 per backorder
            pr.total_cost = pr.holding_cost + pr.backorder_cost
            
            # Update inventory for next round
            inventory.current_stock = pr.inventory_after
            inventory.backorders = pr.backorders_after
        
        timestamp = datetime.utcnow()
        game_round.ended_at = timestamp
        game_round.completed_at = timestamp
        self.db.commit()
    
    def get_current_round(self, game_id: int) -> Optional[GameRound]:
        """Get the current round for a game."""
        return self.db.query(GameRound).filter(
            GameRound.game_id == game_id,
            GameRound.ended_at.is_(None)
        ).first()

    def finish_game(self, game_id: int) -> Game:
        game = self.db.query(Game).filter(Game.id == game_id).first()
        if not game:
            raise ValueError("Game not found")
        game.status = GameStatus.COMPLETED
        self.db.commit(); self.db.refresh(game)
        return game

    def get_report(self, game_id: int) -> Dict[str, Any]:
        game = self.db.query(Game).filter(Game.id == game_id).first()
        if not game:
            raise ValueError("Game not found")
        cfg = game.config or {}
        engine = cfg.get('engine_state', {})
        totals = {r: {
            'inventory': st.get('inventory',0),
            'backlog': st.get('backlog',0),
            'holding_cost': st.get('holding_cost',0.0),
            'backorder_cost': st.get('backorder_cost',0.0),
            'total_cost': st.get('total_cost',0.0),
        } for r, st in engine.items()}
        total_cost = sum(v['total_cost'] for v in totals.values())
        return {'game_id': game_id, 'status': str(game.status), 'totals': totals, 'total_cost': total_cost}
    
    def calculate_demand(self, game: Game, round_number: int) -> int:
        """Calculate demand for a given round based on the game's demand pattern."""
        pattern = normalize_demand_pattern(game.demand_pattern or {})
        pattern_type = pattern.get('type', 'classic')
        params = pattern.get('params', {}) if isinstance(pattern.get('params', {}), dict) else {}

        if pattern_type == 'classic':
            initial = params.get('initial_demand', DEFAULT_CLASSIC_PARAMS['initial_demand'])
            final = params.get('final_demand', DEFAULT_CLASSIC_PARAMS['final_demand'])
            change_week = params.get('change_week', DEFAULT_CLASSIC_PARAMS['change_week'])
            return final if round_number >= change_week else initial

        elif pattern_type == 'random':
            min_demand = params.get('min_demand', 1)
            max_demand = params.get('max_demand', 10)
            return random.randint(min_demand, max_demand)

        # Default demand
        return DEFAULT_CLASSIC_PARAMS['initial_demand']
    
    def list_games(
        self,
        current_user: User,
        status: Optional[GameStatus] = None,
    ) -> List[GameInDBBase]:
        """Return games visible to the requesting user, handling legacy schemas."""

        columns = set(self._get_game_columns())
        base_projection = [
            ("id", "id"),
            ("name", "name"),
            ("status", "status"),
            ("current_round", "current_round"),
            ("max_rounds", "max_rounds"),
            ("created_at", "created_at"),
            ("updated_at", "updated_at"),
            ("started_at", "started_at"),
            ("completed_at", "completed_at"),
            ("finished_at", "finished_at"),
            ("demand_pattern", "demand_pattern"),
            ("config", "config"),
            ("created_by", "created_by"),
            ("group_id", "group_id"),
        ]

        select_parts: List[str] = []
        for column_name, alias in base_projection:
            if column_name in columns:
                select_parts.append(f"g.{column_name} AS {alias}")
            else:
                select_parts.append(f"NULL AS {alias}")

        select_clause = ", ".join(select_parts)
        query = f"SELECT {select_clause} FROM games g"

        filters: List[str] = []
        params: Dict[str, Any] = {}

        if status:
            status_values = [
                value
                for value in self._schema_status_to_db_values(status)
                if value is not None
            ]
            if status_values:
                placeholders = []
                for idx, value in enumerate(status_values):
                    key = f"status_{idx}"
                    placeholders.append(f":{key}")
                    params[key] = value
                filters.append(f"g.status IN ({', '.join(placeholders)})")

        user_type = self._resolve_user_type(current_user)
        if not current_user.is_superuser and user_type != UserTypeEnum.SYSTEM_ADMIN:
            group_id = getattr(current_user, "group_id", None)
            if user_type == UserTypeEnum.GROUP_ADMIN and group_id and "group_id" in columns:
                filters.append("g.group_id = :group_id")
                params["group_id"] = group_id
            elif "created_by" in columns:
                filters.append("g.created_by = :created_by")
                params["created_by"] = current_user.id

        if filters:
            query += " WHERE " + " AND ".join(filters)

        order_column = "created_at" if "created_at" in columns else "id"
        query += f" ORDER BY g.{order_column} DESC"

        from sqlalchemy import text

        result = self.db.execute(text(query), params)

        games: List[GameInDBBase] = []
        for row in result:
            record = dict(row._mapping)
            if record.get("completed_at") is None and record.get("finished_at") is not None:
                record["completed_at"] = record.get("finished_at")
            games.append(self._serialize_game(SimpleNamespace(**record)))

        return games
    
    def get_game_state(self, game_id: int) -> GameState:
        """Get the current state of a game."""
        from sqlalchemy import text
        
        # Get the game
        game_query = """
            SELECT id, name, status, current_round, max_rounds, 
                   created_at, updated_at, demand_pattern, config
            FROM games 
            WHERE id = :game_id
        """
        game_result = self.db.execute(text(game_query), {"game_id": game_id}).first()
        
        if not game_result:
            raise ValueError("Game not found")
        
        # Get all players for the game
        players_query = """
            SELECT p.id, p.name, p.role, p.is_ai, 
                   COALESCE(pi.current_stock, 0) as current_stock,
                   COALESCE(pi.incoming_shipments, '[]') as incoming_shipments,
                   COALESCE(pi.backorders, 0) as backorders
            FROM players p
            LEFT JOIN player_inventories pi ON p.id = pi.player_id
            WHERE p.game_id = :game_id
        """
        players_result = self.db.execute(text(players_query), {"game_id": game_id})
        
        player_states = []
        for player in players_result:
            player_states.append(PlayerState(
                id=player[0],
                name=player[1],
                role=player[2],
                is_ai=player[3],
                current_stock=player[4],
                incoming_shipments=player[5],
                backorders=player[6],
                total_cost=0  # Would be calculated from player rounds
            ))
        
        current_round = self.get_current_round(game_id)
        
        # Create a default demand pattern if none exists
        try:
            raw_pattern = json.loads(game_result[7]) if game_result[7] else DEFAULT_DEMAND_PATTERN.copy()
        except (json.JSONDecodeError, TypeError):
            raw_pattern = DEFAULT_DEMAND_PATTERN.copy()
        demand_pattern = normalize_demand_pattern(raw_pattern)

        # Unpack optional config
        node_policies = {}
        system_config = {}
        pricing_config = {}
        global_policy = {}
        try:
            cfg = json.loads(game_result[8]) if len(game_result) > 8 and game_result[8] else {}
            if isinstance(cfg, dict):
                node_policies = cfg.get('node_policies', {})
                system_config = cfg.get('system_config', {})
                pricing_config = cfg.get('pricing_config', {})
                global_policy = cfg.get('global_policy', {})
            # Also surface nested in demand_pattern.params if present
            if not node_policies:
                node_policies = demand_pattern.get('params', {}).get('node_policies', {}) if isinstance(demand_pattern, dict) else {}
            if not system_config:
                system_config = demand_pattern.get('params', {}).get('system_config', {}) if isinstance(demand_pattern, dict) else {}
        except Exception:
            pass
            
        return GameState(
            id=game_result[0],
            name=game_result[1],
            status=game_result[2],
            current_round=game_result[3],
            max_rounds=game_result[4],
            players=player_states,
            current_demand=None,  # Will be set by the round
            round_started_at=None,  # Will be set by the round
            round_ends_at=None,  # Will be set by the round
            created_at=game_result[5],
            updated_at=game_result[6],
            started_at=None,  # Not in schema
            completed_at=None,  # Not in schema
            created_by=None,  # Not in schema
            is_public=False,  # Default value
            description="",  # Default value
            demand_pattern=demand_pattern,
            node_policies=node_policies,
            system_config=system_config,
            pricing_config=pricing_config,
            global_policy=global_policy
        )
        # Validate optional global policy if provided
        if getattr(game_data, 'global_policy', None):
            gp = game_data.global_policy
            for k in ['info_delay','ship_delay','init_inventory','holding_cost','backlog_cost','max_inbound_per_link','max_order']:
                if k in gp and gp[k] is not None:
                    _check_range(k, float(gp[k]))
