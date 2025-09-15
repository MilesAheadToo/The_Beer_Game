from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import random
import json

from app.models.game import Game, GameStatus
from app.models.player import Player, PlayerRole
from app.models.supply_chain import PlayerInventory, Order, GameRound, PlayerRound
from app.schemas.game import GameCreate, GameUpdate, GameState, PlayerState, GameStatus
from app.schemas.player import PlayerAssignment, PlayerType, PlayerStrategy
from app.services.agents import AgentManager, AgentType, AgentStrategy as AgentStrategyEnum
from app.api.endpoints.config import _read_cfg as read_system_cfg

class MixedGameService:
    """Service for managing games with mixed human and AI players."""
    
    def __init__(self, db: Session):
        self.db = db
        self.agent_manager = AgentManager()
    
    def create_game(self, game_data: GameCreate, created_by: int = None) -> Game:
        """Create a new game with mixed human/agent players.

        Persists extended configuration into Game.config JSON to avoid schema changes.
        """
        # Create the game
        config: Dict[str, Any] = {
            "demand_pattern": game_data.demand_pattern.dict() if game_data.demand_pattern else {},
            "pricing_config": game_data.pricing_config.dict() if hasattr(game_data, 'pricing_config') else {},
            "node_policies": (game_data.node_policies or {}),
            "system_config": (game_data.system_config or {}),
            "global_policy": (game_data.global_policy or {}),
        }
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
            dp = json.dumps(game_data.demand_pattern.dict()) if getattr(game_data, 'demand_pattern', None) else None
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
        node_policies = cfg.get('node_policies', {})
        roles = ['retailer','wholesaler','distributor','manufacturer','factory']
        # allow different naming in node_policies
        if not node_policies:
            node_policies = {r: {"info_delay": 2, "ship_delay": 2, "init_inventory": 12, "min_order_qty": 0} for r in roles}
        engine = {
            r: {
                "inventory": int(node_policies.get(r, {}).get("init_inventory", 12)),
                "backlog": 0,
                "on_order": 0,
                "info_queue": [0] * int(node_policies.get(r, {}).get("info_delay", 2)),
                "ship_queue": [0] * int(node_policies.get(r, {}).get("ship_delay", 2)),
                "last_order": 0,
                "holding_cost": 0.0,
                "backorder_cost": 0.0,
                "total_cost": 0.0,
            } for r in node_policies.keys()
        }
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
        
        # Get demand for this round (retailer)
        demand = self.calculate_demand(game, game.current_round)
        
        # Create new round
        round = GameRound(
            game_id=game.id,
            round_number=game.current_round,
            customer_demand=demand,
            started_at=datetime.utcnow()
        )
        self.db.add(round)
        self.db.flush()
        
        # Simple engine step using node_policies and global_policy costs
        cfg = game.config or {}
        engine = cfg.get('engine_state', {})
        node_policies = cfg.get('node_policies', {})
        global_policy = cfg.get('global_policy', {})
        hold_cost = float(global_policy.get('holding_cost', 0.5))
        back_cost = float(global_policy.get('backlog_cost', 1.0))
        # roles chain inferred from node_policies order
        chain = list(node_policies.keys())
        if not chain:
            chain = ['retailer','wholesaler','distributor','manufacturer','factory']
        # 1) incoming orders at retailer
        if 'retailer' in engine:
            engine['retailer']['info_queue'].append(int(demand))
        # 2) propagate orders upstream
        for i in range(1, len(chain)):
            dn = chain[i-1]; up = chain[i]
            qty = engine[dn]['last_order'] if dn in engine else 0
            engine[up]['info_queue'].append(int(qty))
        # 3) process info delays
        for r in chain:
            if r not in engine: continue
            q = engine[r]['info_queue'].pop(0) if engine[r]['info_queue'] else 0
            engine[r]['incoming_orders'] = q
        # 4) ship downstream (limited by inventory + ship capacity via ship_queue length)
        for i in range(len(chain)-2, -1, -1):
            up = chain[i+1]; dn = chain[i]
            if up not in engine or dn not in engine: continue
            demand_here = engine[dn]['incoming_orders'] + engine[dn]['backlog']
            ship_cap = None  # could read from node_policies
            can_ship = min(engine[up]['inventory'], demand_here)
            if ship_cap is not None:
                can_ship = min(can_ship, ship_cap)
            engine[dn]['ship_queue'].append(int(can_ship))
        # 5) process ship delays (arrivals)
        for r in chain:
            if r not in engine: continue
            arriving = engine[r]['ship_queue'].pop(0) if engine[r]['ship_queue'] else 0
            inv = engine[r]['inventory'] + arriving
            demand_here = engine[r]['incoming_orders'] + engine[r]['backlog']
            shipped = min(inv, demand_here)
            engine[r]['inventory'] = inv - shipped
            engine[r]['backlog'] = max(0, demand_here - shipped)
            # costs
            engine[r]['holding_cost'] += engine[r]['inventory'] * hold_cost
            engine[r]['backorder_cost'] += engine[r]['backlog'] * back_cost
            engine[r]['total_cost'] = engine[r]['holding_cost'] + engine[r]['backorder_cost']
        # 6) place orders based on simple heuristics (base-stock)
        for r in chain:
            if r not in engine: continue
            st = engine[r]
            pol = node_policies.get(r, {})
            target = int(pol.get('init_inventory', 12) + 2 * pol.get('ship_delay', 2))
            desired = target + st['backlog'] - st['inventory'] - st['on_order']
            order = max(0, int(desired))
            moq = int(pol.get('min_order_qty', 0))
            if moq:
                order = ((order + moq - 1) // moq) * moq
            st['last_order'] = order
            st['on_order'] = max(0, st['on_order'] + order - st.get('incoming_shipments', 0))
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
        
        for player in players:
            # Get the AI agent for this player
            agent_type = AgentType(player.role.lower())
            strategy_value = player.ai_strategy or "naive"
            try:
                strategy_enum = AgentStrategyEnum(strategy_value.lower())
            except ValueError:
                if strategy_value.lower().startswith("llm"):
                    strategy_enum = AgentStrategyEnum.LLM
                else:
                    strategy_enum = AgentStrategyEnum.NAIVE

            override_pct = None
            if strategy_enum == AgentStrategyEnum.DAYBREAK_DTCE_CENTRAL:
                overrides = (game.config or {}).get("daybreak_overrides", {})
                role_key = player.role.value if hasattr(player.role, "value") else str(player.role).lower()
                override_pct = overrides.get(role_key)

            self.agent_manager.set_agent_strategy(
                agent_type,
                strategy_enum,
                llm_model=player.llm_model,
                override_pct=override_pct,
            )
            agent = self.agent_manager.get_agent(agent_type)

            # Get player's current state
            inventory = self.db.query(PlayerInventory).filter(
                PlayerInventory.player_id == player.id
            ).first()
            
            # Get previous round's data for the agent
            previous_round = self.db.query(GameRound).filter(
                GameRound.game_id == game.id,
                GameRound.round_number == game_round.round_number - 1
            ).first()
            
            incoming_shipments = []
            if hasattr(inventory, "incoming_shipments") and inventory.incoming_shipments:
                incoming_shipments = inventory.incoming_shipments
                if not isinstance(incoming_shipments, list):
                    incoming_shipments = []

            local_state = {
                "inventory": getattr(inventory, "current_stock", getattr(inventory, "current_inventory", 0)),
                "backlog": getattr(inventory, "backorders", getattr(inventory, "current_backlog", 0)),
                "incoming_shipments": incoming_shipments,
            }

            # Make decision based on agent's strategy
            order_quantity = agent.make_decision(
                current_round=game_round.round_number,
                current_demand=game_round.customer_demand if player.can_see_demand else None,
                upstream_data={
                    'previous_orders': [pr.order_placed for pr in previous_round.player_rounds] if previous_round else []
                },
                local_state=local_state,
            )

            # Create player round record
            player_round = PlayerRound(
                player_id=player.id,
                round_id=game_round.id,
                order_placed=order_quantity,
                order_received=0,  # Will be updated when upstream ships
                inventory_before=inventory.current_stock,
                inventory_after=inventory.current_stock,  # Will be updated after processing
                backorders_before=inventory.backorders,
                backorders_after=inventory.backorders  # Will be updated after processing
            )
            self.db.add(player_round)
    
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
        
        game_round.ended_at = datetime.utcnow()
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
        pattern = game.demand_pattern or {}
        pattern_type = pattern.get('type', 'classic')
        params = pattern.get('params', {})
        
        if pattern_type == 'classic':
            stable_period = params.get('stable_period', 5)
            step_increase = params.get('step_increase', 4)
            
            if round_number <= stable_period:
                return 4  # Base demand
            else:
                return 4 + step_increase
        
        elif pattern_type == 'random':
            min_demand = params.get('min_demand', 1)
            max_demand = params.get('max_demand', 10)
            return random.randint(min_demand, max_demand)
        
        # Default demand
        return 4
    
    def list_games(self, status: Optional[GameStatus] = None) -> List[Dict[str, Any]]:
        """List all games, optionally filtered by status."""
        from sqlalchemy import text
        
        # Build the base query
        query = """
            SELECT 
                g.id, g.name, g.status, g.current_round, g.max_rounds,
                g.created_at, g.updated_at, g.started_at, g.completed_at,
                g.is_public, g.description, g.created_by, g.demand_pattern, g.config,
                (SELECT COUNT(*) FROM players WHERE game_id = g.id) as player_count
            FROM games g
        """
        
        # Add status filter if provided
        params = {}
        if status:
            query += " WHERE g.status = :status"
            params["status"] = status
            
        # Execute the query
        result = self.db.execute(text(query), params)
        
        # Convert result to list of dicts matching GameInDBBase schema
        games = []
        for row in result:
            try:
                demand_pattern = json.loads(row[11]) if row[11] else {"type": "classic", "params": {}}
            except (json.JSONDecodeError, TypeError):
                demand_pattern = {"type": "classic", "params": {}}
            # Unpack optional config
            node_policies = {}
            system_config = {}
            try:
                cfg = json.loads(row[12]) if row[12] else {}
                if isinstance(cfg, dict):
                    node_policies = cfg.get('node_policies', {})
                    system_config = cfg.get('system_config', {})
                    pricing_config = cfg.get('pricing_config', {})
                    global_policy = cfg.get('global_policy', {})
                    if not node_policies:
                        node_policies = demand_pattern.get('params', {}).get('node_policies', {}) if isinstance(demand_pattern, dict) else {}
                    if not system_config:
                        system_config = demand_pattern.get('params', {}).get('system_config', {}) if isinstance(demand_pattern, dict) else {}
            except Exception:
                pass
                
            game_data = {
                "id": row[0],
                "name": row[1],
                "status": row[2],
                "current_round": row[3] or 0,
                "max_rounds": row[4],
                "demand_pattern": demand_pattern,
                "created_at": row[5],
                "updated_at": row[6],
                "started_at": row[7],
                "completed_at": row[8],
                "is_public": bool(row[9]) if row[9] is not None else False,
                "description": row[10] or "",
                "created_by": row[12],
                "node_policies": node_policies,
                "system_config": system_config,
                "pricing_config": pricing_config if 'pricing_config' in locals() else {},
                "global_policy": global_policy if 'global_policy' in locals() else {},
                "players": []  # Will be populated separately if needed
            }
            
            # Ensure all required fields have values
            for field in ["current_round", "max_rounds"]:
                if game_data[field] is None:
                    game_data[field] = 0
                    
            games.append(game_data)
            
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
            demand_pattern = json.loads(game_result[7]) if game_result[7] else {}
        except (json.JSONDecodeError, TypeError):
            demand_pattern = {}

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
