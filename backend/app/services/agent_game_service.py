from typing import List, Dict, Optional, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, select

from app.models.supply_chain import (
    Game, Player, PlayerInventory, Order, GameRound, PlayerRound, 
    GameStatus, PlayerRole
)
from app.db.session import SessionLocal, get_db
from app.schemas.game import GameCreate, PlayerCreate, GameState, PlayerState, DemandPattern
from app.services.agents import AgentManager, AgentType, AgentStrategy
from app.core.demand_patterns import (
    normalize_demand_pattern,
    DEFAULT_DEMAND_PATTERN,
    DEFAULT_CLASSIC_PARAMS,
)

class AgentGameService:
    """Service for managing the Beer Game with AI agents."""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.agent_manager = AgentManager(can_see_demand=False)
    
    def create_game(self, game_data: GameCreate) -> Game:
        """Create a new game with AI agents."""
        pattern_config = (
            normalize_demand_pattern(game_data.demand_pattern.dict())
            if game_data.demand_pattern
            else normalize_demand_pattern(DEFAULT_DEMAND_PATTERN)
        )

        game = Game(
            name=game_data.name,
            status=GameStatus.CREATED,
            current_round=0,
            max_rounds=game_data.max_rounds,
            demand_pattern=pattern_config
        )
        self.db.add(game)
        self.db.commit()
        self.db.refresh(game)
        
        # Create AI players for each role
        self._create_ai_players(game.id)
        
        return game
    
    def _create_ai_players(self, game_id: int):
        """Create AI players for all roles."""
        roles = [
            (PlayerRole.RETAILER, "AI Retailer"),
            (PlayerRole.WHOLESALER, "AI Wholesaler"),
            (PlayerRole.DISTRIBUTOR, "AI Distributor"),
            (PlayerRole.MANUFACTURER, "AI Manufacturer")
        ]
        
        for role, name in roles:
            player = Player(
                game_id=game_id,
                name=name,
                role=role,
                is_ai=True
            )
            self.db.add(player)
        
        self.db.commit()
    
    def start_game(self, game_id: int) -> Game:
        """Start the game and initialize the first round."""
        game = self.db.query(Game).filter(Game.id == game_id).first()
        if not game:
            raise ValueError("Game not found")
            
        game.status = GameStatus.IN_PROGRESS
        game.current_round = 1
        game.started_at = datetime.utcnow()
        
        # Initialize player inventories
        players = self.db.query(Player).filter(Player.game_id == game_id).all()
        for player in players:
            inventory = PlayerInventory(
                player_id=player.id,
                current_inventory=12,  # Starting inventory
                current_backlog=0,
                incoming_shipment=0,
                outgoing_shipment=0
            )
            self.db.add(inventory)
        
        self.db.commit()
        self.db.refresh(game)
        return game
    
    def play_round(self, game_id: int) -> Dict:
        """Play one round of the game with AI agents making decisions."""
        game = self.db.query(Game).filter(Game.id == game_id).first()
        if not game:
            raise ValueError("Game not found")
            
        if game.status != GameStatus.IN_PROGRESS:
            raise ValueError("Game is not in progress")
            
        # Get current demand based on the pattern
        demand_pattern = DemandPattern(**game.demand_pattern)
        current_demand = self._get_current_demand(game.current_round, demand_pattern)
        
        # Get all players in the correct order (Retailer -> Wholesaler -> Distributor -> Manufacturer)
        players = (
            self.db.query(Player)
            .filter(Player.game_id == game_id)
            .order_by(Player.role)
            .all()
        )
        
        # Process each player's turn
        for player in players:
            self._process_player_turn(player, game, current_demand)
        
        # Advance to next round
        game.current_round += 1
        if game.current_round > game.max_rounds:
            game.status = GameStatus.COMPLETED
            game.completed_at = datetime.utcnow()
        
        self.db.commit()
        return self.get_game_state(game_id)
    
    def _process_player_turn(self, player: Player, game: Game, current_demand: int):
        """Process a single player's turn using AI agent."""
        # Get the AI agent for this player
        agent_type = AgentType(player.role.value)
        agent = self.agent_manager.get_agent(agent_type)
        
        # Get player's current state
        inventory = (
            self.db.query(PlayerInventory)
            .filter(PlayerInventory.player_id == player.id)
            .first()
        )
        
        # Get previous round's data for the agent
        previous_round = (
            self.db.query(GameRound)
            .filter(
                GameRound.game_id == game.id,
                GameRound.round_number == game.current_round - 1
            )
            .first()
        )
        
        # Make decision based on agent's strategy
        incoming_shipments = getattr(inventory, "incoming_shipments", None)
        if incoming_shipments is None:
            single = getattr(inventory, "incoming_shipment", None)
            if single is not None:
                incoming_shipments = [single]
        if not isinstance(incoming_shipments, list):
            incoming_shipments = []

        local_state = {
            "inventory": getattr(inventory, "current_inventory", getattr(inventory, "current_stock", 0)),
            "backlog": getattr(inventory, "current_backlog", getattr(inventory, "backorders", 0)),
            "incoming_shipments": incoming_shipments,
        }

        previous_orders: List[int] = []
        previous_orders_by_role: Dict[str, int] = {}
        if previous_round:
            for pr in previous_round.player_rounds:
                quantity = getattr(pr, "order_quantity", getattr(pr, "order_placed", 0))
                try:
                    previous_orders.append(int(quantity))
                except (TypeError, ValueError):
                    continue
                role_key: Optional[str] = None
                if getattr(pr, "player", None) is not None:
                    role_obj = getattr(pr.player, "role", None)
                    if hasattr(role_obj, "value"):
                        role_key = str(role_obj.value)
                    elif isinstance(role_obj, str):
                        role_key = role_obj.lower()
                if role_key:
                    previous_orders_by_role[role_key] = int(quantity)

        order_quantity = agent.make_decision(
            current_round=game.current_round,
            current_demand=current_demand if player.role == PlayerRole.RETAILER else None,
            upstream_data={
                'previous_orders': previous_orders,
                'previous_orders_by_role': previous_orders_by_role,
            },
            local_state=local_state,
        )
        
        # Create new game round if it doesn't exist
        current_round = (
            self.db.query(GameRound)
            .filter(
                GameRound.game_id == game.id,
                GameRound.round_number == game.current_round
            )
            .first()
        )
        
        if not current_round:
            current_round = GameRound(
                game_id=game.id,
                round_number=game.current_round,
                demand=current_demand
            )
            self.db.add(current_round)
            self.db.flush()
        
        # Record player's action
        player_round = PlayerRound(
            round_id=current_round.id,
            player_id=player.id,
            order_quantity=order_quantity,
            received_quantity=0,  # Will be updated when upstream ships
            inventory=inventory.current_inventory,
            backlog=inventory.current_backlog,
            comment=agent.get_last_explanation_comment(),
        )
        self.db.add(player_round)
        
        # Update inventory based on the order
        # (In a real implementation, you'd update inventory based on lead times, etc.)
        inventory.current_inventory -= order_quantity
        if inventory.current_inventory < 0:
            inventory.current_backlog += abs(inventory.current_inventory)
            inventory.current_inventory = 0
        
        self.db.commit()
    
    def _get_current_demand(self, round_number: int, demand_pattern: DemandPattern) -> int:
        """Get the demand for the current round based on the pattern."""
        if not hasattr(demand_pattern, 'pattern') or not demand_pattern.pattern:
            # Generate pattern if not already generated
            self._generate_demand_pattern(demand_pattern)
        
        # Use pattern if available, otherwise default to 4
        pattern = getattr(demand_pattern, 'pattern', [])
        if round_number - 1 < len(pattern):
            return pattern[round_number - 1]
        return 4  # Default demand
    
    def _generate_demand_pattern(self, demand_pattern: DemandPattern):
        """Generate a demand pattern based on the game settings."""
        normalized = normalize_demand_pattern(demand_pattern.dict())
        if normalized.get('type') == 'classic':
            params = normalized.get('params', {})
            initial = params.get('initial_demand', DEFAULT_CLASSIC_PARAMS['initial_demand'])
            final = params.get('final_demand', DEFAULT_CLASSIC_PARAMS['final_demand'])
            change_week = params.get('change_week', DEFAULT_CLASSIC_PARAMS['change_week'])
            total_rounds = max(change_week, 20)
            pattern: List[int] = []
            for week in range(1, total_rounds + 1):
                pattern.append(final if week >= change_week else initial)

            demand_pattern.pattern = pattern
    
    def get_game_state(self, game_id: int) -> Dict:
        """Get the current state of the game."""
        game = self.db.query(Game).filter(Game.id == game_id).first()
        if not game:
            raise ValueError("Game not found")
            
        players = (
            self.db.query(Player)
            .filter(Player.game_id == game_id)
            .order_by(Player.role)
            .all()
        )
        
        player_states = []
        for player in players:
            inventory = (
                self.db.query(PlayerInventory)
                .filter(PlayerInventory.player_id == player.id)
                .first()
            )
            
            player_states.append({
                'id': player.id,
                'name': player.name,
                'role': player.role,
                'is_ai': player.is_ai,
                'inventory': inventory.current_inventory if inventory else 0,
                'backlog': inventory.current_backlog if inventory else 0,
                'incoming_shipment': inventory.incoming_shipment if inventory else 0,
                'outgoing_shipment': inventory.outgoing_shipment if inventory else 0
            })
        
        return {
            'game_id': game.id,
            'name': game.name,
            'status': game.status,
            'current_round': game.current_round,
            'max_rounds': game.max_rounds,
            'players': player_states,
            'demand_pattern': game.demand_pattern
        }
    
    def set_agent_strategy(
        self,
        role: str,
        strategy: str,
        llm_model: Optional[str] = None,
        override_pct: Optional[float] = None,
    ):
        """Set the strategy for an AI agent."""
        try:
            agent_type = AgentType(role.lower())
            strategy_enum = AgentStrategy(strategy.lower())
            self.agent_manager.set_agent_strategy(
                agent_type,
                strategy_enum,
                llm_model=llm_model,
                override_pct=override_pct,
            )
        except ValueError as e:
            raise ValueError(f"Invalid role or strategy: {e}")
    
    def set_demand_visibility(self, visible: bool):
        """Set whether agents can see the actual customer demand."""
        self.agent_manager.set_demand_visibility(visible)
