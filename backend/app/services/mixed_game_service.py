from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import random

from app.models.game import Game, GameStatus
from app.models.player import Player, PlayerRole
from app.models.supply_chain import PlayerInventory, Order, GameRound, PlayerRound
from app.schemas.game import GameCreate, GameUpdate, GameState, PlayerState, GameStatus
from app.schemas.player import PlayerAssignment, PlayerType, PlayerStrategy
from app.services.agents import AgentManager, AgentType, AgentStrategy as AgentStrategyEnum

class MixedGameService:
    """Service for managing games with mixed human and AI players."""
    
    def __init__(self, db: Session):
        self.db = db
        self.agent_manager = AgentManager()
    
    def create_game(self, game_data: GameCreate, created_by: int = None) -> Game:
        """Create a new game with mixed human/agent players."""
        # Create the game
        game = Game(
            name=game_data.name,
            max_rounds=game_data.max_rounds,
            status=GameStatus.CREATED,
            demand_pattern=game_data.demand_pattern.dict(),
            created_by=created_by,
            is_public=game_data.is_public,
            description=game_data.description
        )
        self.db.add(game)
        self.db.flush()
        
        # Create players based on assignments
        for i, assignment in enumerate(game_data.player_assignments):
            is_ai = assignment.player_type == PlayerType.AGENT
            player = Player(
                game_id=game.id,
                role=assignment.role,
                name=f"{assignment.role.capitalize()} ({'AI' if is_ai else 'Human'})",
                is_ai=is_ai,
                ai_strategy=assignment.strategy.value if is_ai else None,
                can_see_demand=assignment.can_see_demand,
                user_id=assignment.user_id if not is_ai else None
            )
            self.db.add(player)
            
            # Initialize inventory for the player
            inventory = PlayerInventory(
                player=player,
                current_stock=12,  # Starting inventory
                incoming_shipments=[],
                backorders=0
            )
            self.db.add(inventory)
            
            # Initialize AI agent if this is an AI player
            if is_ai:
                agent_type = AgentType(assignment.role.lower())
                strategy = AgentStrategyEnum(assignment.strategy.lower())
                self.agent_manager.set_agent_strategy(agent_type, strategy)
        
        self.db.commit()
        self.db.refresh(game)
        return game
    
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
        
        # Get demand for this round
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
        
        # Let AI players make their moves
        self.process_ai_players(game, round)
        
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
            
            # Make decision based on agent's strategy
            order_quantity = agent.make_decision(
                current_round=game_round.round_number,
                current_demand=game_round.customer_demand if player.can_see_demand else None,
                upstream_data={
                    'previous_orders': [pr.order_placed for pr in previous_round.player_rounds] if previous_round else []
                }
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
                g.created_at, g.updated_at, NULL as completed_at,
                FALSE as is_public, '' as description,
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
        
        # Convert result to list of dicts
        games = []
        for row in result:
            games.append({
                "id": row[0],
                "name": row[1],
                "status": row[2],
                "current_round": row[3],
                "max_rounds": row[4],
                "created_at": row[5],
                "started_at": row[6],
                "completed_at": row[7],
                "is_public": row[8],
                "description": row[9],
                "player_count": row[10]
            })
            
        return games
    
    def get_game_state(self, game_id: int) -> GameState:
        """Get the current state of a game."""
        from sqlalchemy import text
        
        # Get the game
        game_query = """
            SELECT id, name, status, current_round, max_rounds, 
                   created_at, updated_at, demand_pattern
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
            demand_pattern=demand_pattern
        )
