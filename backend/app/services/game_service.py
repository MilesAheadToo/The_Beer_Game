from typing import List, Dict, Optional, Any
import random
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.supply_chain import (
    Game, Player, PlayerInventory, Order, GameRound, PlayerRound, 
    GameStatus, PlayerRole
)
from app.schemas.game import GameCreate, PlayerCreate, GameState, PlayerState
from app.core.demand_patterns import get_demand_pattern, DemandPatternType

class GameService:
    # Cost parameters
    HOLDING_COST_PER_UNIT = 0.5  # Cost to hold one unit of inventory for one round
    BACKORDER_COST_PER_UNIT = 1.0  # Cost per unit backordered
    
    # Game parameters
    INITIAL_INVENTORY = 12
    INITIAL_ORDERS = 4  # Initial orders in the pipeline
    MIN_DEMAND = 0
    MAX_DEMAND = 8
    
    # Lead times for each role (in rounds)
    LEAD_TIMES = {
        PlayerRole.RETAILER: 2,
        PlayerRole.WHOLESALER: 2,
        PlayerRole.DISTRIBUTOR: 2,
        PlayerRole.FACTORY: 2
    }
    
    def __init__(self, db: Session):
        self.db = db
        self._demand_pattern = None  # Will be initialized per game
    
    def create_game(self, game_data: GameCreate) -> Game:
        """Create a new game with the given parameters."""
        # Convert DemandPattern model to dict for storage
        demand_pattern = game_data.demand_pattern.dict() if game_data.demand_pattern else {
            "type": DemandPatternType.CLASSIC,
            "params": {
                "stable_period": 5,
                "step_increase": 4
            }
        }
            
        game = Game(
            name=game_data.name,
            status=GameStatus.CREATED,
            current_round=0,
            max_rounds=game_data.max_rounds or 52,
            demand_pattern=demand_pattern
        )
        self.db.add(game)
        self.db.commit()
        self.db.refresh(game)
        return game
    
    def add_player(self, game_id: int, player_data: PlayerCreate) -> Player:
        """Add a player to the game."""
        # Check if the role is already taken in this game
        existing = self.db.query(Player).filter(
            Player.game_id == game_id,
            Player.role == player_data.role
        ).first()
        
        if existing:
            raise ValueError(f"A {player_data.role} already exists in this game")
        
        player = Player(
            game_id=game_id,
            user_id=player_data.user_id,
            role=player_data.role,
            name=player_data.name,
            is_ai=player_data.is_ai
        )
        
        # Initialize player inventory
        inventory = PlayerInventory(
            player=player,
            current_stock=self.INITIAL_INVENTORY,
            incoming_shipments=[],
            backorders=0,
            cost=0.0
        )
        
        self.db.add(player)
        self.db.add(inventory)
        self.db.commit()
        self.db.refresh(player)
        return player
    
    def start_game(self, game_id: int) -> Game:
        """Start the game and initialize the first round."""
        game = self.db.query(Game).filter(Game.id == game_id).first()
        if not game:
            raise ValueError("Game not found")
            
        if game.status != GameStatus.CREATED:
            raise ValueError("Game has already started or completed")
        
        # Pre-generate the entire demand pattern for the game
        self._demand_pattern = get_demand_pattern(
            pattern_config=game.demand_pattern,
            num_rounds=game.max_rounds
        )
        
        # Store the demand pattern in the game for persistence
        game.demand_pattern = {
            "type": game.demand_pattern.get("type", DemandPatternType.CLASSIC),
            "params": game.demand_pattern.get("params", {"stable_period": 5, "step_increase": 4}),
            "pattern": self._demand_pattern  # Store the full pattern
        }
        
        # Initialize first round with the first demand value
        game_round = GameRound(
            game_id=game_id,
            round_number=1,
            customer_demand=self._generate_demand(1, game.max_rounds)
        )
        
        game.status = GameStatus.IN_PROGRESS
        game.current_round = 1
        
        self.db.add(game_round)
        self.db.commit()
        self.db.refresh(game)
        return game
    
    def submit_order(self, game_id: int, player_id: int, order_quantity: int) -> PlayerRound:
        """Submit an order for the current round."""
        game = self.db.query(Game).filter(Game.id == game_id).first()
        if not game or game.status != GameStatus.IN_PROGRESS:
            raise ValueError("Game is not in progress")
            
        player = self.db.query(Player).filter(Player.id == player_id).first()
        if not player:
            raise ValueError("Player not found")
            
        # Get current round
        current_round = self.db.query(GameRound).filter(
            GameRound.game_id == game_id,
            GameRound.round_number == game.current_round
        ).first()
        
        if not current_round:
            raise ValueError("Current round not found")
            
        # Check if player has already submitted an order for this round
        existing_round = self.db.query(PlayerRound).filter(
            PlayerRound.player_id == player_id,
            PlayerRound.round_id == current_round.id
        ).first()
        
        if existing_round:
            raise ValueError("Player has already submitted an order for this round")
            
        # Create player round
        player_round = PlayerRound(
            player_id=player_id,
            round_id=current_round.id,
            order_placed=order_quantity,
            order_received=0,  # Will be updated when processing the round
            inventory_before=player.inventory.current_stock,
            inventory_after=player.inventory.current_stock,  # Will be updated
            backorders_before=player.inventory.backorders,
            backorders_after=player.inventory.backorders,  # Will be updated
            holding_cost=0.0,  # Will be calculated
            backorder_cost=0.0,  # Will be calculated
            total_cost=0.0  # Will be calculated
        )
        
        self.db.add(player_round)
        self.db.commit()
        self.db.refresh(player_round)
        
        # Check if all players have submitted orders
        if self._all_players_submitted(game_id, current_round.id):
            self.advance_round(game_id)
            
        return player_round
    
    def get_game_state(self, game_id: int) -> GameState:
        """Get the current state of the game."""
        game = self.db.query(Game).filter(Game.id == game_id).first()
        if not game:
            raise ValueError("Game not found")
            
        # Get current round
        current_round = self.db.query(GameRound).filter(
            GameRound.game_id == game_id,
            GameRound.round_number == game.current_round
        ).first()
        
        # Get all players and their states
        players = self.db.query(Player).filter(Player.game_id == game_id).all()
        player_states = []
        
        for player in players:
            player_round = None
            if current_round:
                player_round = self.db.query(PlayerRound).filter(
                    PlayerRound.player_id == player.id,
                    PlayerRound.round_id == current_round.id
                ).first()
                
            player_states.append(PlayerState(
                id=player.id,
                name=player.name,
                role=player.role,
                inventory=player.inventory.current_stock,
                backorders=player.inventory.backorders,
                order_placed=player_round.order_placed if player_round else None,
                order_received=player_round.order_received if player_round else None,
                cost=player_round.total_cost if player_round else 0.0
            ))
            
        # Include demand pattern info in the game state
        demand_pattern_info = {
            "type": game.demand_pattern.get("type", DemandPatternType.CLASSIC),
            "params": game.demand_pattern.get("params", {}),
            "current_demand": current_round.customer_demand if current_round else None,
            "next_demand": self._generate_demand(game.current_round + 1, game.max_rounds) 
                          if game.current_round < game.max_rounds else None
        }
            
        return GameState(
            id=game.id,
            name=game.name,
            status=game.status,
            current_round=game.current_round,
            max_rounds=game.max_rounds,
            players=player_states,
            customer_demand=current_round.customer_demand if current_round else None,
            demand_pattern=demand_pattern_info
        )
    
    def _all_players_submitted(self, game_id: int, round_id: int) -> bool:
        """Check if all players have submitted orders for the current round."""
        player_count = self.db.query(Player).filter(Player.game_id == game_id).count()
        submitted_count = self.db.query(PlayerRound).filter(
            PlayerRound.round_id == round_id
        ).count()
        
        return submitted_count >= player_count
    
    def advance_round(self, game_id: int) -> Game:
        """
        Advance the game to the next round.
        
        Args:
            game_id: The ID of the game to advance
            
        Returns:
            The updated game object
            
        Raises:
            ValueError: If the game is not in progress or current round is not found
        """
        game = self.db.query(Game).filter(Game.id == game_id).first()
        if not game or game.status != GameStatus.IN_PROGRESS:
            raise ValueError("Game is not in progress")
        
        # Process all players' turns for the current round
        current_round = self.db.query(GameRound).filter(
            GameRound.game_id == game_id,
            GameRound.round_number == game.current_round
        ).first()
        
        if not current_round:
            raise ValueError("Current round not found")
        
        # Process each player's turn
        players = self.db.query(Player).filter(Player.game_id == game_id).all()
        for player in players:
            self._process_player_turn(player, current_round)
        
        # Check if game is over
        if game.current_round >= game.max_rounds:
            game.status = GameStatus.COMPLETED
            self.db.commit()
            self.db.refresh(game)
            return game
            
        # Create next round
        game.current_round += 1
        
        try:
            # Get demand for the next round from the pre-generated pattern
            next_demand = self._generate_demand(game.current_round, game.max_rounds)
            
            # Create the next round
            next_round = GameRound(
                game_id=game_id,
                round_number=game.current_round,
                customer_demand=next_demand
            )
            self.db.add(next_round)
            
            # Update game status if this is the last round
            if game.current_round == game.max_rounds:
                game.status = GameStatus.COMPLETED
            
            self.db.commit()
            self.db.refresh(game)
            
            return game
            
        except Exception as e:
            # Rollback in case of error
            self.db.rollback()
            raise ValueError(f"Failed to advance to next round: {str(e)}")
    
    def _process_player_turn(self, player: Player, game_round: GameRound) -> None:
        """Process a single player's turn for the current round."""
        # Get the player's round data
        player_round = self.db.query(PlayerRound).filter(
            PlayerRound.player_id == player.id,
            PlayerRound.round_id == game_round.id
        ).first()
        
        if not player_round:
            # If player didn't submit an order, use 0
            player_round = PlayerRound(
                player_id=player.id,
                round_id=game_round.id,
                order_placed=0,
                order_received=0,
                inventory_before=player.inventory.current_stock,
                inventory_after=player.inventory.current_stock,
                backorders_before=player.inventory.backorders,
                backorders_after=player.inventory.backorders,
                holding_cost=0.0,
                backorder_cost=0.0,
                total_cost=0.0
            )
            self.db.add(player_round)
        
        # Process incoming shipments
        self._process_incoming_shipments(player.inventory, game_round.round_number)
        
        # Calculate inventory and backorder costs
        inventory = player.inventory.current_stock
        backorders = player.inventory.backorders
        
        holding_cost = max(0, inventory) * self.HOLDING_COST_PER_UNIT
        backorder_cost = backorders * self.BACKORDER_COST_PER_UNIT
        total_cost = holding_cost + backorder_cost
        
        # Update player round with costs
        player_round.holding_cost = holding_cost
        player_round.backorder_cost = backorder_cost
        player_round.total_cost = total_cost
        player_round.inventory_after = inventory
        player_round.backorders_after = backorders
        
        # Update player's inventory cost
        player.inventory.cost += total_cost
        
        # Place order with upstream player
        if player.role != PlayerRole.FACTORY:  # Factory doesn't place orders
            self._place_order(player, player_round.order_placed, game_round.round_number)
        
        self.db.commit()
    
    def _process_incoming_shipments(self, inventory: PlayerInventory, current_round: int) -> None:
        """Process any incoming shipments that have arrived."""
        # Get shipments that are due to arrive this round
        shipments_to_receive = []
        remaining_shipments = []
        
        for shipment in inventory.incoming_shipments:
            if shipment['arrival_round'] <= current_round:
                shipments_to_receive.append(shipment['quantity'])
            else:
                remaining_shipments.append(shipment)
        
        # Update inventory with received shipments
        total_received = sum(shipments_to_receive)
        inventory.current_stock += total_received
        inventory.incoming_shipments = remaining_shipments
    
    def _place_order(self, player: Player, quantity: int, current_round: int) -> None:
        """Place an order with the upstream player."""
        upstream_player = self._get_upstream_player(player)
        if not upstream_player:
            return  # No upstream player (shouldn't happen for valid games)
        
        # Calculate arrival round based on lead time
        lead_time = self.LEAD_TIMES.get(upstream_player.role, 1)
        arrival_round = current_round + lead_time
        
        # Add to upstream player's incoming shipments
        upstream_player.inventory.incoming_shipments.append({
            'quantity': quantity,
            'arrival_round': arrival_round
        })
    
    def _get_upstream_player(self, player: Player) -> Optional[Player]:
        """Get the player who is immediately upstream in the supply chain."""
        role_order = [
            PlayerRole.RETAILER,
            PlayerRole.WHOLESALER,
            PlayerRole.DISTRIBUTOR,
            PlayerRole.FACTORY
        ]
        
        current_index = role_order.index(player.role)
        if current_index > 0:
            upstream_role = role_order[current_index - 1]
            return self.db.query(Player).filter(
                Player.game_id == player.game_id,
                Player.role == upstream_role
            ).first()
        return None
    
    def _get_downstream_player(self, player: Player) -> Optional[Player]:
        """Get the player who is immediately downstream in the supply chain."""
        role_order = [
            PlayerRole.RETAILER,
            PlayerRole.WHOLESALER,
            PlayerRole.DISTRIBUTOR,
            PlayerRole.FACTORY
        ]
        
        current_index = role_order.index(player.role)
        if current_index < len(role_order) - 1:
            downstream_role = role_order[current_index + 1]
            return self.db.query(Player).filter(
                Player.game_id == player.game_id,
                Player.role == downstream_role
            ).first()
        return None
    
    def _generate_demand(self, round_number: int, max_rounds: int) -> int:
        """
        Generate demand for the specified round number using the pre-generated pattern.
        
        Args:
            round_number: The current round number (1-based)
            max_rounds: Total number of rounds in the game
            
        Returns:
            The demand for the specified round
            
        Raises:
            ValueError: If round_number is invalid
        """
        if round_number < 1 or round_number > max_rounds:
            raise ValueError(f"Invalid round number: {round_number}. Must be between 1 and {max_rounds}")
        
        # If we have a pre-generated pattern, use it
        if hasattr(self, '_demand_pattern') and self._demand_pattern:
            # Ensure we don't go out of bounds
            idx = min(round_number - 1, len(self._demand_pattern) - 1)
            return self._demand_pattern[idx]
            
        # Fallback to default classic pattern if not pre-generated
        stable_period = 5
        base_demand = 4
        step_increase = 4
        
        if round_number <= stable_period:
            return base_demand
        return base_demand + step_increase
