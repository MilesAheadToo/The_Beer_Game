import sys
import os
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.supply_chain import Game, Player, GameRound, PlayerRole, GameStatus
from app.schemas.game import GameCreate, PlayerCreate
from app.services.game_service import GameService
from app.core.demand_patterns import DemandPatternType

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def setup_database():
    # Create all tables
    from app.db.base import Base
    Base.metadata.create_all(bind=engine)

def create_test_game():
    db = TestingSessionLocal()
    
    try:
        # Create a test game with classical demand pattern
        game_service = GameService(db)
        game_data = GameCreate(
            name="Test Demand Pattern Game",
            max_rounds=10  # Test 10 rounds
        )
        
        # Create game
        game = game_service.create_game(game_data)
        
        # Add test players
        roles = [
            PlayerRole.RETAILER,
            PlayerRole.WHOLESALER,
            PlayerRole.DISTRIBUTOR,
            PlayerRole.FACTORY
        ]
        
        for role in roles:
            player_data = PlayerCreate(
                user_id=1,  # Assuming user 1 exists
                role=role,
                name=f"Test {role.value.capitalize()}",
                is_ai=False
            )
            game_service.add_player(game.id, player_data)
        
        return game.id
    finally:
        db.close()

def test_demand_pattern():
    # Setup test database
    setup_database()
    
    # Create test game and players
    game_id = create_test_game()
    
    db = TestingSessionLocal()
    try:
        game_service = GameService(db)
        
        # Start the game
        game = game_service.start_game(game_id)
        
        # Test demand for each round
        expected_demands = [4, 4, 4, 4, 4, 8, 8, 8, 8, 8]  # First 5 rounds: 4, then 8
        
        for round_num in range(1, game.max_rounds + 1):
            if round_num > 1:
                # Submit dummy orders to advance the round
                players = db.query(Player).filter(Player.game_id == game_id).all()
                for player in players:
                    game_service.submit_order(game_id, player.id, 4)  # Order 4 units each round
                
                # Advance to next round
                game = game_service.advance_round(game_id)
            
            # Get current round
            current_round = db.query(GameRound).filter(
                GameRound.game_id == game_id,
                GameRound.round_number == game.current_round
            ).first()
            
            # Verify demand matches expected pattern
            expected_demand = expected_demands[round_num - 1]  # 0-based index
            assert current_round.customer_demand == expected_demand, \
                f"Round {round_num}: Expected demand {expected_demand}, got {current_round.customer_demand}"
            
            print(f"Round {round_num}: Demand = {current_round.customer_demand} (Expected: {expected_demand})")
        
        print("\n✅ All demand pattern tests passed!")
        
    finally:
        # Clean up
        db.close()
        # Remove test database
        if os.path.exists("test.db"):
            os.remove("test.db")

if __name__ == "__main__":
    test_demand_pattern()
