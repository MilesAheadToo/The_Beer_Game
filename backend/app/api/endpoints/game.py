from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.db.session import get_db
from app.models.supply_chain import Game as GameModel, Player, GameRound, PlayerRound
from app.schemas.game import (
    GameCreate, GameUpdate,
    PlayerCreate, PlayerUpdate, Player as PlayerSchema,
    GameState, OrderCreate, OrderResponse, PlayerRound as PlayerRoundSchema,
    GameRound as GameRoundSchema,
    DemandPattern
)
from app.core.demand_patterns import normalize_demand_pattern, DEFAULT_DEMAND_PATTERN

class PlayerResponse(PlayerSchema):
    """Response model for player data."""
    class Config:
        from_attributes = True

class PlayerRoundResponse(PlayerRoundSchema):
    """Response model for player round data."""
    class Config:
        from_attributes = True

class GameRoundResponse(GameRoundSchema):
    """Response model for game round data."""
    class Config:
        from_attributes = True
from pydantic import BaseModel
from typing import Dict, Any

class GameResponse(BaseModel):
    id: int
    name: str
    status: str
    current_round: int
    max_rounds: int
    demand_pattern: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    @classmethod
    def from_orm(cls, obj):
        # Safely handle demand_pattern conversion
        try:
            if hasattr(obj, 'demand_pattern') and obj.demand_pattern is not None:
                if isinstance(obj.demand_pattern, dict):
                    demand_pattern = dict(obj.demand_pattern)
                    demand_pattern.pop('model_config', None)
                elif hasattr(obj.demand_pattern, '__dict__'):
                    demand_pattern = {
                        k: v
                        for k, v in obj.demand_pattern.__dict__.items()
                        if not k.startswith('_') and k != 'model_config'
                    }
                else:
                    import json

                    try:
                        demand_pattern = json.loads(obj.demand_pattern) if isinstance(obj.demand_pattern, str) else {}
                        if isinstance(demand_pattern, dict):
                            demand_pattern.pop('model_config', None)
                    except json.JSONDecodeError:
                        demand_pattern = DEFAULT_DEMAND_PATTERN.copy()
            else:
                demand_pattern = DEFAULT_DEMAND_PATTERN.copy()

            normalized = normalize_demand_pattern(demand_pattern)

            clean_demand_pattern = {
                'type': str(normalized.get('type', 'classic')),
                'params': dict(normalized.get('params', {})) if isinstance(normalized.get('params', {}), dict) else {},
            }
            
            # Convert SQLAlchemy model to dict
            data = {
                'id': int(obj.id),
                'name': str(obj.name),
                'status': obj.status.value if hasattr(obj.status, 'value') else str(obj.status),
                'current_round': int(obj.current_round) if obj.current_round is not None else 0,
                'max_rounds': int(obj.max_rounds) if obj.max_rounds is not None else 20,
                'demand_pattern': clean_demand_pattern,
                'created_at': obj.created_at,
                'updated_at': obj.updated_at
            }
            
            return cls(**data)
            
        except Exception as e:
            print(f"Error in GameResponse.from_orm: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a minimal valid response if something goes wrong
            return cls(
                id=getattr(obj, 'id', 0),
                name=getattr(obj, 'name', 'Unknown'),
                status=str(getattr(obj, 'status', 'UNKNOWN')),
                current_round=int(getattr(obj, 'current_round', 0)),
                max_rounds=int(getattr(obj, 'max_rounds', 20)),
                demand_pattern={
                    'type': DEFAULT_DEMAND_PATTERN['type'],
                    'params': DEFAULT_DEMAND_PATTERN['params'].copy(),
                },
                created_at=getattr(obj, 'created_at', None),
                updated_at=getattr(obj, 'updated_at', None)
            )
from app.services.game_service import GameService
from app.core.security import get_current_user

router = APIRouter()

# Game endpoints
@router.post("/", response_model=GameResponse, status_code=status.HTTP_201_CREATED)
def create_game(
    game_in: GameCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new game.
    """
    game_service = GameService(db)
    game = game_service.create_game(game_in)
    return GameResponse.from_orm(game)

@router.get("/")
async def list_games(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    List all games - DEBUG VERSION
    """
    try:
        print("Fetching games from database...")
        # Get all games from the database
        games = db.query(GameModel).offset(skip).limit(limit).all()
        print(f"Found {len(games)} games in the database")
        
        if not games:
            print("No games found in the database")
            return []
            
        # Convert each game to a simple dictionary
        response = []
        for game in games:
            try:
                # Create a simple dictionary for the game
                game_dict = {
                    'id': game.id,
                    'name': game.name,
                    'status': game.status.value if hasattr(game.status, 'value') else str(game.status),
                    'current_round': game.current_round,
                    'max_rounds': game.max_rounds,
                    'created_at': game.created_at.isoformat() if game.created_at else None,
                    'updated_at': game.updated_at.isoformat() if game.updated_at else None
                }
                
                # Handle demand_pattern
                if hasattr(game, 'demand_pattern') and game.demand_pattern is not None:
                    if isinstance(game.demand_pattern, dict):
                        demand_pattern = dict(game.demand_pattern)
                        demand_pattern.pop('model_config', None)
                    elif hasattr(game.demand_pattern, '__dict__'):
                        demand_pattern = {
                            k: v
                            for k, v in game.demand_pattern.__dict__.items()
                            if not k.startswith('_') and k != 'model_config'
                        }
                    else:
                        import json

                        try:
                            demand_pattern = json.loads(game.demand_pattern) if isinstance(game.demand_pattern, str) else {}
                            if isinstance(demand_pattern, dict):
                                demand_pattern.pop('model_config', None)
                        except json.JSONDecodeError:
                            demand_pattern = DEFAULT_DEMAND_PATTERN.copy()
                else:
                    demand_pattern = DEFAULT_DEMAND_PATTERN.copy()

                game_dict['demand_pattern'] = normalize_demand_pattern(demand_pattern)
                response.append(game_dict)
                
            except Exception as e:
                print(f"Error processing game {getattr(game, 'id', 'unknown')}: {str(e)}")
                continue
                
        print(f"Successfully processed {len(response)} out of {len(games)} games")
        
        # Print the raw response for debugging
        import json
        print("\nRaw response data:")
        print(json.dumps(response, indent=2, default=str))
        
        return response
        
    except Exception as e:
        print(f"Error in list_games endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing games: {str(e)}"
        )

@router.get("/{game_id}", response_model=GameResponse)
def get_game(
    game_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get a game by ID.
    """
    game = db.query(GameModel).filter(GameModel.id == game_id).first()
    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game not found"
        )
    return GameResponse.from_orm(game)

@router.put("/{game_id}", response_model=GameResponse)
def update_game(
    game_id: int,
    game_in: GameUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Update a game.
    """
    game = db.query(GameModel).filter(GameModel.id == game_id).first()
    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game not found"
        )
    
    update_data = game_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(game, field, value)
    
    db.commit()
    db.refresh(game)
    return GameResponse.from_orm(game)

@router.post("/{game_id}/start", response_model=GameResponse)
def start_game(
    game_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Start a game that is in the 'created' state.
    """
    game_service = GameService(db)
    try:
        game = game_service.start_game(game_id)
        return GameResponse.from_orm(game)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/{game_id}/state", response_model=GameState)
def get_game_state(
    game_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get the current state of a game.
    """
    game_service = GameService(db)
    try:
        return game_service.get_game_state(game_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

# Player endpoints
@router.post("/{game_id}/players", response_model=PlayerResponse, status_code=status.HTTP_201_CREATED)
def add_player(
    game_id: int,
    player_in: PlayerCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Add a player to a game.
    """
    game_service = GameService(db)
    try:
        player = game_service.add_player(game_id, player_in)
        return PlayerResponse.model_validate(player)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/{game_id}/players", response_model=List[PlayerResponse])
def list_players(
    game_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    List all players in a game.
    """
    players = db.query(Player).filter(Player.game_id == game_id).all()
    return [PlayerResponse.model_validate(player) for player in players]

@router.get("/{game_id}/players/{player_id}", response_model=PlayerResponse)
def get_player(
    game_id: int,
    player_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get a player by ID.
    """
    player = db.query(Player).filter(
        Player.id == player_id,
        Player.game_id == game_id
    ).first()
    
    if not player:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Player not found"
        )
    
    return PlayerResponse.from_orm(player)

# Order endpoints
@router.post("/games/{game_id}/players/{player_id}/orders", response_model=PlayerRoundResponse)
async def submit_order(
    game_id: int,
    player_id: int,
    order_in: OrderCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Submit or update an order for the current round.
    
    Players can submit or update their order for the current round until the round ends.
    If the round time expires, any unsubmitted orders will be set to zero.
    """
    game_service = GameService(db)
    try:
        player_round = await game_service.submit_order(game_id, player_id, order_in.quantity, order_in.comment)
        return player_round
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Round endpoints
@router.get("/{game_id}/rounds", response_model=List[GameRoundResponse])
def list_rounds(
    game_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    List all rounds for a game.
    """
    rounds = db.query(GameRound).filter(GameRound.game_id == game_id).all()
    return [GameRoundResponse.model_validate(round) for round in rounds]

@router.get("/{game_id}/rounds/{round_number}", response_model=GameRoundResponse)
def get_round(
    game_id: int,
    round_number: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get a specific round for a game.
    """
    game_round = db.query(GameRound).filter(
        GameRound.game_id == game_id,
        GameRound.round_number == round_number
    ).first()
    
    if not game_round:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Round not found"
        )
    
    return GameRoundResponse.model_validate(game_round)

@router.get("/{game_id}/current-round", response_model=GameRoundResponse)
def get_current_round(
    game_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get the current round for a game.
    """
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game not found"
        )
    
    game_round = db.query(GameRound).filter(
        GameRound.game_id == game_id,
        GameRound.round_number == game.current_round
    ).first()
    
    if not game_round:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Current round not found"
        )
    
    return GameRoundResponse.model_validate(game_round)

@router.get("/games/{game_id}/rounds/current/status", response_model=Dict[str, Any])
async def get_round_submission_status(
    game_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get the submission status of the current round.
    
    Returns:
        A dictionary with submission status and details
    """
    game = db.query(GameModel).filter(GameModel.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Get current round
    current_round = db.query(GameRound).filter(
        GameRound.game_id == game_id,
        GameRound.round_number == game.current_round
    ).first()
    
    if not current_round:
        raise HTTPException(status_code=404, detail="Current round not found")
    
    # Get all players in the game
    players = db.query(Player).filter(Player.game_id == game_id).all()
    total_players = len(players)
    
    # Get players who have submitted for the current round
    submitted_players = db.query(PlayerRound).filter(
        PlayerRound.round_id == current_round.id
    ).all()
    submitted_count = len(submitted_players)
    
    # Get list of players who haven't submitted yet
    submitted_player_ids = [p.player_id for p in submitted_players]
    pending_players = [p for p in players if p.id not in submitted_player_ids]
    
    return {
        "game_id": game_id,
        "round_number": current_round.round_number,
        "is_completed": current_round.is_completed,
        "total_players": total_players,
        "submitted_count": submitted_count,
        "pending_count": total_players - submitted_count,
        "pending_players": [{"id": p.id, "name": p.name, "role": p.role} for p in pending_players],
        "all_submitted": current_round.is_completed
    }

# Player Round endpoints
@router.get("/{game_id}/players/{player_id}/current-round", response_model=PlayerRoundResponse)
def get_player_current_round(
    game_id: int,
    player_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get the current round for a player.
    """
    # Get the current game round
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game not found"
        )
    
    current_round = db.query(GameRound).filter(
        GameRound.game_id == game_id,
        GameRound.round_number == game.current_round
    ).first()
    
    if not current_round:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Current round not found"
        )
    
    # Get the player's round
    player_round = db.query(PlayerRound).filter(
        PlayerRound.player_id == player_id,
        PlayerRound.round_id == current_round.id
    ).first()
    
    if not player_round:
        # If the player hasn't taken their turn yet, create a new player round
        player_round = PlayerRound(
            player_id=player_id,
            round_id=current_round.id,
            order_placed=0,  # Default to 0, will be updated when order is placed
            order_received=0,
            inventory_before=0,
            inventory_after=0,
            backorders_before=0,
            backorders_after=0,
            holding_cost=0.0,
            backorder_cost=0.0,
            total_cost=0.0
        )
        db.add(player_round)
        db.commit()
        db.refresh(player_round)
    
    return player_round
