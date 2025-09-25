from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any

from app.db.session import get_db
from app.schemas.game import (
    GameCreate, 
    GameUpdate, 
    GameState, 
    Game as GameSchema, 
    GameStatus,
    GameInDBBase
)
from app.models.game import Game as GameModel
from app.schemas.player import PlayerAssignment, PlayerResponse
from app.services.mixed_game_service import MixedGameService
from app.core.security import get_current_user
from app.models.user import User

router = APIRouter()

def get_mixed_game_service(db: Session = Depends(get_db)) -> MixedGameService:
    return MixedGameService(db)

@router.post("/mixed-games/", response_model=GameSchema, status_code=status.HTTP_201_CREATED)
def create_mixed_game(
    game_data: GameCreate,
    current_user: User = Depends(get_current_user),
    game_service: MixedGameService = Depends(get_mixed_game_service)
):
    """
    Create a new game with mixed human and AI players.
    
    - **player_assignments**: List of player assignments specifying which roles are human/AI
    - **demand_pattern**: Configuration for customer demand pattern
    - **max_rounds**: Total number of rounds in the game
    """
    try:
        return game_service.create_game(game_data, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/mixed-games/{game_id}/start", response_model=GameSchema)
def start_game(
    game_id: int,
    debug_logging: bool = Body(False, embed=True),
    current_user: User = Depends(get_current_user),
    game_service: MixedGameService = Depends(get_mixed_game_service)
):
    """Start a game that's in the 'created' state."""
    try:
        return game_service.start_game(game_id, debug_logging=debug_logging)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/mixed-games/{game_id}/stop", response_model=GameSchema)
def stop_game(
    game_id: int,
    current_user: User = Depends(get_current_user),
    game_service: MixedGameService = Depends(get_mixed_game_service)
):
    """Stop a game that's in progress."""
    try:
        return game_service.stop_game(game_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/mixed-games/{game_id}/next-round", response_model=GameSchema)
def next_round(
    game_id: int,
    current_user: User = Depends(get_current_user),
    game_service: MixedGameService = Depends(get_mixed_game_service)
):
    """Advance to the next round of the game."""
    try:
        game_service.start_new_round(game_id)
        return game_service.get_game_state(game_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/mixed-games/{game_id}/finish", response_model=GameSchema)
def finish_game(
    game_id: int,
    current_user: User = Depends(get_current_user),
    game_service: MixedGameService = Depends(get_mixed_game_service)
):
    """Finish a game and compute a summary."""
    try:
        return game_service.finish_game(game_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/mixed-games/{game_id}/report", response_model=dict)
def get_game_report(
    game_id: int,
    current_user: User = Depends(get_current_user),
    game_service: MixedGameService = Depends(get_mixed_game_service)
):
    """Get simple endgame report."""
    try:
        return game_service.get_report(game_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@router.get("/mixed-games/{game_id}/state", response_model=GameState)
def get_game_state(
    game_id: int,
    current_user: User = Depends(get_current_user),
    game_service: MixedGameService = Depends(get_mixed_game_service)
):
    """Get the current state of a game."""
    try:
        return game_service.get_game_state(game_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@router.put("/mixed-games/{game_id}", response_model=GameState)
def update_game(
    game_id: int,
    payload: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    game_service: MixedGameService = Depends(get_mixed_game_service)
):
    """Update a game's core configuration, demand pattern, and player assignments."""
    try:
        game_service.update_game(game_id, payload)
        return game_service.get_game_state(game_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.delete("/mixed-games/{game_id}", response_model=dict)
def delete_game(
    game_id: int,
    current_user: User = Depends(get_current_user),
    game_service: MixedGameService = Depends(get_mixed_game_service)
):
    try:
        return game_service.delete_game(game_id, current_user)
    except PermissionError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions to delete this game")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@router.get("/mixed-games/", response_model=List[GameInDBBase])
def list_games(
    status: Optional[GameStatus] = None,
    current_user: User = Depends(get_current_user),
    game_service: MixedGameService = Depends(get_mixed_game_service)
):
    """List all games the current user is allowed to view."""
    return game_service.list_games(current_user=current_user, status=status)
