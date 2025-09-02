from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from app.db.session import get_db
from app.schemas.game import GameCreate, GameUpdate, GameState, PlayerState
from app.services.agent_game_service import AgentGameService
from app.core.security import get_current_user

router = APIRouter()

def get_agent_service(db: Session = Depends(get_db)) -> AgentGameService:
    """Dependency to get an instance of AgentGameService."""
    return AgentGameService(db)

@router.post("/agent-games/", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
def create_agent_game(
    game_in: GameCreate,
    agent_service: AgentGameService = Depends(get_agent_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new game with AI agents.
    
    - **name**: Name of the game
    - **max_rounds**: Maximum number of rounds (default: 20)
    - **demand_pattern**: Configuration for the demand pattern
    """
    try:
        game = agent_service.create_game(game_in)
        return {"message": "Game created successfully", "game_id": game.id}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/agent-games/{game_id}/start", response_model=Dict[str, Any])
def start_agent_game(
    game_id: int,
    agent_service: AgentGameService = Depends(get_agent_service),
    current_user: dict = Depends(get_current_user)
):
    """Start an agent-based game."""
    try:
        game = agent_service.start_game(game_id)
        return {"message": "Game started successfully", "game_id": game.id}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "not found" in str(e).lower() 
            else status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/agent-games/{game_id}/play-round", response_model=Dict[str, Any])
def play_agent_round(
    game_id: int,
    agent_service: AgentGameService = Depends(get_agent_service),
    current_user: dict = Depends(get_current_user)
):
    """Play one round of an agent-based game."""
    try:
        game_state = agent_service.play_round(game_id)
        return {"message": "Round played successfully", "game_state": game_state}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/agent-games/{game_id}/state", response_model=Dict[str, Any])
def get_agent_game_state(
    game_id: int,
    agent_service: AgentGameService = Depends(get_agent_service),
    current_user: dict = Depends(get_current_user)
):
    """Get the current state of an agent-based game."""
    try:
        return agent_service.get_game_state(game_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@router.put("/agent-games/{game_id}/agent-strategy")
def set_agent_strategy(
    game_id: int,
    role: str,
    strategy: str,
    agent_service: AgentGameService = Depends(get_agent_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Set the strategy for an AI agent.
    
    - **role**: The role of the agent (retailer, wholesaler, distributor, factory)
    - **strategy**: The strategy to use (naive, bullwhip, conservative, random)
    """
    try:
        agent_service.set_agent_strategy(role, strategy)
        return {"message": f"Strategy for {role} set to {strategy}"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.put("/agent-games/{game_id}/demand-visibility")
def set_demand_visibility(
    game_id: int,
    visible: bool,
    agent_service: AgentGameService = Depends(get_agent_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Set whether agents can see the actual customer demand.
    
    - **visible**: Whether agents can see the demand (true/false)
    """
    try:
        agent_service.set_demand_visibility(visible)
        return {"message": f"Demand visibility set to {visible}"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
