from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi import APIRouter, status
from typing import Dict, List
import json
import asyncio
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.supply_chain import Game, Player
from app.schemas.game import GameState
from app.services.game_service import GameService

router = APIRouter()

class ConnectionManager:
    """Manages WebSocket connections and broadcasts messages to connected clients."""
    
    def __init__(self):
        self.active_connections: Dict[int, Dict[int, WebSocket]] = {}
        self.lock = asyncio.Lock()
    
    async def connect(self, game_id: int, player_id: int, websocket: WebSocket):
        """Register a new WebSocket connection for a player in a game."""
        await websocket.accept()
        
        async with self.lock:
            if game_id not in self.active_connections:
                self.active_connections[game_id] = {}
            self.active_connections[game_id][player_id] = websocket
    
    def disconnect(self, game_id: int, player_id: int):
        """Remove a WebSocket connection when a player disconnects."""
        if game_id in self.active_connections:
            if player_id in self.active_connections[game_id]:
                del self.active_connections[game_id][player_id]
                # Clean up empty game rooms
                if not self.active_connections[game_id]:
                    del self.active_connections[game_id]
    
    async def broadcast_game_state(self, game_id: int, game_state: GameState):
        """Send the current game state to all connected players in a game."""
        if game_id not in self.active_connections:
            return
            
        message = {
            "type": "game_update",
            "data": game_state.dict()
        }
        
        # Create a list of tasks to send messages to all connected clients
        tasks = []
        for player_id, connection in list(self.active_connections[game_id].items()):
            try:
                tasks.append(
                    asyncio.create_task(
                        connection.send_json(message)
                    )
                )
            except Exception as e:
                # If sending fails, the connection is likely dead
                print(f"Error sending to player {player_id}: {e}")
                self.disconnect(game_id, player_id)
        
        # Wait for all sends to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_personal_message(self, player_id: int, game_id: int, message: dict):
        """Send a message to a specific player in a game."""
        if game_id in self.active_connections:
            if player_id in self.active_connections[game_id]:
                try:
                    await self.active_connections[game_id][player_id].send_json(message)
                except Exception as e:
                    print(f"Error sending to player {player_id}: {e}")
                    self.disconnect(game_id, player_id)

# Global connection manager instance
manager = ConnectionManager()

@router.websocket("/ws/games/{game_id}/players/{player_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    game_id: int,
    player_id: int,
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time game updates.
    
    - Connects a player to the game's WebSocket room
    - Sends the current game state on connection
    - Broadcasts updates to all players when the game state changes
    """
    # Verify the game and player exist
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    player = db.query(Player).filter(
        Player.id == player_id,
        Player.game_id == game_id
    ).first()
    
    if not player:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    # Add the connection to the manager
    await manager.connect(game_id, player_id, websocket)
    
    # Send the current game state
    game_service = GameService(db)
    try:
        game_state = game_service.get_game_state(game_id)
        await manager.send_personal_message(
            player_id,
            game_id,
            {
                "type": "game_state",
                "data": game_state.dict()
            }
        )
        
        # Keep the connection alive and handle incoming messages
        while True:
            try:
                # Wait for a message from the client
                data = await websocket.receive_text()
                
                # Parse the message
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON"
                    })
                    continue
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                
                # Add more message handlers as needed
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
                
    finally:
        # Clean up the connection when done
        manager.disconnect(game_id, player_id)

# Add the WebSocket router to the API router
router.websocket_route("/ws/games/{game_id}/players/{player_id}", name="game_ws")(websocket_endpoint)
