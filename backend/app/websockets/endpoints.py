from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.routing import APIRouter
from typing import Optional
import uuid
import json
import logging
from datetime import datetime

from ..core.security import get_current_user
from ..models.user import User
from ..schemas.websocket import WebSocketMessage
from . import manager

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws/games/{game_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    game_id: int,
    token: str = None,
):
    """WebSocket endpoint for real-time game updates"""
    client_id = str(uuid.uuid4())
    user = None
    
    try:
        # Authenticate the user using the token
        if token:
            try:
                user = await get_current_user(token)
                logger.info(f"WebSocket connection from user {user.id} for game {game_id}")
            except Exception as e:
                logger.warning(f"WebSocket auth failed: {e}")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
        else:
            logger.info(f"Unauthenticated WebSocket connection for game {game_id}")
        
        # Accept the WebSocket connection
        await manager.connect(websocket, game_id, client_id)
        
        # Main message loop
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message["type"] == "order":
                    # Process order
                    await handle_order_message(game_id, client_id, user, message)
                    
                elif message["type"] == "chat":
                    # Broadcast chat message
                    await manager.broadcast({
                        "type": "chat",
                        "user_id": user.id if user else None,
                        "username": user.username if user else "Anonymous",
                        "message": message["message"],
                        "timestamp": datetime.utcnow().isoformat()
                    }, game_id)
                    
                elif message["type"] == "get_state":
                    # Send current game state to the requesting client
                    await manager.send_game_state(game_id, client_id)
                    
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "message": "An error occurred",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
        
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}", exc_info=True)
        
    finally:
        manager.disconnect(game_id, client_id)
        
        # Notify other clients about the disconnection
        await manager.broadcast({
            "type": "player_disconnected",
            "client_id": client_id,
            "user_id": user.id if user else None,
            "timestamp": datetime.utcnow().isoformat()
        }, game_id, exclude_client_id=client_id)

async def handle_order_message(game_id: int, client_id: str, user: Optional[User], message: dict):
    """Handle order messages from clients"""
    from ..services.mixed_game_service import MixedGameService
    from ..db.session import SessionLocal
    
    db = SessionLocal()
    try:
        game_service = MixedGameService(db)
        
        # Validate the order
        required_fields = ["round_number", "quantity", "role"]
        if not all(field in message for field in required_fields):
            await manager.send_personal_message({
                "type": "error",
                "message": "Missing required fields in order",
                "timestamp": datetime.utcnow().isoformat()
            }, game_id, client_id)
            return
            
        # Process the order
        order = {
            "round_number": message["round_number"],
            "quantity": message["quantity"],
            "role": message["role"],
            "user_id": user.id if user else None
        }
        
        result = game_service.process_order(game_id, order)
        
        if result["success"]:
            # Broadcast the updated game state to all clients
            await manager.broadcast_game_state(game_id)
            
            # Send confirmation to the client
            await manager.send_personal_message({
                "type": "order_confirmation",
                "order_id": result["order_id"],
                "message": "Order processed successfully",
                "timestamp": datetime.utcnow().isoformat()
            }, game_id, client_id)
        else:
            # Send error to the client
            await manager.send_personal_message({
                "type": "error",
                "message": result.get("message", "Failed to process order"),
                "timestamp": datetime.utcnow().isoformat()
            }, game_id, client_id)
            
    except Exception as e:
        logger.error(f"Error processing order: {e}", exc_info=True)
        await manager.send_personal_message({
            "type": "error",
            "message": "An error occurred while processing your order",
            "timestamp": datetime.utcnow().isoformat()
        }, game_id, client_id)
        
    finally:
        db.close()
