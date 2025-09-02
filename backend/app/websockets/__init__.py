from fastapi import WebSocket
from typing import Dict, List, Optional
import json
import asyncio
from datetime import datetime
from ..models.game import GameStatus
from ..services.mixed_game_service import MixedGameService
from ..db.session import SessionLocal

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, Dict[str, WebSocket]] = {}
        self.game_rooms: Dict[int, set] = {}

    async def connect(self, websocket: WebSocket, game_id: int, client_id: str):
        await websocket.accept()
        if game_id not in self.active_connections:
            self.active_connections[game_id] = {}
            self.game_rooms[game_id] = set()
        
        self.active_connections[game_id][client_id] = websocket
        self.game_rooms[game_id].add(client_id)
        
        # Send current game state to the new connection
        await self.send_game_state(game_id, client_id)
        
        # Notify other clients about the new connection
        await self.broadcast({
            "type": "player_connected",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat()
        }, game_id, exclude_client_id=client_id)

    def disconnect(self, game_id: int, client_id: str):
        if game_id in self.active_connections and client_id in self.active_connections[game_id]:
            del self.active_connections[game_id][client_id]
            if client_id in self.game_rooms[game_id]:
                self.game_rooms[game_id].remove(client_id)

    async def send_personal_message(self, message: dict, game_id: int, client_id: str):
        if game_id in self.active_connections and client_id in self.active_connections[game_id]:
            await self.active_connections[game_id][client_id].send_json(message)

    async def broadcast(self, message: dict, game_id: int, exclude_client_id: str = None):
        if game_id in self.active_connections:
            for client_id, connection in self.active_connections[game_id].items():
                if client_id != exclude_client_id:
                    try:
                        await connection.send_json(message)
                    except Exception as e:
                        print(f"Error broadcasting to {client_id}: {e}")
                        self.disconnect(game_id, client_id)

    async def send_game_state(self, game_id: int, client_id: str):
        """Send the current game state to a specific client"""
        db = SessionLocal()
        try:
            game_service = MixedGameService(db)
            game_state = game_service.get_game_state(game_id)
            
            if game_state:
                await self.send_personal_message({
                    "type": "game_state",
                    "state": game_state,
                    "timestamp": datetime.utcnow().isoformat()
                }, game_id, client_id)
        except Exception as e:
            print(f"Error getting game state: {e}")
            await self.send_personal_message({
                "type": "error",
                "message": "Failed to load game state",
                "timestamp": datetime.utcnow().isoformat()
            }, game_id, client_id)
        finally:
            db.close()

    async def broadcast_game_state(self, game_id: int):
        """Broadcast the current game state to all connected clients"""
        db = SessionLocal()
        try:
            game_service = MixedGameService(db)
            game_state = game_service.get_game_state(game_id)
            
            if game_state:
                await self.broadcast({
                    "type": "game_state",
                    "state": game_state,
                    "timestamp": datetime.utcnow().isoformat()
                }, game_id)
        except Exception as e:
            print(f"Error broadcasting game state: {e}")
        finally:
            db.close()

# Create a singleton instance
manager = ConnectionManager()
