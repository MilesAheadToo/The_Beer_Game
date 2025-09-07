from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .. import models, schemas

def get_active_game_for_user(db: Session, user_id: int) -> Optional[models.Game]:
    """
    Get the active game for a user. For simplicity, returns the most recently created active game.
    In a production environment, you might want to implement more sophisticated logic.
    """
    return db.query(models.Game).join(models.Player).filter(
        models.Player.user_id == user_id,
        models.Game.status.in_([models.GameStatus.STARTED, models.GameStatus.ROUND_IN_PROGRESS])
    ).order_by(models.Game.created_at.desc()).first()

def get_player_metrics(db: Session, player_id: int, game_id: int) -> Dict[str, Any]:
    """
    Calculate key metrics for a player in a game.
    """
    # Get the player
    player = db.query(models.Player).filter(
        models.Player.id == player_id,
        models.Player.game_id == game_id
    ).first()
    
    if not player:
        return {}
    
    # Calculate some basic metrics
    metrics = {
        'current_inventory': player.current_inventory or 0,
        'inventory_change': 0,  # Would calculate this based on previous round
        'backlog': player.current_backlog or 0,
        'total_cost': player.total_cost or 0,
        'avg_weekly_cost': 0,
        'service_level': 1.0,  # Would calculate based on orders fulfilled
        'service_level_change': 0,
    }
    
    # Calculate average weekly cost if game has been running for multiple rounds
    if player.game.current_round > 0:
        metrics['avg_weekly_cost'] = metrics['total_cost'] / player.game.current_round
    
    return metrics

def get_time_series_metrics(db: Session, player_id: int, game_id: int, role: str) -> List[Dict[str, Any]]:
    """
    Get time series data for a player's performance across rounds.
    """
    # Get all rounds for this game
    rounds = db.query(models.Round).filter(
        models.Round.game_id == game_id
    ).order_by(models.Round.round_number).all()
    
    time_series = []
    
    for round_ in rounds:
        # Get player's actions for this round
        actions = db.query(models.PlayerAction).filter(
            models.PlayerAction.player_id == player_id,
            models.PlayerAction.round_id == round_.id
        ).all()
        
        # Initialize round data
        round_data = {
            'week': round_.round_number,
            'inventory': 0,
            'order': 0,
            'cost': 0,
            'backlog': 0
        }
        
        # Process actions to calculate metrics
        for action in actions:
            if action.action_type == 'order':
                round_data['order'] = action.quantity
            elif action.action_type == 'inventory_update':
                round_data['inventory'] = action.quantity
            elif action.action_type == 'cost_update':
                round_data['cost'] = action.quantity
            elif action.action_type == 'backlog_update':
                round_data['backlog'] = action.quantity
        
        # Add demand/supply based on role
        if role in ['RETAILER', 'MANUFACTURER', 'DISTRIBUTOR']:
            round_data['demand'] = round_data.get('order', 0)  # Simplified - would get from actual demand
            
        if role in ['SUPPLIER', 'MANUFACTURER', 'DISTRIBUTOR']:
            round_data['supply'] = round_data.get('inventory', 0)  # Simplified - would get from actual supply
        
        time_series.append(round_data)
    
    return time_series
