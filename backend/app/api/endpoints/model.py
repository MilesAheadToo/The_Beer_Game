from fastapi import APIRouter, HTTPException, Depends
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, and_, select
from pydantic import BaseModel

from app.db.session import get_db
from app.models.supply_chain import Game, Player, GameRound, PlayerRound
from app.schemas.game import PlayerRound as PlayerRoundSchema

router = APIRouter()

MODEL_PATH = Path("checkpoints/supply_chain_gnn.pth")

def get_model_status() -> Dict[str, Any]:
    """Check if the GNN model exists and return its status."""
    if not MODEL_PATH.exists():
        return {
            "is_trained": False,
            "message": "GNN model has not been trained yet.",
            "model_path": str(MODEL_PATH.absolute())
        }
    
    # Get model metadata
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        model_metadata = {
            "is_trained": True,
            "model_path": str(MODEL_PATH.absolute()),
            "file_size_mb": os.path.getsize(MODEL_PATH) / (1024 * 1024),
            "last_modified": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat(),
            "model_state": "available" if checkpoint.get("model_state_dict") else "incomplete",
            "has_optimizer": "optimizer_state_dict" in checkpoint,
            "epoch": checkpoint.get("epoch", "unknown"),
            "training_loss": checkpoint.get("loss", "unknown"),
        }
        return model_metadata
    except Exception as e:
        return {
            "is_trained": False,
            "message": f"Error loading model: {str(e)}",
            "model_path": str(MODEL_PATH.absolute())
        }

@router.get("/model/status", response_model=Dict[str, Any])
async def get_model_status_endpoint():
    """
    Get the status of the GNN model.
    Returns information about whether the model is trained and its metadata.
    """
    return get_model_status()

@router.get("/games/{game_id}/metrics", response_model=Dict[str, Any])
async def get_game_metrics(
    game_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate and return detailed performance metrics for a completed game.
    Includes costs, inventory metrics, and supply chain performance indicators.
    """
    from app.schemas.metrics import (
        GameMetricsResponse, PlayerPerformance, CostMetrics,
        InventoryMetrics, OrderMetrics, PlayerRoundMetrics, MarginMetrics
    )
    
    # Get the game
    result = await db.execute(select(Game).where(Game.id == game_id))
    game = result.scalars().first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Get all rounds for this game
    rounds_result = await db.execute(
        select(GameRound)
        .where(GameRound.game_id == game_id)
        .order_by(GameRound.round_number)
    )
    rounds = rounds_result.scalars().all()
    
    if not rounds:
        raise HTTPException(status_code=400, detail="No rounds found for this game")
    
    # Get all players in this game
    players_result = await db.execute(select(Player).where(Player.game_id == game_id))
    players = players_result.scalars().all()
    if not players:
        raise HTTPException(status_code=400, detail="No players found for this game")
    
    # Get all player rounds
    player_rounds_result = await db.execute(
        select(PlayerRound)
        .join(GameRound)
        .where(GameRound.game_id == game_id)
    )
    player_rounds = player_rounds_result.scalars().all()
    
    if not rounds:
        raise HTTPException(status_code=400, detail="No rounds completed for this game")
    
    player_performances = []
    total_supply_chain_cost = 0
    total_demand = 0
    
    for player in players:
        player_rounds_data = [pr for pr in player_rounds if pr.player_id == player.id]
        
        # Get pricing for this player's role from game configuration
        role = player.role.lower()
        pricing = game.pricing_config.dict()
        role_pricing = pricing.get(role, {})
        
        if not role_pricing:
            raise HTTPException(
                status_code=400,
                detail=f"No pricing configuration found for role: {role}"
            )
            
        selling_price = role_pricing.get("selling_price")
        standard_cost = role_pricing.get("standard_cost")
        
        if selling_price is None or standard_cost is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid pricing configuration for role: {role}"
            )
        
        # Calculate cost metrics
        total_cost = sum(pr.total_cost for pr in player_rounds_data if pr.total_cost)
        holding_cost = sum(pr.holding_cost for pr in player_rounds_data if pr.holding_cost)
        backorder_cost = sum(pr.backorder_cost for pr in player_rounds_data if pr.backorder_cost)
        operational_cost = holding_cost + backorder_cost
        
        # Calculate inventory metrics
        avg_inventory = sum(pr.inventory_after for pr in player_rounds_data) / len(rounds) if rounds else 0
        stockout_weeks = sum(1 for pr in player_rounds_data if pr.backorder_after > 0)
        
        # Calculate order metrics
        orders = [pr.order_placed for pr in player_rounds_data if pr.order_placed is not None]
        avg_order = sum(orders) / len(orders) if orders else 0
        
        # Calculate demand (for retailer) or orders received (for others)
        demands = [pr.order_received if player.role != "retailer" else pr.order_placed 
                  for pr in player_rounds_data if pr.order_received is not None]
        avg_demand = sum(demands) / len(demands) if demands else 0
        total_demand += avg_demand * len(demands)
        
        # Calculate service level (percentage of demand met from stock)
        service_level = (1 - (stockout_weeks / len(player_rounds_data))) * 100 if player_rounds_data else 0
        
        # Calculate inventory turns (annualized)
        inventory_turns = (total_cost / avg_inventory) * 52 if avg_inventory > 0 else 0
        
        # Calculate order variability (coefficient of variation)
        if avg_order > 0:
            order_std = (sum((o - avg_order) ** 2 for o in orders) / len(orders)) ** 0.5
            order_variability = (order_std / avg_order) * 100
        else:
            order_variability = 0
        
        # Calculate bullwhip effect (if not retailer)
        bullwhip_effect = None
        if player.role != "retailer" and demands and orders:
            demand_variance = (sum((d - avg_demand) ** 2 for d in demands) / len(demands)) ** 2
            order_variance = (sum((o - avg_order) ** 2 for o in orders) / len(orders)) ** 2
            if demand_variance > 0:
                bullwhip_effect = order_variance / demand_variance
        
        # Calculate margin metrics
        total_revenue = 0.0
        total_gross_margin = 0.0
        total_net_margin = 0.0
        total_margin_erosion = 0.0
        
        # Prepare round metrics with margin calculations
        round_metrics = []
        for pr in player_rounds:
            # Calculate units sold (orders received for this player)
            units_sold = pr.order_received if pr.order_received is not None else 0
            
            # Calculate revenue and costs
            revenue = units_sold * selling_price
            gross_margin = revenue - (units_sold * standard_cost)
            operational_cost_round = pr.holding_cost + pr.backorder_cost
            net_margin = gross_margin - operational_cost_round
            
            # Calculate margin erosion (percentage of gross margin lost to costs)
            margin_erosion = (operational_cost_round / gross_margin * 100) if gross_margin > 0 else 0
            
            # Update totals
            total_revenue += revenue
            total_gross_margin += gross_margin
            total_net_margin += net_margin
            total_margin_erosion += margin_erosion
            
            # Create round metrics with margin data
            round_metric = PlayerRoundMetrics(
                round_number=pr.game_round.round_number,
                inventory=pr.inventory_after,
                backorders=pr.backorder_after,
                order_placed=pr.order_placed,
                order_received=pr.order_received,
                holding_cost=pr.holding_cost,
                backorder_cost=pr.backorder_cost,
                total_cost=pr.total_cost,
                revenue=revenue,
                gross_margin=gross_margin,
                net_margin=net_margin,
                margin_erosion=margin_erosion
            )
            round_metrics.append(round_metric)
        
        # Add to total supply chain cost
        total_supply_chain_cost += total_cost
        
        # Calculate average margin erosion
        avg_margin_erosion = total_margin_erosion / len(player_rounds) if player_rounds else 0
        
        # Create player performance object
        player_perf = PlayerPerformance(
            player_id=player.id,
            player_name=player.name,
            role=player.role,
            total_cost=total_cost,
            total_revenue=total_revenue,
            total_gross_margin=total_gross_margin,
            total_net_margin=total_net_margin,
            average_margin_erosion=avg_margin_erosion,
            cost_metrics=CostMetrics(
                total_cost=total_cost,
                holding_cost=holding_cost,
                backorder_cost=backorder_cost,
                average_weekly_cost=total_cost / len(player_rounds) if player_rounds else 0,
                operational_cost=operational_cost
            ),
            margin_metrics=MarginMetrics(
                selling_price=selling_price,
                standard_cost=standard_cost,
                gross_margin=total_gross_margin,
                net_margin=total_net_margin,
                margin_erosion=avg_margin_erosion
            ),
            inventory_metrics=InventoryMetrics(
                average_inventory=avg_inventory,
                inventory_turns=inventory_turns,
                stockout_weeks=stockout_weeks,
                service_level=service_level
            ),
            order_metrics=OrderMetrics(
                average_order=avg_order,
                order_variability=order_variability,
                bullwhip_effect=bullwhip_effect
            ),
            round_metrics=round_metrics
        )
        
        player_performances.append(player_perf)
    
    # Calculate overall metrics
    avg_weekly_demand = total_demand / (len(rounds) * len(players)) if players and rounds else 0
    
    # Calculate overall bullwhip effect (retailer variance vs factory variance)
    retailer_orders = []
    factory_orders = []
    
    for player in player_performances:
        if player.role == "retailer":
            retailer_orders = [m.order_placed for m in player.round_metrics]
        elif player.role == "factory":
            factory_orders = [m.order_placed for m in player.round_metrics]
    
    overall_bullwhip = None
    if retailer_orders and factory_orders and len(retailer_orders) == len(factory_orders):
        avg_retailer = sum(retailer_orders) / len(retailer_orders)
        avg_factory = sum(factory_orders) / len(factory_orders)
        
        if avg_retailer > 0:
            retailer_var = sum((o - avg_retailer) ** 2 for o in retailer_orders) / len(retailer_orders)
            factory_var = sum((o - avg_factory) ** 2 for o in factory_orders) / len(factory_orders)
            overall_bullwhip = factory_var / retailer_var if retailer_var > 0 else None
    
    # Prepare final response
    response = GameMetricsResponse(
        game_id=game.id,
        game_name=game.name,
        total_rounds=len(rounds),
        start_date=rounds[0].created_at if rounds else None,
        end_date=rounds[-1].completed_at if rounds and hasattr(rounds[-1], 'completed_at') else None,
        players=player_performances,
        total_supply_chain_cost=total_supply_chain_cost,
        average_weekly_demand=avg_weekly_demand,
        bullwhip_effect=overall_bullwhip
    )
    
    return response.dict()
