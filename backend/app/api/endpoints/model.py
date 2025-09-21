from fastapi import APIRouter, HTTPException, Depends
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, and_, select
from pydantic import BaseModel
import json

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

class TrainRequest(BaseModel):
    server_host: str = "aiserver.local"
    source: str = "sim"  # 'sim' or 'db'
    window: int = 12
    horizon: int = 1
    epochs: int = 10
    device: Optional[str] = None
    steps_table: str = "beer_game_steps"
    db_url: Optional[str] = None
    dataset_path: Optional[str] = None

@router.post("/model/train", response_model=Dict[str, Any])
async def launch_training(req: TrainRequest):
    """Launch tGNN training. Default server_host is 'aiserver.local'.
    This implementation starts a local background process and returns a handle.
    """
    import subprocess, uuid
    jobs_dir = Path("training_jobs"); jobs_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    log_path = jobs_dir / f"job_{job_id}.log"
    cmd = [
        "python", "scripts/training/train_gnn.py",
        "--source", req.source,
        "--window", str(req.window),
        "--horizon", str(req.horizon),
        "--epochs", str(req.epochs),
        "--save-path", str(MODEL_PATH),
    ]
    if req.device:
        cmd += ["--device", req.device]
    if req.source == "db":
        if req.db_url:
            cmd += ["--db-url", req.db_url]
        cmd += ["--steps-table", req.steps_table]
    if req.dataset_path:
        cmd += ["--dataset", req.dataset_path]
    note = None
    if req.server_host not in ("localhost", "127.0.0.1", "aiserver.local"):
        note = f"Remote host '{req.server_host}' not configured for remote execution; launching locally."
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    # write job metadata
    meta = {
        "pid": proc.pid,
        "cmd": cmd,
        "started_at": datetime.utcnow().isoformat(),
        "log": str(log_path),
        "type": "train",
    }
    with open(jobs_dir / f"job_{job_id}.json", "w") as jf:
        json.dump(meta, jf)
    return {
        "job_id": job_id,
        "log": str(log_path),
        "cmd": " ".join(cmd),
        "note": note,
        "model_path": str(MODEL_PATH.absolute())
    }

@router.get("/model/job/{job_id}/status", response_model=Dict[str, Any])
async def get_job_status(job_id: str):
    jobs_dir = Path("training_jobs")
    meta_path = jobs_dir / f"job_{job_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    meta = json.loads(meta_path.read_text())
    log_path = Path(meta.get("log", ""))
    running = False
    pid = meta.get("pid")
    try:
        if pid:
            # check process liveness (POSIX)
            os.kill(pid, 0)
            running = True
    except Exception:
        running = False
    log_tail = ""
    log_size = 0
    if log_path.exists():
        log_size = log_path.stat().st_size
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()[-50:]
                log_tail = "".join(lines)
        except Exception:
            log_tail = ""
    return {"running": running, "pid": pid, "log_size": log_size, "log_tail": log_tail, **meta}

class GenerateDataRequest(BaseModel):
    num_runs: int = 64
    T: int = 64
    window: int = 12
    horizon: int = 1
    param_ranges: Optional[Dict[str, List[float]]] = None
    distribution: Optional[str] = "uniform"  # 'uniform' or 'normal'
    normal_means: Optional[Dict[str, float]] = None
    normal_stds: Optional[Dict[str, float]] = None
    # New: SimPy tuning
    use_simpy: Optional[bool] = None
    sim_alpha: Optional[float] = None
    sim_wip_k: Optional[float] = None

@router.post("/model/generate-data", response_model=Dict[str, Any])
async def generate_data(req: GenerateDataRequest):
    """Generate synthetic training data (npz) using simulator with optional ranges."""
    import numpy as np
    from app.rl.data_generator import generate_sim_training_windows, BeerGameParams
    # Build ranges
    ranges = None
    if req.distribution == "uniform":
        if req.param_ranges:
            ranges = {k: (float(v[0]), float(v[1])) for k, v in req.param_ranges.items() if isinstance(v, (list, tuple)) and len(v) == 2}
    elif req.distribution == "normal":
        # Approximate normal by uniform over [mean-2std, mean+2std]
        if req.normal_means and req.normal_stds:
            ranges = {}
            for k, mu in req.normal_means.items():
                sigma = float(req.normal_stds.get(k, 0))
                lo, hi = float(mu) - 2 * sigma, float(mu) + 2 * sigma
                if lo > hi:
                    lo, hi = hi, lo
                ranges[k] = (lo, hi)
    X, A, P, Y = generate_sim_training_windows(
            num_runs=req.num_runs,
            T=req.T,
            window=req.window,
            horizon=req.horizon,
            params=BeerGameParams(),
            param_ranges=ranges,
            randomize=True,
            use_simpy=req.use_simpy,
            sim_alpha=float(req.sim_alpha) if req.sim_alpha is not None else 0.3,
            sim_wip_k=float(req.sim_wip_k) if req.sim_wip_k is not None else 1.0,
        )
    out_dir = Path("training_jobs"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(out_path, X=X, A=A, P=P, Y=Y)
    return {"path": str(out_path), "X": list(X.shape), "A": list(A.shape), "P": list(P.shape), "Y": list(Y.shape)}

@router.post("/model/job/{job_id}/stop", response_model=Dict[str, Any])
async def stop_job(job_id: str):
    jobs_dir = Path("training_jobs")
    meta_path = jobs_dir / f"job_{job_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    meta = json.loads(meta_path.read_text())
    pid = meta.get("pid")
    if not pid:
        raise HTTPException(status_code=400, detail="No PID recorded for job")
    try:
        os.kill(pid, 15)  # SIGTERM
        meta["stopped_at"] = datetime.utcnow().isoformat()
        meta_path.write_text(json.dumps(meta))
        return {"stopped": True, "pid": pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop job: {e}")

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
    
    # Calculate overall bullwhip effect (retailer variance vs manufacturer variance)
    retailer_orders = []
    manufacturer_orders = []
    
    for player in player_performances:
        if player.role == "retailer":
            retailer_orders = [m.order_placed for m in player.round_metrics]
        elif player.role == "manufacturer":
            manufacturer_orders = [m.order_placed for m in player.round_metrics]
    
    overall_bullwhip = None
    if retailer_orders and manufacturer_orders and len(retailer_orders) == len(manufacturer_orders):
        avg_retailer = sum(retailer_orders) / len(retailer_orders)
        avg_manufacturer = sum(manufacturer_orders) / len(manufacturer_orders)
        
        if avg_retailer > 0:
            retailer_var = sum((o - avg_retailer) ** 2 for o in retailer_orders) / len(retailer_orders)
            manufacturer_var = sum((o - avg_manufacturer) ** 2 for o in manufacturer_orders) / len(manufacturer_orders)
            overall_bullwhip = manufacturer_var / retailer_var if retailer_var > 0 else None
    
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
