import os
import json
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from typing import List, Dict, Any, Optional, Tuple, Deque, Union

from sqlalchemy.orm import Session

from app.db import base  # noqa: F401
from app.db.session import SessionLocal, engine
from app import crud, schemas, models

# Ensure all models are imported for metadata
from app.db.base_class import Base
Base.metadata.create_all(bind=engine)


@dataclass
class PIHeuristicPolicy:
    """Proportional-Integral ordering policy used for synthetic data generation."""

    target_inv: int
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.1
    forecast_window: int = 4
    integral_clip: Optional[Tuple[float, float]] = None  # e.g., (-1000, 1000)
    min_order: int = 0
    max_order: Optional[int] = None

    _state: Dict[str, Dict[str, object]] = field(default_factory=dict, repr=False)

    def _resolve_role_name(self, role: Union[str, Dict[str, Any]]) -> str:
        if isinstance(role, str):
            return role
        if isinstance(role, dict) and "name" in role:
            return str(role["name"])
        return str(role)

    def _get_state(self, role: Union[str, Dict[str, Any]]) -> Dict[str, object]:
        name = self._resolve_role_name(role)
        if name not in self._state:
            self._state[name] = {"hist": deque(maxlen=self.forecast_window), "integral": 0.0}
        st = self._state[name]
        hist = st["hist"]
        if isinstance(hist, deque) and hist.maxlen != self.forecast_window:
            preserved = list(hist)[-self.forecast_window:]
            st["hist"] = deque(preserved, maxlen=self.forecast_window)
        return st

    def order(
        self,
        *,
        I_prime: int,
        B_next: int,
        inbound_pipeline_sum: int,
        observed_demand: int,
        role: Union[str, Dict[str, Any]],
        week: int,
    ) -> int:
        st = self._get_state(role)
        hist: Deque[int] = st["hist"]  # type: ignore[assignment]
        integral: float = float(st["integral"])  # type: ignore[assignment]

        hist.append(max(0, int(observed_demand)))
        forecast = (sum(hist) / len(hist)) if hist else 0.0

        e_t = float(self.target_inv - I_prime + B_next)
        integral += e_t
        if self.integral_clip is not None:
            lo, hi = self.integral_clip
            integral = max(lo, min(hi, integral))
        st["integral"] = integral

        O = self.alpha * forecast + self.beta * e_t + self.gamma * integral
        if self.max_order is not None:
            O = min(self.max_order, O)
        O = max(self.min_order, O)
        return int(round(max(0.0, O)))

    def produce(self, *, O_t: int, role: Union[str, Dict[str, Any]], week: int) -> int:
        return int(max(0, O_t))

def generate_synthetic_game(num_rounds: int = 100) -> Dict[str, Any]:
    """Generate a synthetic game with realistic supply chain dynamics."""
    # Game parameters
    num_players = 4
    roles = ["retailer", "wholesaler", "distributor", "manufacturer"]
    
    # Base demand pattern (weekly seasonality with some noise)
    base_demand = [8, 7, 9, 10, 12, 15, 20, 18, 16, 14] * (num_rounds // 10 + 1)
    base_demand = base_demand[:num_rounds]
    
    # Add some random spikes and drops
    for i in range(num_rounds):
        if random.random() < 0.1:  # 10% chance of demand spike/drop
            base_demand[i] *= random.choice([0.5, 1.5, 2.0])
    
    # Generate game data
    game_data = {
        "name": f"Synthetic Game {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "description": "Synthetic training data",
        "num_rounds": num_rounds,
        "rounds": []
    }
    
    # Initialize inventories and backlogs
    inventories = {role: 20 for role in roles}  # Start with some inventory
    backlogs = {role: 0 for role in roles}
    in_transit = {role: {i: 0 for i in range(1, 5)} for role in roles}  # Up to 4 periods in transit
    
    # Create PI heuristic policies per role
    policies: Dict[str, PIHeuristicPolicy] = {
        role: PIHeuristicPolicy(
            target_inv=int(2 * base_demand[0]),
            alpha=1.0,
            beta=0.6,
            gamma=0.05,
            forecast_window=4,
            integral_clip=(-500.0, 500.0),
            min_order=0,
            max_order=500,
        )
        for role in roles
    }

    # Generate rounds
    for round_num in range(1, num_rounds + 1):
        round_data = {
            "round_number": round_num,
            "decisions": [],
            "demand": base_demand[round_num - 1] * random.uniform(0.8, 1.2)  # Add some noise
        }
        
        downstream_orders: Dict[str, int] = {}

        # Process each role in the supply chain
        for i, role in enumerate(roles):
            # Calculate incoming shipments (arriving this round)
            incoming = in_transit[role].get(1, 0)
            
            # Update in-transit shipments
            for j in range(1, 4):
                in_transit[role][j] = in_transit[role].get(j + 1, 0)
            in_transit[role][4] = 0
            
            # Calculate available inventory
            available = inventories[role] + incoming
            
            # Determine demand (retailer sees customer demand, upstream roles see orders)
            if role == "retailer":
                demand = int(round(round_data["demand"]))
            else:
                downstream_role = roles[i - 1]
                demand = downstream_orders.get(downstream_role, 0)

            # Calculate fulfilled demand and update backlog
            fulfilled = min(available, demand + backlogs[role])
            new_backlog = max(0, demand + backlogs[role] - fulfilled)

            # Update inventory
            new_inventory = max(0, available - fulfilled)

            # PI heuristic order decision
            target_inventory = int(2 * base_demand[min(round_num - 1, len(base_demand) - 1)])
            policy = policies[role]
            policy.target_inv = target_inventory
            inbound_sum = int(sum(in_transit[role].values()))
            order_quantity = policy.order(
                I_prime=int(new_inventory),
                B_next=int(new_backlog),
                inbound_pipeline_sum=inbound_sum,
                observed_demand=int(demand),
                role=role,
                week=round_num,
            )

            # Add order to in-transit for upstream
            if role != "manufacturer":  # Manufacturer has infinite supply
                upstream_role = roles[i + 1] if i < len(roles) - 1 else "manufacturer"
                in_transit[upstream_role][2] = in_transit[upstream_role].get(2, 0) + order_quantity

            # Record decision
            decision = {
                "role": role,
                "inventory": int(new_inventory),
                "order_quantity": int(order_quantity),
                "demand": int(demand),
                "backlog": int(new_backlog),
                "incoming_shipment": int(incoming),
                "fulfilled_demand": int(fulfilled)
            }
            round_data["decisions"].append(decision)
            
            # Update state for next round
            inventories[role] = new_inventory
            backlogs[role] = new_backlog
            downstream_orders[role] = order_quantity
        
        game_data["rounds"].append(round_data)
    
    return game_data

def save_synthetic_game(db: Session, game_data: Dict[str, Any]) -> models.Game:
    """Save synthetic game to the database."""
    # Create game
    game_in = schemas.GameCreate(
        name=game_data["name"],
        description=game_data["description"],
        max_players=4,
        num_rounds=game_data["num_rounds"],
        is_public=False,
        is_completed=True
    )
    game = crud.game.create(db, obj_in=game_in)
    
    # Create players (AI players)
    roles = ["retailer", "wholesaler", "distributor", "manufacturer"]
    for i, role in enumerate(roles):
        player_in = schemas.PlayerCreate(
            user_id=None,  # AI player
            game_id=game.id,
            role=role,
            is_ai=True,
            ai_strategy="synthetic_data"
        )
        crud.player.create(db, obj_in=player_in)
    
    # Create rounds and decisions
    for round_num, round_data in enumerate(game_data["rounds"], 1):
        round_in = schemas.RoundCreate(
            game_id=game.id,
            round_number=round_num,
            is_completed=True
        )
        db_round = crud.round.create(db, obj_in=round_in)
        
        for decision in round_data["decisions"]:
            player = crud.player.get_by_game_and_role(
                db, game_id=game.id, role=decision["role"]
            )
            decision_in = schemas.DecisionCreate(
                round_id=db_round.id,
                player_id=player.id,
                role=decision["role"],
                order_quantity=decision["order_quantity"],
                current_inventory=decision["inventory"],
                demand=decision["demand"],
                backlog=decision["backlog"],
                incoming_shipment=decision["incoming_shipment"],
                cost=0.0,  # Not used in training
                timestamp=datetime.utcnow() - timedelta(days=len(game_data["rounds"]) - round_num)
            )
            crud.decision.create(db, obj_in=decision_in)
    
    return game

def generate_and_save_games(num_games: int = 10, rounds_per_game: int = 100):
    """Generate and save multiple synthetic games."""
    db = SessionLocal()
    try:
        for i in range(num_games):
            print(f"Generating game {i+1}/{num_games}...")
            game_data = generate_synthetic_game(num_rounds=rounds_per_game)
            game = save_synthetic_game(db, game_data)
            print(f"Saved game {game.name} with ID {game.id}")
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error generating games: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    # Generate 20 games with 100 rounds each
    generate_and_save_games(num_games=20, rounds_per_game=100)
    print("Synthetic data generation complete!")
