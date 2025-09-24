import logging
import os
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from llm_agent.daybreak_client import DaybreakStrategistSession

class LLMStrategy(Enum):
    """Different Daybreak LLM prompting strategies for the Beer Game."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"

class LLMAgent:
    """Daybreak LLM-based agent that delegates to the Strategist assistant."""

    def __init__(
        self,
        role: str,
        strategy: LLMStrategy = LLMStrategy.BALANCED,
        model: str = "gpt-4",
    ):
        self.role = role
        self.strategy = strategy
        default_model = os.getenv("DAYBREAK_LLM_MODEL")
        self.model = model or default_model or "gpt-4"
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.session = DaybreakStrategistSession(model=self.model)
        self.last_explanation: Optional[str] = None
        self._last_ship_plan: Optional[int] = None

    def make_decision(
        self,
        current_round: int,
        current_inventory: int,
        backorders: int,
        incoming_shipments: list,
        demand_history: list,
        order_history: list,
        current_demand: Optional[int] = None,
        upstream_data: Optional[Dict[str, Any]] = None
    ) -> int:
        """Make a decision on how many units to order."""
        self.last_explanation = None
        self._last_ship_plan = None
        payload = None
        if isinstance(upstream_data, dict):
            payload = upstream_data.get("llm_payload")

        if payload:
            try:
                decision = self.session.decide(payload)
                order_quantity = max(0, int(decision.get("order_upstream", 0)))
                self._last_ship_plan = max(0, int(decision.get("ship_to_downstream", 0)))
                rationale = decision.get("rationale", "")
                ship_fragment = (
                    f" | Proposed ship downstream: {self._last_ship_plan}"
                    if self._last_ship_plan is not None
                    else ""
                )
                self.last_explanation = f"{rationale}{ship_fragment}".strip()
                return order_quantity
            except Exception as exc:
                print(f"Error calling Daybreak strategist: {exc}")
                return self._fallback_strategy(current_inventory, backorders, current_demand)

        return self._fallback_strategy(current_inventory, backorders, current_demand)

    def _fallback_strategy(
        self, 
        current_inventory: int, 
        backorders: int,
        current_demand: Optional[int] = None
    ) -> int:
        """Fallback strategy if the Daybreak LLM call fails."""
        if current_demand is None:
            current_demand = 8  # Default average demand
            
        if self.strategy == LLMStrategy.CONSERVATIVE:
            return max(0, current_demand * 2 - current_inventory + backorders)
        elif self.strategy == LLMStrategy.AGGRESSIVE:
            return max(0, current_demand - current_inventory + backorders)
        else:  # BALANCED or ADAPTIVE
            return max(0, int(current_demand * 1.5) - current_inventory + backorders)


logger = logging.getLogger(__name__)


def check_daybreak_llm_access(
    *,
    model: Optional[str] = None,
    request_timeout: float = 5.0,
) -> Tuple[bool, str]:
    """Probe the configured Daybreak LLM endpoint to confirm availability."""

    if not os.getenv("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY environment variable not set"

    target_model = model or os.getenv("DAYBREAK_LLM_MODEL") or "gpt-4o-mini"

    try:
        session = DaybreakStrategistSession(model=target_model)
        session.reset()
        # Minimal ping state adhering to the strategist schema
        ping_state: Dict[str, Any] = {
            "role": "retailer",
            "week": 0,
            "toggles": {
                "customer_demand_history_sharing": False,
                "volatility_signal_sharing": False,
                "downstream_inventory_visibility": False,
            },
            "parameters": {
                "holding_cost": 0.5,
                "backlog_cost": 0.5,
                "L_order": 2,
                "L_ship": 2,
                "L_prod": 4,
            },
            "local_state": {
                "on_hand": 12,
                "backlog": 0,
                "incoming_orders_this_week": 0,
                "received_shipment_this_week": 0,
                "pipeline_orders_upstream": [0, 0],
                "pipeline_shipments_inbound": [0, 0],
                "optional": {},
            },
        }
        session.decide(ping_state)
        return True, target_model
    except Exception as exc:  # pragma: no cover - depends on external service
        logger.warning("Daybreak LLM probe failed for model %s: %s", target_model, exc)
        return False, str(exc)

# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Example usage
    agent = LLMAgent(role="retailer", strategy=LLMStrategy.BALANCED, model="gpt-4o-mini")
    
    # Example game state
    order = agent.make_decision(
        current_round=1,
        current_inventory=12,
        backorders=0,
        incoming_shipments=[4, 4],
        demand_history=[8, 8, 8, 8, 12],
        order_history=[8, 8, 8, 8, 12],
        current_demand=8,
        upstream_data={
            "wholesaler_inventory": 24,
            "recent_lead_time": 2,
            "market_conditions": "stable"
        }
    )
    
    print(f"Agent decided to order: {order} units")
