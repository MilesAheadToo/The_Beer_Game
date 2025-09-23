from typing import List, Dict, Optional, Any, Tuple
import random
import statistics
from enum import Enum
from collections import deque
from dataclasses import dataclass, field

from .beer_game_xai_explain import (
    Obs,
    Forecast,
    RoleParams,
    SupervisorContext,
    explain_role_decision,
    explain_supervisor_adjustment,
)

# Import the Daybreak LLM agent only when needed to avoid unnecessary dependencies
try:  # pragma: no cover - optional import
    from .llm_agent import LLMAgent, LLMStrategy
except Exception:  # pragma: no cover - tests may not have openai deps
    LLMAgent = None  # type: ignore
    LLMStrategy = None  # type: ignore

class AgentType(Enum):
    RETAILER = "retailer"
    WHOLESALER = "wholesaler"
    DISTRIBUTOR = "distributor"
    MANUFACTURER = "manufacturer"

class AgentStrategy(Enum):
    NAIVE = "naive"  # Simple strategy, always orders based on current demand
    BULLWHIP = "bullwhip"  # Tends to over-order when demand increases
    CONSERVATIVE = "conservative"  # Maintains stable orders
    RANDOM = "random"  # Random ordering for baseline
    PI = "pi_heuristic"  # Proportional-integral controller
    LLM = "llm"  # Daybreak LLM strategy
    DAYBREAK_DTCE = "daybreak_dtce"  # Decentralized twin coordinated ensemble
    DAYBREAK_DTCE_CENTRAL = "daybreak_dtce_central"  # DTCE with central override
    DAYBREAK_DTCE_GLOBAL = "daybreak_dtce_global"  # Single agent orchestrating the network


@dataclass
class PIControllerState:
    history: deque = field(default_factory=lambda: deque(maxlen=4))
    integral: float = 0.0


class DaybreakCoordinator:
    """Coordinates decentralized Daybreak agents with an optional override."""

    def __init__(self, default_override: float = 0.05, history_length: int = 8):
        self.default_override = self._clamp(default_override)
        self.history_length = history_length
        self.override_pct: Dict[AgentType, float] = {}
        self.order_history: Dict[AgentType, deque] = {}

    @staticmethod
    def _clamp(pct: Optional[float]) -> float:
        """Clamp override percentage to the 5%-50% range."""
        try:
            pct_val = float(pct) if pct is not None else 0.05
        except (TypeError, ValueError):
            pct_val = 0.05
        pct_val = abs(pct_val)
        pct_val = max(0.05, min(pct_val, 0.5))
        return pct_val

    def set_override_pct(self, agent_type: AgentType, pct: Optional[float]) -> None:
        """Set an override percentage for a specific agent."""
        if pct is None:
            self.override_pct.pop(agent_type, None)
            return
        self.override_pct[agent_type] = self._clamp(pct)

    def get_override_pct(self, agent_type: AgentType) -> float:
        return self.override_pct.get(agent_type, self.default_override)

    def _record_order(self, agent_type: AgentType, order: int) -> None:
        history = self.order_history.setdefault(agent_type, deque(maxlen=self.history_length))
        history.append(order)

    def register_decision(self, agent_type: AgentType, order: int) -> None:
        """Record the final decision from a decentralized agent."""
        self._record_order(agent_type, order)

    def _network_average(self, exclude: AgentType) -> Optional[float]:
        values = [history[-1] for atype, history in self.order_history.items() if atype != exclude and history]
        if values:
            return sum(values) / len(values)
        return None

    def apply_override(
        self,
        agent_type: AgentType,
        base_order: float,
        context: Optional[Dict[str, Any]] = None,
        week: Optional[int] = None,
    ) -> Tuple[int, Optional[str]]:
        """Apply the centralized override to the base order and return an explanation."""

        base = max(0.0, float(base_order))
        network_avg = self._network_average(agent_type)
        target = base
        reasons: List[str] = []
        global_notes: List[str] = []

        if network_avg is not None:
            target = (base + network_avg) / 2.0
            reasons.append(f"align toward network avg {network_avg:.1f}")

        backlog = float(context.get("backlog", 0)) if context else 0.0
        inventory = float(context.get("inventory", 0)) if context else 0.0

        if backlog > inventory:
            target = max(target, base + (backlog - inventory))
            reasons.append("backlog exceeds on-hand")
        elif inventory > backlog * 2 and inventory > 0:
            target = min(target, base - (inventory - backlog) / 2.0)
            reasons.append("inventory well above backlog")

        pct = self.get_override_pct(agent_type)
        max_adjustment = base * pct
        adjustment = target - base
        if adjustment > 0:
            adjustment = min(adjustment, max_adjustment)
        else:
            adjustment = max(adjustment, -max_adjustment)

        adjusted = max(0.0, base + adjustment)
        final_order = int(round(adjusted))
        self._record_order(agent_type, final_order)

        pre_qty = int(round(base))
        if context:
            global_notes.append(f"inventory {inventory:.1f}, backlog {backlog:.1f}")
            pipeline = context.get("pipeline")
            if pipeline is not None:
                global_notes.append(f"pipeline {float(pipeline):.1f}")

        week_val = week if week is not None else int(context.get("week", 0)) if context else 0
        supervisor_ctx = SupervisorContext(
            max_scale_pct=pct * 100,
            rule="stability_smoothing",
            reasons=reasons,
        )
        explanation = explain_supervisor_adjustment(
            role=agent_type.name.replace("_", " ").title(),
            week=week_val,
            pre_qty=pre_qty,
            post_qty=final_order,
            ctx=supervisor_ctx,
            global_notes=global_notes or None,
        )

        return final_order, explanation


class DaybreakGlobalController:
    """Orchestrates a single Daybreak agent across the entire supply chain."""

    def __init__(self, history_length: int = 12):
        self.history_length = max(3, int(history_length))
        self.round_marker: Optional[int] = None
        self.base_orders: Dict[AgentType, float] = {}
        self.context: Dict[AgentType, Dict[str, Any]] = {}
        self.plan: Dict[AgentType, int] = {}
        self.last_orders: Dict[AgentType, deque] = {}
        self.network_targets: deque = deque(maxlen=self.history_length)

    def _reset_round(self, round_number: int) -> None:
        if self.round_marker != round_number:
            self.round_marker = round_number
            self.base_orders.clear()
            self.context.clear()
            self.plan.clear()

    def _determine_target_flow(self) -> float:
        if AgentType.RETAILER in self.base_orders:
            return max(0.0, float(self.base_orders[AgentType.RETAILER]))
        if self.base_orders:
            return max(0.0, sum(self.base_orders.values()) / len(self.base_orders))
        if self.network_targets:
            return max(0.0, float(self.network_targets[-1]))
        return 0.0

    def _peer_anchor(self, requesting: AgentType) -> Optional[float]:
        peers = [qty for role, qty in self.plan.items() if role != requesting]
        if peers:
            return sum(peers) / len(peers)
        return None

    def plan_order(
        self,
        agent_type: AgentType,
        round_number: int,
        base_order: float,
        context: Optional[Dict[str, Any]] = None,
        prev_order: Optional[int] = None,
    ) -> Tuple[int, Optional[str]]:
        self._reset_round(round_number)

        safe_base = max(0.0, float(base_order))
        ctx = context or {}
        self.base_orders[agent_type] = safe_base
        self.context[agent_type] = ctx

        target_flow = self._determine_target_flow()
        peer_anchor = self._peer_anchor(agent_type)
        if peer_anchor is not None:
            target_flow = 0.6 * target_flow + 0.4 * peer_anchor

        def _to_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        backlog = _to_float(ctx.get("backlog"))
        inventory = _to_float(ctx.get("inventory"))
        pipeline = _to_float(ctx.get("pipeline"))

        adjusted = 0.65 * safe_base + 0.35 * target_flow
        reasons: List[str] = []

        if backlog > inventory:
            adjusted += backlog - inventory
            reasons.append("protect service level (backlog > inventory)")
        elif inventory > backlog and inventory > 0:
            reduction = min(inventory - backlog, adjusted * 0.3)
            if reduction > 0:
                adjusted -= reduction
                reasons.append("trim excess inventory")

        if pipeline > target_flow * 1.5 and pipeline > 0:
            dampen = min(pipeline - target_flow * 1.5, adjusted * 0.25)
            if dampen > 0:
                adjusted -= dampen
                reasons.append("pipeline congestion")

        prev_reference = prev_order
        if prev_reference is None:
            prev_history = self.last_orders.get(agent_type)
            if prev_history:
                prev_reference = prev_history[-1]

        if prev_reference is not None:
            max_step = max(3.0, abs(prev_reference) * 0.5)
            upper = prev_reference + max_step
            lower = max(0.0, prev_reference - max_step)
            if adjusted > upper:
                adjusted = upper
                reasons.append("limit step-up for stability")
            elif adjusted < lower:
                adjusted = lower
                reasons.append("limit step-down for stability")

        final_qty = int(round(max(0.0, adjusted)))
        self.plan[agent_type] = final_qty
        history = self.last_orders.setdefault(agent_type, deque(maxlen=self.history_length))
        history.append(final_qty)
        self.network_targets.append(target_flow)

        global_notes: List[str] = [f"target flow {target_flow:.1f}"]
        if peer_anchor is not None:
            global_notes.append(f"peer avg {peer_anchor:.1f}")
        if pipeline > 0:
            global_notes.append(f"pipeline {pipeline:.1f}")

        supervisor_ctx = SupervisorContext(
            max_scale_pct=100.0,
            rule="global_balancing",
            reasons=reasons,
        )
        explanation = explain_supervisor_adjustment(
            role=f"{agent_type.name.replace('_', ' ').title()} (Global)",
            week=round_number,
            pre_qty=int(round(safe_base)),
            post_qty=final_qty,
            ctx=supervisor_ctx,
            global_notes=global_notes or None,
        )

        return final_qty, explanation

class BeerGameAgent:
    def __init__(
        self,
        agent_id: int,
        agent_type: AgentType,
        strategy: AgentStrategy = AgentStrategy.NAIVE,
        can_see_demand: bool = False,
        initial_inventory: int = 12,
        initial_orders: int = 4,
        llm_model: Optional[str] = None,
        central_coordinator: Optional[DaybreakCoordinator] = None,
        global_controller: Optional[DaybreakGlobalController] = None,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.strategy = strategy
        self.can_see_demand = can_see_demand
        self.inventory = initial_inventory
        self.backlog = 0
        self.pipeline = [initial_orders] * 2  # Orders in the pipeline (2 rounds of lead time)
        self.order_history = []
        self.demand_history = []
        self.last_order = initial_orders
        # Daybreak LLM specific configuration
        self.llm_model = llm_model
        self._llm_agent: Optional[LLMAgent] = None
        # Optional centralized coordinator for Daybreak variants
        self.central_coordinator = central_coordinator
        # Optional global coordinator when a single agent manages all roles
        self.global_controller = global_controller
        self.last_explanation: Optional[str] = None
        # PI controller state
        self._pi_state = PIControllerState()
        self._pi_alpha = 1.0
        self._pi_beta = 0.6
        self._pi_gamma = 0.05
        self._pi_target_multiplier = 2.0
        self._pi_integral_clip: Tuple[float, float] = (-500.0, 500.0)
        self._pi_max_order: Optional[int] = 500
        self.reset_for_strategy()

    def make_decision(
        self,
        current_round: int,
        current_demand: Optional[int] = None,
        upstream_data: Optional[Dict] = None,
        local_state: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Make an order decision based on the agent's strategy and available information.
        
        Args:
            current_round: Current game round
            current_demand: Current customer demand (only visible to retailer if configured)
            upstream_data: Data from upstream (e.g., orders from downstream)
            
        Returns:
            int: The order quantity
        """
        # Update demand history if visible
        if current_demand is not None and (self.agent_type == AgentType.RETAILER or self.can_see_demand):
            self.demand_history.append(current_demand)

        # Normalize local state inputs for advanced strategies
        local_state = local_state or {}
        inventory_level = int(local_state.get("inventory", self.inventory))
        backlog_level = int(local_state.get("backlog", self.backlog))
        inventory_level = max(0, inventory_level)
        backlog_level = max(0, backlog_level)

        shipments_raw = local_state.get("incoming_shipments", self.pipeline)
        processed_shipments: List[float] = []
        if isinstance(shipments_raw, (list, tuple)):
            for item in shipments_raw:
                if isinstance(item, (int, float)):
                    processed_shipments.append(float(item))
                elif isinstance(item, dict):
                    qty = item.get("quantity") or item.get("qty")
                    if qty is not None:
                        try:
                            processed_shipments.append(float(qty))
                        except (TypeError, ValueError):
                            continue
        if not processed_shipments:
            processed_shipments = [float(x) for x in self.pipeline]

        # Keep internal state aligned with the most recent observation
        self.inventory = inventory_level
        self.backlog = backlog_level

        prev_order = self.last_order
        self.last_explanation = None

        # Make decision based on strategy
        if self.strategy == AgentStrategy.NAIVE:
            order = self._naive_strategy(current_demand, backlog_level, inventory_level)
        elif self.strategy == AgentStrategy.BULLWHIP:
            order = self._bullwhip_strategy(current_demand, upstream_data)
        elif self.strategy == AgentStrategy.CONSERVATIVE:
            order = self._conservative_strategy(current_demand)
        elif self.strategy == AgentStrategy.PI:
            order = self._pi_strategy(
                current_round,
                current_demand,
                upstream_data or {},
                inventory_level,
                backlog_level,
                processed_shipments,
            )
        elif self.strategy == AgentStrategy.LLM:
            order = self._llm_strategy(current_round, current_demand, upstream_data)
        elif self.strategy == AgentStrategy.DAYBREAK_DTCE:
            order = self._daybreak_dtce_strategy(
                current_round,
                prev_order,
                current_demand,
                upstream_data,
                inventory_level,
                backlog_level,
                processed_shipments,
            )
        elif self.strategy == AgentStrategy.DAYBREAK_DTCE_CENTRAL:
            order = self._daybreak_central_strategy(
                current_round,
                prev_order,
                current_demand,
                upstream_data,
                inventory_level,
                backlog_level,
                processed_shipments,
            )
        elif self.strategy == AgentStrategy.DAYBREAK_DTCE_GLOBAL:
            order = self._daybreak_global_strategy(
                current_round,
                prev_order,
                current_demand,
                upstream_data,
                inventory_level,
                backlog_level,
                processed_shipments,
            )
        else:  # RANDOM fallback
            order = self._random_strategy()

        order = max(0, int(round(order)))
        self.last_order = order
        self.order_history.append(order)
        return order

    def reset_for_strategy(self) -> None:
        self._pi_state = PIControllerState()

    def _get_downstream_role_name(self) -> Optional[str]:
        mapping = {
            AgentType.WHOLESALER: AgentType.RETAILER.value,
            AgentType.DISTRIBUTOR: AgentType.WHOLESALER.value,
            AgentType.MANUFACTURER: AgentType.DISTRIBUTOR.value,
        }
        return mapping.get(self.agent_type)
    
    def _naive_strategy(
        self,
        current_demand: Optional[int],
        backlog: float,
        inventory: float,
    ) -> int:
        """Order enough to cover demand plus backlog, adjusted for on-hand inventory."""
        demand = max(0.0, float(current_demand or 0.0))
        backlog_val = max(0.0, float(backlog or 0.0))
        inventory_val = max(0.0, float(inventory or 0.0))

        # If we have no signal at all, fall back to previous order
        if demand == 0.0 and backlog_val == 0.0 and inventory_val == 0.0:
            return max(0, int(round(self.last_order)))

        shortfall = max(0.0, backlog_val + demand - inventory_val)
        target = max(demand, shortfall)
        return max(0, int(round(target)))
    
    def _bullwhip_strategy(self, current_demand: Optional[int], upstream_data: Optional[Dict]) -> int:
        """Tend to over-order when demand increases."""
        if not self.demand_history:
            return self.last_order
            
        avg_demand = sum(self.demand_history) / len(self.demand_history)
        last_demand = self.demand_history[-1]
        
        # If demand is increasing, over-order
        if last_demand > avg_demand * 1.2:  # 20% increase
            return int(last_demand * 1.5)
        return last_demand
    
    def _conservative_strategy(self, current_demand: Optional[int]) -> int:
        """Maintain stable orders, avoid large fluctuations."""
        if not self.order_history:
            return 4  # Default order
            
        # Moving average of last 3 orders
        recent_orders = self.order_history[-3:] if len(self.order_history) >= 3 else self.order_history
        return int(sum(recent_orders) / len(recent_orders))

    def _pi_strategy(
        self,
        current_round: int,
        current_demand: Optional[int],
        upstream_data: Dict[str, Any],
        inventory_level: int,
        backlog_level: int,
        incoming_shipments: List[float],
    ) -> int:
        downstream_role = self._get_downstream_role_name()
        previous_orders_by_role: Dict[str, int] = upstream_data.get("previous_orders_by_role", {})

        if current_demand is not None and (
            self.agent_type == AgentType.RETAILER or self.can_see_demand
        ):
            observed_demand = current_demand
        elif downstream_role:
            observed_demand = previous_orders_by_role.get(downstream_role)
        else:
            observed_demand = None

        if observed_demand is None:
            observed_demand = self._pi_state.history[-1] if self._pi_state.history else self.last_order

        observed_demand = max(0, int(round(observed_demand)))
        self._pi_state.history.append(observed_demand)
        forecast = sum(self._pi_state.history) / len(self._pi_state.history) if self._pi_state.history else float(observed_demand)

        inbound_sum = float(sum(incoming_shipments)) if incoming_shipments else 0.0
        inventory_position = float(inventory_level - backlog_level) + inbound_sum

        target_inventory = float(max(0, int(round(self._pi_target_multiplier * max(forecast, 1.0)))))
        error = target_inventory - inventory_position

        self._pi_state.integral += error
        lo, hi = self._pi_integral_clip
        self._pi_state.integral = max(lo, min(hi, self._pi_state.integral))

        order = (
            self._pi_alpha * forecast
            + self._pi_beta * error
            + self._pi_gamma * self._pi_state.integral
        )

        if self._pi_max_order is not None:
            order = min(self._pi_max_order, order)

        order = max(0.0, order)
        self.demand_history.append(observed_demand)
        self.last_explanation = (
            f"PI heuristic | forecast {forecast:.1f} | error {error:.1f} | integral {self._pi_state.integral:.1f}"
        )
        return int(round(order))

    def _random_strategy(self) -> int:
        """Make random orders for baseline testing."""
        return random.randint(1, 8)

    def _llm_strategy(
        self,
        current_round: int,
        current_demand: Optional[int],
        upstream_data: Optional[Dict],
    ) -> int:
        """Use a Daybreak LLM-backed agent to decide the order quantity."""
        if LLMAgent is None:  # Daybreak LLM dependencies not available
            return self._naive_strategy(current_demand)

        if self._llm_agent is None:
            try:
                model = self.llm_model or "gpt-4"
                self._llm_agent = LLMAgent(
                    role=self.agent_type.value,
                    strategy=LLMStrategy.BALANCED if LLMStrategy else None,
                    model=model,
                )
            except Exception:
                # Fallback to simple strategy if Daybreak LLM initialization fails
                return self._naive_strategy(current_demand)

        try:
            return self._llm_agent.make_decision(
                current_round=current_round,
                current_inventory=self.inventory,
                backorders=self.backlog,
                incoming_shipments=self.pipeline,
                demand_history=self.demand_history,
                order_history=self.order_history,
                current_demand=current_demand,
                upstream_data=upstream_data,
            )
        except Exception:
            # Fallback if the Daybreak LLM call fails for any reason
            return self._naive_strategy(current_demand)

    def _compute_daybreak_base(
        self,
        current_demand: Optional[int],
        upstream_data: Optional[Dict],
        inventory_level: int,
        backlog_level: int,
        incoming_shipments: List[float],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute the baseline Daybreak DTCE recommendation."""
        alpha = 0.35
        if self.demand_history:
            forecast = float(self.demand_history[0])
            for demand in self.demand_history[1:]:
                forecast = alpha * float(demand) + (1 - alpha) * forecast
        elif current_demand is not None:
            forecast = float(current_demand)
        else:
            forecast = float(self.last_order or 4)

        pipeline = sum(incoming_shipments[:2]) if incoming_shipments else 0.0
        inventory_position = float(inventory_level) + pipeline - float(backlog_level)
        safety_stock = max(0.0, forecast * 0.5)
        base_target = forecast * 2.0 + safety_stock + float(backlog_level)
        base_order = base_target - inventory_position

        upstream_orders = (upstream_data or {}).get('previous_orders') if upstream_data else None
        upstream_avg = None
        if upstream_orders:
            recent_upstream = upstream_orders[-3:]
            if recent_upstream:
                upstream_avg = sum(recent_upstream) / len(recent_upstream)

        recent_local_orders = self.order_history[-3:] if self.order_history else []
        local_avg = sum(recent_local_orders) / len(recent_local_orders) if recent_local_orders else float(self.last_order)

        smoothing_anchor = upstream_avg if upstream_avg is not None else local_avg
        if smoothing_anchor is not None:
            base_order = 0.7 * base_order + 0.3 * float(smoothing_anchor)

        base_order = max(0.0, base_order)
        context = {
            "forecast": forecast,
            "inventory": float(inventory_level),
            "backlog": float(backlog_level),
            "pipeline": pipeline,
            "upstream_avg": upstream_avg,
            "local_avg": local_avg,
        }
        return base_order, context

    def _daybreak_dtce_strategy(
        self,
        current_round: int,
        prev_order: Optional[int],
        current_demand: Optional[int],
        upstream_data: Optional[Dict],
        inventory_level: int,
        backlog_level: int,
        incoming_shipments: List[float],
    ) -> int:
        base_order, context = self._compute_daybreak_base(
            current_demand,
            upstream_data,
            inventory_level,
            backlog_level,
            incoming_shipments,
        )
        order = max(0, int(round(base_order)))
        explanation = self._build_daybreak_explanation(
            week=current_round,
            inventory_level=inventory_level,
            backlog_level=backlog_level,
            incoming_shipments=incoming_shipments,
            context=context,
            action_qty=order,
            prev_action=prev_order,
            base_order=base_order,
        )
        self.last_explanation = explanation
        if self.central_coordinator:
            self.central_coordinator.register_decision(self.agent_type, order)
        return order

    def _daybreak_central_strategy(
        self,
        current_round: int,
        prev_order: Optional[int],
        current_demand: Optional[int],
        upstream_data: Optional[Dict],
        inventory_level: int,
        backlog_level: int,
        incoming_shipments: List[float],
    ) -> int:
        base_order, context = self._compute_daybreak_base(
            current_demand,
            upstream_data,
            inventory_level,
            backlog_level,
            incoming_shipments,
        )
        final_order = max(0, int(round(base_order)))
        supervisor_explanation: Optional[str] = None
        if self.central_coordinator:
            final_order, supervisor_explanation = self.central_coordinator.apply_override(
                self.agent_type,
                base_order,
                context,
                week=current_round,
            )

        role_explanation = self._build_daybreak_explanation(
            week=current_round,
            inventory_level=inventory_level,
            backlog_level=backlog_level,
            incoming_shipments=incoming_shipments,
            context=context,
            action_qty=final_order,
            prev_action=prev_order,
            base_order=base_order,
        )

        if supervisor_explanation:
            if role_explanation:
                role_explanation = f"{role_explanation}\n\n{supervisor_explanation}"
            else:
                role_explanation = supervisor_explanation

        self.last_explanation = role_explanation
        return final_order

    def _daybreak_global_strategy(
        self,
        current_round: int,
        prev_order: Optional[int],
        current_demand: Optional[int],
        upstream_data: Optional[Dict],
        inventory_level: int,
        backlog_level: int,
        incoming_shipments: List[float],
    ) -> int:
        base_order, context = self._compute_daybreak_base(
            current_demand,
            upstream_data,
            inventory_level,
            backlog_level,
            incoming_shipments,
        )
        final_order = max(0, int(round(base_order)))
        global_explanation: Optional[str] = None

        if self.global_controller:
            final_order, global_explanation = self.global_controller.plan_order(
                self.agent_type,
                current_round,
                base_order,
                context,
                prev_order,
            )

        role_explanation = self._build_daybreak_explanation(
            week=current_round,
            inventory_level=inventory_level,
            backlog_level=backlog_level,
            incoming_shipments=incoming_shipments,
            context=context,
            action_qty=final_order,
            prev_action=prev_order,
            base_order=base_order,
        )

        if global_explanation:
            if role_explanation:
                role_explanation = f"{role_explanation}\n\n{global_explanation}"
            else:
                role_explanation = global_explanation

        self.last_explanation = role_explanation

        if self.central_coordinator:
            self.central_coordinator.register_decision(self.agent_type, final_order)

        return final_order

    def _build_daybreak_explanation(
        self,
        week: int,
        inventory_level: int,
        backlog_level: int,
        incoming_shipments: List[float],
        context: Dict[str, Any],
        action_qty: int,
        prev_action: Optional[int],
        base_order: float,
    ) -> str:
        try:
            lead_time = max(1, len(incoming_shipments) or 2)
            pipeline_orders = [int(round(x)) for x in self.order_history[-lead_time:]]
            if not pipeline_orders and self.pipeline:
                pipeline_orders = [int(round(x)) for x in self.pipeline[:lead_time]]
            if not pipeline_orders:
                pipeline_orders = [0] * lead_time

            pipeline_shipments = [int(round(x)) for x in incoming_shipments[:lead_time]]
            if not pipeline_shipments:
                pipeline_shipments = [0] * lead_time

            last_k_in = [int(round(x)) for x in self.demand_history[-lead_time:]] if self.demand_history else []
            last_k_out = [int(round(x)) for x in self.order_history[-lead_time:]] if self.order_history else []
            notes = self._format_daybreak_notes(context)

            obs = Obs(
                on_hand=int(inventory_level),
                backlog=int(backlog_level),
                pipeline_orders=pipeline_orders,
                pipeline_shipments=pipeline_shipments,
                last_k_orders_in=last_k_in,
                last_k_shipments_in=pipeline_shipments,
                last_k_orders_out=last_k_out,
                notes=notes,
            )

            forecast_mean = float(context.get("forecast", 0.0)) if context else 0.0
            forecast_mean_vec = [forecast_mean] * lead_time
            demand_window = self.demand_history[-max(lead_time, 3):]
            forecast_std_vec: Optional[List[float]] = None
            if demand_window and len(demand_window) >= 2:
                std_val = float(statistics.pstdev([float(x) for x in demand_window]))
                if std_val > 0:
                    forecast_std_vec = [std_val] * lead_time

            forecast = Forecast(mean=forecast_mean_vec, std=forecast_std_vec)
            params = RoleParams(
                lead_time=lead_time,
                service_level=0.95,
                capacity_cap=None,
                smoothing_lambda=0.0,
            )

            attribution = self._daybreak_actor_attribution(context, base_order, backlog_level)
            whatifs = self._daybreak_whatifs(inventory_level, backlog_level)

            explanation = explain_role_decision(
                role=self.agent_type.name.replace("_", " ").title(),
                week=week,
                obs=obs,
                action_qty=action_qty,
                forecast=forecast,
                params=params,
                shadow_policy="base_stock",
                actor_attribution=attribution,
                whatif_cfg=whatifs,
                prev_action_qty=prev_action,
            )
            return explanation
        except Exception:
            return f"Decision (Week {week}, {self.agent_type.name.replace('_', ' ').title()}): order **{action_qty}** units upstream."

    def _format_daybreak_notes(self, context: Optional[Dict[str, Any]]) -> Optional[str]:
        if not context:
            return None
        notes: List[str] = []
        upstream_avg = context.get("upstream_avg")
        if upstream_avg is not None:
            notes.append(f"upstream avg {float(upstream_avg):.1f}")
        local_avg = context.get("local_avg")
        if local_avg is not None:
            notes.append(f"recent order avg {float(local_avg):.1f}")
        return "; ".join(notes) if notes else None

    def _daybreak_actor_attribution(
        self,
        context: Optional[Dict[str, Any]],
        base_order: float,
        backlog_level: int,
    ) -> Optional[Dict[str, float]]:
        if not base_order:
            return None
        denom = max(1.0, abs(float(base_order)))
        attribution: Dict[str, float] = {}

        forecast_val = float(context.get("forecast", 0.0)) if context else 0.0
        if forecast_val:
            attribution["forecast_pull"] = max(-1.0, min(forecast_val / denom, 1.0))

        pipeline_val = float(context.get("pipeline", 0.0)) if context else 0.0
        if pipeline_val:
            attribution["pipeline_cover"] = max(-1.0, min(-pipeline_val / denom, 1.0))

        if backlog_level:
            attribution["backlog_pressure"] = max(-1.0, min(backlog_level / denom, 1.0))

        return attribution or None

    def _daybreak_whatifs(
        self,
        inventory_level: int,
        backlog_level: int,
    ) -> Optional[Dict[str, float]]:
        whatifs: Dict[str, float] = {}
        if backlog_level > inventory_level:
            whatifs["demand_scale"] = 1.2
        elif inventory_level > backlog_level * 2 and inventory_level > 0:
            whatifs["demand_scale"] = 0.8
        return whatifs or None

    def get_last_explanation_comment(self) -> Optional[str]:
        if not self.last_explanation:
            return None
        flattened = " | ".join(part.strip() for part in self.last_explanation.splitlines() if part.strip())
        if len(flattened) > 255:
            return f"{flattened[:252]}..."
        return flattened

    def update_inventory(self, incoming_shipment: int, outgoing_shipment: int):
        """Update inventory and backlog based on incoming and outgoing shipments."""
        self.inventory = self.inventory + incoming_shipment - outgoing_shipment
        if self.inventory < 0:
            self.backlog += abs(self.inventory)
            self.inventory = 0
        else:
            self.backlog = max(0, self.backlog - outgoing_shipment)


class AgentManager:
    """Manages multiple agents in the supply chain."""

    def __init__(self, can_see_demand: bool = False):
        self.agents: Dict[AgentType, BeerGameAgent] = {}
        self.can_see_demand = can_see_demand
        self.daybreak_coordinator = DaybreakCoordinator()
        self.daybreak_global_controller = DaybreakGlobalController()
        self.initialize_agents()

    def initialize_agents(self):
        """Initialize agents for each role in the supply chain."""
        self.agents[AgentType.RETAILER] = BeerGameAgent(
            agent_id=1,
            agent_type=AgentType.RETAILER,
            strategy=AgentStrategy.NAIVE,
            can_see_demand=True,  # Retailer can always see demand
            llm_model="gpt-4o-mini",
            central_coordinator=self.daybreak_coordinator,
            global_controller=self.daybreak_global_controller,
        )

        self.agents[AgentType.WHOLESALER] = BeerGameAgent(
            agent_id=2,
            agent_type=AgentType.WHOLESALER,
            strategy=AgentStrategy.NAIVE,
            can_see_demand=self.can_see_demand,
            llm_model="gpt-4o-mini",
            central_coordinator=self.daybreak_coordinator,
            global_controller=self.daybreak_global_controller,
        )

        self.agents[AgentType.DISTRIBUTOR] = BeerGameAgent(
            agent_id=3,
            agent_type=AgentType.DISTRIBUTOR,
            strategy=AgentStrategy.NAIVE,
            can_see_demand=self.can_see_demand,
            llm_model="gpt-4o-mini",
            central_coordinator=self.daybreak_coordinator,
            global_controller=self.daybreak_global_controller,
        )

        self.agents[AgentType.MANUFACTURER] = BeerGameAgent(
            agent_id=4,
            agent_type=AgentType.MANUFACTURER,
            strategy=AgentStrategy.NAIVE,
            can_see_demand=self.can_see_demand,
            llm_model="gpt-4o-mini",
            central_coordinator=self.daybreak_coordinator,
            global_controller=self.daybreak_global_controller,
        )

    def get_agent(self, agent_type: AgentType) -> BeerGameAgent:
        """Get agent by type."""
        return self.agents.get(agent_type)

    def set_agent_strategy(
        self,
        agent_type: AgentType,
        strategy: AgentStrategy,
        llm_model: Optional[str] = None,
        override_pct: Optional[float] = None,
    ):
        """Set strategy and optional Daybreak LLM model for a specific agent."""
        if agent_type in self.agents:
            agent = self.agents[agent_type]
            agent.strategy = strategy
            agent.central_coordinator = self.daybreak_coordinator
            agent.global_controller = self.daybreak_global_controller
            if llm_model is not None:
                agent.llm_model = llm_model
            if strategy == AgentStrategy.DAYBREAK_DTCE_CENTRAL:
                if override_pct is not None:
                    self.daybreak_coordinator.set_override_pct(agent_type, override_pct)
            else:
                self.daybreak_coordinator.set_override_pct(agent_type, None)
            agent.reset_for_strategy()
    
    def set_demand_visibility(self, visible: bool):
        """Set whether agents can see the actual customer demand."""
        self.can_see_demand = visible
        for agent in self.agents.values():
            # Don't change visibility for retailer (always sees demand)
            if agent.agent_type != AgentType.RETAILER:
                agent.can_see_demand = visible
    
    def get_agent_states(self) -> Dict[str, Dict]:
        """Get current state of all agents."""
        return {
            agent_type.value: {
                'inventory': agent.inventory,
                'backlog': agent.backlog,
                'last_order': agent.last_order,
                'order_history': agent.order_history,
                'strategy': agent.strategy.value
            }
            for agent_type, agent in self.agents.items()
        }
