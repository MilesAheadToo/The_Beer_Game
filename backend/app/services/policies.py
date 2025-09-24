"""Order policies for the Beer Game engine.

This module implements a tiny policy abstraction that can be plugged into the
Beer Game simulation engine.  Policies consume a dictionary of observations for
the current node and return the order quantity that should be placed upstream
for the current tick.

Two policies are provided:

* :class:`NaiveEchoPolicy` – echoes the order that arrived from the downstream
  partner in the previous tick.  This matches the classic "naïve" benchmark in
  which each role simply replaces what was requested from them.
* :class:`PIPolicy` – a lightweight proportional–integral controller that tries
  to keep the inventory position close to a base-stock target.  The controller
  operates on the inventory position (on-hand + pipeline − backlog), which is
  the standard control signal for the Beer Game.

The policies expose a very small state interface so that their internal state
can be serialised along with the engine state between requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


class OrderPolicy:
    """Base interface for order policies."""

    def order(self, obs: Dict[str, Any]) -> int:
        """Return the order quantity for the current period."""

    # The default implementations below make the policy stateless.  Policies
    # that maintain internal state (e.g. the PI controller) override the
    # methods to expose their state so that the engine can persist it.

    def get_state(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the policy state."""
        return {}

    def set_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore the policy state from :func:`get_state`."""
        # Stateless by default – nothing to restore.
        _ = state


class NaiveEchoPolicy(OrderPolicy):
    """Echo the most recent incoming order from the downstream partner."""

    def order(self, obs: Dict[str, Any]) -> int:
        last_incoming_order = int(obs.get("last_incoming_order", 0))
        return max(0, last_incoming_order)


@dataclass
class PIState:
    integral_error: float = 0.0


class PIPolicy(OrderPolicy):
    """Simple PI controller operating on inventory position."""

    def __init__(
        self,
        base_stock: int,
        kp: float = 0.6,
        ki: float = 0.1,
        clamp_min: int = 0,
        clamp_max: Optional[int] = None,
    ) -> None:
        self.base_stock = int(base_stock)
        self.kp = float(kp)
        self.ki = float(ki)
        self.clamp_min = int(clamp_min)
        self.clamp_max = None if clamp_max is None else int(clamp_max)
        self.state = PIState()

    def order(self, obs: Dict[str, Any]) -> int:
        on_hand = int(obs.get("inventory", 0))
        backlog = int(obs.get("backlog", 0))
        pipeline = int(obs.get("pipeline_on_order", 0))
        demand_anchor = int(obs.get("last_incoming_order", 0))

        inv_position = on_hand + pipeline - backlog
        error = self.base_stock - inv_position
        self.state.integral_error += error

        control = self.kp * error + self.ki * self.state.integral_error
        quantity = max(0, int(round(demand_anchor + control)))

        if self.clamp_max is not None:
            quantity = max(self.clamp_min, min(quantity, self.clamp_max))
        else:
            quantity = max(self.clamp_min, quantity)
        return quantity

    def get_state(self) -> Dict[str, Any]:
        return {"integral_error": self.state.integral_error}

    def set_state(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            self.state = PIState()
            return
        try:
            self.state.integral_error = float(state.get("integral_error", 0.0))
        except (TypeError, ValueError):
            self.state.integral_error = 0.0

