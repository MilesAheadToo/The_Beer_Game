"""Core Beer Game simulation engine used for agent-driven games."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, List

from .policies import NaiveEchoPolicy, OrderPolicy

# Lead times are kept constant for now.  They can be made configurable in the
# future if needed.
LEAD_TIME = 2  # inbound shipment delay (periods)
ORDER_LEAD = 2  # order transmission delay upstream (periods)


class Node:
    """Represents a single role in the Beer Game supply chain."""

    def __init__(
        self,
        name: str,
        policy: OrderPolicy,
        *,
        base_stock: int = 20,
        inventory: int = 12,
        backlog: int = 0,
        pipeline_shipments: Iterable[int] | None = None,
        order_pipe: Iterable[int] | None = None,
        last_incoming_order: int = 0,
        cost: float = 0.0,
    ) -> None:
        self.name = name
        self.policy = policy
        self.base_stock = int(base_stock)
        self.inv = int(inventory)
        self.backlog = int(backlog)
        self.pipeline_shipments: Deque[int] = deque(
            pipeline_shipments or [0] * LEAD_TIME, maxlen=LEAD_TIME
        )
        if len(self.pipeline_shipments) < LEAD_TIME:
            while len(self.pipeline_shipments) < LEAD_TIME:
                self.pipeline_shipments.appendleft(0)
        self.order_pipe: Deque[int] = deque(order_pipe or [0] * ORDER_LEAD, maxlen=ORDER_LEAD)
        if len(self.order_pipe) < ORDER_LEAD:
            while len(self.order_pipe) < ORDER_LEAD:
                self.order_pipe.appendleft(0)
        self.last_incoming_order = int(last_incoming_order)
        self.cost = float(cost)

    @property
    def pipeline_on_order(self) -> int:
        return sum(self.pipeline_shipments)

    def receive_shipments(self) -> int:
        arrived = self.pipeline_shipments.popleft()
        self.inv += arrived
        self.pipeline_shipments.append(0)
        return arrived

    def fulfill(self, demand: int) -> int:
        shipped = min(self.inv, demand)
        self.inv -= shipped
        unmet = demand - shipped
        self.backlog += unmet
        return shipped

    def step_order(self) -> int:
        observation = {
            "inventory": self.inv,
            "backlog": self.backlog,
            "pipeline_on_order": self.pipeline_on_order,
            "last_incoming_order": self.last_incoming_order,
            "base_stock": self.base_stock,
        }
        return max(0, int(self.policy.order(observation)))

    def accrue_costs(self, holding_cost: float = 1.0, backlog_cost: float = 2.0) -> float:
        previous = self.cost
        self.cost += holding_cost * max(self.inv, 0) + backlog_cost * self.backlog
        return self.cost - previous

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inventory": self.inv,
            "backlog": self.backlog,
            "pipeline_shipments": list(self.pipeline_shipments),
            "order_pipe": list(self.order_pipe),
            "last_incoming_order": self.last_incoming_order,
            "cost": self.cost,
            "base_stock": self.base_stock,
            "policy_state": self.policy.get_state(),
        }

    @classmethod
    def from_dict(
        cls,
        name: str,
        policy: OrderPolicy,
        state: Dict[str, Any],
    ) -> "Node":
        node = cls(
            name,
            policy,
            base_stock=int(state.get("base_stock", 20)),
            inventory=int(state.get("inventory", 12)),
            backlog=int(state.get("backlog", 0)),
            pipeline_shipments=state.get("pipeline_shipments"),
            order_pipe=state.get("order_pipe"),
            last_incoming_order=int(state.get("last_incoming_order", 0)),
            cost=float(state.get("cost", 0.0)),
        )
        policy.set_state(state.get("policy_state"))
        return node


class BeerLine:
    """Beer Game supply chain consisting of four sequential roles."""

    ROLES = ["Retailer", "Wholesaler", "Distributor", "Factory"]

    def __init__(
        self,
        *,
        role_policies: Dict[str, OrderPolicy] | None = None,
        base_stocks: Dict[str, int] | None = None,
        state: Dict[str, Dict[str, Any]] | None = None,
    ) -> None:
        role_policies = role_policies or {}
        base_stocks = base_stocks or {}

        self.nodes: List[Node] = []
        for role in self.ROLES:
            policy = role_policies.get(role, NaiveEchoPolicy())
            if state and role in state:
                node = Node.from_dict(role, policy, state[role])
            else:
                node = Node(role, policy, base_stock=int(base_stocks.get(role, 20)))
            if base_stocks and role in base_stocks:
                node.base_stock = int(base_stocks[role])
            self.nodes.append(node)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {node.name: node.to_dict() for node in self.nodes}

    def tick(self, customer_demand: int) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}

        # 1) Incoming shipments arrive
        for node in self.nodes:
            arrived = node.receive_shipments()
            stats[node.name] = {
                "incoming_shipment": arrived,
                "inventory_before": node.inv,
                "backlog_before": node.backlog,
            }

        # 2) Determine demand each node must satisfy this period
        demands: List[int] = [0] * len(self.nodes)
        demands[0] = customer_demand + self.nodes[0].backlog
        for idx in range(1, len(self.nodes)):
            due = self.nodes[idx - 1].order_pipe[0]
            demands[idx] = self.nodes[idx].backlog + due
            stats[self.nodes[idx].name]["order_due"] = due
        stats[self.nodes[0].name]["order_due"] = customer_demand

        # 3) Ship downstream and queue shipments for delivery after lead time
        for idx, node in enumerate(self.nodes):
            shipped = node.fulfill(demands[idx])
            stats[node.name]["demand"] = demands[idx]
            stats[node.name]["shipped"] = shipped
            stats[node.name]["inventory_after"] = node.inv
            stats[node.name]["backlog_after"] = node.backlog
            if idx > 0:
                self.nodes[idx - 1].pipeline_shipments[-1] += shipped
            stats[node.name]["outgoing_shipment"] = shipped

        # 4) Advance order pipelines upstream
        for idx in range(len(self.nodes) - 1):
            due_upstream = self.nodes[idx].order_pipe.popleft()
            self.nodes[idx].order_pipe.append(0)
            self.nodes[idx + 1].last_incoming_order = due_upstream
            stats[self.nodes[idx + 1].name]["last_incoming_order"] = due_upstream

        self.nodes[0].last_incoming_order = customer_demand
        stats[self.nodes[0].name]["last_incoming_order"] = customer_demand

        # 5) Place new orders
        for idx, node in enumerate(self.nodes):
            quantity = node.step_order()
            stats[node.name]["order_placed"] = quantity
            if idx < len(self.nodes) - 1:
                node.order_pipe[-1] += quantity
            else:
                # Factory "produces" goods that will arrive after the shipment lead
                node.pipeline_shipments[-1] += quantity
            stats[node.name]["pipeline_on_order"] = node.pipeline_on_order
            stats[node.name]["order_pipe"] = list(node.order_pipe)

        # 6) Accrue costs
        for node in self.nodes:
            cost_added = node.accrue_costs()
            stats[node.name]["cost_added"] = cost_added
            stats[node.name]["total_cost"] = node.cost
            stats[node.name]["pipeline_shipments"] = list(node.pipeline_shipments)

        return stats

    def summary(self) -> Dict[str, Dict[str, Any]]:
        return {
            node.name: {
                "inventory": node.inv,
                "backlog": node.backlog,
                "pipeline": list(node.pipeline_shipments),
                "orders_in_transit": list(node.order_pipe),
                "cost": node.cost,
            }
            for node in self.nodes
        }

