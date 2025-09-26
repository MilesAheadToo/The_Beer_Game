"""Core Beer Game simulation engine used for agent-driven games."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, List

from .policies import NaiveEchoPolicy, OrderPolicy

# Classic Beer Game lead times (can be made configurable via supply chain config)
DEFAULT_SHIPMENT_LEAD_TIME = 2  # inbound shipment delay (periods)
DEFAULT_ORDER_LEAD_TIME = 2  # information / order transmission delay (periods)


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
        shipment_lead_time: int = DEFAULT_SHIPMENT_LEAD_TIME,
        order_lead_time: int = DEFAULT_ORDER_LEAD_TIME,
    ) -> None:
        self.name = name
        self.policy = policy
        self.base_stock = int(base_stock)
        self.inventory = int(inventory)
        self.backlog = int(backlog)

        self.shipment_lead_time = max(1, int(shipment_lead_time) if shipment_lead_time is not None else 1)
        self.order_lead_time = max(1, int(order_lead_time) if order_lead_time is not None else 1)

        self.pipeline_shipments: Deque[int] = deque(
            list(pipeline_shipments)
            if pipeline_shipments is not None
            else [0] * self.shipment_lead_time,
            maxlen=self.shipment_lead_time,
        )
        while len(self.pipeline_shipments) < self.shipment_lead_time:
            self.pipeline_shipments.appendleft(0)

        self.order_pipe: Deque[int] = deque(
            list(order_pipe)
            if order_pipe is not None
            else [0] * self.order_lead_time,
            maxlen=self.order_lead_time,
        )
        while len(self.order_pipe) < self.order_lead_time:
            self.order_pipe.appendleft(0)

        self.last_incoming_order = int(last_incoming_order)
        self.cost = float(cost)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def pipeline_on_order(self) -> int:
        """Total inventory scheduled to arrive in future periods."""

        return sum(self.pipeline_shipments)

    @property
    def inventory_position(self) -> int:
        """Inventory position used by classic Beer Game policies."""

        return self.inventory + self.pipeline_on_order - self.backlog

    # ------------------------------------------------------------------
    # State transition helpers
    # ------------------------------------------------------------------
    def receive_shipment(self) -> int:
        """Advance the shipment pipeline and add arrivals to on-hand inventory."""

        arrived = self.pipeline_shipments.popleft()
        self.pipeline_shipments.append(0)
        self.inventory += arrived
        return arrived

    def shift_order_pipe(self) -> int:
        """Advance the outbound order pipeline (orders travelling upstream)."""

        due_upstream = self.order_pipe.popleft()
        self.order_pipe.append(0)
        return due_upstream

    def schedule_inbound_shipment(self, quantity: int) -> None:
        """Queue a shipment that will arrive after the shipment lead time."""

        qty = int(quantity)
        if qty > 0:
            self.pipeline_shipments[-1] += qty

    def schedule_order(self, quantity: int) -> None:
        """Queue an order that will reach the upstream partner after the delay."""

        qty = int(quantity)
        if qty > 0:
            self.order_pipe[-1] += qty

    def decide_order(self) -> int:
        """Call the node's policy to determine the order for this week."""

        observation = {
            "inventory": self.inventory,
            "backlog": self.backlog,
            "pipeline_on_order": self.pipeline_on_order,
            "last_incoming_order": self.last_incoming_order,
            "base_stock": self.base_stock,
            "inventory_position": self.inventory_position,
        }
        try:
            quantity = self.policy.order(observation)
        except Exception:
            quantity = 0
        return max(0, int(round(quantity)))

    def accrue_costs(self, holding_cost: float = 1.0, backlog_cost: float = 2.0) -> float:
        """Accrue weekly holding and backlog costs."""

        previous = self.cost
        holding = holding_cost * max(self.inventory, 0)
        backlog_penalty = backlog_cost * max(self.backlog, 0)
        self.cost += holding + backlog_penalty
        return self.cost - previous

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "inventory": self.inventory,
            "backlog": self.backlog,
            "pipeline_shipments": list(self.pipeline_shipments),
            "order_pipe": list(self.order_pipe),
            "last_incoming_order": self.last_incoming_order,
            "cost": self.cost,
            "base_stock": self.base_stock,
            "policy_state": self.policy.get_state(),
            "shipment_lead_time": self.shipment_lead_time,
            "order_lead_time": self.order_lead_time,
        }

    @classmethod
    def from_dict(
        cls,
        name: str,
        policy: OrderPolicy,
        state: Dict[str, Any],
        *,
        shipment_lead_time: int = DEFAULT_SHIPMENT_LEAD_TIME,
        order_lead_time: int = DEFAULT_ORDER_LEAD_TIME,
    ) -> "Node":
        state_shipment_lead = state.get("shipment_lead_time", shipment_lead_time)
        state_order_lead = state.get("order_lead_time", order_lead_time)
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
            shipment_lead_time=int(state_shipment_lead),
            order_lead_time=int(state_order_lead),
        )
        policy.set_state(state.get("policy_state"))
        return node


class BeerLine:
    """Beer Game supply chain consisting of four sequential roles."""

    #: Material flows downstream following these lanes.
    #: Each tuple represents (from_node, to_node) for shipments. Orders travel
    #: back upstream, i.e. the opposite direction of these edges.
    MATERIAL_LANES: List[tuple[str, str]] = [
        ("Manufacturer", "Distributor"),
        ("Distributor", "Wholesaler"),
        ("Wholesaler", "Retailer"),
    ]

    _ROLE_CACHE: List[str] | None = None

    def __init__(
        self,
        *,
        role_policies: Dict[str, OrderPolicy] | None = None,
        base_stocks: Dict[str, int] | None = None,
        state: Dict[str, Dict[str, Any]] | None = None,
        shipment_lead_time: int = DEFAULT_SHIPMENT_LEAD_TIME,
        order_lead_time: int = DEFAULT_ORDER_LEAD_TIME,
    ) -> None:
        role_policies = role_policies or {}
        base_stocks = base_stocks or {}

        self.role_names = self.role_sequence_names()
        self.shipment_lead_time = max(1, int(shipment_lead_time) if shipment_lead_time is not None else 1)
        self.order_lead_time = max(1, int(order_lead_time) if order_lead_time is not None else 1)

        self.nodes: List[Node] = []
        for role in self.role_names:
            policy = role_policies.get(role, NaiveEchoPolicy())
            if state and role in state:
                node = Node.from_dict(
                    role,
                    policy,
                    state[role],
                    shipment_lead_time=self.shipment_lead_time,
                    order_lead_time=self.order_lead_time,
                )
            else:
                node = Node(
                    role,
                    policy,
                    base_stock=int(base_stocks.get(role, 20)),
                    shipment_lead_time=self.shipment_lead_time,
                    order_lead_time=self.order_lead_time,
                )
            if base_stocks and role in base_stocks:
                node.base_stock = int(base_stocks[role])
            self.nodes.append(node)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {node.name: node.to_dict() for node in self.nodes}

    # ------------------------------------------------------------------
    # Lane/role helpers
    # ------------------------------------------------------------------
    @classmethod
    def _derive_role_sequence(cls) -> List[str]:
        """Return the downstream-to-upstream role order implied by the lanes."""

        nodes = set()
        adjacency: Dict[str, List[str]] = {}
        indegree: Dict[str, int] = {}
        for upstream, downstream in cls.MATERIAL_LANES:
            nodes.add(upstream)
            nodes.add(downstream)
            adjacency.setdefault(upstream, []).append(downstream)
            indegree.setdefault(upstream, 0)
            indegree[downstream] = indegree.get(downstream, 0) + 1

        queue: Deque[str] = deque(sorted(node for node in nodes if indegree.get(node, 0) == 0))
        topo: List[str] = []
        while queue:
            current = queue.popleft()
            topo.append(current)
            for neighbour in adjacency.get(current, []):
                indegree[neighbour] -= 1
                if indegree[neighbour] == 0:
                    queue.append(neighbour)

        if len(topo) != len(nodes):
            raise ValueError("Material lanes contain a cycle; cannot derive role order")

        # The topological order walks with shipments (upstream -> downstream);
        # we keep nodes indexed from downstream -> upstream because customer
        # demand enters at index 0 (Retailer) in the classic Beer Game.
        return list(reversed(topo))

    @classmethod
    def role_sequence_names(cls) -> List[str]:
        if cls._ROLE_CACHE is None:
            cls._ROLE_CACHE = cls._derive_role_sequence()
        # Return a shallow copy to avoid accidental mutation by callers.
        return list(cls._ROLE_CACHE)

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------
    def tick(self, customer_demand: int) -> Dict[str, Dict[str, Any]]:
        """Advance the supply chain by one week following the Beer Game cookbook."""

        demand = max(0, int(customer_demand))
        stats: Dict[str, Dict[str, Any]] = {}

        # Step 1 – Receive inbound shipments (advance shipment pipelines)
        for node in self.nodes:
            inventory_previous = node.inventory
            arrived = node.receive_shipment()
            stats[node.name] = {
                "incoming_shipment": arrived,
                "inventory_previous": inventory_previous,
                "inventory_before": node.inventory,
                "backlog_before": node.backlog,
            }

        # Step 2 – Observe inbound orders and advance outbound order pipelines
        incoming_orders: List[int] = [0] * len(self.nodes)
        incoming_orders[0] = demand
        self.nodes[0].last_incoming_order = demand
        stats[self.nodes[0].name]["incoming_order"] = demand
        stats[self.nodes[0].name]["order_due"] = demand
        stats[self.nodes[0].name]["last_incoming_order"] = demand

        for idx in range(len(self.nodes) - 1):
            downstream = self.nodes[idx]
            upstream = self.nodes[idx + 1]
            due_upstream = downstream.shift_order_pipe()
            incoming_orders[idx + 1] = due_upstream
            upstream.last_incoming_order = due_upstream
            stats[upstream.name]["incoming_order"] = due_upstream
            stats[upstream.name]["order_due"] = due_upstream
            stats[upstream.name]["last_incoming_order"] = due_upstream
            stats[downstream.name]["order_pipe"] = list(downstream.order_pipe)

        # Ensure the manufacturer has order_pipe stats even though it has no upstream
        stats[self.nodes[-1].name]["order_pipe"] = list(self.nodes[-1].order_pipe)

        # Step 3 – Ship to downstream partner, prioritising backlog first
        for idx, node in enumerate(self.nodes):
            need = node.backlog + incoming_orders[idx]
            shipped = min(node.inventory, need)
            node.inventory -= shipped
            node.backlog = max(need - shipped, 0)

            stats[node.name].update(
                {
                    "demand": need,
                    "shipped": shipped,
                    "outgoing_shipment": shipped,
                    "inventory_after": node.inventory,
                    "backlog_after": node.backlog,
                    "inventory_position": node.inventory - node.backlog,
                }
            )

            if idx > 0:
                downstream = self.nodes[idx - 1]
                downstream.schedule_inbound_shipment(shipped)

        # Step 4/5 – Decide new orders and queue them in the outbound pipelines
        for idx, node in enumerate(self.nodes):
            order_qty = node.decide_order()
            stats[node.name]["order_placed"] = order_qty

            if idx < len(self.nodes) - 1:
                node.schedule_order(order_qty)
                stats[node.name]["order_pipe"] = list(node.order_pipe)
            else:
                # Manufacturer starts production that feeds its own shipment pipeline
                node.schedule_inbound_shipment(order_qty)

            stats[node.name]["pipeline_on_order"] = node.pipeline_on_order
            stats[node.name]["inventory_position_with_pipeline"] = node.inventory_position
            stats[node.name]["inventory_position"] = node.inventory - node.backlog

        # Step 6/7 – Accrue costs and expose pipeline snapshots
        for node in self.nodes:
            cost_added = node.accrue_costs()
            stats[node.name]["cost_added"] = cost_added
            stats[node.name]["total_cost"] = node.cost
            stats[node.name]["pipeline_shipments"] = list(node.pipeline_shipments)

        return stats

    def summary(self) -> Dict[str, Dict[str, Any]]:
        return {
            node.name: {
                "inventory": node.inventory,
                "backlog": node.backlog,
                "pipeline": list(node.pipeline_shipments),
                "orders_in_transit": list(node.order_pipe),
                "cost": node.cost,
            }
            for node in self.nodes
        }
