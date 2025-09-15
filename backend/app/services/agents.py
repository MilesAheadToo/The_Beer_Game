from typing import List, Dict, Optional
import random
from enum import Enum

# Import the LLM agent only when needed to avoid unnecessary dependencies
try:  # pragma: no cover - optional import
    from .llm_agent import LLMAgent, LLMStrategy
except Exception:  # pragma: no cover - tests may not have openai deps
    LLMAgent = None  # type: ignore
    LLMStrategy = None  # type: ignore

class AgentType(Enum):
    RETAILER = "retailer"
    WHOLESALER = "wholesaler"
    DISTRIBUTOR = "distributor"
    FACTORY = "factory"

class AgentStrategy(Enum):
    NAIVE = "naive"  # Simple strategy, always orders based on current demand
    BULLWHIP = "bullwhip"  # Tends to over-order when demand increases
    CONSERVATIVE = "conservative"  # Maintains stable orders
    RANDOM = "random"  # Random ordering for baseline
    LLM = "llm"  # Large Language Model strategy

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
        # LLM specific configuration
        self.llm_model = llm_model
        self._llm_agent: Optional[LLMAgent] = None
        
    def make_decision(
        self, 
        current_round: int,
        current_demand: Optional[int] = None,
        upstream_data: Optional[Dict] = None
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
        
        # Make decision based on strategy
        if self.strategy == AgentStrategy.NAIVE:
            order = self._naive_strategy(current_demand)
        elif self.strategy == AgentStrategy.BULLWHIP:
            order = self._bullwhip_strategy(current_demand, upstream_data)
        elif self.strategy == AgentStrategy.CONSERVATIVE:
            order = self._conservative_strategy(current_demand)
        elif self.strategy == AgentStrategy.LLM:
            order = self._llm_strategy(current_round, current_demand, upstream_data)
        else:  # RANDOM
            order = self._random_strategy()
            
        self.last_order = order
        self.order_history.append(order)
        return order
    
    def _naive_strategy(self, current_demand: Optional[int]) -> int:
        """Order exactly what was demanded this round."""
        if current_demand is not None and (self.agent_type == AgentType.RETAILER or self.can_see_demand):
            return current_demand
        return self.last_order  # Default to last order if no demand info
    
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
    
    def _random_strategy(self) -> int:
        """Make random orders for baseline testing."""
        return random.randint(1, 8)

    def _llm_strategy(
        self,
        current_round: int,
        current_demand: Optional[int],
        upstream_data: Optional[Dict],
    ) -> int:
        """Use an LLM-backed agent to decide the order quantity."""
        if LLMAgent is None:  # LLM dependencies not available
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
                # Fallback to simple strategy if initialization fails
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
            # Fallback if the LLM call fails for any reason
            return self._naive_strategy(current_demand)
    
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
        self.initialize_agents()
    
    def initialize_agents(self):
        """Initialize agents for each role in the supply chain."""
        self.agents[AgentType.RETAILER] = BeerGameAgent(
            agent_id=1,
            agent_type=AgentType.RETAILER,
            strategy=AgentStrategy.NAIVE,
            can_see_demand=True,  # Retailer can always see demand
            llm_model="gpt-4o-mini",
        )
        
        self.agents[AgentType.WHOLESALER] = BeerGameAgent(
            agent_id=2,
            agent_type=AgentType.WHOLESALER,
            strategy=AgentStrategy.NAIVE,
            can_see_demand=self.can_see_demand,
            llm_model="gpt-4o-mini",
        )
        
        self.agents[AgentType.DISTRIBUTOR] = BeerGameAgent(
            agent_id=3,
            agent_type=AgentType.DISTRIBUTOR,
            strategy=AgentStrategy.NAIVE,
            can_see_demand=self.can_see_demand,
            llm_model="gpt-4o-mini",
        )
        
        self.agents[AgentType.FACTORY] = BeerGameAgent(
            agent_id=4,
            agent_type=AgentType.FACTORY,
            strategy=AgentStrategy.NAIVE,
            can_see_demand=self.can_see_demand,
            llm_model="gpt-4o-mini",
        )
    
    def get_agent(self, agent_type: AgentType) -> BeerGameAgent:
        """Get agent by type."""
        return self.agents.get(agent_type)
    
    def set_agent_strategy(
        self,
        agent_type: AgentType,
        strategy: AgentStrategy,
        llm_model: Optional[str] = None,
    ):
        """Set strategy and optional LLM model for a specific agent."""
        if agent_type in self.agents:
            agent = self.agents[agent_type]
            agent.strategy = strategy
            if llm_model is not None:
                agent.llm_model = llm_model
    
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
