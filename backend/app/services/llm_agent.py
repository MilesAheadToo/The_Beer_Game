import os
from typing import Dict, Any, Optional
import openai
from enum import Enum
import json

class LLMStrategy(Enum):
    """Different LLM prompting strategies for the Beer Game."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"

class LLMAgent:
    """LLM-based agent for the Beer Game that uses OpenAI's API."""

    def __init__(
        self,
        role: str,
        strategy: LLMStrategy = LLMStrategy.BALANCED,
        model: str = "gpt-4",
    ):
        self.role = role
        self.strategy = strategy
        self.model = model
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        openai.api_key = self.openai_api_key
        self.conversation_history = []
        self.initialize_agent()
    
    def initialize_agent(self):
        """Initialize the agent with role-specific instructions."""
        role_context = {
            "retailer": "You are the Retailer in the Beer Game supply chain. "
                       "You receive customer demand and place orders to the Wholesaler. "
                       "Your goal is to minimize costs while maintaining good service levels.",
            "wholesaler": "You are the Wholesaler in the Beer Game supply chain. "
                         "You receive orders from the Retailer and place orders to the Distributor. "
                         "You need to balance inventory costs with order fulfillment.",
            "distributor": "You are the Distributor in the Beer Game supply chain. "
                          "You receive orders from the Wholesaler and place orders to the Manufacturer. "
                          "You need to manage the bullwhip effect in the supply chain.",
            "manufacturer": "You are the Manufacturer in the Beer Game supply chain. "
                           "You receive orders from the Distributor and produce finished goods. "
                           "You need to manage production capacity and lead times effectively."
        }
        
        strategy_context = {
            LLMStrategy.CONSERVATIVE: "You are a conservative player who prioritizes maintaining high inventory levels "
                                   "to avoid stockouts, even if it means higher holding costs.",
            LLMStrategy.BALANCED: "You are a balanced player who tries to maintain moderate inventory levels "
                                "while being responsive to demand changes.",
            LLMStrategy.AGGRESSIVE: "You are an aggressive player who keeps inventory levels low to minimize holding costs, "
                                 "but this increases the risk of stockouts.",
            LLMStrategy.ADAPTIVE: "You are an adaptive player who adjusts strategy based on the current game state, "
                               "demand patterns, and supply chain dynamics."
        }
        
        self.system_prompt = f"""
        {role_context.get(self.role.lower(), "")}
        {strategy_context.get(self.strategy, "")}
        
        Game Rules:
        1. Each round, you'll receive information about current inventory, backorders, and incoming shipments.
        2. You need to decide how many units to order from your immediate upstream partner.
        3. Your goal is to minimize total costs, which include:
           - Holding costs: $0.5 per unit per round
           - Backorder costs: $2 per unit per round
        4. There is a 2-round delay between placing an order and receiving it.
        5. You'll receive information about recent demand patterns.
        
        Respond with a JSON object containing:
        {
            "order_quantity": number,  // Number of units to order (must be >= 0)
            "reasoning": string       // Your reasoning for this decision
        }
        """
        
        # Initialize conversation with system message
        self.add_message("system", self.system_prompt)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
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
        # Prepare the prompt with current game state
        prompt = f"""Current Game State (Round {current_round}):
        - Current Inventory: {current_inventory} units
        - Current Backorders: {backorders} units
        - Incoming Shipments (next 2 rounds): {incoming_shipments}
        - Recent Demand: {demand_history[-5:] if len(demand_history) > 0 else 'No history'}
        - Your Recent Orders: {order_history[-5:] if len(order_history) > 0 else 'No history'}
        """
        
        if current_demand is not None:
            prompt += f"\n- Current Customer Demand: {current_demand} units"
        
        if upstream_data:
            prompt += "\n\nUpstream Information:"
            for key, value in upstream_data.items():
                prompt += f"\n- {key.replace('_', ' ').title()}: {value}"
        
        prompt += "\n\nHow many units would you like to order this round?"
        
        try:
            # Add user message to conversation
            self.add_message("user", prompt)
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7 if self.strategy == LLMStrategy.ADAPTIVE else 0.3,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            # Extract and parse the response
            response_content = response.choices[0].message.content
            self.add_message("assistant", response_content)
            
            # Parse the JSON response
            try:
                response_json = json.loads(response_content)
                order_quantity = max(0, int(response_json.get("order_quantity", 0)))
                reasoning = response_json.get("reasoning", "No reasoning provided")
                
                print(f"LLM Agent ({self.role}, {self.strategy.value}) order: {order_quantity}")
                print(f"Reasoning: {reasoning}")
                
                return order_quantity
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing LLM response: {e}")
                print(f"Response was: {response_content}")
                return 0
                
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Fallback to a simple strategy if API call fails
            return self._fallback_strategy(current_inventory, backorders, current_demand)
    
    def _fallback_strategy(
        self, 
        current_inventory: int, 
        backorders: int,
        current_demand: Optional[int] = None
    ) -> int:
        """Fallback strategy if LLM call fails."""
        if current_demand is None:
            current_demand = 8  # Default average demand
            
        if self.strategy == LLMStrategy.CONSERVATIVE:
            return max(0, current_demand * 2 - current_inventory + backorders)
        elif self.strategy == LLMStrategy.AGGRESSIVE:
            return max(0, current_demand - current_inventory + backorders)
        else:  # BALANCED or ADAPTIVE
            return max(0, int(current_demand * 1.5) - current_inventory + backorders)

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
