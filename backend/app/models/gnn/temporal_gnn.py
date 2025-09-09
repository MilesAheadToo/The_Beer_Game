import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Dict, Optional, Tuple, List
import numpy as np

class TemporalGNN(nn.Module):
    """
    A simple temporal graph neural network layer that combines graph and temporal information.
    This is a simplified version for debugging purposes.
    """
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 1, heads: int = 1):
        super().__init__()
        self.gat = GATv2Conv(in_channels, out_channels, heads=1, edge_dim=edge_dim)
        self.gru = nn.GRU(out_channels, out_channels, batch_first=True)
        self.edge_dim = edge_dim
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        # x shape: [batch_size, seq_len, num_nodes, in_channels] or [seq_len, num_nodes, in_channels]
        # We need to process each time step separately to avoid dimension issues
        
        # Add batch dimension if missing
        if x.dim() == 3:  # [seq_len, num_nodes, in_channels]
            x = x.unsqueeze(0)  # [1, seq_len, num_nodes, in_channels]
            
        batch_size, seq_len, num_nodes, in_channels = x.size()
        
        # Process each time step separately
        h_list = []
        for t in range(seq_len):
            # Get node features for this time step: [batch_size, num_nodes, in_channels]
            x_t = x[:, t]  # [batch_size, num_nodes, in_channels]
            
            # Process each graph in the batch separately
            batch_h = []
            for b in range(batch_size):
                # Get features for this batch element: [num_nodes, in_channels]
                x_bt = x_t[b]  # [num_nodes, in_channels]
                
                # Apply GAT with edge attributes if provided
                if edge_attr is not None:
                    h_bt = self.gat(x_bt, edge_index, edge_attr)  # [num_nodes, out_channels]
                else:
                    h_bt = self.gat(x_bt, edge_index)  # [num_nodes, out_channels]
                
                batch_h.append(h_bt.unsqueeze(0))  # Add batch dimension back
                
            # Stack batch items: [batch_size, num_nodes, out_channels]
            h_t = torch.cat(batch_h, dim=0)
            h_list.append(h_t.unsqueeze(1))  # Add sequence dimension
        
        # Stack along sequence dimension: [batch_size, seq_len, num_nodes, out_channels]
        h = torch.cat(h_list, dim=1)
        
        # Apply GRU
        h = h.permute(0, 2, 1, 3)  # [batch_size, num_nodes, seq_len, out_channels]
        h = h.reshape(batch_size * num_nodes, seq_len, -1)  # [batch_size * num_nodes, seq_len, out_channels]
        h, _ = self.gru(h)  # [batch_size * num_nodes, seq_len, out_channels]
        h = h.reshape(batch_size, num_nodes, seq_len, -1)  # [batch_size, num_nodes, seq_len, out_channels]
        h = h.permute(0, 2, 1, 3)  # [batch_size, seq_len, num_nodes, out_channels]
        
        return h

class SupplyChainTemporalGNN(nn.Module):
    """
    Temporal Graph Neural Network for supply chain forecasting and decision making.
    Combines GNN with temporal attention for modeling supply chain dynamics.
    """
    
    def __init__(
        self,
        node_features: int = 5,  # inventory, orders, demand, backlog, incoming_shipments
        edge_features: int = 3,  # lead_time, cost, relationship_strength
        hidden_dim: int = 32,   # Reduced from 64 to make it easier to debug
        num_layers: int = 2,    # Reduced from 3 to make it simpler
        num_heads: int = 1,     # Using single head for simplicity
        dropout: float = 0.1,
        seq_len: int = 10,      # Number of time steps to consider
        num_nodes: int = 4,     # retailer, wholesaler, distributor, manufacturer
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        
        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # Store edge dimension
        self.edge_dim = edge_features
        
        # Temporal GNN layers
        self.tgnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.tgnn_layers.append(
                TemporalGNN(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    edge_dim=edge_features,
                    heads=1  # Using single head for simplicity
                )
            )
        
        # Output heads
        self.order_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Predict order quantity
        )
        
        self.demand_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Predict next demand
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        hx: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Node features [batch_size, seq_len, num_nodes, node_features] or [seq_len, num_nodes, node_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            hx: Optional hidden state
            
        Returns:
            Dictionary with 'order_quantity' and 'demand_forecast' predictions
        """
        # Ensure input has 4 dimensions [batch_size, seq_len, num_nodes, node_features]
        print(f"Input x shape: {x.shape}")
        
        # Handle 6D input [1, 1, 1, seq_len, num_nodes, node_features] -> [1, seq_len, num_nodes, node_features]
        if x.dim() == 6:
            # Remove the extra dimensions at indices 1 and 2
            x = x.squeeze(1).squeeze(1)
            print(f"After removing extra dims (6D case), x shape: {x.shape}")
        # Handle 5D input [1, 1, seq_len, num_nodes, node_features] -> [1, seq_len, num_nodes, node_features]
        elif x.dim() == 5:
            # Remove the extra dimension at index 1
            x = x.squeeze(1)
            print(f"After removing extra dim (5D case), x shape: {x.shape}")
        # Handle 3D input [seq_len, num_nodes, node_features] -> [1, seq_len, num_nodes, node_features]
        elif x.dim() == 3:
            # Add batch dimension if missing
            x = x.unsqueeze(0)  # [1, seq_len, num_nodes, node_features]
            print(f"After adding batch dim, x shape: {x.shape}")
        
        try:
            batch_size, seq_len, num_nodes, node_features = x.size()
            print(f"Batch size: {batch_size}, Seq len: {seq_len}, Num nodes: {num_nodes}, Node features: {node_features}")
        except Exception as e:
            print(f"Error unpacking x size: {e}")
            print(f"x size: {x.size()}")
            raise
        
        # Project input features
        x = self.input_proj(x)  # [batch_size, seq_len, num_nodes, hidden_dim]
        print(f"After input projection, x shape: {x.shape}")
        
        # Apply temporal GNN layers
        for tgnn in self.tgnn_layers:
            # Process each time step separately to avoid dimension issues
            time_step_outputs = []
            for t in range(seq_len):
                # Get features for this time step
                x_t = x[:, t]  # [batch_size, num_nodes, hidden_dim]
                
                # Apply temporal GNN
                h_t = tgnn(x_t, edge_index, edge_attr)  # [batch_size, num_nodes, hidden_dim]
                time_step_outputs.append(h_t.unsqueeze(1))  # Add time dimension back
                
            # Stack time steps
            x = torch.cat(time_step_outputs, dim=1)  # [batch_size, seq_len, num_nodes, hidden_dim]
            
        # Take the last time step's output
        x = x[:, -1]  # [batch_size, num_nodes, hidden_dim]
        
        # Process each node separately to avoid dimension issues
        order_outputs = []
        demand_outputs = []
        
        for i in range(x.size(1)):  # Iterate over num_nodes
            # Get features for this node [batch_size, hidden_dim]
            node_features = x[:, i, :]
            
            # Get order quantity for this node
            order = self.order_head(node_features)  # [batch_size, 1]
            order_outputs.append(order)
            
            # Get demand forecast for this node
            demand = self.demand_head(node_features)  # [batch_size, 1]
            demand_outputs.append(demand)
        
        # Stack outputs: [batch_size, num_nodes]
        order_quantity = torch.cat(order_outputs, dim=1)  # [batch_size, num_nodes]
        demand_forecast = torch.cat(demand_outputs, dim=1)  # [batch_size, num_nodes]
        
        return {
            'order_quantity': order_quantity,
            'demand_forecast': demand_forecast
        }

class SupplyChainAgent:
    """
    Agent that uses the TemporalGNN for supply chain decisions.
    """
    def __init__(
        self,
        node_id: int,
        model: Optional[SupplyChainTemporalGNN] = None,
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.node_id = node_id
        # Normalize device into torch.device
        self.device = torch.device(device)
        
        if model is None:
            self.model = SupplyChainTemporalGNN().to(device)
        else:
            self.model = model.to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.hidden_state = None
    
    def reset_hidden_state(self):
        """Reset the hidden state of the model's RNN components."""
        self.hidden_state = None
    
    def act(self, observation: Dict[str, torch.Tensor], training: bool = True) -> torch.Tensor:
        """
        Generate an action (order quantity) based on the observation.
        
        Args:
            observation: Dictionary containing:
                - node_features: [batch_size, seq_len, num_nodes, node_features]
                - edge_index: [2, num_edges]
                - edge_attr: [num_edges, edge_features] (optional)
            training: Whether to use training mode (affects dropout, batch norm, etc.)
            
        Returns:
            action: [batch_size] tensor of order quantities
        """
        self.model.train(training)
        
        # Move data to device
        node_features = observation['node_features'].to(self.device, non_blocking=True)
        edge_index = observation['edge_index'].to(self.device, non_blocking=True)
        edge_attr = observation.get('edge_attr')
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device, non_blocking=True)
        
        # Process one sample at a time
        batch_size = node_features.size(0)
        all_actions = []
        
        for i in range(batch_size):
            # Process one sample at a time
            sample = {
                'node_features': node_features[i:i+1],  # Keep batch dim
                'edge_index': edge_index,
                'edge_attr': edge_attr
            }
            
            # Forward pass for this sample
            with torch.set_grad_enabled(training):
                outputs = self.model(
                    x=sample['node_features'],
                    edge_index=sample['edge_index'],
                    edge_attr=sample['edge_attr'],
                    hx=self.hidden_state
                )
                
                # Get order quantity for this node
                order_quantity = outputs['order_quantity'][0, self.node_id]  # [1] -> scalar
                all_actions.append(order_quantity.unsqueeze(0))
                
                # Update hidden state if using RNN
                if hasattr(self.model, 'hidden_state'):
                    self.hidden_state = outputs.get('hidden_state')
        
        # Stack all actions
        return torch.cat(all_actions, dim=0).detach().cpu()
    
    def update(
        self,
        observations: List[Dict[str, torch.Tensor]],
        actions: List[torch.Tensor],
        rewards: List[float],
        next_observations: List[Dict[str, torch.Tensor]],
        dones: List[bool],
        gamma: float = 0.99
    ) -> Dict[str, float]:
        """
        Update the model using temporal difference learning.
        
        Args:
            observations: List of observation dictionaries
            actions: List of actions taken (indices of order quantities)
            rewards: List of rewards received
            next_observations: List of next observation dictionaries
            dones: List of done flags
            gamma: Discount factor
            
        Returns:
            Dictionary containing loss and other metrics
        """
        self.model.train()
        
        # Process one sample at a time
        total_loss = 0.0
        total_q_value = 0.0
        total_target_q_value = 0.0
        num_samples = len(observations)
        
        for i in range(num_samples):
            # Get current sample
            obs = observations[i]
            next_obs = next_observations[i]
            action = actions[i]  # This is a tensor of shape [batch_size]
            reward = rewards[i]  # This is a scalar
            done = dones[i]     # This is a boolean
            
            # Convert to tensors and move to device
            node_features = obs['node_features'].to(self.device, non_blocking=True)  # [batch_size, seq_len, num_nodes, node_features]
            next_node_features = next_obs['node_features'].to(self.device, non_blocking=True)
            edge_index = obs['edge_index'].to(self.device, non_blocking=True)
            edge_attr = obs.get('edge_attr')
            if edge_attr is not None:
                edge_attr = edge_attr.to(self.device, non_blocking=True)
            
            # Ensure action is a tensor with proper shape [batch_size]
            if torch.is_tensor(action):
                action_tensor = action.to(self.device, non_blocking=True)
            else:
                action_tensor = torch.tensor(action, dtype=torch.long, device=self.device)
            
            # Ensure reward and done are tensors with proper shapes
            if torch.is_tensor(reward):
                reward_tensor = reward.to(self.device, non_blocking=True)
            else:
                reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
                
            if torch.is_tensor(done):
                done_tensor = done.to(self.device, non_blocking=True)
            else:
                done_tensor = torch.tensor(done, dtype=torch.float32, device=self.device)
            
            # Forward pass for current state
            outputs = self.model(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                hx=self.hidden_state
            )
            
            # Get the order quantity tensor [batch_size, num_nodes]
            order_quantity = outputs['order_quantity']  # [1, 4]
            
            # Ensure action indices are within valid range [0, num_nodes-1] and have int64 dtype
            action_tensor = action_tensor.clamp(0, order_quantity.size(1) - 1).to(torch.int64)
            
            # For the current test case, we have batch_size=1 and num_nodes=4
            # The action_tensor has shape [4], representing actions for each node
            # We need to select the Q-value for each node's action
            
            # Reshape order_quantity to [batch_size * num_nodes] and gather using action indices
            # The action_tensor should have the same shape as the flattened order_quantity
            q_values = order_quantity.view(-1).gather(0, action_tensor.view(-1).to(torch.int64))
            
            # Get target Q-values
            with torch.no_grad():
                next_outputs = self.model(
                    x=next_node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    hx=self.hidden_state
                )
                # Max Q-value for next state
                next_q_values = next_outputs['order_quantity'].max(1)[0]
                
                # Ensure shapes match for the TD target
                if reward_tensor.dim() == 0:
                    reward_tensor = reward_tensor.unsqueeze(0)
                if done_tensor.dim() == 0:
                    done_tensor = done_tensor.unsqueeze(0)
                    
                target_q_values = reward_tensor + gamma * (1 - done_tensor) * next_q_values
            
            # Compute loss for this sample
            loss = self.criterion(q_values, target_q_values)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_q_value += q_values.mean().item()
            total_target_q_value += target_q_values.mean().item()
        
        # Return average metrics
        return {
            'total_loss': total_loss / num_samples,
            'q_value': total_q_value / num_samples,
            'target_q_value': total_target_q_value / num_samples
        }

def create_supply_chain_agents(
    num_agents: int = 4,
    shared_model: bool = True,
    **kwargs
) -> List[SupplyChainAgent]:
    """
    Create a list of supply chain agents.
    
    Args:
        num_agents: Number of agents to create (one per node in the supply chain)
        shared_model: Whether agents should share the same model parameters
        **kwargs: Additional arguments to pass to SupplyChainAgent
        
    Returns:
        List of SupplyChainAgent instances
    """
    if shared_model:
        # Create a single shared model
        model = SupplyChainTemporalGNN(**kwargs)
        agents = [
            SupplyChainAgent(node_id=i, model=model, **kwargs)
            for i in range(num_agents)
        ]
    else:
        # Create separate models for each agent
        agents = [
            SupplyChainAgent(node_id=i, model=None, **kwargs)
            for i in range(num_agents)
        ]
    
    return agents
