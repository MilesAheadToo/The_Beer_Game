import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Dict, Optional, Tuple, List, Union, Any
import numpy as np

from app.utils.device import (
    get_available_device, 
    to_device, 
    device_scope,
    is_cuda_available,
    empty_cache
)

# Type aliases
Tensor = torch.Tensor
Device = Union[str, torch.device]

class TemporalGNN(nn.Module):
    """
    A simple temporal graph neural network layer that combines graph and temporal information.
    This is a simplified version for debugging purposes.
    """
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 3, heads: int = 1,
                 device: Optional[Device] = None):
        super().__init__()
        self.device = get_available_device(device)
        self.edge_dim = edge_dim  # Should match the edge_attr dimension (3 in our case)
        self.in_channels = in_channels
        
        # Ensure the device is properly set
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
            
        # Move all parameters to the specified device during initialization
        self._register_load_state_dict_pre_hook(self._move_to_device_hook)
        self.out_channels = out_channels
        
        # Initialize layers with proper device handling
        # Note: edge_dim should match the dimension of edge_attr (3 in our case)
        self.gat = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=1,
            edge_dim=edge_dim,
            add_self_loops=True
        )
        
        self.gru = nn.GRU(
            input_size=out_channels,
            hidden_size=out_channels,
            batch_first=True
        )
        
        # Move to device after initialization
        self.to(self.device)
        
    def _move_to_device_hook(self, state_dict, prefix, *args, **kwargs):
        """Ensure all parameters and buffers are moved to the correct device."""
        for key, param in self.named_parameters():
            if param is not None:
                param.data = param.data.to(self.device)
        for key, buf in self.named_buffers():
            if buf is not None:
                buf.data = buf.data.to(self.device)
                
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        # Ensure all input tensors are on the correct device
        with device_scope(self.device):
            x = to_device(x, self.device, non_blocking=True)
            edge_index = to_device(edge_index, self.device, non_blocking=True)
            if edge_attr is not None:
                edge_attr = to_device(edge_attr, self.device, non_blocking=True)
                
            # Ensure model parameters are on the correct device
            self.gat = self.gat.to(self.device)
            self.gru = self.gru.to(self.device)
            
            # x shape: [batch_size, seq_len, num_nodes, in_channels] or [seq_len, num_nodes, in_channels]
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
                
                # Ensure all tensors are on the same device and have correct dimensions
                device = next(self.parameters()).device
                x_bt = x_bt.to(device)
                edge_idx = edge_index.to(device)
                
                # Ensure edge_attr has the correct shape [num_edges, edge_dim]
                if edge_attr is not None:
                    edge_attr_dev = edge_attr.to(device)
                    # Ensure edge_attr has shape [num_edges, edge_dim]
                    if edge_attr_dev.dim() == 1:
                        edge_attr_dev = edge_attr_dev.unsqueeze(-1)  # [num_edges, 1]
                    elif edge_attr_dev.dim() > 2:
                        edge_attr_dev = edge_attr_dev.view(edge_attr_dev.size(0), -1)  # Flatten to [num_edges, edge_dim]
                else:
                    edge_attr_dev = None
                
                # Ensure GAT is on the same device
                self.gat = self.gat.to(device)
                
                # Apply GAT with edge attributes if provided
                try:
                    if edge_attr_dev is not None:
                        # Ensure edge_attr has the correct shape [num_edges, edge_dim]
                        if edge_attr_dev.size(1) != self.edge_dim:
                            # If edge_attr has wrong dimension, project it to the correct dimension
                            if hasattr(self, 'edge_proj'):
                                edge_attr_dev = self.edge_proj(edge_attr_dev)
                            else:
                                # Create a projection layer if it doesn't exist
                                self.edge_proj = nn.Linear(edge_attr_dev.size(1), self.edge_dim).to(device)
                                edge_attr_dev = self.edge_proj(edge_attr_dev)
                        
                        h_bt = self.gat(x_bt, edge_idx, edge_attr=edge_attr_dev)  # [num_nodes, out_channels]
                    else:
                        h_bt = self.gat(x_bt, edge_idx)  # [num_nodes, out_channels]
                except RuntimeError as e:
                    print(f"Error in GAT forward pass:")
                    print(f"x_bt device: {x_bt.device}, shape: {x_bt.shape}")
                    print(f"edge_index device: {edge_idx.device}, shape: {edge_idx.shape}")
                    if edge_attr_dev is not None:
                        print(f"edge_attr device: {edge_attr_dev.device}, shape: {edge_attr_dev.shape}")
                    print(f"GAT device: {next(self.gat.parameters()).device if next(self.gat.parameters(), None) is not None else 'no parameters'}")
                    raise
                
                batch_h.append(h_bt.unsqueeze(0))  # Add batch dimension back
                
            # Stack batch items: [batch_size, num_nodes, out_channels]
            h_t = torch.cat(batch_h, dim=0)
            h_list.append(h_t.unsqueeze(1))  # Add sequence dimension
        
        # Stack along sequence dimension: [batch_size, seq_len, num_nodes, out_channels]
        h = torch.cat(h_list, dim=1)
        
        # Apply GRU
        h = h.permute(0, 2, 1, 3)  # [batch_size, num_nodes, seq_len, out_channels]
        h = h.reshape(batch_size * num_nodes, seq_len, -1)  # [batch_size * num_nodes, seq_len, out_channels]
        
        # Ensure GRU is on the correct device
        self.gru = self.gru.to(h.device)
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
        device: Optional[Device] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        
        # Initialize device handling
        self.device = get_available_device(device)
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
            
        # Store edge dimension
        self.edge_dim = edge_features
        
        # Register hook to ensure parameters are moved to the correct device
        self._register_load_state_dict_pre_hook(self._move_to_device_hook)
        
        # Initialize layers
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim) if edge_features > 0 else None
        
        # Create temporal GNN layers
        self.tgnn_layers = nn.ModuleList([
            TemporalGNN(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=edge_features,
                heads=1,  # Using single head for simplicity
                device=self.device
            ) for _ in range(num_layers)
        ])
        
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
        
        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # Initialize weights and move to device
        self._init_weights()
        self.to(self.device)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _move_to_device_hook(self, state_dict, prefix, *args, **kwargs):
        """Ensure all parameters and buffers are moved to the correct device."""
        for key, param in self.named_parameters():
            if param is not None:
                param.data = param.data.to(self.device)
        for key, buf in self.named_buffers():
            if buf is not None:
                buf.data = buf.data.to(self.device)
                
    def forward(
        self,
        node_features: torch.Tensor,  # [batch_size, seq_len, num_nodes, node_features] or [batch_size, 1, seq_len, num_nodes, node_features]
        edge_index: torch.Tensor,     # [2, num_edges]
        edge_attr: Optional[torch.Tensor] = None  # [num_edges, edge_features]
    ) -> Dict[str, torch.Tensor]:
        with device_scope(self.device):
            # Move all inputs to the correct device
            node_features = to_device(node_features, self.device, non_blocking=True)
            edge_index = to_device(edge_index, self.device, non_blocking=True)
            if edge_attr is not None:
                edge_attr = to_device(edge_attr, self.device, non_blocking=True)
            
            # Debug shapes
            print(f"Input node_features shape: {node_features.shape}")
            print(f"Input edge_index shape: {edge_index.shape}")
            if edge_attr is not None:
                print(f"Input edge_attr shape: {edge_attr.shape}")
            
            # Handle both 5D and 6D input shapes
            original_shape = node_features.shape
            if len(original_shape) == 6:
                # Reshape from [batch_size, 1, 1, seq_len, num_nodes, node_features] to [batch_size, seq_len, num_nodes, node_features]
                node_features = node_features.squeeze(1).squeeze(1)
                print(f"Reshaped node_features from {original_shape} to {node_features.shape}")
            elif len(original_shape) == 5:
                # Reshape from [batch_size, 1, seq_len, num_nodes, node_features] to [batch_size, seq_len, num_nodes, node_features]
                node_features = node_features.squeeze(1)
                print(f"Reshaped node_features from {original_shape} to {node_features.shape}")
            
            try:
                batch_size, seq_len, num_nodes, node_feat_dim = node_features.size()
                print(f"Successfully unpacked shape: batch_size={batch_size}, seq_len={seq_len}, num_nodes={num_nodes}, node_feat_dim={node_feat_dim}")
            except Exception as e:
                print(f"Error unpacking node_features shape: {node_features.shape}")
                raise
            
            try:
                # Encode node and edge features
                h = self.node_encoder(node_features)  # [batch_size, seq_len, num_nodes, hidden_dim]
                
                # Encode edge attributes if provided
                if edge_attr is not None and self.edge_encoder is not None:
                    edge_embeddings = self.edge_encoder(edge_attr)  # [num_edges, hidden_dim]
                else:
                    edge_embeddings = None
                    
                # Apply temporal GNN layers
                for layer in self.tgnn_layers:
                    h = layer(h, edge_index, edge_embeddings)  # [batch_size, seq_len, num_nodes, hidden_dim]
                    
                # Get the last time step's hidden state for each node
                last_hidden = h[:, -1]  # [batch_size, num_nodes, hidden_dim]
                
                # Predict order quantities and demand forecasts
                order_quantities = self.order_head(last_hidden).squeeze(-1)  # [batch_size, num_nodes]
                demand_forecasts = self.demand_head(last_hidden).squeeze(-1)  # [batch_size, num_nodes]
                
                # Return node embeddings and predictions
                return {
                    'order_quantity': order_quantities,  # [batch_size, num_nodes]
                    'demand_forecast': demand_forecasts,  # [batch_size, num_nodes]
                    'node_embeddings': last_hidden  # [batch_size, num_nodes, hidden_dim]
                }
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and is_cuda_available():
                    # Clear CUDA cache and retry once
                    empty_cache()
                    return self.forward(node_features, edge_index, edge_attr)
                raise
                
    def forward_original(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        hx: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        # Ensure tensors are on the correct device
        x = to_device(x, self.device)
        edge_index = to_device(edge_index, self.device)
        if edge_attr is not None:
            edge_attr = to_device(edge_attr, self.device)
        if hx is not None:
            hx = to_device(hx, self.device)
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
    Handles device management and training/inference for a single node in the supply chain.
    """
    def __init__(
        self,
        node_id: int,
        model: Optional[SupplyChainTemporalGNN] = None,
        learning_rate: float = 1e-3,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.node_id = node_id
        
        # Get device, defaulting to auto-detect if not specified
        self.device = get_available_device(device)
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        
        # Initialize model
        if model is None:
            self.model = SupplyChainTemporalGNN(device=self.device)
        else:
            self.model = model.to(self.device)
        
        # Initialize optimizer with parameters on the correct device
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        self.criterion = nn.MSELoss().to(self.device)
        self.hidden_state = None
        
        # Set model to evaluation mode by default
        self.model.eval()
    
    def reset_hidden_state(self):
        """Reset the hidden state of the model's RNN components."""
        self.hidden_state = None
    
    def act(self, observation: Dict[str, torch.Tensor], training: bool = False) -> torch.Tensor:
        """
        Generate an action (order quantity) based on the observation.
        
        Args:
            observation: Dictionary containing:
                - node_features: [batch_size, seq_len, num_nodes, node_features]
                - edge_index: [2, num_edges]
                - edge_attr: [num_edges, edge_features] (optional)
            training: Whether to use training mode (affects dropout, batch norm, etc.)
            
        Returns:
            action: [batch_size] tensor of order quantities on CPU
        """
        with device_scope(self.device):
            self.model.train(training)
            
            try:
                # Ensure model is on the correct device
                self.model = self.model.to(self.device)
                
                # Move data to device with non-blocking transfer if possible
                node_features = to_device(observation['node_features'], self.device, non_blocking=True)
                edge_index = to_device(observation['edge_index'], self.device, non_blocking=True)
                edge_attr = to_device(observation.get('edge_attr'), self.device, non_blocking=True)
                
                # Process one sample at a time for stability
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
                            node_features=sample['node_features'],
                            edge_index=sample['edge_index'],
                            edge_attr=sample['edge_attr']
                        )
                        
                        # Get order quantity for this node and ensure it's on CPU
                        order_quantity = outputs['order_quantity'][0, self.node_id]  # [1] -> scalar
                        all_actions.append(order_quantity.unsqueeze(0).cpu())
                        
                        # Update hidden state if using RNN
                        if hasattr(self, 'hidden_state') and 'hidden_state' in outputs:
                            self.hidden_state = outputs['hidden_state']
                
                # Stack all actions (already on CPU)
                return torch.cat(all_actions, dim=0).detach()
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and is_cuda_available():
                    # Clear CUDA cache and retry once
                    empty_cache()
                    return self.act(observation, training)
                raise
    
    def update(
        self,
        observations: List[Dict[str, torch.Tensor]],
        actions: List[torch.Tensor],
        rewards: List[float],
        next_observations: List[Dict[str, torch.Tensor]],
        dones: List[bool],
        gamma: float = 0.99,
        clip_grad_norm: Optional[float] = 1.0
    ) -> Dict[str, float]:
        with device_scope(self.device):
            # Ensure model is in training mode and on the correct device
            self.model.train()
            self.model = self.model.to(self.device)
            
            # Move data to device
            obs_batch = []
            next_obs_batch = []
            
            # Use a single device transfer for all tensors
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                for obs, next_obs in zip(observations, next_observations):
                    # Move observation data to device
                    obs_device = {
                        'node_features': to_device(obs['node_features'], self.device, non_blocking=True),
                        'edge_index': to_device(obs['edge_index'], self.device, non_blocking=True)
                    }
                    if 'edge_attr' in obs and obs['edge_attr'] is not None:
                        obs_device['edge_attr'] = to_device(obs['edge_attr'], self.device, non_blocking=True)
                    obs_batch.append(obs_device)
                    
                    # Move next observation data to device
                    next_obs_device = {
                        'node_features': to_device(next_obs['node_features'], self.device, non_blocking=True),
                        'edge_index': to_device(next_obs['edge_index'], self.device, non_blocking=True)
                    }
                    if 'edge_attr' in next_obs and next_obs.get('edge_attr') is not None:
                        next_obs_device['edge_attr'] = to_device(next_obs['edge_attr'], self.device, non_blocking=True)
                    next_obs_batch.append(next_obs_device)
            
            # Initialize metrics
            metrics = {
                'loss': 0.0,
                'q_value': 0.0,
                'target_q_value': 0.0,
                'grad_norm': 0.0
            }
            num_samples = len(observations)
            
            # Process one sample at a time
            for i in range(num_samples):
                try:
                    # Get current observation
                    obs_data = obs_batch[i]
                    next_obs_data = next_obs_batch[i]
                    
                    # Ensure node_features has shape [batch_size, seq_len, num_nodes, node_features]
                    node_features = obs_data['node_features']
                    if len(node_features.shape) == 3:
                        node_features = node_features.unsqueeze(1)  # Add seq_len dimension if missing
                    
                    next_node_features = next_obs_data['node_features']
                    if len(next_node_features.shape) == 3:
                        next_node_features = next_node_features.unsqueeze(1)
                    
                    # Create observation dictionaries
                    obs = {
                        'node_features': node_features,
                        'edge_index': obs_data['edge_index'],
                        'edge_attr': obs_data.get('edge_attr')
                    }
                    
                    next_obs = {
                        'node_features': next_node_features,
                        'edge_index': next_obs_data['edge_index'],
                        'edge_attr': next_obs_data.get('edge_attr')
                    }
                    
                    # Convert to tensors if needed
                    action = to_device(actions[i], self.device, non_blocking=True)
                    reward = to_device(torch.tensor(rewards[i], dtype=torch.float32), self.device, non_blocking=True)
                    done = to_device(torch.tensor(dones[i], dtype=torch.float32), self.device, non_blocking=True)
                    
                    # Ensure action has proper shape and type
                    if not isinstance(action, torch.Tensor):
                        action = torch.tensor(action, dtype=torch.long, device=self.device)
                    
                    # Forward pass for current state
                    self.optimizer.zero_grad()
                    
                    # Debug shapes
                    print(f"node_features shape: {obs['node_features'].shape}")
                    print(f"edge_index shape: {obs['edge_index'].shape}")
                    if obs['edge_attr'] is not None:
                        print(f"edge_attr shape: {obs['edge_attr'].shape}")
                    
                    outputs = self.model(
                        node_features=obs['node_features'],
                        edge_index=obs['edge_index'],
                        edge_attr=obs['edge_attr']
                    )
                    
                    # Get Q-values and select the taken actions
                    # q_values shape: [batch_size, num_nodes] where num_nodes=4 (retailer, wholesaler, distributor, manufacturer)
                    q_values = outputs['order_quantity']
                    
                    # Select Q-values for the current agent's node
                    # Since each agent corresponds to a specific node in the graph,
                    # we use the node_id to select the appropriate Q-value
                    current_q = q_values[:, self.node_id].unsqueeze(1)  # [batch_size, 1]
                    
                    # Compute target Q-values using target network (if available) or current network
                    with torch.no_grad():
                        next_outputs = self.model(
                            node_features=next_obs['node_features'],
                            edge_index=next_obs['edge_index'],
                            edge_attr=next_obs['edge_attr']
                        )
                        next_q_values = next_outputs['order_quantity']
                        # Select next Q-values for the current agent's node
                        next_q = next_q_values[:, self.node_id]  # [batch_size]
                        target_q = reward + (1 - done) * gamma * next_q
                    
                    # Compute loss
                    loss = self.criterion(current_q.squeeze(), target_q.detach())
                    
                    # Backward pass and optimize
                    loss.backward()
                    
                    # Clip gradients if specified
                    if clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            clip_grad_norm
                        )
                        metrics['grad_norm'] += grad_norm.item()
                    
                    # Update parameters
                    self.optimizer.step()
                    
                    # Update metrics
                    metrics['loss'] += loss.item()
                    metrics['q_value'] += current_q.mean().item()
                    metrics['target_q_value'] += target_q.mean().item()
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() and is_cuda_available():
                        # Clear CUDA cache and retry once
                        empty_cache()
                        return self.update(
                            observations[i:], 
                            actions[i:], 
                            rewards[i:], 
                            next_observations[i:], 
                            dones[i:], 
                            gamma, 
                            clip_grad_norm
                        )
                    raise
            
            # Average metrics
            for k in metrics:
                metrics[k] /= max(1, num_samples)
                
            return metrics  

def create_supply_chain_agents(
    num_agents: int = 4,
    shared_model: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> List[SupplyChainAgent]:
    """
    Create a list of supply chain agents with proper GPU support.
    
    Args:
        num_agents: Number of agents to create (one per node in the supply chain)
        shared_model: Whether agents should share the same model parameters
        device: Device to place the model on (None for auto-detect)
        **kwargs: Additional arguments to pass to SupplyChainAgent
        
    Returns:
        List of SupplyChainAgent instances
    """
    # Get the device and ensure it's a torch.device
    device = get_available_device(device)
    if isinstance(device, str):
        device = torch.device(device)
    
    # Print device info for debugging
    print(f"Creating {num_agents} agents on device: {device}")
    if device.type == 'cuda':
        print(f"  CUDA Device: {torch.cuda.get_device_name(device)}")
        print(f"  CUDA Memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f}MB allocated")
        print(f"  CUDA Memory: {torch.cuda.memory_reserved(device) / 1024**2:.2f}MB reserved")
    
    # Create shared model if needed
    try:
        if shared_model:
            print("Creating shared model for all agents...")
            model = SupplyChainTemporalGNN(device=device, **kwargs)
            agents = [
                SupplyChainAgent(node_id=i, model=model, device=device, **kwargs)
                for i in range(num_agents)
            ]
            print("Shared model created successfully.")
        else:
            # Create separate models for each agent
            print(f"Creating {num_agents} separate models...")
            agents = []
            for i in range(num_agents):
                print(f"  Creating model for agent {i}...")
                agent_model = SupplyChainTemporalGNN(device=device, **kwargs)
                agent = SupplyChainAgent(
                    node_id=i,
                    model=agent_model,
                    device=device,
                    **kwargs
                )
                agents.append(agent)
                # Free up memory
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            print("All agent models created successfully.")
            
    except RuntimeError as e:
        print(f"Error creating models: {e}")
        if 'CUDA out of memory' in str(e):
            print("CUDA out of memory error detected. Trying to free up memory...")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print("Please reduce model size or batch size and try again.")
        raise
    
    return agents
