import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from typing import List, Dict, Tuple, Optional

from app.models.gnn.temporal_gnn import SupplyChainTemporalGNN, SupplyChainAgent, create_supply_chain_agents
from app.core.logging import setup_logging

# Set up logging
logger = setup_logging(__name__)

def load_synthetic_data(data_path: str, num_episodes: int = 100) -> List[dict]:
    """Load synthetic data from JSON file.
    
    Args:
        data_path: Path to the JSON file containing synthetic data
        num_episodes: Maximum number of episodes to load
        
    Returns:
        List of game data dictionaries
    """
    with open(data_path, 'r') as f:
        games = json.load(f)
    return games[:num_episodes]

def prepare_training_data(
    data_path: str, 
    num_episodes: int = 100, 
    seq_len: int = 10
) -> List[Dict]:
    """Prepare training data from synthetic game data.
    
    Args:
        data_path: Path to the JSON file containing synthetic data
        num_episodes: Number of episodes to use for training
        seq_len: Length of the sequence to use for each sample
        
    Returns:
        List of training samples with observations, actions, rewards, etc.
    """
    # Load games from JSON file
    games = load_synthetic_data(data_path, num_episodes)
    training_data = []
    
    for game in games:
        rounds = game.get('rounds', [])
        
        # Skip games with too few rounds
        if len(rounds) < seq_len + 1:
            continue
            
        # Convert to sequential samples
        for i in range(len(rounds) - seq_len):
            # Get sequence of rounds
            sequence = rounds[i:i + seq_len + 1]
            
            # Extract node features for each role in each round
            node_features = []
            actions = []
            rewards = []
            
            for round_data in sequence:
                round_number = round_data['round_number']
                decisions = round_data.get('decisions', [])
                
                # Create node features for each role
                role_features = {}
                for decision in decisions:
                    role = decision['role']
                    role_features[role] = [
                        decision.get('inventory', 0),
                        decision.get('backlog', 0),
                        decision.get('incoming_shipment', 0),
                        decision.get('demand', 0),
                        decision.get('order_quantity', 0)
                    ]
                    
                    # Store action (order quantity) for this role
                    if 'order_quantity' in decision:
                        actions.append(decision['order_quantity'])
                    
                    # Simple reward function (negative of total cost)
                    cost = decision.get('inventory_cost', 0) + decision.get('backlog_cost', 0)
                    rewards.append(-cost)
                
                # Ensure all roles are present
                for role in ['retailer', 'wholesaler', 'distributor', 'manufacturer']:
                    if role not in role_features:
                        role_features[role] = [0] * 5  # Default features if missing
                
                # Create node features tensor with shape [num_nodes, node_features]
                # Order: retailer, wholesaler, distributor, manufacturer
                features = torch.tensor([
                    role_features['retailer'],
                    role_features['wholesaler'],
                    role_features['distributor'],
                    role_features['manufacturer']
                ], dtype=torch.float32)
                
                node_features.append(features)
            
            # Create edge indices (fully connected graph)
            num_nodes = 4  # 4 roles
            edge_index = []
            
            # Add edges between consecutive nodes in the supply chain
            for i in range(num_nodes - 1):
                edge_index.append([i, i+1])  # Forward edge
                edge_index.append([i+1, i])  # Backward edge
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            # Edge attributes: [lead_time, cost, relationship_strength]
            edge_attr = []
            for _ in edge_index.T:
                edge_attr.append([1.0, 1.0, 1.0])  # Placeholder values
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
            
            # Convert to tensors
            # node_features is already a list of tensors, stack them along the sequence dimension
            node_features = torch.stack(node_features, dim=0)  # [seq_len, num_nodes, node_features]
            actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)  # [seq_len, 1]
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)  # [seq_len, 1]
            
            # Add to training data
            training_data.append({
                'node_features': node_features,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'actions': actions,
                'rewards': rewards
            })
    
    return training_data

class SupplyChainDataset(torch.utils.data.Dataset):
    """Custom dataset for supply chain data."""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Collate function for DataLoader to handle sequences of node features."""
    # Get sequence length (should be the same for all samples)
    seq_len = len(batch[0]['node_features'])
    
    # Initialize lists to store batched data
    batched_node_features = []
    batched_actions = []
    batched_rewards = []
    
    # Stack sequences for each timestep
    for t in range(seq_len):
        # Get node features for this timestep across all samples in batch
        node_features_t = torch.stack([item['node_features'][t] for item in batch])
        batched_node_features.append(node_features_t)
        
        # Get actions and rewards for this timestep
        actions_t = torch.stack([item['actions'][t] for item in batch])
        rewards_t = torch.stack([item['rewards'][t] for item in batch])
        
        batched_actions.append(actions_t)
        batched_rewards.append(rewards_t)
    
    # Stack along sequence dimension
    node_features = torch.stack(batched_node_features, dim=1)  # [batch_size, seq_len, num_nodes, node_features]
    actions = torch.stack(batched_actions, dim=1)  # [batch_size, seq_len]
    rewards = torch.stack(batched_rewards, dim=1)  # [batch_size, seq_len]
    
    # Edge index and attributes (same for all samples)
    edge_index = batch[0]['edge_index']
    edge_attr = batch[0].get('edge_attr', None)
    
    return {
        'node_features': node_features,
        'actions': actions,
        'rewards': rewards,
        'edge_index': edge_index,
        'edge_attr': edge_attr
    }

def train_agents(
    data_path: str,
    num_episodes: int = 100,
    batch_size: int = 32,
    seq_len: int = 10,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    save_dir: str = "models/tgnn",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Train the TemporalGNN agents.
    
    Args:
        data_path: Path to the JSON file containing training data
        num_episodes: Number of episodes to use for training
        batch_size: Batch size for training
        seq_len: Length of the sequence to use for each sample
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        save_dir: Directory to save the trained models
        device: Device to use for training ('cuda' or 'cpu')
    """
    import os
    from torch.utils.data import DataLoader, TensorDataset
    from torch.nn.utils.rnn import pad_sequence
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up TensorBoard
    log_dir = os.path.join("runs", f"tgnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir)
    
    # Create agents (one for each role, sharing the same model)
    model = SupplyChainTemporalGNN(
        node_features=5,  # inventory, orders, demand, backlog, incoming_shipments
        edge_features=3,  # lead_time, cost, relationship_strength
        hidden_dim=32,
        num_layers=2,
        num_heads=1,
        dropout=0.1,
        seq_len=seq_len,
        num_nodes=4  # retailer, wholesaler, distributor, manufacturer
    )
    
    agents = [
        SupplyChainAgent(
            node_id=i,
            model=model if i == 0 else None,  # Share the same model instance
            learning_rate=learning_rate,
            device=device
        )
        for i in range(4)  # 4 roles: retailer, wholesaler, distributor, manufacturer
    ]
    
    # Prepare training data
    logger.info("Preparing training data...")
    training_data = prepare_training_data(data_path, num_episodes, seq_len)
    logger.info(f"Prepared {len(training_data)} training samples")
    
    if not training_data:
        logger.error("No training data available. Exiting.")
        return
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        
        # Shuffle training data
        np.random.shuffle(training_data)
        
        # Split into training and validation sets (80/20)
        split_idx = int(0.8 * len(training_data))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        # Create datasets
        train_dataset = SupplyChainDataset(train_data)
        val_dataset = SupplyChainDataset(val_data)
        
        # Create data loaders with custom collate function
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Training
        for agent in agents:
            agent.model.train()
        
        train_loss = 0.0
        for batch in train_loader:
            # Unpack batch
            observations = batch['node_features']
            actions = batch['actions']
            rewards = batch['rewards']
            
            # Handle next_observations - if not provided, shift observations by 1
            if 'next_node_features' in batch:
                next_observations = batch['next_node_features']
            else:
                next_observations = [obs.clone() for obs in observations[1:]]
                next_observations.append(observations[-1].clone())
            
            # Process each timestep in the sequence
            batch_loss = 0.0
            for t in range(len(observations) - 1):
                # Update each agent
                for i, agent in enumerate(agents):
                    # Prepare data for this timestep and agent
                    # Ensure node_features has shape [batch_size=1, seq_len=1, num_nodes=4, node_features=5]
                    node_features = observations[t].unsqueeze(0).unsqueeze(1)  # Add batch and seq_len dimensions
                    
                    obs = {
                        'node_features': node_features.to(device),
                        'edge_index': batch['edge_index'].to(device),
                        'edge_attr': batch['edge_attr'].to(device) if 'edge_attr' in batch else None
                    }
                    
                    next_node_features = next_observations[t].unsqueeze(0).unsqueeze(1)
                    next_obs = {
                        'node_features': next_node_features.to(device),
                        'edge_index': batch['edge_index'].to(device),
                        'edge_attr': batch['edge_attr'].to(device) if 'edge_attr' in batch else None
                    }
                    
                    # Update agent
                    loss = agent.update(
                        observations=[obs],
                        actions=[actions[t][i]],
                        rewards=[rewards[t]],
                        next_observations=[next_obs],
                        dones=[t == len(observations) - 2]  # Last timestep in sequence
                    )
                    
                    batch_loss += loss['total_loss']
                    
                    # Log training loss
                    writer.add_scalar(f'train/agent_{i}/loss', loss['total_loss'], global_step)
                
                global_step += 1
            
            train_loss += batch_loss / len(observations)
        
        # Validation
        for agent in agents:
            agent.model.eval()
        
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # Unpack batch
                observations = batch['node_features']
                actions = batch['actions']
                rewards = batch['rewards']
                
                # Handle next_observations - if not provided, shift observations by 1
                if 'next_node_features' in batch:
                    next_observations = batch['next_node_features']
                else:
                    next_observations = [obs.clone() for obs in observations[1:]]
                    next_observations.append(observations[-1].clone())
                
                # Process each timestep in the sequence
                batch_val_loss = 0.0
                for t in range(len(observations) - 1):
                    # Evaluate each agent
                    for i, agent in enumerate(agents):
                        # Prepare data for this timestep and agent
                        # Ensure node_features has shape [batch_size=1, seq_len=1, num_nodes=4, node_features=5]
                        node_features = observations[t].unsqueeze(0).unsqueeze(1)  # Add batch and seq_len dimensions
                        
                        obs = {
                            'node_features': node_features.to(device),
                            'edge_index': batch['edge_index'].to(device),
                            'edge_attr': batch['edge_attr'].to(device) if 'edge_attr' in batch else None
                        }
                        
                        next_node_features = next_observations[t].unsqueeze(0).unsqueeze(1)
                        next_obs = {
                            'node_features': next_node_features.to(device),
                            'edge_index': batch['edge_index'].to(device),
                            'edge_attr': batch['edge_attr'].to(device) if 'edge_attr' in batch else None
                        }
                        
                        # Get model outputs
                        with torch.no_grad():
                            outputs = agent.model(
                                obs['node_features'].unsqueeze(0).to(device),
                                obs['edge_index'].to(device),
                                obs['edge_attr'].to(device) if 'edge_attr' in obs else None
                            )
                            
                            # Get order quantities and demand forecasts
                            order_quantities = outputs['order_quantity']
                            demand_forecasts = outputs['demand_forecast']
                            
                            # Calculate loss (simplified for validation)
                            action = actions[t][i].to(device)
                            # Ensure action has the same shape as order_quantities for the current node
                            action = action.unsqueeze(1)  # [batch_size, 1]
                            action_loss = F.mse_loss(order_quantities, action.float())
                            
                            # For value loss, we'll use the mean of demand_forecasts across nodes
                            # and compare to the reward
                            mean_demand = demand_forecasts.mean(dim=1, keepdim=True)  # [batch_size, 1]
                            
                            # Handle different reward formats
                            if isinstance(rewards[t], (list, np.ndarray, torch.Tensor)):
                                reward_value = float(rewards[t][0]) if len(rewards[t]) > 0 else 0.0
                            else:
                                reward_value = float(rewards[t])
                                
                            reward_tensor = torch.tensor([[reward_value]], device=device).float()  # [1, 1]
                            value_loss = F.mse_loss(mean_demand, reward_tensor)
                            
                            # Combine losses
                            loss = action_loss + 0.5 * value_loss
                            
                            batch_val_loss += loss.item()
                
                val_loss += batch_val_loss / (len(observations) * len(agents))
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            for i, agent in enumerate(agents):
                save_path = os.path.join(save_dir, f"agent_{i}_best.pt")
                torch.save(agent.model.state_dict(), save_path)
            logger.info(f"Saved new best model with val loss: {val_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': agents[0].model.state_dict(),
                'optimizer_state_dict': agents[0].optimizer.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final models
    for i, agent in enumerate(agents):
        save_path = os.path.join(save_dir, f"agent_{i}_final.pt")
        torch.save(agent.model.state_dict(), save_path)
    
    logger.info("Training complete. Models saved to {}".format(os.path.abspath(save_dir)))
    return agents

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train TemporalGNN agents on supply chain data')
    parser.add_argument('--data_path', type=str, default='data/synthetic_games_20250903_211141.json',
                        help='Path to the JSON file containing training data')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes to use for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--seq_len', type=int, default=10,
                        help='Length of the sequence to use for each sample')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer')
    parser.add_argument('--save_dir', type=str, default='models/tgnn',
                        help='Directory to save the trained models')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Run training
    train_agents(
        data_path=args.data_path,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        device=args.device
    )
