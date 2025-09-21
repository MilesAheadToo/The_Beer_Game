import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.rl.config import BeerGameParams
from app.rl.data_generator import (
    DbLookupConfig,
    generate_sim_training_windows,
    load_sequences_from_db,
)
from app.rl.policy import SimpleTemporalHead

# ---- A tiny backbone stub (replace with your temporal GNN) -------------------
class TinyBackbone(nn.Module):
    """
    Stand-in for your temporal GNN:
    - Input: X [B, T, N, F]
    - Output: H [B, T, N, H]
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        B, T, N, F = x.shape
        x = x.reshape(B * T * N, F)
        h = self.net(x)
        return h.reshape(B, T, N, -1)

def get_data(
    source: str,
    window: int,
    horizon: int,
    db_url: Optional[str],
    steps_table: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load or generate training data.
    
    Args:
        source: 'db' to load from database, 'sim' to generate synthetic data
        window: number of time steps in each input sequence
        horizon: number of future steps to predict
        db_url: database connection string
        steps_table: name of the table containing game steps
        
    Returns:
        X: [num_windows, window, num_nodes, num_features] input sequences
        A: [2, num_nodes, num_nodes] adjacency matrices for graph structure
        P: [num_windows, 0] global context (unused)
        Y: [num_windows, horizon, num_nodes] target action indices
    """
    params = BeerGameParams()
    if source == "db":
        if not db_url:
            db_url = os.getenv("DATABASE_URL", "")
        cfg = DbLookupConfig(database_url=db_url, steps_table=steps_table)
        return load_sequences_from_db(cfg, params=params, game_ids=None, window=window, horizon=horizon)
    elif source == "sim":
        return generate_sim_training_windows(
            num_runs=2056, T=64, window=window, horizon=horizon, params=params
        )
    else:
        raise ValueError(f"Unknown source: {source}")

def train_epoch(model, head, X, Y, optimizer, device):
    """Train for one epoch."""
    model.train()
    head.train()
    criterion = nn.CrossEntropyLoss()
    total = 0.0

    # Flatten windows/time; teacher-forcing on next-step action indices
    B, T, N, F = X.shape
    H = Y.shape[1]  # horizon
    assert H == 1, "Set horizon=1 for next-step prediction training."

    x = torch.from_numpy(X).float().to(device)      # [B, T, N, F]
    y = torch.from_numpy(Y[:, 0]).long().to(device) # [B, N]

    h = model(x)              # [B, T, N, H]
    h_cur = h[:, -1]          # last observed step [B, N, H]
    logits = head(h_cur)      # [B, N, A]

    loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total += float(loss.item())

    return total

def main():
    parser = argparse.ArgumentParser(description='Train a GNN for the Beer Game')
    parser.add_argument("--source", choices=["db", "sim"], default="db",
                      help="Data source: 'db' for database, 'sim' for simulator")
    parser.add_argument("--db-url", default=None,
                      help="Database connection URL (default: use DATABASE_URL env var)")
    parser.add_argument("--steps-table", default="beer_game_steps",
                      help="Name of the table containing game steps")
    parser.add_argument("--window", type=int, default=12,
                      help="Number of time steps in each input sequence")
    parser.add_argument("--horizon", type=int, default=1,
                      help="Number of future steps to predict")
    parser.add_argument("--epochs", type=int, default=10,
                      help="Number of training epochs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to train on (cuda/cpu)")
    parser.add_argument("--save-path", default="artifacts/temporal_gnn.pt",
                      help="Path to save the trained model")
    parser.add_argument("--dataset", default=None,
                      help="Optional path to an .npz dataset with arrays X,A,P,Y. If provided, overrides --source/--db-url.")
    args = parser.parse_args()

    # --- Load/Generate data
    if args.dataset:
        print(f"Loading dataset from {args.dataset}...")
        data = np.load(args.dataset)
        required = {"X", "A", "P", "Y"}
        if not required.issubset(set(data.files)):
            raise RuntimeError(f"Dataset {args.dataset} missing required arrays {required}. Found: {set(data.files)}")
        X, A, P, Y = data["X"], data["A"], data["P"], data["Y"]
    else:
        print(f"Loading data from {args.source}...")
        X, A, P, Y = get_data(
            source=args.source,
            window=args.window,
            horizon=args.horizon,
            db_url=args.db_url,
            steps_table=args.steps_table,
        )
    print(f"Loaded {len(X)} training samples")
    print(f"Input shape: {X.shape}, Target shape: {Y.shape}")

    requested_device = args.device
    resolved_device = requested_device

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "CUDA requested but not available; falling back to CPU.",
            file=sys.stderr,
        )
        resolved_device = "cpu"

    try:
        device = torch.device(resolved_device)
    except (RuntimeError, ValueError) as exc:
        print(
            f"Failed to initialize device '{resolved_device}' ({exc}); using CPU instead.",
            file=sys.stderr,
        )
        resolved_device = "cpu"
        device = torch.device(resolved_device)

    print(f"Using device: {resolved_device}")
    in_dim = X.shape[-1]

    model = TinyBackbone(in_dim=in_dim, hidden_dim=64).to(device)
    head = SimpleTemporalHead(hidden_dim=64).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Head parameters: {sum(p.numel() for p in head.parameters())}")

    optim_all = optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, head, X, Y, optim_all, device)
        print(f"[epoch {epoch:02d}/{args.epochs}] loss={loss:.4f}")

    # Save model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {"backbone_state_dict": model.state_dict(), 
         "head_state_dict": head.state_dict(),
         "in_dim": in_dim,
         "hidden_dim": 64,
        },
        args.save_path,
    )
    print(f"Saved model to {args.save_path}")

if __name__ == "__main__":
    main()
