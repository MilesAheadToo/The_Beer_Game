#!/usr/bin/env python3
"""Generate Daybreak training data using the SimPy backend for the default configuration."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from sqlalchemy import select

from app.core.config import settings
from app.rl.data_generator import generate_sim_training_windows
from app.db.session import async_session_factory
from app.models.supply_chain_config import SupplyChainConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_CONFIG_NAME = "Default TBG"
BACKEND_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = BACKEND_ROOT / "training_jobs"


@dataclass
class TrainingParams:
    num_runs: int
    timesteps: int
    window: int
    horizon: int
    sim_alpha: float
    sim_wip_k: float
    use_simpy: bool = True


def _generate_training_dataset(config_id: int, params: TrainingParams) -> Dict[str, Any]:
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)

    X, A, P, Y = generate_sim_training_windows(
        num_runs=params.num_runs,
        T=params.timesteps,
        window=params.window,
        horizon=params.horizon,
        supply_chain_config_id=config_id,
        db_url=settings.SQLALCHEMY_DATABASE_URI or None,
        use_simpy=params.use_simpy,
        sim_alpha=params.sim_alpha,
        sim_wip_k=params.sim_wip_k,
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dataset_path = TRAINING_ROOT / f"dataset_cfg{config_id}_{timestamp}.npz"
    np.savez(dataset_path, X=X, A=A, P=P, Y=Y)

    return {
        "path": str(dataset_path),
        "samples": int(X.shape[0]),
        "window": params.window,
        "horizon": params.horizon,
    }


async def _resolve_config(config_id: Optional[int], config_name: str) -> SupplyChainConfig:
    if async_session_factory is None:
        raise RuntimeError("Async session factory is not configured; cannot access the database.")

    async with async_session_factory() as session:
        stmt = (
            select(SupplyChainConfig).where(SupplyChainConfig.id == config_id)
            if config_id is not None
            else select(SupplyChainConfig).where(SupplyChainConfig.name == config_name)
        )
        result = await session.execute(stmt)
        config = result.scalars().first()
        if not config:
            identifier = f"id={config_id}" if config_id is not None else f"name='{config_name}'"
            raise RuntimeError(f"Supply chain configuration not found ({identifier}).")
        await session.commit()
        return config


def _latest_dataset_path(config_id: int) -> Optional[Path]:
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    matches = sorted(TRAINING_ROOT.glob(f"dataset_cfg{config_id}_*.npz"))
    return matches[-1] if matches else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", type=int, default=None, help="Target supply chain configuration ID.")
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Supply chain configuration name to use when an explicit ID is not provided.",
    )
    parser.add_argument("--num-runs", type=int, default=64, help="Number of simulated runs.")
    parser.add_argument("--timesteps", type=int, default=64, help="Number of periods per simulation run.")
    parser.add_argument("--window", type=int, default=12, help="Input window length.")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon for training labels.")
    parser.add_argument("--sim-alpha", type=float, default=0.3, help="SimPy smoothing factor.")
    parser.add_argument("--sim-wip-k", type=float, default=1.0, help="SimPy WIP gain parameter.")
    parser.add_argument("--force", action="store_true", help="Regenerate even if a dataset already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = asyncio.run(_resolve_config(args.config_id, args.config_name))

    latest_dataset = _latest_dataset_path(config.id)
    if latest_dataset and not args.force:
        logger.info("Existing dataset found for config %s at %s; skipping generation.", config.id, latest_dataset)
        output = {
            "status": "skipped",
            "path": str(latest_dataset),
            "config_id": config.id,
        }
        print(json.dumps(output))
        return

    params = TrainingParams(
        num_runs=int(args.num_runs),
        timesteps=int(args.timesteps),
        window=int(args.window),
        horizon=int(args.horizon),
        sim_alpha=float(args.sim_alpha),
        sim_wip_k=float(args.sim_wip_k),
    )

    dataset_info = _generate_training_dataset(config.id, params)
    dataset_info.update({
        "status": "created",
        "config_id": config.id,
    })
    logger.info(
        "Generated dataset for config %s with %s samples at %s",
        config.id,
        dataset_info.get("samples"),
        dataset_info.get("path"),
    )
    print(json.dumps(dataset_info))


if __name__ == "__main__":
    main()
