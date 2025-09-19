#!/usr/bin/env python3
"""Train the Daybreak agent on the latest dataset using the GPU (if available)."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from sqlalchemy import select

from app.api.endpoints.supply_chain_config import (  # type: ignore
    ConfigTrainingRequest,
    MODEL_ROOT,
    TRAINING_ROOT,
    _run_training_process,
)
from app.db.session import async_session_factory
from app.models.supply_chain_config import SupplyChainConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_CONFIG_NAME = "Default TBG"


async def _resolve_config(config_id: Optional[int], config_name: str) -> SupplyChainConfig:
    if async_session_factory is None:
        raise RuntimeError("Async session factory is not configured; cannot access the database.")

    async with async_session_factory() as session:
        stmt = (
            select(SupplyChainConfig)
            .where(SupplyChainConfig.id == config_id)
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
    matches = sorted(TRAINING_ROOT.glob(f"dataset_cfg{config_id}_*.npz"))
    return matches[-1] if matches else None


def _model_path(config_id: int) -> Path:
    model_dir = MODEL_ROOT / f"config_{config_id}"
    return model_dir / "temporal_gnn.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", type=int, default=None, help="Target supply chain configuration ID.")
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Supply chain configuration name to use when an explicit ID is not provided.",
    )
    parser.add_argument("--dataset", default=None, help="Override dataset path. Uses latest dataset when omitted.")
    parser.add_argument("--window", type=int, default=12, help="Input window length to train with.")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon for training labels.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--device", default="cuda", help="Torch device to use for training (default: cuda).")
    parser.add_argument("--force", action="store_true", help="Retrain even if a model checkpoint already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = asyncio.run(_resolve_config(args.config_id, args.config_name))

    dataset_path = Path(args.dataset) if args.dataset else _latest_dataset_path(config.id)
    if not dataset_path or not dataset_path.exists():
        raise RuntimeError(
            "No training dataset found. Generate a dataset before running GPU training."
        )

    model_path = _model_path(config.id)
    if model_path.exists() and not args.force:
        logger.info("Model already exists at %s; skipping training (use --force to retrain).", model_path)
        output = {
            "status": "skipped",
            "model_path": str(model_path),
            "config_id": config.id,
        }
        print(json.dumps(output))
        return

    params = ConfigTrainingRequest(
        window=int(args.window),
        horizon=int(args.horizon),
        epochs=int(args.epochs),
        device=args.device,
    )

    result = _run_training_process(config.id, dataset_path, params)
    result.update({
        "status": "trained",
        "config_id": config.id,
        "dataset": str(dataset_path),
    })
    logger.info("Training complete. Model stored at %s", result.get("model_path"))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
