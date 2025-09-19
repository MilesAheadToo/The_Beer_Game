#!/usr/bin/env python3
"""Train the Daybreak agent on the latest dataset using the GPU (if available)."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from sqlalchemy import select
from sqlalchemy.orm import Session as SyncSession

from app.core.config import settings
from app.db.session import sync_engine
from app.models.supply_chain_config import SupplyChainConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_CONFIG_NAME = "Default TBG"
BACKEND_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = BACKEND_ROOT / "training_jobs"
MODEL_ROOT = BACKEND_ROOT / "checkpoints" / "supply_chain_configs"
TRAIN_SCRIPT = BACKEND_ROOT / "scripts" / "training" / "train_gnn.py"


@dataclass
class TrainingParams:
    window: int
    horizon: int
    epochs: int
    device: str
    dataset: Path


def _latest_dataset_path(config_id: int) -> Optional[Path]:
    matches = sorted(TRAINING_ROOT.glob(f"dataset_cfg{config_id}_*.npz"))
    return matches[-1] if matches else None


def _model_path(config_id: int) -> Path:
    model_dir = MODEL_ROOT / f"config_{config_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "temporal_gnn.pt"


def _run_training_process(config_id: int, params: TrainingParams) -> Dict[str, Any]:
    log_path = params.dataset.parent / f"train_{Path(params.dataset).stem}.log"

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--source",
        "sim",
        "--window",
        str(params.window),
        "--horizon",
        str(params.horizon),
        "--epochs",
        str(params.epochs),
        "--save-path",
        str(_model_path(config_id)),
        "--dataset",
        str(params.dataset),
    ]
    if params.device:
        cmd.extend(["--device", params.device])

    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    log_output = stdout + ("\n" + stderr if stderr else "")
    log_path.write_text(log_output)

    return {
        "model_path": str(_model_path(config_id)),
        "log": log_output,
        "log_path": str(log_path),
        "command": cmd,
    }


def _resolve_config_id(config_id: Optional[int], config_name: str) -> int:
    if sync_engine is None:
        raise RuntimeError("Synchronous database engine is not configured; cannot access the database.")

    with SyncSession(sync_engine) as session:
        stmt = (
            select(SupplyChainConfig.id).where(SupplyChainConfig.id == config_id)
            if config_id is not None
            else select(SupplyChainConfig.id).where(SupplyChainConfig.name == config_name)
        )
        result = session.execute(stmt).scalar()
        if result is None:
            identifier = f"id={config_id}" if config_id is not None else f"name='{config_name}'"
            raise RuntimeError(f"Supply chain configuration not found ({identifier}).")
        return int(result)


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
    config_id = _resolve_config_id(args.config_id, args.config_name)

    dataset_path = Path(args.dataset) if args.dataset else _latest_dataset_path(config_id)
    if not dataset_path or not dataset_path.exists():
        raise RuntimeError(
            "No training dataset found. Generate a dataset before running GPU training."
        )

    model_path = _model_path(config_id)
    if model_path.exists() and not args.force:
        logger.info(
            "Model already exists at %s; skipping training (use --force to retrain).",
            model_path,
        )
        output = {
            "status": "skipped",
            "model_path": str(model_path),
            "config_id": config_id,
        }
        print(json.dumps(output))
        return

    params = TrainingParams(
        window=int(args.window),
        horizon=int(args.horizon),
        epochs=int(args.epochs),
        device=args.device,
        dataset=dataset_path,
    )

    result = _run_training_process(config_id, params)
    result.update({
        "status": "trained",
        "config_id": config_id,
        "dataset": str(dataset_path),
    })
    logger.info("Training complete. Model stored at %s", result.get("model_path"))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
