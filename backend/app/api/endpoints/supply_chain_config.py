import hashlib
import json
import subprocess
import sys
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, Body, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload

from app import crud, models, schemas
from app.api import deps
from app.core.config import settings
from app.db.session import SessionLocal
from app.services.supply_chain_config_service import SupplyChainConfigService
from app.models.supply_chain_config import NodeType, SupplyChainConfig, SupplyChainTrainingArtifact
from app.models.user import UserTypeEnum
from app.schemas.game import GameCreate
from app.rl.data_generator import generate_sim_training_windows

logger = logging.getLogger(__name__)
router = APIRouter()

BACKEND_ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = BACKEND_ROOT / "training_jobs"
MODEL_ROOT = BACKEND_ROOT / "checkpoints" / "supply_chain_configs"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", value.strip().lower()).strip("-")
    return slug or "config"


class ConfigTrainingRequest(BaseModel):
    num_runs: int = Field(32, ge=4, le=512, description="Number of simulation runs to generate")
    T: int = Field(64, ge=16, le=512, description="Number of periods in each simulation run")
    window: int = Field(12, ge=1, le=128, description="Input window length for training samples")
    horizon: int = Field(1, ge=1, le=8, description="Forecast horizon for the temporal model")
    epochs: int = Field(5, ge=1, le=500, description="Training epochs for the temporal GNN")
    device: Optional[str] = Field(None, description="Optional device hint passed to the trainer (e.g. 'cpu', 'cuda')")
    use_simpy: Optional[bool] = Field(None, description="Override simulator backend; defaults to environment setting")
    sim_alpha: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Smoothing factor for SimPy simulator")
    sim_wip_k: Optional[float] = Field(1.0, ge=0.0, le=10.0, description="WIP gain parameter for SimPy simulator")


class ConfigTrainingResponse(BaseModel):
    status: str
    message: str
    dataset_path: str
    model_path: str
    trained_at: Optional[datetime]
    log: str

# --- Helper functions ---

def get_config_or_404(db: Session, config_id: int):
    config = crud.supply_chain_config.get(db, id=config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Configuration not found"
        )
    return config

def get_item_or_404(db: Session, item_id: int, config_id: int):
    item = crud.item.get(db, id=item_id)
    if not item or item.config_id != config_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found in this configuration"
        )
    return item

def get_node_or_404(db: Session, node_id: int, config_id: int):
    node = crud.node.get(db, id=node_id)
    if not node or node.config_id != config_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Node not found in this configuration"
        )
    return node


def get_lane_or_404(db: Session, lane_id: int, config_id: int):
    lane = crud.lane.get(db, id=lane_id)
    if not lane or lane.config_id != config_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lane not found in this configuration",
        )
    return lane


def get_item_node_config_or_404(db: Session, config_id: int, config_entry_id: int):
    entry = crud.item_node_config.get(db, id=config_entry_id)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item-node configuration not found",
        )

    node = crud.node.get(db, id=entry.node_id)
    item = crud.item.get(db, id=entry.item_id)
    if not node or node.config_id != config_id or not item or item.config_id != config_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item-node configuration not found in this configuration",
        )
    return entry


def get_market_demand_or_404(db: Session, config_id: int, demand_id: int):
    demand = crud.market_demand.get(db, id=demand_id)
    if not demand or demand.config_id != config_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market demand entry not found in this configuration",
        )
    return demand


def _lane_to_payload(lane: models.Lane) -> Dict[str, Any]:
    lead_time_days = lane.lead_time_days or {}
    min_lead = lead_time_days.get("min")
    max_lead = lead_time_days.get("max")

    if min_lead is not None and max_lead is not None and min_lead == max_lead:
        lead_time = min_lead
    elif min_lead is not None:
        lead_time = min_lead
    elif max_lead is not None:
        lead_time = max_lead
    else:
        lead_time = None

    return {
        "id": lane.id,
        "from_node_id": lane.upstream_node_id,
        "to_node_id": lane.downstream_node_id,
        "capacity": lane.capacity,
        "lead_time_days": lead_time_days,
        "lead_time": lead_time,
        "cost_per_unit": None,
    }


def _coerce_lane_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    upstream = payload.get("from_node_id", payload.get("upstream_node_id"))
    downstream = payload.get("to_node_id", payload.get("downstream_node_id"))

    lead_time_days = payload.get("lead_time_days")
    if not lead_time_days and payload.get("lead_time") is not None:
        try:
            value = int(payload["lead_time"])
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Lead time must be a whole number",
            )
        lead_time_days = {"min": value, "max": value}

    data = {
        "upstream_node_id": upstream,
        "downstream_node_id": downstream,
        "capacity": payload.get("capacity"),
        "lead_time_days": lead_time_days,
    }

    return data


def _get_user_admin_group_id(db: Session, user: models.User) -> Optional[int]:
    """Return the group ID managed by the provided user, if any."""
    if user.is_superuser:
        return None

    # First, check if the user is explicitly registered as the group's primary admin
    direct_group = (
        db.query(models.Group)
        .filter(models.Group.admin_id == user.id)
        .first()
    )
    if direct_group:
        return direct_group.id

    user_type = getattr(user, "user_type", None)
    if isinstance(user_type, str):
        try:
            user_type = UserTypeEnum(user_type)
        except ValueError:
            user_type = None

    if user_type == UserTypeEnum.GROUP_ADMIN and user.group_id:
        group = (
            db.query(models.Group)
            .filter(models.Group.id == user.group_id)
            .first()
        )
        if group:
            return group.id

    return None


def _ensure_user_can_manage_config(
    db: Session,
    user: models.User,
    config: SupplyChainConfig,
):
    """Ensure the current user can manage the provided configuration."""
    if user.is_superuser:
        return

    admin_group_id = _get_user_admin_group_id(db, user)
    if not admin_group_id or config.group_id != admin_group_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this configuration",
        )


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _compute_config_hash(db: Session, config_id: int) -> Optional[str]:
    config = crud.supply_chain_config.get(db, id=config_id)
    if not config:
        return None

    items = crud.item.get_multi_by_config(db, config_id=config_id, limit=1000)
    nodes = crud.node.get_multi_by_config(db, config_id=config_id, limit=1000)
    lanes = crud.lane.get_by_config(db, config_id=config_id)
    item_node_configs = crud.item_node_config.get_by_config(db, config_id=config_id)
    market_demands = crud.market_demand.get_by_config(db, config_id=config_id)

    payload = {
        "config": {
            "name": config.name,
            "description": config.description,
            "is_active": bool(config.is_active),
            "group_id": config.group_id,
        },
        "items": [
            {
                "name": item.name,
                "unit_cost_range": item.unit_cost_range,
            }
            for item in sorted(items, key=lambda obj: obj.id)
        ],
        "nodes": [
            {
                "name": node.name,
                "type": getattr(node.type, "value", str(node.type)),
            }
            for node in sorted(nodes, key=lambda obj: obj.id)
        ],
        "lanes": [
            {
                "upstream": lane.upstream_node_id,
                "downstream": lane.downstream_node_id,
                "capacity": lane.capacity,
                "lead_time_days": lane.lead_time_days,
            }
            for lane in sorted(lanes, key=lambda obj: obj.id)
        ],
        "item_node_configs": [
            {
                "item_id": inc.item_id,
                "node_id": inc.node_id,
                "inventory_target_range": inc.inventory_target_range,
                "initial_inventory_range": inc.initial_inventory_range,
                "holding_cost_range": inc.holding_cost_range,
                "backlog_cost_range": inc.backlog_cost_range,
                "selling_price_range": inc.selling_price_range,
            }
            for inc in sorted(item_node_configs, key=lambda obj: obj.id)
        ],
        "market_demands": [
            {
                "item_id": md.item_id,
                "retailer_id": md.retailer_id,
                "demand_pattern": md.demand_pattern,
            }
            for md in sorted(market_demands, key=lambda obj: obj.id)
        ],
    }

    encoded = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mark_config_requires_training(
    db: Session,
    config: SupplyChainConfig,
    status_label: str = "pending",
) -> SupplyChainConfig:
    config.needs_training = True
    config.training_status = status_label
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


def _create_training_artifact(
    db: Session,
    config_id: int,
    dataset_path: Path,
    model_path: Path,
) -> SupplyChainTrainingArtifact:
    artifact = SupplyChainTrainingArtifact(
        config_id=config_id,
        dataset_name=dataset_path.name,
        model_name=model_path.name,
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


def _set_training_outcome(
    db: Session,
    config: SupplyChainConfig,
    *,
    status_label: str,
    needs_training: bool,
    trained_at: Optional[datetime] = None,
    model_path: Optional[str] = None,
    config_hash: Optional[str] = None,
) -> SupplyChainConfig:
    config.training_status = status_label
    config.needs_training = needs_training
    config.trained_at = trained_at
    config.trained_model_path = model_path
    config.last_trained_config_hash = config_hash
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


def _generate_training_dataset(
    config: SupplyChainConfig,
    params: ConfigTrainingRequest,
) -> Dict[str, Any]:
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    slug = _slugify(config.name)
    dataset_filename = f"{slug}_dataset.npz"
    dataset_path = TRAINING_ROOT / dataset_filename

    X, A, P, Y = generate_sim_training_windows(
        num_runs=int(params.num_runs),
        T=int(params.T),
        window=int(params.window),
        horizon=int(params.horizon),
        supply_chain_config_id=config.id,
        db_url=settings.SQLALCHEMY_DATABASE_URI or None,
        use_simpy=params.use_simpy,
        sim_alpha=float(params.sim_alpha) if params.sim_alpha is not None else 0.3,
        sim_wip_k=float(params.sim_wip_k) if params.sim_wip_k is not None else 1.0,
    )
    np.savez(dataset_path, X=X, A=A, P=P, Y=Y)

    return {
        "path": str(dataset_path),
        "filename": dataset_filename,
        "samples": int(X.shape[0]),
        "window": int(params.window),
        "horizon": int(params.horizon),
    }


def _run_training_process(
    config: SupplyChainConfig,
    dataset_path: Path,
    params: ConfigTrainingRequest,
) -> Dict[str, Any]:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    slug = _slugify(config.name)

    model_filename = f"{slug}_temporal_gnn.pt"
    model_path = MODEL_ROOT / model_filename
    log_path = MODEL_ROOT / f"{slug}_train.log"
    script_path = BACKEND_ROOT / "scripts" / "training" / "train_gnn.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--source",
        "sim",
        "--window",
        str(params.window),
        "--horizon",
        str(params.horizon),
        "--epochs",
        str(params.epochs),
        "--save-path",
        str(model_path),
        "--dataset",
        str(dataset_path),
    ]
    if params.device:
        cmd.extend(["--device", params.device])

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        log_output = stdout + ("\n" + stderr if stderr else "")
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        log_output = stdout + ("\n" + stderr if stderr else "")
        log_path.write_text(log_output)
        raise RuntimeError(log_output.strip() or str(exc)) from exc

    log_path.write_text(log_output)

    return {
        "model_path": str(model_path),
        "log": log_output,
        "log_path": str(log_path),
        "command": cmd,
    }


def _enqueue_training(
    background_tasks: Optional[BackgroundTasks],
    config_id: int,
    params: Optional[ConfigTrainingRequest] = None,
) -> None:
    payload = (params or ConfigTrainingRequest()).dict()
    if background_tasks is None:
        _train_config_task(config_id, payload)
    else:
        background_tasks.add_task(_train_config_task, config_id, payload)


def _train_config_task(config_id: int, params_data: Dict[str, Any]) -> None:
    """Run dataset generation and training for a configuration in the background."""
    db = SessionLocal()
    try:
        params = ConfigTrainingRequest(**params_data)
        config = crud.supply_chain_config.get(db, id=config_id)
        if not config:
            logger.warning("Config %s no longer exists; skipping training", config_id)
            return

        try:
            dataset_info = _generate_training_dataset(config, params)
        except Exception:
            logger.exception("Dataset generation failed for config %s", config_id)
            fresh = crud.supply_chain_config.get(db, id=config_id)
            if fresh:
                _set_training_outcome(
                    db,
                    fresh,
                    status_label="failed",
                    needs_training=True,
                    trained_at=fresh.trained_at,
                    model_path=fresh.trained_model_path,
                    config_hash=fresh.last_trained_config_hash,
                )
            return

        config = _set_training_outcome(
            db,
            config,
            status_label="in_progress",
            needs_training=False,
            trained_at=None,
            model_path=config.trained_model_path,
            config_hash=config.last_trained_config_hash,
        )

        try:
            dataset_path = Path(dataset_info["path"])
            training_info = _run_training_process(config, dataset_path, params)
            config_hash = _compute_config_hash(db, config_id)
            config = _set_training_outcome(
                db,
                config,
                status_label="trained",
                needs_training=False,
                trained_at=datetime.utcnow(),
                model_path=training_info["model_path"],
                config_hash=config_hash,
            )
            _create_training_artifact(db, config_id, dataset_path, Path(training_info["model_path"]))
        except Exception:
            logger.exception("Training process failed for config %s", config_id)
            fresh = crud.supply_chain_config.get(db, id=config_id)
            if fresh:
                _set_training_outcome(
                    db,
                    fresh,
                    status_label="failed",
                    needs_training=True,
                    trained_at=fresh.trained_at,
                    model_path=fresh.trained_model_path,
                    config_hash=fresh.last_trained_config_hash,
                )
    finally:
        db.close()

# --- Configuration Endpoints ---
# --- Configuration Endpoints ---

@router.get("/", response_model=List[schemas.SupplyChainConfig])
def read_configs(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    """Retrieve all supply chain configurations."""
    if current_user.is_superuser:
        return crud.supply_chain_config.get_multi(db, skip=skip, limit=limit)

    admin_group_id = _get_user_admin_group_id(db, current_user)
    if not admin_group_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view these configurations",
        )

    return crud.supply_chain_config.get_multi(
        db,
        skip=skip,
        limit=limit,
        group_id=admin_group_id,
    )

@router.get("/active", response_model=schemas.SupplyChainConfig)
def read_active_config(
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_active_user),
):
    """Get the active supply chain configuration."""
    if current_user.is_superuser:
        config = crud.supply_chain_config.get_active(db)
    else:
        admin_group_id = _get_user_admin_group_id(db, current_user)
        if not admin_group_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to view this configuration",
            )

        config = (
            db.query(SupplyChainConfig)
            .filter(
                SupplyChainConfig.group_id == admin_group_id,
                SupplyChainConfig.is_active == True,
            )
            .first()
        )

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active configuration found",
        )
    return config

@router.post("/", response_model=schemas.SupplyChainConfig, status_code=status.HTTP_201_CREATED)
def create_config(
    *,
    db: Session = Depends(deps.get_db),
    config_in: schemas.SupplyChainConfigCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
    background_tasks: BackgroundTasks,
):
    """Create a new supply chain configuration."""
    admin_group_id = _get_user_admin_group_id(db, current_user)

    if current_user.is_superuser:
        target_group_id = config_in.group_id
        if target_group_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Group is required to create a configuration",
            )

        if not db.query(models.Group).filter(models.Group.id == target_group_id).first():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Specified group not found",
            )

        payload = config_in
    else:
        if not admin_group_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to create configurations",
            )

        payload = config_in.copy(update={"group_id": admin_group_id})

    cfg = crud.supply_chain_config.create(db, obj_in=payload)
    # Attach creator if column exists
    try:
        cfg.created_by = current_user.id
        db.add(cfg)
        db.commit()
        db.refresh(cfg)
    except Exception:
        pass
    cfg = _mark_config_requires_training(db, cfg)
    _enqueue_training(background_tasks, cfg.id)
    return cfg

@router.get("/{config_id}", response_model=schemas.SupplyChainConfig)
def read_config(
    config_id: int,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_active_user),
):
    """Get a specific configuration by ID."""
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    return config

@router.put("/{config_id}", response_model=schemas.SupplyChainConfig)
def update_config(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    config_in: schemas.SupplyChainConfigUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
    background_tasks: BackgroundTasks,
):
    """Update a configuration."""
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)

    update_data = config_in.dict(exclude_unset=True)

    if current_user.is_superuser:
        if "group_id" in update_data:
            new_group_id = update_data["group_id"]
            if new_group_id is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Group cannot be null",
                )
            if not db.query(models.Group).filter(models.Group.id == new_group_id).first():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Specified group not found",
                )
    else:
        if "group_id" in update_data and update_data["group_id"] != config.group_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You cannot reassign this configuration to another group",
            )
        update_data.pop("group_id", None)

    updated = crud.supply_chain_config.update(db, db_obj=config, obj_in=update_data)
    changed_keys = set(update_data.keys())
    if not changed_keys or changed_keys <= {"is_active"}:
        return updated
    updated = _mark_config_requires_training(db, updated)
    _enqueue_training(background_tasks, updated.id)
    return updated


@router.post("/{config_id}/train", response_model=ConfigTrainingResponse)
def train_supply_chain_config(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    params: ConfigTrainingRequest = Body(default_factory=ConfigTrainingRequest),
    current_user: models.User = Depends(deps.get_current_active_user),
):
    """Generate synthetic training data and train the temporal GNN for a configuration."""
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)

    if config.training_status == "in_progress":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training is already running for this configuration.",
        )

    dataset_info = _generate_training_dataset(config, params)

    # Mark as in progress before launching the trainer
    _set_training_outcome(
        db,
        config,
        status_label="in_progress",
        needs_training=False,
        trained_at=None,
        model_path=config.trained_model_path,
        config_hash=config.last_trained_config_hash,
    )

    try:
        dataset_path = Path(dataset_info["path"])
        training_info = _run_training_process(
            config,
            dataset_path,
            params,
        )
        config_hash = _compute_config_hash(db, config_id)
        updated = _set_training_outcome(
            db,
            config,
            status_label="trained",
            needs_training=False,
            trained_at=datetime.utcnow(),
            model_path=training_info["model_path"],
            config_hash=config_hash,
        )
        _create_training_artifact(db, config.id, dataset_path, Path(training_info["model_path"]))
    except Exception as exc:  # noqa: BLE001 - surface training failure to client
        _set_training_outcome(
            db,
            config,
            status_label="failed",
            needs_training=True,
            trained_at=config.trained_at,
            model_path=config.trained_model_path,
            config_hash=config.last_trained_config_hash,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {exc}",
        ) from exc

    return ConfigTrainingResponse(
        status=updated.training_status,
        message="Training completed successfully.",
        dataset_path=dataset_info["path"],
        model_path=updated.trained_model_path or training_info["model_path"],
        trained_at=updated.trained_at,
        log=training_info["log"],
    )

@router.delete("/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_config(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    """Delete a configuration."""
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    crud.supply_chain_config.remove(db, id=config_id)
    return None

# --- Item Endpoints ---

@router.get("/{config_id}/items/", response_model=List[schemas.Item])
def read_items(
    config_id: int,
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    """Get all items for a configuration."""
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    return crud.item.get_multi_by_config(db, config_id=config_id, skip=skip, limit=limit)

@router.post("/{config_id}/items/", response_model=schemas.Item, status_code=status.HTTP_201_CREATED)
def create_item(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    item_in: schemas.ItemCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    """Create a new item in a configuration."""
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)

    # Check for duplicate name
    if crud.item.get_by_name(db, name=item_in.name, config_id=config_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An item with this name already exists in this configuration"
        )

    created = crud.item.create_with_config(db, obj_in=item_in, config_id=config_id)
    _mark_config_requires_training(db, config)
    return created

@router.get("/{config_id}/items/{item_id}", response_model=schemas.Item)
def read_item(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    item_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    return get_item_or_404(db, item_id, config_id)


@router.put("/{config_id}/items/{item_id}", response_model=schemas.Item)
def update_item(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    item_id: int,
    item_in: schemas.ItemUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    db_item = get_item_or_404(db, item_id, config_id)

    new_name = item_in.name.strip() if isinstance(item_in.name, str) else db_item.name
    if new_name and new_name != db_item.name:
        existing = crud.item.get_by_name(db, name=new_name, config_id=config_id)
        if existing and existing.id != db_item.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An item with this name already exists in this configuration",
            )

    updated = crud.item.update(db, db_obj=db_item, obj_in=item_in)
    _mark_config_requires_training(db, config)
    return updated


@router.delete("/{config_id}/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    item_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    db_item = get_item_or_404(db, item_id, config_id)

    crud.item.remove(db, id=db_item.id)
    _mark_config_requires_training(db, config)
    return None

# ... (Additional endpoints for items, nodes, lanes, item_node_configs, and market_demands)
# The full implementation would include similar CRUD endpoints for all models
# including proper error handling and permissions

# --- Node Endpoints ---

@router.get("/{config_id}/nodes/", response_model=List[schemas.Node])
def read_nodes(
    config_id: int,
    node_type: Optional[NodeType] = None,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_active_user),
):
    """Get all nodes for a configuration, optionally filtered by type."""
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)

    if node_type:
        return crud.node.get_by_type(db, node_type=node_type, config_id=config_id)
    return crud.node.get_multi_by_config(db, config_id=config_id)

@router.post("/{config_id}/nodes/", response_model=schemas.Node, status_code=status.HTTP_201_CREATED)
def create_node(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    node_in: schemas.NodeCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    """Create a new node in a configuration."""
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)

    # Check for duplicate name and type
    if crud.node.get_by_name_and_type(
        db, name=node_in.name, node_type=node_in.type, config_id=config_id
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A node with this name and type already exists in this configuration"
        )
    
    created = crud.node.create_with_config(db, obj_in=node_in, config_id=config_id)
    _mark_config_requires_training(db, config)
    return created


@router.get("/{config_id}/nodes/{node_id}", response_model=schemas.Node)
def read_node(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    node_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    return get_node_or_404(db, node_id, config_id)


@router.put("/{config_id}/nodes/{node_id}", response_model=schemas.Node)
def update_node(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    node_id: int,
    node_in: schemas.NodeUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    db_node = get_node_or_404(db, node_id, config_id)

    new_name = node_in.name.strip() if isinstance(node_in.name, str) else db_node.name
    new_type = node_in.type or db_node.type

    if (new_name != db_node.name) or (new_type != db_node.type):
        existing = crud.node.get_by_name_and_type(
            db,
            name=new_name,
            node_type=new_type,
            config_id=config_id,
        )
        if existing and existing.id != db_node.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A node with this name and type already exists in this configuration",
            )

    updated = crud.node.update(db, db_obj=db_node, obj_in=node_in)
    _mark_config_requires_training(db, config)
    return updated


@router.delete("/{config_id}/nodes/{node_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_node(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    node_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    db_node = get_node_or_404(db, node_id, config_id)

    if db_node.upstream_lanes or db_node.downstream_lanes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Remove associated lanes before deleting this node",
        )

    crud.node.remove(db, id=db_node.id)
    _mark_config_requires_training(db, config)
    return None


# --- Lane Endpoints ---

@router.get("/{config_id}/lanes")
def read_lanes(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> List[Dict[str, Any]]:
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    lanes = crud.lane.get_by_config(db, config_id=config_id)
    return [_lane_to_payload(lane) for lane in lanes]


@router.get("/{config_id}/lanes/{lane_id}")
def read_lane(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    lane_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Dict[str, Any]:
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    lane = get_lane_or_404(db, lane_id, config_id)
    return _lane_to_payload(lane)


@router.post("/{config_id}/lanes", status_code=status.HTTP_201_CREATED)
def create_lane(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    lane_in: Dict[str, Any] = Body(...),
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Dict[str, Any]:
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)

    lane_data = _coerce_lane_payload(lane_in)
    upstream_id = lane_data.get("upstream_node_id")
    downstream_id = lane_data.get("downstream_node_id")
    capacity = lane_data.get("capacity")
    lead_time_days = lane_data.get("lead_time_days")

    if not upstream_id or not downstream_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both source and destination nodes are required",
        )

    upstream_node = get_node_or_404(db, upstream_id, config_id)
    downstream_node = get_node_or_404(db, downstream_id, config_id)

    if upstream_node.id == downstream_node.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Source and destination nodes must be different",
        )

    if lead_time_days is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Lead time information is required",
        )

    if capacity is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Capacity is required",
        )

    existing = crud.lane.get_by_nodes(
        db,
        upstream_node_id=upstream_node.id,
        downstream_node_id=downstream_node.id,
        config_id=config_id,
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A lane between these nodes already exists",
        )

    lane_payload = schemas.LaneCreate(**lane_data)
    created = crud.lane.create_with_config(db, obj_in=lane_payload, config_id=config_id)
    _mark_config_requires_training(db, config)
    return _lane_to_payload(created)


@router.put("/{config_id}/lanes/{lane_id}")
def update_lane(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    lane_id: int,
    lane_in: Dict[str, Any] = Body(...),
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Dict[str, Any]:
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    lane = get_lane_or_404(db, lane_id, config_id)

    update_payload: Dict[str, Any] = {}
    if "capacity" in lane_in and lane_in["capacity"] is not None:
        try:
            update_payload["capacity"] = int(lane_in["capacity"])
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Capacity must be a whole number",
            )

    lead_time_days = lane_in.get("lead_time_days")
    if not lead_time_days and lane_in.get("lead_time") is not None:
        try:
            value = int(lane_in["lead_time"])
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Lead time must be a whole number",
            )
        lead_time_days = {"min": value, "max": value}
    if lead_time_days is not None:
        update_payload["lead_time_days"] = lead_time_days

    if not update_payload:
        return _lane_to_payload(lane)

    lane_update = schemas.LaneUpdate(**update_payload)
    updated = crud.lane.update(db, db_obj=lane, obj_in=lane_update)
    _mark_config_requires_training(db, config)
    return _lane_to_payload(updated)


@router.delete("/{config_id}/lanes/{lane_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_lane(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    lane_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    lane = get_lane_or_404(db, lane_id, config_id)

    crud.lane.remove(db, id=lane.id)
    _mark_config_requires_training(db, config)
    return None


# --- Item-Node Configuration Endpoints ---

@router.get("/{config_id}/item-node-configs", response_model=List[schemas.ItemNodeConfig])
def read_item_node_configs(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    return crud.item_node_config.get_by_config(db, config_id=config_id)


@router.post(
    "/{config_id}/item-node-configs",
    response_model=schemas.ItemNodeConfig,
    status_code=status.HTTP_201_CREATED,
)
def create_item_node_config(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    config_in: schemas.ItemNodeConfigCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)

    item = crud.item.get(db, id=config_in.item_id)
    node = crud.node.get(db, id=config_in.node_id)
    if not item or item.config_id != config_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Item must belong to this configuration",
        )
    if not node or node.config_id != config_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Node must belong to this configuration",
        )

    existing = crud.item_node_config.get_by_item_and_node(
        db,
        item_id=config_in.item_id,
        node_id=config_in.node_id,
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This item already has configuration for the specified node",
        )

    created = crud.item_node_config.create(db, obj_in=config_in)
    _mark_config_requires_training(db, config)
    return created


@router.put(
    "/{config_id}/item-node-configs/{config_entry_id}",
    response_model=schemas.ItemNodeConfig,
)
def update_item_node_config(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    config_entry_id: int,
    config_in: schemas.ItemNodeConfigUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    entry = get_item_node_config_or_404(db, config_id, config_entry_id)

    updated = crud.item_node_config.update(db, db_obj=entry, obj_in=config_in)
    _mark_config_requires_training(db, config)
    return updated


@router.delete(
    "/{config_id}/item-node-configs/{config_entry_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_item_node_config(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    config_entry_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    entry = get_item_node_config_or_404(db, config_id, config_entry_id)

    crud.item_node_config.remove(db, id=entry.id)
    _mark_config_requires_training(db, config)
    return None


# --- Market Demand Endpoints ---

@router.get(
    "/{config_id}/market-demands",
    response_model=List[schemas.MarketDemand],
)
def read_market_demands(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    return crud.market_demand.get_by_config(db, config_id=config_id)


@router.post(
    "/{config_id}/market-demands",
    response_model=schemas.MarketDemand,
    status_code=status.HTTP_201_CREATED,
)
def create_market_demand(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    demand_in: schemas.MarketDemandCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)

    item = crud.item.get(db, id=demand_in.item_id)
    retailer = crud.node.get(db, id=demand_in.retailer_id)

    if not item or item.config_id != config_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Item must belong to this configuration",
        )

    if not retailer or retailer.config_id != config_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Retailer must belong to this configuration",
        )

    existing = crud.market_demand.get_by_item_and_retailer(
        db,
        item_id=demand_in.item_id,
        retailer_id=demand_in.retailer_id,
        config_id=config_id,
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A market demand entry already exists for this item and retailer",
        )

    created = crud.market_demand.create(db, obj_in=demand_in)
    _mark_config_requires_training(db, config)
    return created


@router.put(
    "/{config_id}/market-demands/{demand_id}",
    response_model=schemas.MarketDemand,
)
def update_market_demand(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    demand_id: int,
    demand_in: schemas.MarketDemandUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    demand = get_market_demand_or_404(db, config_id, demand_id)

    updated = crud.market_demand.update(db, db_obj=demand, obj_in=demand_in)
    _mark_config_requires_training(db, config)
    return updated


@router.delete(
    "/{config_id}/market-demands/{demand_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_market_demand(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    demand_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
):
    config = get_config_or_404(db, config_id)
    _ensure_user_can_manage_config(db, current_user, config)
    demand = get_market_demand_or_404(db, config_id, demand_id)

    crud.market_demand.remove(db, id=demand.id)
    _mark_config_requires_training(db, config)
    return None
# --- Game Integration Endpoints ---

@router.post("/{config_id}/create-game", response_model=Dict[str, Any])
def create_game_from_config(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    game_data: Dict[str, Any] = Body(
        default={
            "name": "New Game",
            "description": "Game created from supply chain configuration",
            "max_rounds": 52,
            "is_public": True
        },
        example={
            "name": "New Game",
            "description": "Game created from supply chain configuration",
            "max_rounds": 52,
            "is_public": True
        }
    ),
    current_user: models.User = Depends(deps.get_current_active_user),
):
    """
    Create a game configuration from a supply chain configuration.
    
    This endpoint generates a game configuration based on the supply chain configuration
    with the specified ID. The configuration includes node policies, demand patterns,
    and other settings derived from the supply chain model.
    
    Returns a game configuration that can be used to create a new game.
    """
    # Verify the configuration exists
    config = db.query(SupplyChainConfig).filter(SupplyChainConfig.id == config_id).first()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supply chain configuration with ID {config_id} not found"
        )

    _ensure_user_can_manage_config(db, current_user, config)

    # Use the service to create the game configuration
    service = SupplyChainConfigService(db)
    try:
        game_config = service.create_game_from_config(config_id, game_data)
        return game_config
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create game configuration: {str(e)}"
        )

# ... (Additional CRUD endpoints for nodes, lanes, item_node_configs, and market_demands)
