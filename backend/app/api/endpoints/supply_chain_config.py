from typing import List, Optional, Dict, Any, Iterable, Set
from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps
from app.core.config import settings
from app.services.supply_chain_config_service import SupplyChainConfigService
from app.models.supply_chain_config import NodeType, SupplyChainConfig
from app.schemas.game import GameCreate

router = APIRouter()

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


def _normalize_roles(roles: Optional[Iterable[str]]) -> Set[str]:
    """Normalize role strings to a comparable set."""
    normalized: Set[str] = set()
    if not roles:
        return normalized

    for role in roles:
        if isinstance(role, str):
            normalized.add(role.strip().lower().replace(" ", "").replace("_", ""))
    return normalized


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

    # Fall back to role-based group admins tied to their group's membership
    normalized_roles = _normalize_roles(getattr(user, "roles", []))
    is_group_admin = "groupadmin" in normalized_roles or "admin" in normalized_roles
    if is_group_admin and user.group_id:
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

    return crud.supply_chain_config.update(db, db_obj=config, obj_in=update_data)

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

    return crud.item.create_with_config(db, obj_in=item_in, config_id=config_id)

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
    
    return crud.node.create_with_config(db, obj_in=node_in, config_id=config_id)

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
