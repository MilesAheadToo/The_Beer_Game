from typing import List, Optional, Dict, Any
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

# --- Configuration Endpoints ---

@router.get("/", response_model=List[schemas.SupplyChainConfig])
def read_configs(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
):
    """Retrieve all supply chain configurations."""
    return crud.supply_chain_config.get_multi(db, skip=skip, limit=limit)

@router.get("/active", response_model=schemas.SupplyChainConfig)
def read_active_config(db: Session = Depends(deps.get_db)):
    """Get the active supply chain configuration."""
    config = crud.supply_chain_config.get_active(db)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active configuration found"
        )
    return config

@router.post("/", response_model=schemas.SupplyChainConfig, status_code=status.HTTP_201_CREATED)
def create_config(
    *,
    db: Session = Depends(deps.get_db),
    config_in: schemas.SupplyChainConfigCreate,
    current_user: models.User = Depends(deps.get_current_active_superuser),
):
    """Create a new supply chain configuration."""
    return crud.supply_chain_config.create(db, obj_in=config_in)

@router.get("/{config_id}", response_model=schemas.SupplyChainConfig)
def read_config(
    config_id: int,
    db: Session = Depends(deps.get_db),
):
    """Get a specific configuration by ID."""
    return get_config_or_404(db, config_id)

@router.put("/{config_id}", response_model=schemas.SupplyChainConfig)
def update_config(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    config_in: schemas.SupplyChainConfigUpdate,
    current_user: models.User = Depends(deps.get_current_active_superuser),
):
    """Update a configuration."""
    config = get_config_or_404(db, config_id)
    return crud.supply_chain_config.update(db, db_obj=config, obj_in=config_in)

@router.delete("/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_config(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    current_user: models.User = Depends(deps.get_current_active_superuser),
):
    """Delete a configuration."""
    config = get_config_or_404(db, config_id)
    crud.supply_chain_config.remove(db, id=config_id)
    return None

# --- Item Endpoints ---

@router.get("/{config_id}/items/", response_model=List[schemas.Item])
def read_items(
    config_id: int,
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
):
    """Get all items for a configuration."""
    get_config_or_404(db, config_id)  # Verify config exists
    return crud.item.get_multi_by_config(db, config_id=config_id, skip=skip, limit=limit)

@router.post("/{config_id}/items/", response_model=schemas.Item, status_code=status.HTTP_201_CREATED)
def create_item(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    item_in: schemas.ItemCreate,
    current_user: models.User = Depends(deps.get_current_active_superuser),
):
    """Create a new item in a configuration."""
    get_config_or_404(db, config_id)  # Verify config exists
    
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
):
    """Get all nodes for a configuration, optionally filtered by type."""
    get_config_or_404(db, config_id)  # Verify config exists
    
    if node_type:
        return crud.node.get_by_type(db, node_type=node_type, config_id=config_id)
    return crud.node.get_multi_by_config(db, config_id=config_id)

@router.post("/{config_id}/nodes/", response_model=schemas.Node, status_code=status.HTTP_201_CREATED)
def create_node(
    *,
    db: Session = Depends(deps.get_db),
    config_id: int,
    node_in: schemas.NodeCreate,
    current_user: models.User = Depends(deps.get_current_active_superuser),
):
    """Create a new node in a configuration."""
    get_config_or_404(db, config_id)  # Verify config exists
    
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
