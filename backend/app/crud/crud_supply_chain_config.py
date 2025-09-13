from typing import List, Optional, Dict, Any, Type, TypeVar, Generic
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session, Query, joinedload
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel

from app.models.supply_chain_config import (
    SupplyChainConfig, Item, Node, Lane, ItemNodeConfig, MarketDemand, NodeType
)
from app.schemas.supply_chain_config import (
    SupplyChainConfigCreate, SupplyChainConfigUpdate,
    ItemCreate, ItemUpdate,
    NodeCreate, NodeUpdate,
    LaneCreate, LaneUpdate,
    ItemNodeConfigCreate, ItemNodeConfigUpdate,
    MarketDemandCreate, MarketDemandUpdate
)
from app.crud.base import CRUDBase

ModelType = TypeVar("ModelType", bound=Any)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)

class CRUDSupplyChainConfig(CRUDBase[SupplyChainConfig, SupplyChainConfigCreate, SupplyChainConfigUpdate]):
    def get_active(self, db: Session) -> Optional[SupplyChainConfig]:
        return db.query(self.model).filter(self.model.is_active == True).first()

    def get_multi_by_creator(
        self, db: Session, *, creator_id: int, skip: int = 0, limit: int = 100
    ) -> List[SupplyChainConfig]:
        """Return configs created by a specific user."""
        return (
            db.query(self.model)
            .filter(self.model.created_by == creator_id)
            .offset(skip)
            .limit(limit)
            .all()
        )
            
    def create(self, db: Session, *, obj_in: SupplyChainConfigCreate) -> SupplyChainConfig:
        # Ensure only one active config
        if obj_in.is_active:
            db.query(self.model).filter(self.model.is_active == True).update({"is_active": False})
        return super().create(db, obj_in=obj_in)
    
    def update(
        self, db: Session, *, db_obj: SupplyChainConfig, obj_in: SupplyChainConfigUpdate
    ) -> SupplyChainConfig:
        # Handle is_active update
        if obj_in.is_active is True and not db_obj.is_active:
            # Deactivate other active configs
            db.query(self.model).filter(
                self.model.id != db_obj.id,
                self.model.is_active == True
            ).update({"is_active": False})
        
        return super().update(db, db_obj=db_obj, obj_in=obj_in)

class CRUDItem(CRUDBase[Item, ItemCreate, ItemUpdate]):
    def get_by_name(self, db: Session, *, name: str, config_id: int) -> Optional[Item]:
        return db.query(self.model).filter(
            self.model.name == name,
            self.model.config_id == config_id
        ).first()

class CRUDNode(CRUDBase[Node, NodeCreate, NodeUpdate]):
    def get_by_name_and_type(self, db: Session, *, name: str, node_type: NodeType, config_id: int) -> Optional[Node]:
        return db.query(self.model).filter(
            self.model.name == name,
            self.model.type == node_type,
            self.model.config_id == config_id
        ).first()
    
    def get_by_type(self, db: Session, *, node_type: NodeType, config_id: int) -> List[Node]:
        return db.query(self.model).filter(
            self.model.type == node_type,
            self.model.config_id == config_id
        ).all()

class CRUDLane(CRUDBase[Lane, LaneCreate, LaneUpdate]):
    def get_by_nodes(
        self, 
        db: Session, 
        *, 
        upstream_node_id: int, 
        downstream_node_id: int,
        config_id: int
    ) -> Optional[Lane]:
        return db.query(self.model).filter(
            self.model.upstream_node_id == upstream_node_id,
            self.model.downstream_node_id == downstream_node_id,
            self.model.config_id == config_id
        ).first()
    
    def get_by_config(self, db: Session, *, config_id: int) -> List[Lane]:
        return db.query(self.model).options(
            joinedload(self.model.upstream_node),
            joinedload(self.model.downstream_node)
        ).filter(self.model.config_id == config_id).all()

class CRUDItemNodeConfig(CRUDBase[ItemNodeConfig, ItemNodeConfigCreate, ItemNodeConfigUpdate]):
    def get_by_item_and_node(
        self, 
        db: Session, 
        *, 
        item_id: int, 
        node_id: int
    ) -> Optional[ItemNodeConfig]:
        return db.query(self.model).filter(
            self.model.item_id == item_id,
            self.model.node_id == node_id
        ).first()
    
    def get_by_config(self, db: Session, *, config_id: int) -> List[ItemNodeConfig]:
        return db.query(self.model).join(Node).filter(
            Node.config_id == config_id
        ).options(
            joinedload(self.model.item),
            joinedload(self.model.node)
        ).all()

class CRUDMarketDemand(CRUDBase[MarketDemand, MarketDemandCreate, MarketDemandUpdate]):
    def get_by_item_and_retailer(
        self, 
        db: Session, 
        *, 
        item_id: int, 
        retailer_id: int,
        config_id: int
    ) -> Optional[MarketDemand]:
        return db.query(self.model).filter(
            self.model.item_id == item_id,
            self.model.retailer_id == retailer_id,
            self.model.config_id == config_id
        ).first()
    
    def get_by_config(self, db: Session, *, config_id: int) -> List[MarketDemand]:
        return db.query(self.model).options(
            joinedload(self.model.item),
            joinedload(self.model.retailer)
        ).filter(self.model.config_id == config_id).all()

# Initialize CRUD classes
supply_chain_config = CRUDSupplyChainConfig(SupplyChainConfig)
item = CRUDItem(Item)
node = CRUDNode(Node)
lane = CRUDLane(Lane)
item_node_config = CRUDItemNodeConfig(ItemNodeConfig)
market_demand = CRUDMarketDemand(MarketDemand)
