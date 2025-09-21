from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    Enum,
    JSON,
    Boolean,
    UniqueConstraint,
    DateTime,
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declared_attr
from enum import Enum as PyEnum
from typing import List, Optional, TYPE_CHECKING
import datetime
from .base import Base

if TYPE_CHECKING:
    from .group import Group

class NodeType(str, PyEnum):
    RETAILER = "retailer"
    WHOLESALER = "wholesaler"
    DISTRIBUTOR = "distributor"
    MANUFACTURER = "manufacturer"

class SupplyChainConfig(Base):
    """Core configuration for the supply chain"""
    __tablename__ = "supply_chain_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, default="Default Configuration")
    description = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=True)
    group_id = Column(Integer, ForeignKey('groups.id', ondelete='CASCADE'), nullable=False)
    
    # Relationships
    items = relationship("Item", back_populates="config", cascade="all, delete-orphan")
    nodes = relationship("Node", back_populates="config", cascade="all, delete-orphan")
    lanes = relationship("Lane", back_populates="config", cascade="all, delete-orphan")
    market_demands = relationship("MarketDemand", back_populates="config", cascade="all, delete-orphan")
    group = relationship("Group", back_populates="supply_chain_configs")

    # Training metadata
    needs_training = Column(Boolean, nullable=False, default=True)
    training_status = Column(String(50), nullable=False, default="pending")
    trained_at = Column(DateTime, nullable=True)
    trained_model_path = Column(String(255), nullable=True)
    last_trained_config_hash = Column(String(128), nullable=True)

class Item(Base):
    """Products in the supply chain"""
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True, index=True)
    config_id = Column(Integer, ForeignKey("supply_chain_configs.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    unit_cost_range = Column(JSON, default={"min": 0, "max": 100})  # Cost range for training
    
    # Relationships
    config = relationship("SupplyChainConfig", back_populates="items")
    node_configs = relationship("ItemNodeConfig", back_populates="item", cascade="all, delete-orphan")

class Node(Base):
    """Nodes in the supply chain (retailer, distributor, etc.)"""
    __tablename__ = "nodes"
    
    id = Column(Integer, primary_key=True, index=True)
    config_id = Column(Integer, ForeignKey("supply_chain_configs.id"), nullable=False)
    name = Column(String(100), nullable=False)
    type = Column(Enum(NodeType), nullable=False)
    
    # Relationships
    config = relationship("SupplyChainConfig", back_populates="nodes")
    upstream_lanes = relationship("Lane", foreign_keys="Lane.downstream_node_id", back_populates="downstream_node")
    downstream_lanes = relationship("Lane", foreign_keys="Lane.upstream_node_id", back_populates="upstream_node")
    item_configs = relationship("ItemNodeConfig", back_populates="node", cascade="all, delete-orphan")

class Lane(Base):
    """Connections between nodes with capacities and lead times"""
    __tablename__ = "lanes"
    
    id = Column(Integer, primary_key=True, index=True)
    config_id = Column(Integer, ForeignKey("supply_chain_configs.id"), nullable=False)
    upstream_node_id = Column(Integer, ForeignKey("nodes.id"), nullable=False)
    downstream_node_id = Column(Integer, ForeignKey("nodes.id"), nullable=False)
    
    # Capacity in units per day
    capacity = Column(Integer, nullable=False)
    
    # Lead time in days (fixed or range for training)
    lead_time_days = Column(JSON, default={"min": 1, "max": 3})
    
    # Relationships
    config = relationship("SupplyChainConfig", back_populates="lanes")
    upstream_node = relationship("Node", foreign_keys=[upstream_node_id], back_populates="downstream_lanes")
    downstream_node = relationship("Node", foreign_keys=[downstream_node_id], back_populates="upstream_lanes")
    
    # Ensure we don't have duplicate lanes
    __table_args__ = (
        UniqueConstraint('upstream_node_id', 'downstream_node_id', name='_node_connection_uc'),
    )

class ItemNodeConfig(Base):
    """Configuration for items at specific nodes"""
    __tablename__ = "item_node_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=False)
    node_id = Column(Integer, ForeignKey("nodes.id"), nullable=False)
    
    # Ranges for training data generation
    inventory_target_range = Column(JSON, default={"min": 10, "max": 50})
    initial_inventory_range = Column(JSON, default={"min": 5, "max": 30})
    holding_cost_range = Column(JSON, default={"min": 0.1, "max": 1.0})  # Cost per unit per day
    backlog_cost_range = Column(JSON, default={"min": 0.5, "max": 2.0})  # Cost per unit per day
    selling_price_range = Column(JSON, default={"min": 5.0, "max": 20.0})
    
    # Relationships
    item = relationship("Item", back_populates="node_configs")
    node = relationship("Node", back_populates="item_configs")
    
    # Ensure we don't have duplicate configs
    __table_args__ = (
        UniqueConstraint('item_id', 'node_id', name='_item_node_uc'),
    )

class MarketDemand(Base):
    """Market demand configuration per item per retailer"""
    __tablename__ = "market_demands"
    
    id = Column(Integer, primary_key=True, index=True)
    config_id = Column(Integer, ForeignKey("supply_chain_configs.id"), nullable=False)
    item_id = Column(Integer, ForeignKey("items.id"), nullable=False)
    retailer_id = Column(Integer, ForeignKey("nodes.id"), nullable=False)
    
    # Demand pattern configuration (e.g., normal, uniform, seasonal)
    demand_pattern = Column(JSON, default={
        "type": "normal",  # normal, uniform, seasonal, etc.
        "mean": 10,        # for normal distribution
        "stddev": 2,       # for normal distribution
        "min": 5,          # for uniform distribution
        "max": 15,         # for uniform distribution
        "seasonality": {   # for seasonal patterns
            "period": 7,   # e.g., weekly seasonality
            "amplitude": 2 # strength of seasonality
        }
    })
    
    # Relationships
    config = relationship("SupplyChainConfig", back_populates="market_demands")
    item = relationship("Item")
    retailer = relationship("Node")
