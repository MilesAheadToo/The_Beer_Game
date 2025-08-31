from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from app.db.base import Base
import datetime

class NodeType(Base):
    __tablename__ = "node_types"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)  # e.g., 'Retailer', 'Distributor', 'Factory'
    description = Column(String(255))
    
    nodes = relationship("Node", back_populates="node_type")

class Node(Base):
    __tablename__ = "nodes"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    node_type_id = Column(Integer, ForeignKey("node_types.id"))
    capacity = Column(JSON)  # {distribution: 'lognormal', params: {mean: X, sigma: Y}}
    lead_time = Column(JSON)  # {distribution: 'lognormal', params: {mean: X, sigma: Y}}
    throughput = Column(JSON)  # {distribution: 'lognormal', params: {mean: X, sigma: Y}}
    
    node_type = relationship("NodeType", back_populates="nodes")
    incoming_edges = relationship("Edge", foreign_keys="[Edge.destination_id]", back_populates="destination")
    outgoing_edges = relationship("Edge", foreign_keys="[Edge.source_id]", back_populates="source")
    inventory = relationship("Inventory", back_populates="node")

class Edge(Base):
    __tablename__ = "edges"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("nodes.id"), nullable=False)
    destination_id = Column(Integer, ForeignKey("nodes.id"), nullable=False)
    cost_per_unit = Column(Float, default=0.0)
    transport_lead_time = Column(JSON)  # {distribution: 'lognormal', params: {mean: X, sigma: Y}}
    
    source = relationship("Node", foreign_keys=[source_id], back_populates="outgoing_edges")
    destination = relationship("Node", foreign_keys=[destination_id], back_populates="incoming_edges")

class Inventory(Base):
    __tablename__ = "inventory"
    
    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(Integer, ForeignKey("nodes.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, default=0)
    safety_stock = Column(Integer, default=0)
    reorder_point = Column(Integer, default=0)
    
    node = relationship("Node", back_populates="inventory")
    product = relationship("Product", back_populates="inventory")

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(255))
    unit_cost = Column(Float, default=0.0)
    
    inventory = relationship("Inventory", back_populates="product")

class SimulationRun(Base):
    __tablename__ = "simulation_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(255))
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    parameters = Column(JSON)  # Store simulation parameters
    
    steps = relationship("SimulationStep", back_populates="simulation_run")

class SimulationStep(Base):
    __tablename__ = "simulation_steps"
    
    id = Column(Integer, primary_key=True, index=True)
    simulation_run_id = Column(Integer, ForeignKey("simulation_runs.id"), nullable=False)
    step_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    state = Column(JSON)  # Store the complete state of the simulation at this step
    
    simulation_run = relationship("SimulationRun", back_populates="steps")
