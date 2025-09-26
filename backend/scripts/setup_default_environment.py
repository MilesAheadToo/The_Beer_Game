#!/usr/bin/env python3
"""
Script to set up the default environment with a group admin, default group, 
supply chain configuration, and a game with AI players.
"""
import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))

import asyncio
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))

from sqlalchemy import select, update, insert, delete, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.db.session import async_session_factory, engine, Base
from app.models.user import User, UserBase
from app.models.group import Group
from app.models.supply_chain_config import SupplyChainConfig, Node, Lane, ItemNodeConfig, MarketDemand, NodeType, Item
from app.models.game import Game, GameStatus
from app.models.player import Player, PlayerRole
from app.core.security import get_password_hash
from datetime import datetime

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_default_environment():
    """Create the default environment with group admin, group, and game."""
    # Create all tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with async_session_factory() as db:
        try:
            # Check if group admin already exists
            result = await db.execute(
                select(User).where(User.email == "groupadmin@daybreak.ai")
            )
            group_admin = result.scalars().first()
            
            if not group_admin:
                # Create group admin user
                group_admin = User(
                    username="groupadmin",
                    email="groupadmin@daybreak.ai",
                    hashed_password=get_password_hash("Daybreak@2025"),
                    full_name="Group Admin",
                    is_superuser=False,
                    is_active=True
                )
                db.add(group_admin)
                await db.flush()  # Flush to get the ID
                logger.info("✅ Created group admin user: groupadmin@daybreak.ai / Daybreak@2025")
        
            # Check if default group exists
            result = await db.execute(
                select(Group).where(Group.name == "Default TBG")
            )
            default_group = result.scalars().first()
            
            if not default_group:
                # Create default group
                default_group = Group(
                    name="Default TBG",
                    description="Default Group for The Beer Game",
                    admin_id=group_admin.id
                )
                db.add(default_group)
                await db.flush()  # Flush to get the ID
                logger.info(f"✅ Created default group: {default_group.name}")
                
                # Update group admin with group_id
                group_admin.group_id = default_group.id
                await db.flush()
                logger.info(f"✅ Updated group admin with group_id: {group_admin.group_id}")
            
            # Check if default supply chain config exists
            result = await db.execute(
                select(SupplyChainConfig).where(SupplyChainConfig.name == "Default TBG")
            )
            default_config = result.scalars().first()
            
            if not default_config:
                # Create default supply chain configuration
                default_config = SupplyChainConfig(
                    name="Default TBG",
                    description="Default supply chain configuration",
                    group_id=default_group.id,
                    created_by=group_admin.id
                )
                db.add(default_config)
                await db.flush()
            
                # Create nodes
                nodes = [
                    {"name": "Market Supply", "node_type": NodeType.MARKET_SUPPLY, "position_x": -1, "position_y": 0, "role": None},
                    {"name": "Manufacturer", "node_type": NodeType.MANUFACTURER, "position_x": 0, "position_y": 0, "role": PlayerRole.MANUFACTURER},
                    {"name": "Distributor", "node_type": NodeType.DISTRIBUTOR, "position_x": 1, "position_y": 0, "role": PlayerRole.DISTRIBUTOR},
                    {"name": "Wholesaler", "node_type": NodeType.WHOLESALER, "position_x": 2, "position_y": 0, "role": PlayerRole.WHOLESALER},
                    {"name": "Retailer", "node_type": NodeType.RETAILER, "position_x": 3, "position_y": 0, "role": PlayerRole.RETAILER},
                    {"name": "Market Demand", "node_type": NodeType.MARKET_DEMAND, "position_x": 4, "position_y": 0, "role": None},
                ]
                
                node_objs = []
                for node_data in nodes:
                    node = Node(
                        name=node_data["name"],
                        node_type=node_data["node_type"],
                        position_x=node_data["position_x"],
                        position_y=node_data["position_y"],
                        config_id=default_config.id
                    )
                    db.add(node)
                    node_objs.append(node)
                
                await db.flush()
                
                # Create lanes between nodes
                for i in range(len(node_objs) - 1):
                    lane = Lane(
                        source_id=node_objs[i].id,
                        target_id=node_objs[i+1].id,
                        config_id=default_config.id,
                        lead_time=0 if node_objs[i].node_type in {NodeType.MARKET_SUPPLY} or node_objs[i+1].node_type in {NodeType.MARKET_DEMAND} else 1,
                        service_level=0.95
                    )
                    db.add(lane)
                
                # Create default item
                item = Item(name="Beer", description="Standard beer product")
                db.add(item)
                await db.flush()
                
                # Create item-node configurations
                for node in node_objs:
                    if node.node_type in {NodeType.MARKET_SUPPLY, NodeType.MARKET_DEMAND}:
                        continue
                    inc = ItemNodeConfig(
                        item_id=item.id,
                        node_id=node.id,
                        config_id=default_config.id,
                        holding_cost=1.0,
                        backlog_cost=2.0,
                        initial_inventory=12,
                        order_up_to=30,
                        reorder_point=10
                    )
                    db.add(inc)
                
                # Create market demand
                market_demand = MarketDemand(
                    config_id=default_config.id,
                    item_id=item.id,
                    mean_demand=8,
                    std_demand=2,
                    pattern_type="NORMAL"
                )
                db.add(market_demand)
            
                logger.info(f"✅ Created default supply chain configuration: {default_config.name}")
            
            # Create AI users for each role
            ai_users = {}
            for role in ["retailer", "wholesaler", "distributor", "manufacturer"]:
                ai_user = User(
                    username=f"ai_{role}",
                    email=f"ai_{role}@daybreak.ai",
                    hashed_password=get_password_hash("Daybreak@2025"),
                    full_name=f"AI {role.capitalize()}",
                    is_superuser=False,
                    is_active=True,
                    group_id=default_group.id  # Add AI users to the default group
                )
                db.add(ai_user)
                await db.flush()
                ai_users[role] = ai_user
                logger.info(f"✅ Created AI player: {ai_user.username}")
            
            # Check if default game exists
            result = await db.execute(
                select(Game).where(Game.name == "The Beer Game")
            )
            default_game = result.scalars().first()
            
            if not default_game:
                # Create default game
                default_game = Game(
                    name="The Beer Game",
                    description="Default beer game with AI players",
                    max_rounds=50,
                    current_round=0,
                    status=GameStatus.CREATED,
                    supply_chain_config_id=default_config.id,
                    group_id=default_group.id,
                    created_by=group_admin.id
                )
                db.add(default_game)
                await db.flush()
                
                # Create players for the game
                for role, user in ai_users.items():
                    player = Player(
                        game_id=default_game.id,
                        user_id=user.id,
                        role=PlayerRole[role.upper()],
                        is_ai=True,
                        strategy="naive"  # Simple ordering strategy
                    )
                    db.add(player)
                
                logger.info(f"✅ Created default game: {default_game.name}")
            
            # Commit all changes at the end
            await db.commit()
            logger.info("✅ Successfully set up default environment")

        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Error setting up default environment: {e}")
            raise

if __name__ == "__main__":
    import asyncio
    
    logger.info("\n[+] Setting up default environment...")
    try:
        asyncio.run(create_default_environment())
    except Exception as e:
        logger.error(f"Failed to set up default environment: {e}")
        sys.exit(1)
