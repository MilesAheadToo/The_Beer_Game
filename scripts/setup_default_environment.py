#!/usr/bin/env python3
"""
Script to set up a default environment for The Beer Game.

This script creates:
1. A group admin user
2. A default group
3. A default supply chain configuration
4. AI players for each role
5. A default game with the AI players
"""
import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))

from sqlalchemy import select, update, insert, delete, func, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.db.session import async_session_factory, engine, Base
from app.models.user import User
from app.models.group import Group
from app.models.supply_chain_config import SupplyChainConfig, Node, Lane, ItemNodeConfig, MarketDemand, NodeType, Item
from app.models.game import Game, GameStatus
from app.models.player import Player
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
    async with async_session_factory() as db:
        try:
            # Check if group admin already exists using raw SQL to avoid ORM issues
            result = await db.execute(
                text("SELECT id FROM users WHERE email = :email"),
                {"email": "groupadmin@daybreak.ai"}
            )
            group_admin_id = result.scalar_one_or_none()
            
            if not group_admin_id:
                # Create group admin user using raw SQL
                result = await db.execute(
                    text("""
                    INSERT INTO users (username, email, hashed_password, full_name, is_superuser, is_active, created_at, updated_at)
                    VALUES (:username, :email, :hashed_password, :full_name, :is_superuser, :is_active, NOW(), NOW())
                    """),
                    {
                        "username": "groupadmin",
                        "email": "groupadmin@daybreak.ai",
                        "hashed_password": get_password_hash("Daybreak@2025"),
                        "full_name": "Group Admin",
                        "is_superuser": False,
                        "is_active": True
                    }
                )
                group_admin_id = result.lastrowid
                logger.info("✅ Created group admin user: groupadmin@daybreak.ai / Daybreak@2025")
            
            # Get the group admin user object
            result = await db.execute(
                select(User).where(User.id == group_admin_id)
            )
            group_admin = result.scalars().first()
        
            # Check if default group exists
            result = await db.execute(
                select(Group).where(Group.name == "Default TBG")
            )
            default_group = result.scalars().first()
            
            if not default_group:
                # Create default group
                result = await db.execute(
                    text("""
                    INSERT INTO `groups` (name, description, admin_id, created_at, updated_at)
                    VALUES (:name, :description, :admin_id, NOW(), NOW())
                    """),
                    {
                        "name": "Default TBG",
                        "description": "Default Group for The Beer Game",
                        "admin_id": group_admin.id
                    }
                )
                default_group_id = result.lastrowid
                logger.info(f"✅ Created default group: Default TBG")
                
                # Update group admin with group_id
                await db.execute(
                    text("UPDATE users SET group_id = :group_id WHERE id = :user_id"),
                    {"group_id": default_group_id, "user_id": group_admin.id}
                )
                logger.info("✅ Updated group admin with group_id")
                
                # Get the default group object
                result = await db.execute(
                    select(Group).where(Group.id == default_group_id)
                )
                default_group = result.scalars().first()
            
            # Check if default supply chain config exists
            result = await db.execute(
                text("SELECT id FROM supply_chain_configs WHERE name = :name"),
                {"name": "Default TBG"}
            )
            default_config_id = result.scalar_one_or_none()
            
            if not default_config_id:
                # Create default supply chain configuration using raw SQL
                result = await db.execute(
                    text("""
                    INSERT INTO supply_chain_configs 
                    (name, description, created_by, created_at, updated_at)
                    VALUES (:name, :description, :created_by, NOW(), NOW())
                    """),
                    {
                        "name": "Default TBG",
                        "description": "Default supply chain configuration for The Beer Game",
                        "created_by": group_admin.id
                    }
                )
                default_config_id = result.lastrowid
                logger.info("✅ Created default supply chain configuration")
                
                # Create nodes
                nodes = [
                    {"name": "Retailer", "type": "RETAILER", "position_x": 100, "position_y": 100},
                    {"name": "Distributor", "type": "DISTRIBUTOR", "position_x": 300, "position_y": 100},
                    {"name": "Manufacturer", "type": "MANUFACTURER", "position_x": 500, "position_y": 100},
                    {"name": "Supplier", "type": "SUPPLIER", "position_x": 700, "position_y": 100}
                ]
                
                node_ids = {}
                for node in nodes:
                    result = await db.execute(
                        text("""
                        INSERT INTO nodes 
                        (name, type, position_x, position_y, config_id, created_at, updated_at)
                        VALUES (:name, :type, :position_x, :position_y, :config_id, NOW(), NOW())
                        """),
                        {
                            "name": node["name"],
                            "type": node["type"],
                            "position_x": node["position_x"],
                            "position_y": node["position_y"],
                            "config_id": default_config_id
                        }
                    )
                    node_ids[node["name"]] = result.lastrowid
                
                # Create lanes between nodes
                lanes = [
                    {"from_node": "Retailer", "to_node": "Distributor"},
                    {"from_node": "Distributor", "to_node": "Manufacturer"},
                    {"from_node": "Manufacturer", "to_node": "Supplier"}
                ]
                
                for lane in lanes:
                    await db.execute(
                        text("""
                        INSERT INTO lanes 
                        (from_node_id, to_node_id, config_id, created_at, updated_at)
                        VALUES (:from_node_id, :to_node_id, :config_id, NOW(), NOW())
                        """),
                        {
                            "from_node_id": node_ids[lane["from_node"]],
                            "to_node_id": node_ids[lane["to_node"]],
                            "config_id": default_config_id
                        }
                    )
                
                # Create items
                result = await db.execute(
                    text("""
                    INSERT INTO items 
                    (name, description, config_id, created_at, updated_at)
                    VALUES (:name, :description, :config_id, NOW(), NOW())
                    RETURNING id
                    """),
                    {
                        "name": "Beer",
                        "description": "Case of Beer (24 bottles)",
                        "config_id": default_config_id
                    }
                )
                item_id = result.scalar_one()
                
                # Create market demand
                await db.execute(
                    text("""
                    INSERT INTO market_demands 
                    (config_id, item_id, mean_demand, std_demand, pattern_type, created_at, updated_at)
                    VALUES (:config_id, :item_id, :mean_demand, :std_demand, :pattern_type, NOW(), NOW())
                    """),
                    {
                        "config_id": default_config_id,
                        "item_id": item_id,
                        "mean_demand": 8,
                        "std_demand": 2,
                        "pattern_type": "NORMAL"
                    }
                )
                
                logger.info("✅ Created supply chain items and market demand")
                
                # Get the default config object
                result = await db.execute(
                    select(SupplyChainConfig).where(SupplyChainConfig.id == default_config_id)
                )
                default_config = result.scalars().first()
            
            # Create AI users for each role if they don't exist
            ai_users = {}
            roles = ["retailer", "distributor", "manufacturer", "supplier"]
            
            for role in roles:
                # Check if AI user exists
                result = await db.execute(
                    text("SELECT id FROM users WHERE email = :email"),
                    {"email": f"ai_{role}@daybreak.ai"}
                )
                user_id = result.scalar_one_or_none()
                
                if not user_id:
                    # Create AI user
                    result = await db.execute(
                        text("""
                        INSERT INTO users 
                        (username, email, hashed_password, full_name, is_superuser, is_active, group_id, created_at, updated_at)
                        VALUES (:username, :email, :hashed_password, :full_name, :is_superuser, :is_active, :group_id, NOW(), NOW())
                        RETURNING id
                        """),
                        {
                            "username": f"ai_{role}",
                            "email": f"ai_{role}@daybreak.ai",
                            "hashed_password": get_password_hash("Daybreak@2025"),
                            "full_name": f"AI {role.capitalize()}",
                            "is_superuser": False,
                            "is_active": True,
                            "group_id": default_group.id
                        }
                    )
                    user_id = result.scalar_one()
                    logger.info(f"✅ Created AI player: ai_{role}")
                
                ai_users[role] = {"id": user_id, "email": f"ai_{role}@daybreak.ai"}
            
            # Check if default game exists
            result = await db.execute(
                text("SELECT id FROM games WHERE name = :name"),
                {"name": "The Beer Game"}
            )
            game_id = result.scalar_one_or_none()
            
            if not game_id:
                # Create default game
                result = await db.execute(
                    text("""
                    INSERT INTO games 
                    (name, description, max_rounds, current_round, status, config_id, group_id, created_by, created_at, updated_at)
                    VALUES (:name, :description, :max_rounds, :current_round, :status, :config_id, :group_id, :created_by, NOW(), NOW())
                    RETURNING id
                    """),
                    {
                        "name": "The Beer Game",
                        "description": "Default Beer Game with AI players",
                        "max_rounds": 50,
                        "current_round": 0,
                        "status": "CREATED",
                        "config_id": default_config.id,
                        "group_id": default_group.id,
                        "created_by": group_admin.id
                    }
                )
                game_id = result.scalar_one()
                logger.info("✅ Created default game")
                
                # Create players for the game
                for role, user in ai_users.items():
                    await db.execute(
                        text("""
                        INSERT INTO players 
                        (game_id, user_id, role, is_ai, strategy, created_at, updated_at)
                        VALUES (:game_id, :user_id, :role, :is_ai, :strategy, NOW(), NOW())
                        """),
                        {
                            "game_id": game_id,
                            "user_id": user["id"],
                            "role": role.upper(),
                            "is_ai": True,
                            "strategy": "naive"
                        }
                    )
                
                logger.info("✅ Added AI players to the game")
            
            # Commit all changes
            await db.commit()
            logger.info("✅ Successfully set up default environment")
            
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Error setting up default environment: {e}")
            raise
                    {"name": "Retailer", "node_type": NodeType.RETAILER, "position_x": 0, "position_y": 0, "role": PlayerRole.RETAILER},
                    {"name": "Distributor", "node_type": NodeType.DISTRIBUTOR, "position_x": 1, "position_y": 0, "role": PlayerRole.DISTRIBUTOR},
                    {"name": "Manufacturer", "node_type": NodeType.MANUFACTURER, "position_x": 2, "position_y": 0, "role": PlayerRole.MANUFACTURER},
                    {"name": "Supplier", "node_type": NodeType.SUPPLIER, "position_x": 3, "position_y": 0, "role": PlayerRole.SUPPLIER},
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
                        lead_time=1,
                        service_level=0.95
                    )
                    db.add(lane)
                
                # Create default item
                item = Item(name="Beer", description="Standard beer product")
                db.add(item)
                await db.flush()
                
                # Create item-node configurations
                for node in node_objs:
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
            for role in ["retailer", "distributor", "manufacturer", "supplier"]:
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
                    config_id=default_config.id,
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
