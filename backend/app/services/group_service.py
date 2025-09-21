import logging

from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from ..models import (
    Group,
    User,
    SupplyChainConfig,
    Game,
    GameStatus,
    Player,
    PlayerRole,
    PlayerType,
    PlayerStrategy,
)
from ..models.user import UserTypeEnum
from ..models.supply_chain_config import (
    Item,
    Node,
    Lane,
    ItemNodeConfig,
    MarketDemand,
    NodeType,
)
from ..schemas.group import GroupCreate, GroupUpdate
from ..core.security import get_password_hash
from .supply_chain_config_service import SupplyChainConfigService

logger = logging.getLogger(__name__)

class GroupService:
    def __init__(self, db: Session):
        self.db = db

    def get_groups(self):
        return self.db.query(Group).all()

    def get_group(self, group_id: int) -> Group:
        group = self.db.query(Group).filter(Group.id == group_id).first()
        if not group:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Group not found",
            )
        return group

    def create_group(self, group_in: GroupCreate) -> Group:
        admin_data = group_in.admin
        hashed_password = get_password_hash(admin_data.password)
        try:
            admin_user = User(
                username=admin_data.username,
                email=admin_data.email,
                full_name=admin_data.full_name,
                hashed_password=hashed_password,
                user_type=UserTypeEnum.GROUP_ADMIN,
                is_active=True,
                is_superuser=False,
            )
            self.db.add(admin_user)
            self.db.flush()

            group = Group(
                name=group_in.name,
                description=group_in.description,
                logo=group_in.logo,
                admin_id=admin_user.id,
            )
            self.db.add(group)
            self.db.flush()

            admin_user.group_id = group.id
            self.db.add(admin_user)

            sc_config = SupplyChainConfig(
                name="Default TBG",
                description="Default supply chain configuration",
                created_by=admin_user.id,
                group_id=group.id,
                is_active=True
            )
            self.db.add(sc_config)
            self.db.flush()

            item = Item(
                config_id=sc_config.id,
                name="Case of Beer",
                description="Standard product for the Beer Game"
            )
            self.db.add(item)
            self.db.flush()

            node_specs = [
                ("Retailer", NodeType.RETAILER),
                ("Wholesaler", NodeType.WHOLESALER),
                ("Distributor", NodeType.DISTRIBUTOR),
                ("Manufacturer", NodeType.MANUFACTURER),
            ]
            nodes = {}
            for name, node_type in node_specs:
                node = Node(
                    config_id=sc_config.id,
                    name=name,
                    type=node_type,
                )
                self.db.add(node)
                self.db.flush()
                nodes[node_type] = node

            lane_specs = [
                (NodeType.MANUFACTURER, NodeType.DISTRIBUTOR),
                (NodeType.DISTRIBUTOR, NodeType.WHOLESALER),
                (NodeType.WHOLESALER, NodeType.RETAILER),
            ]
            for upstream_type, downstream_type in lane_specs:
                lane = Lane(
                    config_id=sc_config.id,
                    upstream_node_id=nodes[upstream_type].id,
                    downstream_node_id=nodes[downstream_type].id,
                    capacity=9999,
                    lead_time_days={"min": 2, "max": 10},
                )
                self.db.add(lane)

            self.db.flush()

            for node in nodes.values():
                node_config = ItemNodeConfig(
                    item_id=item.id,
                    node_id=node.id,
                    inventory_target_range={"min": 10, "max": 20},
                    initial_inventory_range={"min": 5, "max": 30},
                    holding_cost_range={"min": 1.0, "max": 5.0},
                    backlog_cost_range={"min": 5.0, "max": 10.0},
                    selling_price_range={"min": 25.0, "max": 50.0},
                )
                self.db.add(node_config)

            market_demand = MarketDemand(
                config_id=sc_config.id,
                item_id=item.id,
                retailer_id=nodes[NodeType.RETAILER].id,
                demand_pattern={"type": "constant", "params": {"value": 4}},
            )
            self.db.add(market_demand)
            self.db.flush()

            config_service = SupplyChainConfigService(self.db)
            game_config = config_service.create_game_from_config(
                sc_config.id,
                {"name": "The Beer Game", "max_rounds": 50},
            )

            game = Game(
                name=game_config.get("name", "The Beer Game"),
                created_by=admin_user.id,
                group_id=group.id,
                status=GameStatus.CREATED,
                max_rounds=game_config.get("max_rounds", 52),
                config=game_config,
                demand_pattern=game_config.get("demand_pattern", {}),
            )
            self.db.add(game)
            self.db.flush()

            group_suffix = f"g{group.id}"
            player_password_hash = get_password_hash("Daybreak@2025")
            default_users = [
                {
                    "username": f"retailer_{group_suffix}",
                    "email": f"retailer+{group_suffix}@daybreak.ai",
                    "full_name": "Retailer",
                    "role": PlayerRole.RETAILER,
                },
                {
                    "username": f"distributor_{group_suffix}",
                    "email": f"distributor+{group_suffix}@daybreak.ai",
                    "full_name": "Distributor",
                    "role": PlayerRole.DISTRIBUTOR,
                },
                {
                    "username": f"manufacturer_{group_suffix}",
                    "email": f"manufacturer+{group_suffix}@daybreak.ai",
                    "full_name": "Manufacturer",
                    "role": PlayerRole.MANUFACTURER,
                },
                {
                    "username": f"wholesaler_{group_suffix}",
                    "email": f"wholesaler+{group_suffix}@daybreak.ai",
                    "full_name": "Wholesaler",
                    "role": PlayerRole.WHOLESALER,
                },
            ]

            player_users = []
            for spec in default_users:
                user = User(
                    username=spec["username"],
                    email=spec["email"],
                    full_name=spec["full_name"],
                    hashed_password=player_password_hash,
                    user_type=UserTypeEnum.PLAYER,
                    group_id=group.id,
                    is_active=True,
                    is_superuser=False,
                )
                self.db.add(user)
                self.db.flush()
                player_users.append((user, spec["role"], spec["full_name"]))

            players = []
            for user_obj, role_enum, display_name in player_users:
                player = Player(
                    game_id=game.id,
                    user_id=user_obj.id,
                    name=display_name,
                    role=role_enum,
                    type=PlayerType.AI,
                    strategy=PlayerStrategy.MANUAL,
                    is_ai=True,
                    ai_strategy="naive",
                )
                players.append(player)

            self.db.add_all(players)

            game.role_assignments = {
                role_enum.value: {
                    "is_ai": True,
                    "agent_config_id": None,
                    "user_id": user_obj.id,
                    "strategy": "naive",
                }
                for user_obj, role_enum, _ in player_users
            }
            self.db.add(game)

            self.db.commit()
            self.db.refresh(group)
            return group
        except Exception:
            self.db.rollback()
            logger.exception("Failed to create group %s", group_in.name)
            raise HTTPException(status_code=500, detail="Error creating group")

    def update_group(self, group_id: int, group_update: GroupUpdate) -> Group:
        group = self.db.query(Group).filter(Group.id == group_id).first()
        if not group:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found")
        for field, value in group_update.dict(exclude_unset=True).items():
            setattr(group, field, value)
        self.db.commit()
        self.db.refresh(group)
        return group

    def delete_group(self, group_id: int):
        group = self.db.query(Group).filter(Group.id == group_id).first()
        if not group:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found")
        self.db.delete(group)
        self.db.commit()
        return {"message": "Group deleted"}
