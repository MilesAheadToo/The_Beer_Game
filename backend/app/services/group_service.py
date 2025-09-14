from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
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
from ..schemas.group import GroupCreate, GroupUpdate
from ..core.security import get_password_hash

class GroupService:
    def __init__(self, db: Session):
        self.db = db

    def get_groups(self):
        return self.db.query(Group).all()

    def create_group(self, group_in: GroupCreate) -> Group:
        admin_data = group_in.admin
        hashed_password = get_password_hash(admin_data.password)
        group = Group(name=group_in.name, description=group_in.description, logo=group_in.logo)
        try:
            self.db.add(group)
            self.db.flush()

            admin_user = User(
                username=admin_data.username,
                email=admin_data.email,
                full_name=admin_data.full_name,
                hashed_password=hashed_password,
                roles=["admin"],
                group_id=group.id,
                is_active=True,
                is_superuser=False
            )
            self.db.add(admin_user)
            self.db.flush()

            group.admin_id = admin_user.id
            self.db.add(group)

            sc_config = SupplyChainConfig(
                name="Default TBG",
                description="Default supply chain configuration",
                created_by=admin_user.id,
                group_id=group.id,
                is_active=False
            )
            game = Game(
                name="The Beer Game",
                created_by=admin_user.id,
                group_id=group.id,
                status=GameStatus.CREATED
            )
            self.db.add_all([sc_config, game])
            self.db.flush()

            # Create default players and corresponding user accounts
            default_roles = [
                ("retailer", PlayerRole.RETAILER),
                ("wholesaler", PlayerRole.WHOLESALER),
                ("distributor", PlayerRole.DISTRIBUTOR),
                ("manufacturer", PlayerRole.MANUFACTURER),
            ]

            player_users = []
            for username, role_enum in default_roles:
                user = User(
                    username=username,
                    email=f"{username}@daybreak.ai",
                    full_name=username.capitalize(),
                    hashed_password=get_password_hash("Daybreak2025"),
                    roles=["player"],
                    group_id=group.id,
                    is_active=True,
                    is_superuser=False,
                )
                self.db.add(user)
                self.db.flush()
                player_users.append((user, role_enum))

            players = []
            for user_obj, role_enum in player_users:
                player = Player(
                    game_id=game.id,
                    user_id=user_obj.id,
                    name=user_obj.full_name or user_obj.username,
                    role=role_enum,
                    type=PlayerType.AI,
                    strategy=PlayerStrategy.LLM_BASIC,
                )
                players.append(player)

            self.db.add_all(players)
            self.db.commit()
            self.db.refresh(group)
            return group
        except SQLAlchemyError as e:
            self.db.rollback()
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
