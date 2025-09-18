#!/usr/bin/env python3
"""Seed the default Daybreak group, configuration, and game with naive AI players."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

# Ensure the backend package is importable when running via `python backend/scripts/...`
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles

from app.core.config import settings
from app.db.base_class import SessionLocal
from app.models import Base as ModelBase
from app.models import (
    AgentConfig,
    Game,
    GameStatus,
    Group,
    Item,
    ItemNodeConfig,
    Lane,
    MarketDemand,
    Node,
    NodeType,
    Player,
    PlayerRole,
    PlayerStrategy,
    PlayerType,
    SupplyChainConfig,
    User,
)
from app.models.user import UserTypeEnum
from app.services.supply_chain_config_service import SupplyChainConfigService
from app.core.security import get_password_hash

DEFAULT_GROUP_NAME = "Default TBG"
DEFAULT_GROUP_DESCRIPTION = "Default Daybreak Beer Game group"
DEFAULT_ADMIN_USERNAME = "groupadmin"
DEFAULT_ADMIN_EMAIL = "groupadmin@daybreak.ai"
DEFAULT_ADMIN_FULL_NAME = "Group Administrator"
DEFAULT_PASSWORD = "Daybreak@2025"
DEFAULT_GAME_NAME = "The Beer Game"
DEFAULT_AGENT_TYPE = "naive"

FALLBACK_DB_FILENAME = "seed_default_group.sqlite"
FALLBACK_DB_PATH = BACKEND_ROOT / FALLBACK_DB_FILENAME


@compiles(JSONB, "sqlite")
def _compile_jsonb_for_sqlite(type_, compiler, **kw):
    """Render PostgreSQL JSONB columns as JSON when using SQLite fallback."""

    return "JSON"


def create_sqlite_session_factory() -> Tuple[sessionmaker, Path]:
    """Create a SQLite session factory for fallback seeding."""
    FALLBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    fallback_uri = f"sqlite:///{FALLBACK_DB_PATH}"
    engine = create_engine(
        fallback_uri,
        connect_args={"check_same_thread": False},
    )
    # Use the SQLAlchemy Base from the models package so all tables are registered
    ModelBase.metadata.create_all(bind=engine)
    factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False,
    )
    return factory, FALLBACK_DB_PATH


def mask_db_uri(uri: str) -> str:
    """Return a database URI with any password information masked."""
    if not uri:
        return uri

    try:
        url = make_url(uri)
        if url.password:
            url = url.set(password="***")
        return str(url)
    except Exception:
        return uri


def ensure_group(session: Session) -> Tuple[Group, bool]:
    """Create the default group and admin if they do not already exist."""
    existing_group = (
        session.query(Group).filter(Group.name == DEFAULT_GROUP_NAME).first()
    )
    if existing_group:
        admin = existing_group.admin
        if admin:
            updated = False
            if admin.user_type != UserTypeEnum.GROUP_ADMIN:
                admin.user_type = UserTypeEnum.GROUP_ADMIN
                updated = True
            if admin.is_superuser:
                admin.is_superuser = False
                updated = True
            if admin.group_id != existing_group.id:
                admin.group_id = existing_group.id
                updated = True
            if updated:
                session.add(admin)
                session.flush()
                print(
                    f"[info] Updated admin settings for group '{existing_group.name}' (id={existing_group.id})."
                )
        print(
            f"[info] Group '{DEFAULT_GROUP_NAME}' already exists (id={existing_group.id})."
        )
        return existing_group, False

    print("[info] Creating default group and administrator user...")

    admin_user = (
        session.query(User).filter(User.email == DEFAULT_ADMIN_EMAIL).first()
    )
    admin_created = False
    if admin_user is None:
        admin_user = User(
            username=DEFAULT_ADMIN_USERNAME,
            email=DEFAULT_ADMIN_EMAIL,
            full_name=DEFAULT_ADMIN_FULL_NAME,
            hashed_password=get_password_hash(DEFAULT_PASSWORD),
            is_active=True,
            is_superuser=False,
            user_type=UserTypeEnum.GROUP_ADMIN,
        )
        session.add(admin_user)
        session.flush()
        admin_created = True
    else:
        # Ensure existing admin has the expected settings
        updated = False
        if admin_user.username != DEFAULT_ADMIN_USERNAME:
            admin_user.username = DEFAULT_ADMIN_USERNAME
            updated = True
        if admin_user.full_name != DEFAULT_ADMIN_FULL_NAME:
            admin_user.full_name = DEFAULT_ADMIN_FULL_NAME
            updated = True
        if admin_user.is_superuser:
            admin_user.is_superuser = False
            updated = True
        if admin_user.user_type != UserTypeEnum.GROUP_ADMIN:
            admin_user.user_type = UserTypeEnum.GROUP_ADMIN
            updated = True
        if not admin_user.is_active:
            admin_user.is_active = True
            updated = True
        if updated:
            session.add(admin_user)
            session.flush()

    group = Group(
        name=DEFAULT_GROUP_NAME,
        description=DEFAULT_GROUP_DESCRIPTION,
        admin_id=admin_user.id,
    )
    session.add(group)
    session.flush()

    if admin_user.group_id != group.id:
        admin_user.group_id = group.id
        session.add(admin_user)
        session.flush()

    print(
        f"[success] Created group '{group.name}' (id={group.id}) and admin "
        f"'{DEFAULT_ADMIN_EMAIL}'{' (new user)' if admin_created else ''}."
    )

    return group, True


def ensure_supply_chain_config(session: Session, group: Group) -> SupplyChainConfig:
    """Ensure the default supply chain configuration exists for the group."""
    config = (
        session.query(SupplyChainConfig)
        .filter(SupplyChainConfig.group_id == group.id)
        .order_by(SupplyChainConfig.id.asc())
        .first()
    )
    if config:
        print(
            f"[info] Supply chain configuration already exists (id={config.id})."
        )
        return config

    print("[info] Creating default supply chain configuration...")
    config = SupplyChainConfig(
        name="Default TBG",
        description="Default supply chain configuration",
        created_by=group.admin_id,
        group_id=group.id,
        is_active=True,
    )
    session.add(config)
    session.flush()

    item = Item(
        config_id=config.id,
        name="Case of Beer",
        description="Standard product for the Beer Game",
    )
    session.add(item)
    session.flush()

    node_specs = [
        ("Retailer", NodeType.RETAILER),
        ("Distributor", NodeType.DISTRIBUTOR),
        ("Manufacturer", NodeType.MANUFACTURER),
        ("Supplier", NodeType.SUPPLIER),
    ]
    nodes: Dict[NodeType, Node] = {}
    for name, node_type in node_specs:
        node = Node(
            config_id=config.id,
            name=name,
            type=node_type,
        )
        session.add(node)
        session.flush()
        nodes[node_type] = node

    lane_specs = [
        (NodeType.SUPPLIER, NodeType.MANUFACTURER),
        (NodeType.MANUFACTURER, NodeType.DISTRIBUTOR),
        (NodeType.DISTRIBUTOR, NodeType.RETAILER),
    ]
    for upstream_type, downstream_type in lane_specs:
        lane = Lane(
            config_id=config.id,
            upstream_node_id=nodes[upstream_type].id,
            downstream_node_id=nodes[downstream_type].id,
            capacity=9999,
            lead_time_days={"min": 2, "max": 10},
        )
        session.add(lane)

    session.flush()

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
        session.add(node_config)

    market_demand = MarketDemand(
        config_id=config.id,
        item_id=item.id,
        retailer_id=nodes[NodeType.RETAILER].id,
        demand_pattern={"type": "constant", "params": {"value": 4}},
    )
    session.add(market_demand)
    session.flush()

    print(
        f"[success] Created supply chain configuration (id={config.id}) for group {group.id}."
    )
    return config


def ensure_default_game(session: Session, group: Group) -> Game:
    """Ensure the default game exists for the supplied group."""
    game = (
        session.query(Game)
        .filter(Game.group_id == group.id, Game.name == DEFAULT_GAME_NAME)
        .first()
    )
    if game:
        print(
            f"[info] Game '{DEFAULT_GAME_NAME}' already exists (id={game.id})."
        )
        return game

    print("[info] Creating default game from supply chain configuration...")
    sc_config = ensure_supply_chain_config(session, group)

    config_service = SupplyChainConfigService(session)
    game_config = config_service.create_game_from_config(
        sc_config.id, {"name": DEFAULT_GAME_NAME, "max_rounds": 50}
    )

    game = Game(
        name=game_config.get("name", DEFAULT_GAME_NAME),
        created_by=sc_config.created_by or group.admin_id,
        group_id=group.id,
        status=GameStatus.CREATED,
        max_rounds=game_config.get("max_rounds", 52),
        config=game_config,
        demand_pattern=game_config.get("demand_pattern", {}),
    )
    session.add(game)
    session.flush()
    print(f"[success] Created game '{game.name}' (id={game.id}).")
    return game


def _ensure_default_players(session: Session, game: Game) -> None:
    """Create placeholder AI players if none exist for the game."""
    players = session.query(Player).filter(Player.game_id == game.id).all()
    if players:
        return

    print("[info] Creating default players for the game...")
    role_specs = [
        ("retailer", PlayerRole.RETAILER),
        ("wholesaler", PlayerRole.WHOLESALER),
        ("distributor", PlayerRole.DISTRIBUTOR),
        ("manufacturer", PlayerRole.MANUFACTURER),
    ]
    for display_name, role in role_specs:
        player = Player(
            game_id=game.id,
            name=f"{display_name.title()} (AI)",
            role=role,
            type=PlayerType.AI,
            strategy=PlayerStrategy.MANUAL,
            is_ai=True,
            ai_strategy=DEFAULT_AGENT_TYPE,
            can_see_demand=(role == PlayerRole.RETAILER),
        )
        session.add(player)
    session.flush()


def ensure_naive_agents(session: Session, game: Game) -> None:
    """Assign naive AI agents to each role in the game."""
    _ensure_default_players(session, game)

    players = session.query(Player).filter(Player.game_id == game.id).all()
    if not players:
        raise RuntimeError("No players available to configure for the game.")

    role_assignments: Dict[str, Dict[str, int | bool | None]] = {}
    for player in players:
        player.is_ai = True
        player.type = PlayerType.AI
        player.ai_strategy = DEFAULT_AGENT_TYPE
        player.user_id = None
        if player.strategy != PlayerStrategy.MANUAL:
            player.strategy = PlayerStrategy.MANUAL
        session.add(player)

        agent_config = (
            session.query(AgentConfig)
            .filter(
                AgentConfig.game_id == game.id, AgentConfig.role == player.role.value
            )
            .first()
        )
        if agent_config:
            agent_config.agent_type = DEFAULT_AGENT_TYPE
            agent_config.config = agent_config.config or {}
            session.add(agent_config)
        else:
            agent_config = AgentConfig(
                game_id=game.id,
                role=player.role.value,
                agent_type=DEFAULT_AGENT_TYPE,
                config={},
            )
            session.add(agent_config)
            session.flush()

        role_assignments[player.role.value] = {
            "is_ai": True,
            "agent_config_id": agent_config.id,
            "user_id": None,
        }

    if "wholesaler" in role_assignments and "supplier" not in role_assignments:
        role_assignments["supplier"] = dict(role_assignments["wholesaler"])

    game.role_assignments = role_assignments
    session.add(game)
    print(
        "[success] Assigned naive agents to roles: "
        + ", ".join(sorted(role_assignments.keys()))
    )


def ensure_role_users(session: Session, group: Group) -> None:
    """Ensure default role users exist with the expected configuration."""
    expected_emails = {
        "retailer": f"retailer+g{group.id}@daybreak.ai",
        "wholesaler": f"wholesaler+g{group.id}@daybreak.ai",
        "distributor": f"distributor+g{group.id}@daybreak.ai",
        "factory": f"factory+g{group.id}@daybreak.ai",
    }

    for role, email in expected_emails.items():
        user = (
            session.query(User)
            .filter(User.email == email)
            .first()
        )

        username = f"{role}_{group.id}"
        full_name = f"{role.capitalize()} User"
        if user is None:
            user = User(
                username=username,
                email=email,
                full_name=full_name,
                hashed_password=get_password_hash(DEFAULT_PASSWORD),
                is_active=True,
                is_superuser=False,
                user_type=UserTypeEnum.PLAYER,
                group_id=group.id,
            )
            session.add(user)
            session.flush()
            print(f"[info] Created {role} user: {email}")
            continue

        updated = False
        if user.username != username:
            user.username = username
            updated = True
        if user.full_name != full_name:
            user.full_name = full_name
            updated = True
        if user.group_id != group.id:
            user.group_id = group.id
            updated = True
        if user.user_type != UserTypeEnum.PLAYER:
            user.user_type = UserTypeEnum.PLAYER
            updated = True
        if not user.is_active:
            user.is_active = True
            updated = True
        if user.is_superuser:
            user.is_superuser = False
            updated = True

        if updated:
            session.add(user)
            session.flush()
            print(f"[info] Updated {role} user: {email}")
        else:
            print(f"[info] {role} user already exists: {email}")


def seed_default_data(session: Session) -> None:
    """Run the seeding workflow using the provided session."""
    group, _ = ensure_group(session)
    ensure_role_users(session, group)
    game = ensure_default_game(session, group)
    ensure_naive_agents(session, game)


def run_seed_with_session(session_factory: Callable[[], Session]) -> None:
    """Execute the seeding process using the supplied session factory."""
    session: Session | None = None
    try:
        session = session_factory()
        seed_default_data(session)
        session.commit()
    except Exception:
        if session is not None:
            session.rollback()
        raise
    finally:
        if session is not None:
            session.close()


def main() -> None:
    from app.db.session import SQLALCHEMY_DATABASE_URI
    print(f"[DEBUG] Database URL from settings: {settings.SQLALCHEMY_DATABASE_URI}")
    print(f"[DEBUG] Database URL from session: {SQLALCHEMY_DATABASE_URI}")
    
    configured_uri = settings.SQLALCHEMY_DATABASE_URI
    configured_label = mask_db_uri(configured_uri)
    print(f"[info] Attempting to seed default data using: {configured_label or 'default settings'}")

    try:
        run_seed_with_session(SessionLocal)
        print("[done] Default group, users, and game are ready.")
        if configured_label:
            print(f"[info] Data stored in: {configured_label}")
        return
    except Exception as exc:
        print("\n[error] Failed to seed the database. Please check the following:")
        print("1. Make sure the MariaDB container is running")
        print("2. Verify the database credentials in your .env file")
        print("3. Check that the database 'beer_game' exists and is accessible")
        print(f"\nError details: {exc}")
        raise


if __name__ == "__main__":
    main()
