#!/usr/bin/env python3
"""Seed the default Daybreak group, configuration, and game with naive AI players."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import subprocess
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

# Ensure the backend package is importable when running via `python backend/scripts/...`
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

SCRIPTS_ROOT = Path(__file__).resolve().parent
TRAINING_SCRIPTS_DIR = SCRIPTS_ROOT / "training"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", value.strip().lower()).strip("-")
    return slug or "config"

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
    PlayerAction,
    PlayerRole,
    PlayerStrategy,
    PlayerType,
    Round,
    SupplyChainConfig,
    SupplyChainTrainingArtifact,
    SupervisorAction,
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


def _normalize_user_type(value: Any) -> UserTypeEnum:
    if isinstance(value, UserTypeEnum):
        return value
    if isinstance(value, str):
        try:
            return UserTypeEnum(value)
        except ValueError:
            try:
                return UserTypeEnum[value]
            except KeyError:
                return UserTypeEnum.PLAYER
    return UserTypeEnum.PLAYER


def _role_key(role: PlayerRole) -> str:
    return role.name.lower()


DAYBREAK_AGENT_SPECS = [
    {
        "name": "Daybreak DTCE Showcase",
        "agent_type": "daybreak_dtce",
        "description": "Decentralised Daybreak temporal GNN agents running per role.",
        "override_pct": None,
    },
    {
        "name": "Daybreak DTCE Central Showcase",
        "agent_type": "daybreak_dtce_central",
        "description": "Daybreak agents coordinated with a central override (10% adjustment).",
        "override_pct": 0.1,
    },
    {
        "name": "Daybreak DTCE Global Showcase",
        "agent_type": "daybreak_dtce_global",
        "description": "Single Daybreak global controller orchestrating the full supply chain.",
        "override_pct": None,
    },
    {
        "name": "Daybreak LLM Cooperative Showcase",
        "agent_type": "llm_adaptive",
        "description": "Daybreak LLM-driven agents with full information sharing across the supply chain.",
        "override_pct": None,
        "llm_model": "gpt-5-mini",
        "can_see_demand_all": True,
    },
]

ROLE_SPECS = [
    ("Retailer", PlayerRole.RETAILER),
    ("Wholesaler", PlayerRole.WHOLESALER),
    ("Distributor", PlayerRole.DISTRIBUTOR),
    ("Manufacturer", PlayerRole.MANUFACTURER),
]

ROLE_EMAIL_TEMPLATES = {
    PlayerRole.RETAILER: "retailer+g{group_id}@daybreak.ai",
    PlayerRole.WHOLESALER: "wholesaler+g{group_id}@daybreak.ai",
    PlayerRole.DISTRIBUTOR: "distributor+g{group_id}@daybreak.ai",
    PlayerRole.MANUFACTURER: "manufacturer+g{group_id}@daybreak.ai",
}


@dataclass
class SeedOptions:
    reset_games: bool = False
    force_dataset: bool = False
    force_training: bool = False
    create_daybreak_games: bool = True
    use_naive_agents: bool = True

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
            if _normalize_user_type(admin.user_type) != UserTypeEnum.GROUP_ADMIN:
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
        if _normalize_user_type(admin_user.user_type) != UserTypeEnum.GROUP_ADMIN:
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
        _apply_default_supply_chain_settings(session, config)
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
        ("Wholesaler", NodeType.WHOLESALER),
        ("Distributor", NodeType.DISTRIBUTOR),
        ("Manufacturer", NodeType.MANUFACTURER),
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
        (NodeType.MANUFACTURER, NodeType.DISTRIBUTOR),
        (NodeType.DISTRIBUTOR, NodeType.WHOLESALER),
        (NodeType.WHOLESALER, NodeType.RETAILER),
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
    _apply_default_supply_chain_settings(session, config)
    return config


def _apply_default_supply_chain_settings(session: Session, config: SupplyChainConfig) -> None:
    """Normalize supply chain configuration settings to project defaults."""

    config.is_active = True
    config.description = config.description or "Default supply chain configuration"
    session.add(config)

    nodes = session.query(Node).filter(Node.config_id == config.id).all()

    default_inventory_range = {"min": 10, "max": 20}
    initial_inventory_range = {"min": 5, "max": 5}
    holding_cost_range = {"min": 0.5, "max": 0.5}
    backlog_cost_range = {"min": 5.0, "max": 5.0}
    selling_price_range = {"min": 25.0, "max": 25.0}

    for node in nodes:
        for node_config in node.item_configs:
            node_config.inventory_target_range = dict(default_inventory_range)
            node_config.initial_inventory_range = dict(initial_inventory_range)
            node_config.holding_cost_range = dict(holding_cost_range)
            node_config.backlog_cost_range = dict(backlog_cost_range)
            node_config.selling_price_range = dict(selling_price_range)
            session.add(node_config)

    lanes = session.query(Lane).filter(Lane.config_id == config.id).all()
    for lane in lanes:
        lane.capacity = lane.capacity or 9999
        lane.lead_time_days = {"min": 1, "max": 1}
        session.add(lane)

    market_demands = session.query(MarketDemand).filter(MarketDemand.config_id == config.id).all()
    for md in market_demands:
        md.demand_pattern = {
            "type": "classic",
            "params": {
                "initial_demand": 4,
                "change_week": 15,
                "final_demand": 12,
            },
        }
        session.add(md)

    session.flush()

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
        existing_config = game.config or {}
        if isinstance(existing_config, str):
            try:
                existing_config = json.loads(existing_config)
            except json.JSONDecodeError:
                existing_config = {}
        existing_config.setdefault("progression_mode", "supervised")
        game.config = json.loads(json.dumps(existing_config))
        session.add(game)
        return game

    print("[info] Creating default game from supply chain configuration...")
    sc_config = ensure_supply_chain_config(session, group)

    config_service = SupplyChainConfigService(session)
    game_config = config_service.create_game_from_config(
        sc_config.id, {"name": DEFAULT_GAME_NAME, "max_rounds": 40}
    )
    game_config["progression_mode"] = "supervised"

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


def configure_human_players_for_game(
    session: Session,
    group: Group,
    game: Game,
) -> None:
    """Ensure the default game uses human players mapped to role-specific accounts."""

    players_by_role: Dict[PlayerRole, Player] = {
        player.role: player
        for player in session.query(Player).filter(Player.game_id == game.id).all()
    }

    role_assignments: Dict[str, Dict[str, int | bool | None]] = {}

    for display_name, role in ROLE_SPECS:
        email_template = ROLE_EMAIL_TEMPLATES.get(role)
        if not email_template:
            continue

        email = email_template.format(group_id=group.id)
        user = session.query(User).filter(User.email == email).first()
        if not user:
            print(f"[warn] Skipping role '{role.name.title()}' — user {email} not found.")
            continue

        player = players_by_role.get(role)
        if player is None:
            player = Player(
                game_id=game.id,
                role=role,
                name=display_name,
                type=PlayerType.HUMAN,
                strategy=PlayerStrategy.MANUAL,
                is_ai=False,
                ai_strategy=None,
                can_see_demand=(role == PlayerRole.RETAILER),
                user_id=user.id,
            )
            session.add(player)
            session.flush()
            players_by_role[role] = player
        else:
            player.name = display_name
            player.type = PlayerType.HUMAN
            player.strategy = PlayerStrategy.MANUAL
            player.is_ai = False
            player.ai_strategy = None
            player.can_see_demand = role == PlayerRole.RETAILER
            player.user_id = user.id
            player.llm_model = None
        session.add(player)

        role_assignments[_role_key(role)] = {
            "is_ai": False,
            "agent_config_id": None,
            "user_id": user.id,
        }

    if "supplier" in role_assignments:
        role_assignments.setdefault("wholesaler", dict(role_assignments["supplier"]))
        role_assignments.pop("supplier", None)

    # Remove any lingering agent configs from previous runs
    session.query(AgentConfig).filter(AgentConfig.game_id == game.id).delete(
        synchronize_session=False
    )

    try:
        config_payload = game.config or {}
        if isinstance(config_payload, str):
            config_payload = json.loads(config_payload)
    except json.JSONDecodeError:
        config_payload = {}

    config_payload["progression_mode"] = "supervised"
    game.config = json.loads(json.dumps(config_payload))
    game.role_assignments = role_assignments
    session.add(game)
    session.flush()

    print(
        "[success] Configured human players for default game roles: "
        + ", ".join(sorted(role_assignments.keys()))
    )


def _ensure_default_players(session: Session, game: Game) -> None:
    """Create placeholder AI players if none exist for the game."""
    players = session.query(Player).filter(Player.game_id == game.id).all()
    if players:
        return

    print("[info] Creating default players for the game...")
    for display_name, role in ROLE_SPECS:
        player = Player(
            game_id=game.id,
            name=f"{display_name} (AI)",
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
    """Assign naive AI agents to each role in the game and switch to auto progression."""
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

        role_assignments[_role_key(player.role)] = {
            "is_ai": True,
            "agent_config_id": agent_config.id,
            "user_id": None,
        }

    if "supplier" in role_assignments:
        role_assignments.setdefault("wholesaler", dict(role_assignments["supplier"]))
        role_assignments.pop("supplier", None)

    try:
        config_payload = game.config or {}
        if isinstance(config_payload, str):
            config_payload = json.loads(config_payload)
    except json.JSONDecodeError:
        config_payload = {}

    config_payload["progression_mode"] = "unsupervised"
    game.config = json.loads(json.dumps(config_payload))
    game.role_assignments = role_assignments
    session.add(game)
    session.flush()
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
        "manufacturer": f"manufacturer+g{group.id}@daybreak.ai",
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
        if _normalize_user_type(user.user_type) != UserTypeEnum.PLAYER:
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


def _resolve_training_device(preferred: str = "cuda") -> str:
    """Return an available training device, falling back to CPU when needed."""

    try:
        import torch  # type: ignore
    except Exception:
        print("[warn] PyTorch not available; defaulting training to CPU.")
        return "cpu"

    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "cuda":
        print("[warn] CUDA unavailable; training will run on CPU.")
    return "cuda" if torch.cuda.is_available() else "cpu"


def _run_post_seed_tasks(
    session: Session,
    config: SupplyChainConfig,
    *,
    force_dataset: bool = False,
    force_training: bool = False,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Run data generation and training workflows for the supplied configuration ID."""

    if not TRAINING_SCRIPTS_DIR.exists():
        print("[warn] Training scripts directory not found; skipping automatic training.")
        return {"dataset": None, "model": None, "device": None}

    config_id = config.id
    slug = _slugify(config.name)

    artifacts: Dict[str, Optional[Dict[str, Any]]] = {
        "dataset": None,
        "model": None,
        "device": None,
    }

    dataset_cmd = [
        sys.executable,
        str(TRAINING_SCRIPTS_DIR / "generate_simpy_dataset.py"),
        "--config-id",
        str(config_id),
    ]
    if force_dataset:
        dataset_cmd.append("--force")

    print(f"[info] Generate SimPy dataset (config_id={config_id})...")
    dataset_completed = subprocess.run(
        dataset_cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    dataset_stdout = dataset_completed.stdout.strip()
    dataset_payload: Optional[Dict[str, Any]] = None
    if dataset_stdout:
        try:
            dataset_payload = json.loads(dataset_stdout.splitlines()[-1])
        except json.JSONDecodeError:
            print(dataset_stdout)
    if dataset_completed.stderr:
        print(dataset_completed.stderr.strip(), file=sys.stderr)

    if dataset_payload:
        status = dataset_payload.get("status", "unknown")
        print(f"[success] Generate SimPy dataset: {status}")
        if dataset_payload.get("path"):
            original_path = Path(dataset_payload["path"])
            target_path = original_path.with_name(f"{slug}_dataset.npz")
            if original_path != target_path:
                if target_path.exists():
                    target_path.unlink()
                original_path.rename(target_path)
                dataset_payload["path"] = str(target_path)
            dataset_payload["filename"] = Path(dataset_payload["path"]).name
            print(f"          Dataset: {dataset_payload['path']}")
        artifacts["dataset"] = dataset_payload

    device = _resolve_training_device()
    artifacts["device"] = {"preferred": "cuda", "resolved": device}

    training_cmd = [
        sys.executable,
        str(TRAINING_SCRIPTS_DIR / "train_gpu_default.py"),
        "--config-id",
        str(config_id),
        "--device",
        device,
    ]
    dataset_path = (dataset_payload or {}).get("path")
    if dataset_path:
        training_cmd.extend(["--dataset", dataset_path])
    if force_training:
        training_cmd.append("--force")

    print(f"[info] Train temporal GNN ({device}) (config_id={config_id})...")
    training_completed = subprocess.run(
        training_cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    training_stdout = training_completed.stdout.strip()
    training_payload: Optional[Dict[str, Any]] = None
    if training_stdout:
        try:
            training_payload = json.loads(training_stdout.splitlines()[-1])
        except json.JSONDecodeError:
            print(training_stdout)
    if training_completed.stderr:
        print(training_completed.stderr.strip(), file=sys.stderr)

    if training_payload:
        status = training_payload.get("status", "unknown")
        print(f"[success] Train temporal GNN: {status}")
        if training_payload.get("model_path"):
            model_path = Path(training_payload["model_path"])
            target_model_path = model_path.with_name(f"{slug}_temporal_gnn.pt")
            if model_path != target_model_path:
                if target_model_path.exists():
                    target_model_path.unlink()
                model_path.rename(target_model_path)
                training_payload["model_path"] = str(target_model_path)
            print(f"          Model: {training_payload['model_path']}")
        if dataset_payload and training_payload:
            artifact = SupplyChainTrainingArtifact(
                config_id=config_id,
                dataset_name=Path(dataset_payload["path"]).name,
                model_name=Path(training_payload["model_path"]).name,
            )
            session.add(artifact)
            session.commit()
        artifacts["model"] = training_payload

    return artifacts


def _purge_existing_games(session: Session) -> None:
    """Remove all game records and related AI configurations."""

    print("[info] Removing existing games, players, and agent configurations...")
    session.query(PlayerAction).delete(synchronize_session=False)
    session.query(Round).delete(synchronize_session=False)
    session.query(SupervisorAction).delete(synchronize_session=False)
    session.query(AgentConfig).delete(synchronize_session=False)
    session.query(Player).delete(synchronize_session=False)
    session.query(Game).delete(synchronize_session=False)
    session.flush()


def _configure_game_agents(
    session: Session,
    game: Game,
    agent_type: str,
    *,
    override_pct: Optional[float] = None,
    llm_model: Optional[str] = None,
    can_see_demand_all: bool = False,
) -> None:
    """Ensure players and agent configs exist for a game using the specified agent type."""

    existing_players = {
        player.role: player
        for player in session.query(Player).filter(Player.game_id == game.id)
    }

    assignments: Dict[str, Dict[str, Optional[int]]] = {}

    for display_name, role in ROLE_SPECS:
        player = existing_players.get(role)
        if player is None:
            player = Player(
                game_id=game.id,
                name=f"{display_name} ({agent_type})",
                role=role,
            )

        player.type = PlayerType.AI
        player.is_ai = True
        player.ai_strategy = agent_type
        player.strategy = PlayerStrategy.MANUAL
        player.user_id = None
        player.can_see_demand = True if can_see_demand_all else (role == PlayerRole.RETAILER)
        if llm_model:
            player.llm_model = llm_model
        session.add(player)
        session.flush()

        agent_config = (
            session.query(AgentConfig)
            .filter(AgentConfig.game_id == game.id, AgentConfig.role == role.value)
            .first()
        )
        if agent_config is None:
            agent_config = AgentConfig(
                game_id=game.id,
                role=role.value,
                agent_type=agent_type,
                config={"llm_model": llm_model} if llm_model else {},
            )
        else:
            agent_config.agent_type = agent_type
            agent_config.config = agent_config.config or {}
            if llm_model:
                agent_config.config["llm_model"] = llm_model

        session.add(agent_config)
        session.flush()

        assignments[_role_key(role)] = {
            "is_ai": True,
            "agent_config_id": agent_config.id,
            "user_id": None,
            "strategy": agent_type,
        }

    if override_pct is not None:
        overrides = {_role_key(role): override_pct for _, role in ROLE_SPECS}
        game_config = game.config or {}
        game_config.setdefault("daybreak_overrides", {}).update(overrides)
        game.config = json.loads(json.dumps(game_config))

    game.role_assignments = assignments
    session.add(game)


def ensure_daybreak_games(
    session: Session,
    group: Group,
    config: SupplyChainConfig,
    artifacts: Dict[str, Optional[Dict[str, Any]]],
    *,
    recreate: bool,
) -> None:
    """Create or update showcase games for the Daybreak agent variants."""

    config_service = SupplyChainConfigService(session)
    dataset_info = artifacts.get("dataset") or {}
    model_info = artifacts.get("model") or {}

    for spec in DAYBREAK_AGENT_SPECS:
        game = (
            session.query(Game)
            .filter(Game.group_id == group.id, Game.name == spec["name"])
            .first()
        )

        if game and recreate:
            print(f"[info] Recreating showcase game '{spec['name']}' (id={game.id}).")
            session.delete(game)
            session.flush()
            game = None

        if game is None:
            print(f"[info] Creating showcase game '{spec['name']}'...")
            base_config = config_service.create_game_from_config(
                config.id,
                {
                    "name": spec["name"],
                    "max_rounds": 40,
                    "is_public": True,
                    "description": spec["description"],
                },
            )
            base_config["progression_mode"] = "unsupervised"
            base_config.setdefault("daybreak", {})
            base_config["daybreak"].update(
                {
                    "strategy": spec["agent_type"],
                    "dataset": dataset_info.get("path"),
                    "model_path": model_info.get("model_path"),
                }
            )
            if spec.get("llm_model"):
                base_config["daybreak"]["llm_model"] = spec["llm_model"]
                base_config.setdefault("info_sharing", {}).update({"visibility": "full"})
            if spec["override_pct"] is not None:
                overrides = {_role_key(role): spec["override_pct"] for _, role in ROLE_SPECS}
                base_config.setdefault("daybreak_overrides", {}).update(overrides)

            game = Game(
                name=spec["name"],
                created_by=config.created_by or group.admin_id,
                group_id=group.id,
                status=GameStatus.CREATED,
                max_rounds=base_config.get("max_rounds", 40),
                config=base_config,
                demand_pattern=base_config.get("demand_pattern", {}),
                description=spec["description"],
            )
            session.add(game)
            session.flush()
        else:
            print(
                f"[info] Updating showcase game '{spec['name']}' (id={game.id}) with latest training artifacts."
            )
            game_config = game.config or {}
            if isinstance(game_config, str):
                try:
                    game_config = json.loads(game_config)
                except json.JSONDecodeError:
                    game_config = {}
            game_config.setdefault("daybreak", {})
            game_config["daybreak"].update(
                {
                    "strategy": spec["agent_type"],
                    "dataset": dataset_info.get("path"),
                    "model_path": model_info.get("model_path"),
                }
            )
            if spec.get("llm_model"):
                game_config["daybreak"]["llm_model"] = spec["llm_model"]
                game_config.setdefault("info_sharing", {}).update({"visibility": "full"})
            if spec["override_pct"] is not None:
                overrides = {_role_key(role): spec["override_pct"] for _, role in ROLE_SPECS}
                game_config.setdefault("daybreak_overrides", {}).update(overrides)
            game_config["progression_mode"] = "unsupervised"
            game_config["max_rounds"] = 40
            game.config = json.loads(json.dumps(game_config))
            session.add(game)

        _configure_game_agents(
            session,
            game,
            spec["agent_type"],
            override_pct=spec.get("override_pct"),
            llm_model=spec.get("llm_model"),
            can_see_demand_all=spec.get("can_see_demand_all", False),
        )



def seed_default_data(session: Session, options: Optional[SeedOptions] = None) -> None:
    """Run the seeding workflow using the provided session."""

    if options is None:
        options = SeedOptions()

    if options.reset_games:
        _purge_existing_games(session)

    group, _ = ensure_group(session)
    ensure_role_users(session, group)
    game = ensure_default_game(session, group)
    if options.use_naive_agents:
        ensure_naive_agents(session, game)
    else:
        configure_human_players_for_game(session, group, game)

    # Persist the base bootstrap objects so downstream helper scripts see them
    session.flush()
    session.commit()

    config = (
        session.query(SupplyChainConfig)
        .filter(SupplyChainConfig.group_id == group.id)
        .order_by(SupplyChainConfig.id.asc())
        .first()
    )

    artifacts: Dict[str, Optional[Dict[str, Any]]] = {
        "dataset": None,
        "model": None,
        "device": None,
    }
    if config:
        artifacts = _run_post_seed_tasks(
            session,
            config,
            force_dataset=options.force_dataset,
            force_training=options.force_training,
        )

        model_artifact = artifacts.get("model") or {}
        if config:
            config.needs_training = False
            config.training_status = model_artifact.get("status", "trained")
            config.trained_model_path = model_artifact.get("model_path")
            config.trained_at = datetime.utcnow()
            session.add(config)

        if options.create_daybreak_games:
            ensure_daybreak_games(
                session,
                group,
                config,
                artifacts,
                recreate=options.reset_games,
            )


def run_seed_with_session(session_factory: Callable[[], Session], options: SeedOptions) -> None:
    """Execute the seeding process using the supplied session factory."""
    session: Session | None = None
    try:
        session = session_factory()
        seed_default_data(session, options)
        session.commit()
    except Exception:
        if session is not None:
            session.rollback()
        raise
    finally:
        if session is not None:
            session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Daybreak default data and agent games")
    parser.add_argument(
        "--reset-games",
        action="store_true",
        help="Delete all existing games before recreating defaults and showcases.",
    )
    parser.add_argument(
        "--skip-daybreak-games",
        action="store_true",
        help="Skip creation/update of Daybreak showcase games.",
    )
    parser.add_argument(
        "--force-training",
        action="store_true",
        help="Retrain the temporal GNN even if a checkpoint already exists.",
    )
    parser.add_argument(
        "--force-dataset",
        action="store_true",
        help="Regenerate the SimPy dataset even if cached output exists.",
    )
    parser.add_argument(
        "--use-human-players",
        action="store_true",
        help="Assign the default Beer Game to human accounts instead of naive AI agents.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = SeedOptions(
        reset_games=args.reset_games,
        force_dataset=args.force_dataset or args.reset_games,
        force_training=args.force_training or args.reset_games,
        create_daybreak_games=not args.skip_daybreak_games,
        use_naive_agents=not args.use_human_players,
    )
    print(f"[DEBUG] Database URL from settings: {settings.SQLALCHEMY_DATABASE_URI}")

    configured_uri = settings.SQLALCHEMY_DATABASE_URI
    configured_label = mask_db_uri(configured_uri)
    print(f"[info] Attempting to seed default data using: {configured_label or 'default settings'}")

    try:
        run_seed_with_session(SessionLocal, options)
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
