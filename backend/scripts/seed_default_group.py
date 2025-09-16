#!/usr/bin/env python3
"""Seed the default Daybreak group, configuration, and game with naive AI players."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

# Ensure the backend package is importable when running via `python backend/scripts/...`
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from sqlalchemy.orm import Session

from app.db.base_class import SessionLocal
from app.models import (
    Group,
    Game,
    GameStatus,
    Player,
    PlayerRole,
    PlayerStrategy,
    PlayerType,
    User,
)
from app.models.agent_config import AgentConfig
from app.models.supply_chain_config import SupplyChainConfig
from app.schemas.group import GroupCreate
from app.schemas.user import UserCreate
from app.services.group_service import GroupService
from app.services.supply_chain_config_service import SupplyChainConfigService

DEFAULT_GROUP_NAME = "Default TBG"
DEFAULT_GROUP_DESCRIPTION = "Default Daybreak Beer Game group"
DEFAULT_ADMIN_USERNAME = "groupadmin"
DEFAULT_ADMIN_EMAIL = "groupadmin@daybreak.ai"
DEFAULT_ADMIN_FULL_NAME = "Group Administrator"
DEFAULT_PASSWORD = "Daybreak@2025"
DEFAULT_GAME_NAME = "The Beer Game"
DEFAULT_AGENT_TYPE = "naive"


def ensure_group(session: Session) -> Tuple[Group, bool]:
    """Create the default group and admin if they do not already exist."""
    existing_group = (
        session.query(Group).filter(Group.name == DEFAULT_GROUP_NAME).first()
    )
    if existing_group:
        print(
            f"[info] Group '{DEFAULT_GROUP_NAME}' already exists (id={existing_group.id})."
        )
        return existing_group, False

    print("[info] Creating default group and administrator user...")
    service = GroupService(session)
    group_create = GroupCreate(
        name=DEFAULT_GROUP_NAME,
        description=DEFAULT_GROUP_DESCRIPTION,
        admin=UserCreate(
            username=DEFAULT_ADMIN_USERNAME,
            email=DEFAULT_ADMIN_EMAIL,
            full_name=DEFAULT_ADMIN_FULL_NAME,
            password=DEFAULT_PASSWORD,
        ),
    )
    group = service.create_group(group_create)
    session.refresh(group)
    print(
        f"[success] Created group '{group.name}' (id={group.id}) and admin "
        f"'{DEFAULT_ADMIN_EMAIL}'."
    )
    return group, True


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
    sc_config = (
        session.query(SupplyChainConfig)
        .filter(SupplyChainConfig.group_id == group.id)
        .order_by(SupplyChainConfig.id.asc())
        .first()
    )
    if sc_config is None:
        raise RuntimeError(
            "No supply chain configuration found for the default group."
        )

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
    """Log the status of default role users to confirm their existence."""
    expected_emails = {
        "retailer": f"retailer+g{group.id}@daybreak.ai",
        "distributor": f"distributor+g{group.id}@daybreak.ai",
        "manufacturer": f"manufacturer+g{group.id}@daybreak.ai",
        "supplier": f"supplier+g{group.id}@daybreak.ai",
    }
    for role, email in expected_emails.items():
        exists = (
            session.query(User)
            .filter(User.group_id == group.id, User.email == email)
            .first()
            is not None
        )
        status = "found" if exists else "missing"
        print(f"[info] {role.title()} user {status} ({email}).")


def main() -> None:
    session: Session = SessionLocal()
    try:
        group, _ = ensure_group(session)
        ensure_role_users(session, group)
        game = ensure_default_game(session, group)
        ensure_naive_agents(session, game)
        session.commit()
        print("[done] Default group, users, and game are ready.")
    except Exception as exc:  # pragma: no cover - runtime safeguard
        session.rollback()
        print(f"[error] {exc}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
