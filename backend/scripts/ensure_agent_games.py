"""Ensure showcase games exist for specific agent strategies."""
from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import sessionmaker

from app.db.session import sync_engine
from app.models.game import Game, GameStatus
from app.services.supply_chain_config_service import SupplyChainConfigService
from scripts.seed_default_group import (
    ensure_group,
    ensure_supply_chain_config,
    ensure_ai_agents,
    DEFAULT_LLM_MODEL,
)

SHOWCASE_GAMES = [
    {
        "name": "The Beer Game - Naiive",
        "description": "Baseline Beer Game using Naiive agent strategy for all roles.",
        "strategy": "naive",
        "llm_model": None,
    },
    {
        "name": "The Beer Game - LLM",
        "description": "Beer Game where all roles are controlled by the Daybreak LLM agent.",
        "strategy": "llm",
        "llm_model": DEFAULT_LLM_MODEL,
    },
]

def ensure_agent_game(session, service: SupplyChainConfigService, group, config, *, name: str, description: str, strategy: str, llm_model: Optional[str]) -> None:
    game = (
        session.query(Game)
        .filter(Game.group_id == group.id, Game.name == name)
        .first()
    )
    if game is None:
        base_config = service.create_game_from_config(
            config.id,
            {
                "name": name,
                "description": description,
                "max_rounds": 40,
                "is_public": True,
            },
        )
        game = Game(
            name=name,
            description=description,
            created_by=group.admin_id,
            group_id=group.id,
            status=GameStatus.CREATED,
            max_rounds=base_config.get("max_rounds", 40),
            config=base_config,
            demand_pattern=base_config.get("demand_pattern", {}),
        )
        session.add(game)
        session.flush()

    ensure_ai_agents(session, game, strategy, llm_model)
    session.commit()

def main() -> None:
    Session = sessionmaker(bind=sync_engine, autoflush=False, autocommit=False)
    session = Session()
    try:
        group, _ = ensure_group(session)
        config = ensure_supply_chain_config(session, group)
        service = SupplyChainConfigService(session)

        for spec in SHOWCASE_GAMES:
            ensure_agent_game(
                session,
                service,
                group,
                config,
                name=spec["name"],
                description=spec["description"],
                strategy=spec["strategy"],
                llm_model=spec["llm_model"],
            )
    finally:
        session.close()


if __name__ == "__main__":
    main()
