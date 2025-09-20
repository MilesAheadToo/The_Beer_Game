"""Auto-play Daybreak showcase games and capture agent explanations."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any

from main import (
    SessionLocal,
    _coerce_game_config,
    _ensure_round,
    _ensure_simulation_state,
    _pending_orders,
    _finalize_round_if_ready,
    _touch_game,
    _save_game_config,
    _compute_customer_demand,
)
try:
    from scripts.export_round_history import export_game as export_round_history
except ImportError:  # pragma: no cover - fallback when executed from package root
    from export_round_history import export_game as export_round_history
from app.models.game import Game, GameStatus as DbGameStatus, PlayerAction
from app.models.player import Player
from app.services.agents import AgentManager, AgentType, AgentStrategy as AgentStrategyEnum


ROLES = ["retailer", "wholesaler", "distributor", "manufacturer"]


def _role_key(player: Player) -> str:
    return str(player.role.value if hasattr(player.role, "value") else player.role).lower()


def _agent_type_for(role: str) -> AgentType:
    mapping = {
        "factory": AgentType.FACTORY,
        "manufacturer": AgentType.FACTORY,
        "distributor": AgentType.DISTRIBUTOR,
        "wholesaler": AgentType.WHOLESALER,
        "retailer": AgentType.RETAILER,
    }
    if role not in mapping:
        raise ValueError(f"Unsupported role {role}")
    return mapping[role]


def _strategy_for(player: Player) -> AgentStrategyEnum:
    raw = (player.ai_strategy or "daybreak_dtce").lower()
    try:
        return AgentStrategyEnum(raw)
    except ValueError:
        if raw.startswith("llm"):
            return AgentStrategyEnum.LLM
        return AgentStrategyEnum.DAYBREAK_DTCE


def auto_play_daybreak_games() -> None:
    session = SessionLocal()
    try:
        games = (
            session.query(Game)
            .filter(Game.name.like('Daybreak%'))
            .order_by(Game.id)
            .all()
        )

        for game in games:
            print(f"=== Simulating game {game.id}: {game.name} ===")
            config = _coerce_game_config(game)
            config['progression_mode'] = 'unsupervised'
            _ensure_simulation_state(config)
            _pending_orders(config).clear()
            _save_game_config(session, game, config)
            session.flush()

            if game.status == DbGameStatus.CREATED:
                if not game.current_round or game.current_round <= 0:
                    game.current_round = 1
                round_rec = _ensure_round(session, game, game.current_round)
                round_rec.status = 'in_progress'
                round_rec.started_at = datetime.utcnow()
                game.status = DbGameStatus.ROUND_IN_PROGRESS
                _touch_game(game)
                _save_game_config(session, game, config)
                session.add(round_rec)
                session.add(game)
                session.commit()

            agent_manager = AgentManager()
            players = session.query(Player).filter(Player.game_id == game.id).all()
            overrides = config.get('daybreak_overrides') or {}
            info_sharing = config.get('info_sharing') or {}
            full_visibility = str(info_sharing.get('visibility', '')).lower() == 'full'

            for player in players:
                if not player.is_ai:
                    continue
                role_key = _role_key(player)
                agent_type = _agent_type_for(role_key)
                strategy = _strategy_for(player)
                agent_manager.set_agent_strategy(
                    agent_type,
                    strategy,
                    llm_model=player.llm_model,
                    override_pct=overrides.get(role_key),
                )

            while True:
                session.refresh(game)
                if game.status == DbGameStatus.FINISHED:
                    print(f"  Game finished at round {game.current_round}")
                    break

                config = _coerce_game_config(game)
                _ensure_simulation_state(config)
                pending = _pending_orders(config)
                pending.clear()
                round_record = _ensure_round(session, game, game.current_round or 1)
                if round_record.status != 'in_progress':
                    round_record.status = 'in_progress'
                    round_record.started_at = datetime.utcnow()

                demand = _compute_customer_demand(game, round_record.round_number)
                history = config.get('history', [])
                last_orders = history[-1]['orders'] if history else {}
                sim_state = config.get('simulation_state', {})
                now_iso = datetime.utcnow().isoformat() + 'Z'

                for player in players:
                    if not player.is_ai:
                        continue
                    role_key = _role_key(player)
                    agent = agent_manager.get_agent(_agent_type_for(role_key))

                    local_state = {
                        'inventory': sim_state.get('inventory', {}).get(role_key, player.inventory),
                        'backlog': sim_state.get('backlog', {}).get(role_key, player.backlog),
                        'incoming_shipments': sim_state.get('incoming', {}).get(role_key, []),
                    }
                    previous_qty = last_orders.get(role_key, {}).get('quantity', 0)
                    upstream_data = {'previous_orders': [previous_qty]}
                    visible_demand = demand if (player.can_see_demand or role_key == 'retailer' or full_visibility) else None

                    quantity = int(max(0, agent.make_decision(
                        current_round=round_record.round_number,
                        current_demand=visible_demand,
                        upstream_data=upstream_data,
                        local_state=local_state,
                    )))

                    action = session.query(PlayerAction).filter(
                        PlayerAction.game_id == game.id,
                        PlayerAction.round_id == round_record.id,
                        PlayerAction.player_id == player.id,
                        PlayerAction.action_type == 'order',
                    ).first()
                    if action:
                        action.quantity = quantity
                        action.created_at = datetime.utcnow()
                    else:
                        action = PlayerAction(
                            game_id=game.id,
                            round_id=round_record.id,
                            player_id=player.id,
                            action_type='order',
                            quantity=quantity,
                            created_at=datetime.utcnow(),
                        )
                        session.add(action)

                    explanation = agent.get_last_explanation_comment()
                    pending[role_key] = {
                        'player_id': player.id,
                        'quantity': quantity,
                        'comment': explanation or f'Daybreak decision: order {quantity} units.',
                        'submitted_at': now_iso,
                    }

                session.flush()
                _save_game_config(session, game, config)
                progressed = _finalize_round_if_ready(session, game, config, round_record, force=True)
                session.add(game)
                session.flush()
                session.commit()

                if not progressed:
                    print(f"  Warning: round {round_record.round_number} did not finalize; forcing advance")
                    game.current_round = (game.current_round or 0) + 1
                    session.commit()

            export_round_history(game.id, os.environ.get("ROUND_EXPORT_DIR", "/app/exports"))

        session.commit()
    finally:
        session.close()


if __name__ == "__main__":
    auto_play_daybreak_games()
