"""Lightweight helper for interacting with the Daybreak Beer Game Strategist."""

from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Optional

from .daybreak_client import DaybreakStrategistSession


_SESSION: Optional[DaybreakStrategistSession] = None
_SESSION_LOCK = Lock()


def _resolve_model(preferred: Optional[str] = None) -> str:
    from os import getenv

    return (
        preferred
        or getenv("DAYBREAK_LLM_MODEL")
        or getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    )


def _get_session(model: Optional[str] = None) -> DaybreakStrategistSession:
    """Return a singleton session for the supplied model."""

    global _SESSION
    resolved = _resolve_model(model)

    with _SESSION_LOCK:
        if _SESSION is None or _SESSION.model != resolved:
            _SESSION = DaybreakStrategistSession(model=resolved)
        return _SESSION


def call_beer_game_gpt(
    state: Dict[str, Any],
    *,
    model: Optional[str] = None,
    reset_thread: bool = False,
) -> Dict[str, Any]:
    """Submit a single Beer Game state snapshot and return the assistant decision."""

    if not isinstance(state, dict):
        raise TypeError(
            "call_beer_game_gpt expects a state dictionary matching the strategist schema"
        )

    session = _get_session(model)
    if reset_thread:
        session.reset()
    return session.decide(state)


def get_last_decision() -> Optional[Dict[str, Any]]:
    """Expose the most recent assistant response for debugging or telemetry."""

    if _SESSION is None:
        return None
    return _SESSION.last_decision


__all__ = ["call_beer_game_gpt", "get_last_decision", "DaybreakStrategistSession"]

