"""Shared session lifecycle utilities for the Daybreak Beer Game Strategist."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from .daybreak_instructions import DAYBREAK_STRATEGIST_INSTRUCTIONS


_CLIENT: Optional[OpenAI] = None
_CLIENT_LOCK = Lock()

_SESSION_CACHE: Dict[Tuple[str, str], str] = {}
_SESSION_LOCK = Lock()


def _get_client() -> OpenAI:
    """Return a singleton OpenAI client configured via environment variables."""

    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    with _CLIENT_LOCK:
        if _CLIENT is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            _CLIENT = OpenAI(api_key=api_key)
        return _CLIENT


def _resolve_session_id(model: str, instructions: str) -> str:
    """Create (or reuse) the session backing the Daybreak strategist."""

    explicit = (
        os.getenv("DAYBREAK_RESPONSES_SESSION_ID")
        or os.getenv("DAYBREAK_SESSION_ID")
        or os.getenv("DAYBREAK_ASSISTANT_ID")
        or os.getenv("DAYBREAK_STRATEGIST_ASSISTANT_ID")
        or os.getenv("GPT_ID")
    )
    if explicit:
        return explicit

    cache_key = (model, instructions)
    if cache_key in _SESSION_CACHE:
        return _SESSION_CACHE[cache_key]

    with _SESSION_LOCK:
        if cache_key in _SESSION_CACHE:
            return _SESSION_CACHE[cache_key]

        client = _get_client()
        session = client.responses.sessions.create(
            model=model,
            instructions=instructions,
        )
        _SESSION_CACHE[cache_key] = session.id
        return session.id


def _extract_json_block(text: str) -> Dict[str, Any]:
    """Parse the JSON payload returned by the model."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError(f"Assistant did not return JSON: {text}")
        return json.loads(match.group(0))


@dataclass
class DaybreakStrategistSession:
    """Stateful helper that owns the Responses session lifecycle."""

    model: str
    instructions: str = DAYBREAK_STRATEGIST_INSTRUCTIONS
    session_id: Optional[str] = None
    _client: Optional[OpenAI] = None
    _last_decision: Optional[Dict[str, Any]] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = _get_client()
        return self._client

    def start(self) -> str:
        """Initialise a Responses session for a new Beer Game run."""

        if not self.session_id:
            session_id = _resolve_session_id(self.model, self.instructions)
            self.session_id = session_id
        return self.session_id

    def reset(self) -> None:
        """Discard the current conversation (start a fresh game)."""

        if self.session_id:
            cache_key = (self.model, self.instructions)
            with _SESSION_LOCK:
                if _SESSION_CACHE.get(cache_key) == self.session_id:
                    _SESSION_CACHE.pop(cache_key, None)
        self.session_id = None
        self._last_decision = None

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Send one turn's state and capture the model's JSON decision."""

        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary matching the strategist schema")

        if not self.session_id:
            self.start()

        payload = json.dumps(state, separators=(",", ":"))
        prompt = (
            "Here is the current state as JSON. Respond ONLY with the required JSON object.\n\n"
            f"```json\n{payload}\n```"
        )

        response = self.client.responses.create(
            session=self.session_id,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        )

        text_chunks = []
        output_text = getattr(response, "output_text", None)
        if output_text:
            text_chunks.append(output_text)
        else:
            for item in getattr(response, "output", []) or []:
                contents = getattr(item, "content", []) or []
                for content in contents:
                    if getattr(content, "type", None) not in {"output_text", "text"}:
                        continue
                    text_obj = getattr(content, "text", None)
                    if text_obj is None:
                        continue
                    value = getattr(text_obj, "value", None)
                    if value is None and isinstance(text_obj, str):
                        value = text_obj
                    elif value is None:
                        value = getattr(text_obj, "text", None)
                    if value:
                        text_chunks.append(value)

        if not text_chunks:
            raise RuntimeError("Daybreak LLM response did not include text output")

        decision = _extract_json_block("\n".join(text_chunks))

        for key in ("order_upstream", "ship_to_downstream", "rationale"):
            if key not in decision:
                raise ValueError(f"Assistant response missing '{key}': {decision}")

        self._last_decision = decision
        return decision

    @property
    def last_decision(self) -> Optional[Dict[str, Any]]:
        return self._last_decision

