"""Shared assistant lifecycle utilities for the Daybreak Beer Game Strategist."""

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

_ASSISTANT_CACHE: Dict[Tuple[str, str], str] = {}
_ASSISTANT_LOCK = Lock()


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


def _resolve_assistant_id(model: str, instructions: str) -> str:
    """Create (or reuse) the assistant backing the Daybreak strategist."""

    # Allow explicit override if the deployment already provisioned an assistant
    # in the OpenAI dashboard (or exported one from the bootstrap helper).
    explicit = (
        os.getenv("DAYBREAK_ASSISTANT_ID")
        or os.getenv("DAYBREAK_STRATEGIST_ASSISTANT_ID")
        or os.getenv("GPT_ID")
    )
    if explicit:
        return explicit

    # Custom GPT share links expose the assistant identifier directly. Users often
    # set that ID as the "model" which should short-circuit the assistant creation
    # path below – the platform will reject attempts to create a new assistant with
    # a `g-` identifier. Accept both `g-` (custom GPT) and `asst_` prefixes.
    if model.startswith("g-") or model.startswith("asst_"):
        return model

    cache_key = (model, instructions)
    if cache_key in _ASSISTANT_CACHE:
        return _ASSISTANT_CACHE[cache_key]

    with _ASSISTANT_LOCK:
        if cache_key in _ASSISTANT_CACHE:
            return _ASSISTANT_CACHE[cache_key]

        client = _get_client()
        assistant = client.beta.assistants.create(
            name="Daybreak Beer Game Strategist",
            model=model,
            instructions=instructions,
        )
        _ASSISTANT_CACHE[cache_key] = assistant.id
        return assistant.id


def _extract_json_block(text: str) -> Dict[str, Any]:
    """Parse the JSON payload returned by the assistant."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError(f"Assistant did not return JSON: {text}")
        return json.loads(match.group(0))


@dataclass
class DaybreakStrategistSession:
    """Stateful helper that owns the assistant + thread lifecycle."""

    model: str
    instructions: str = DAYBREAK_STRATEGIST_INSTRUCTIONS
    thread_id: Optional[str] = None
    _assistant_id: Optional[str] = None
    _client: Optional[OpenAI] = None
    _last_decision: Optional[Dict[str, Any]] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = _get_client()
        return self._client

    @property
    def assistant_id(self) -> str:
        if self._assistant_id is None:
            self._assistant_id = _resolve_assistant_id(self.model, self.instructions)
        return self._assistant_id

    def start(self) -> str:
        """Initialise a thread for a new Beer Game run."""

        thread = self.client.beta.threads.create()
        self.thread_id = thread.id
        return self.thread_id

    def reset(self) -> None:
        """Discard the current conversation thread (start a fresh game)."""

        self.thread_id = None
        self._last_decision = None

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Send one turn's state and capture the assistant's JSON decision."""

        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary matching the strategist schema")

        if not self.thread_id:
            self.start()

        payload = json.dumps(state, separators=(",", ":"))
        prompt = (
            "Here is the current state as JSON. Respond ONLY with the required JSON object.\n\n"
            f"```json\n{payload}\n```"
        )

        self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=prompt,
        )

        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
        )

        if run.status != "completed":
            raise RuntimeError(f"Daybreak LLM run failed with status '{run.status}'")

        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread_id,
            limit=1,
        )

        if not messages.data:
            raise RuntimeError("Daybreak LLM did not return any messages")

        content = messages.data[0].content
        if not content:
            raise RuntimeError("Daybreak LLM response was empty")

        text_chunks = [block.text.value for block in content if getattr(block, "type", None) == "text"]
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

