"""Shared session lifecycle utilities for the Daybreak Beer Game Strategist."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

from .daybreak_instructions import DAYBREAK_STRATEGIST_INSTRUCTIONS


_CLIENT: Optional[OpenAI] = None
_CLIENT_LOCK = Lock()

_SESSION_CACHE: Dict[Tuple[str, str, Tuple[str, ...]], str] = {}
_SESSION_LOCK = Lock()

_RESOURCE_UPLOAD_CACHE: Dict[Tuple[str, str], str] = {}


def _iter_resource_files(
    package: str = "backend.llm_agent.resources",
) -> Iterable[Tuple[str, Path]]:
    """Yield (relative_path, absolute_path) pairs for packaged assets."""

    try:
        package_root = resources.files(package)
    except (ModuleNotFoundError, FileNotFoundError):
        return []

    file_entries: List[Tuple[str, Path]] = []
    with resources.as_file(package_root) as resolved_root:
        root_path = Path(resolved_root)
        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.name == "__init__.py":
                continue
            relative = file_path.relative_to(root_path).as_posix()
            file_entries.append((relative, file_path))
    return file_entries


def _ensure_resource_attachments(client: OpenAI) -> List[Dict[str, str]]:
    """Upload bundled resources once and return attachment descriptors."""

    attachments: List[Dict[str, str]] = []
    for relative_path, absolute_path in _iter_resource_files():
        data = absolute_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        cache_key = (relative_path, digest)
        file_id = _RESOURCE_UPLOAD_CACHE.get(cache_key)
        if not file_id:
            buffer = io.BytesIO(data)
            buffer.name = relative_path  # type: ignore[attr-defined]
            uploaded = client.files.create(file=buffer, purpose="assistants")
            file_id = uploaded.id
            _RESOURCE_UPLOAD_CACHE[cache_key] = file_id
        attachments.append({"file_id": file_id})
    return attachments


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


def _resolve_session_id(
    model: str,
    instructions: str,
    attachments: Optional[List[Dict[str, str]]] = None,
) -> str:
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

    attachment_ids = tuple(sorted(att["file_id"] for att in attachments or []))
    cache_key = (model, instructions, attachment_ids)
    if cache_key in _SESSION_CACHE:
        return _SESSION_CACHE[cache_key]

    with _SESSION_LOCK:
        if cache_key in _SESSION_CACHE:
            return _SESSION_CACHE[cache_key]

        client = _get_client()
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
        }
        if attachments:
            create_kwargs["attachments"] = attachments

        session = client.responses.sessions.create(**create_kwargs)
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
    _resource_attachments: Optional[List[Dict[str, str]]] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = _get_client()
        return self._client

    def start(self) -> str:
        """Initialise a Responses session for a new Beer Game run."""

        if not self.session_id:
            if self._resource_attachments is None:
                self._resource_attachments = _ensure_resource_attachments(self.client)

            attachments = self._resource_attachments or []
            session_id = _resolve_session_id(
                self.model,
                self.instructions,
                attachments if attachments else None,
            )
            self.session_id = session_id
        return self.session_id

    def reset(self) -> None:
        """Discard the current conversation (start a fresh game)."""

        if self.session_id:
            attachment_ids = tuple(
                sorted(att["file_id"] for att in (self._resource_attachments or []))
            )
            cache_key = (self.model, self.instructions, attachment_ids)
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

