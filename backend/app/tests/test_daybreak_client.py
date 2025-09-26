import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class _OpenAIStub:
    def __init__(self, *args, **kwargs):
        raise AssertionError("OpenAI client should be patched in tests")

sys.modules.setdefault("openai", SimpleNamespace(OpenAI=_OpenAIStub))

from llm_agent.daybreak_instructions import DAYBREAK_STRATEGIST_INSTRUCTIONS
from llm_agent import daybreak_client


class DummyResponsesSessions:
    def __init__(self):
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(id=f"session-{len(self.create_calls)}")


class DummyResponses:
    def __init__(self, response_text: str):
        self.sessions = DummyResponsesSessions()
        self.response_text = response_text
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        text_block = SimpleNamespace(
            type="output_text",
            text=SimpleNamespace(value=self.response_text),
        )
        message = SimpleNamespace(type="message", content=[text_block])
        return SimpleNamespace(output=[message], output_text=self.response_text)


class DummyClient:
    def __init__(self, response_text: str):
        self.responses = DummyResponses(response_text)


def _minimal_state() -> dict:
    return {
        "role": "retailer",
        "week": 1,
        "toggles": {
            "customer_demand_history_sharing": False,
            "volatility_signal_sharing": False,
            "downstream_inventory_visibility": False,
        },
        "parameters": {
            "holding_cost": 0.5,
            "backlog_cost": 0.5,
            "L_order": 2,
            "L_ship": 2,
            "L_prod": 4,
        },
        "local_state": {
            "on_hand": 12,
            "backlog": 0,
            "incoming_orders_this_week": 4,
            "received_shipment_this_week": 0,
            "pipeline_orders_upstream": [0, 0],
            "pipeline_shipments_inbound": [0, 0],
            "optional": {},
        },
    }


def test_instructions_payload_contains_role_name():
    assert "Daybreak Beer Game Strategist" in DAYBREAK_STRATEGIST_INSTRUCTIONS


def test_daybreak_session_decide_reuses_session(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("DAYBREAK_RESPONSES_SESSION_ID", raising=False)
    monkeypatch.delenv("DAYBREAK_SESSION_ID", raising=False)
    monkeypatch.delenv("DAYBREAK_ASSISTANT_ID", raising=False)
    monkeypatch.delenv("DAYBREAK_STRATEGIST_ASSISTANT_ID", raising=False)
    monkeypatch.delenv("GPT_ID", raising=False)

    response = json.dumps(
        {
            "order_upstream": 7,
            "ship_to_downstream": 5,
            "rationale": "stub decision",
        }
    )
    client = DummyClient(response)

    monkeypatch.setattr(daybreak_client, "_CLIENT", None)
    monkeypatch.setattr(daybreak_client, "_SESSION_CACHE", {})
    monkeypatch.setattr(daybreak_client, "_get_client", lambda: client)

    session = daybreak_client.DaybreakStrategistSession(model="gpt-test")
    state = _minimal_state()

    result_one = session.decide(state)
    assert result_one == {
        "order_upstream": 7,
        "ship_to_downstream": 5,
        "rationale": "stub decision",
    }
    assert session.last_decision == result_one
    # Session should be created exactly once with the expected instruction block
    assert len(client.responses.sessions.create_calls) == 1
    assert (
        client.responses.sessions.create_calls[0]["instructions"]
        == DAYBREAK_STRATEGIST_INSTRUCTIONS
    )
    first_session_id = session.session_id
    assert first_session_id is not None

    payload = json.dumps(state, separators=(",", ":"))
    expected_prompt = (
        "Here is the current state as JSON. Respond ONLY with the required JSON object.\n\n"
        f"```json\n{payload}\n```"
    )
    assert len(client.responses.create_calls) == 1
    assert client.responses.create_calls[0] == {
        "session": first_session_id,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": expected_prompt,
                    }
                ],
            }
        ],
    }

    # Second decision should reuse the same session
    result_two = session.decide(state)
    assert result_two == result_one
    assert len(client.responses.sessions.create_calls) == 1
    assert len(client.responses.create_calls) == 2
    for call in client.responses.create_calls:
        assert call["session"] == first_session_id
        assert call["input"][0]["content"][0]["text"] == expected_prompt


def test_daybreak_session_reset_creates_new_session(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("DAYBREAK_RESPONSES_SESSION_ID", raising=False)
    monkeypatch.delenv("DAYBREAK_SESSION_ID", raising=False)

    response = json.dumps(
        {
            "order_upstream": 2,
            "ship_to_downstream": 1,
            "rationale": "stub",
        }
    )
    client = DummyClient(response)

    monkeypatch.setattr(daybreak_client, "_CLIENT", None)
    monkeypatch.setattr(daybreak_client, "_SESSION_CACHE", {})
    monkeypatch.setattr(daybreak_client, "_get_client", lambda: client)

    session = daybreak_client.DaybreakStrategistSession(model="gpt-reset")
    state = _minimal_state()

    first = session.decide(state)
    first_session_id = session.session_id
    session.reset()
    assert session.session_id is None
    second = session.decide(state)

    assert first == second
    # Session creation should have been invoked twice due to reset
    assert len(client.responses.sessions.create_calls) == 2
    assert session.session_id != first_session_id
