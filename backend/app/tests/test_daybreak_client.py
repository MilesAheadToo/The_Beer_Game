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


class DummyAssistants:
    def __init__(self):
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(id=f"asst-{len(self.create_calls)}")


class DummyMessages:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.create_calls = []
        self.list_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(id=f"msg-{len(self.create_calls)}")

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        text_block = SimpleNamespace(
            type="text",
            text=SimpleNamespace(value=self.response_text),
        )
        message = SimpleNamespace(content=[text_block])
        return SimpleNamespace(data=[message])


class DummyRuns:
    def __init__(self, status: str = "completed"):
        self.status = status
        self.create_calls = []

    def create_and_poll(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(status=self.status)


class DummyThreads:
    def __init__(self, messages: DummyMessages, runs: DummyRuns):
        self.messages = messages
        self.runs = runs
        self.create_calls = []

    def create(self):
        thread_id = f"thread-{len(self.create_calls) + 1}"
        self.create_calls.append({"id": thread_id})
        return SimpleNamespace(id=thread_id)


class DummyBeta:
    def __init__(self, response_text: str):
        self.assistants = DummyAssistants()
        self._messages = DummyMessages(response_text)
        self._runs = DummyRuns()
        self.threads = DummyThreads(self._messages, self._runs)

    @property
    def messages(self):  # pragma: no cover - compatibility shim
        return self._messages


class DummyClient:
    def __init__(self, response_text: str):
        beta = DummyBeta(response_text)
        # The real client exposes resources under client.beta
        self.beta = SimpleNamespace(
            assistants=beta.assistants,
            threads=beta.threads,
        )
        self._beta_resources = beta


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


def test_daybreak_session_decide_reuses_assistant(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
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
    monkeypatch.setattr(daybreak_client, "_ASSISTANT_CACHE", {})
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
    # Assistant should be created exactly once with the expected instruction block
    assert len(client.beta.assistants.create_calls) == 1
    assert client.beta.assistants.create_calls[0]["instructions"] == DAYBREAK_STRATEGIST_INSTRUCTIONS
    # Thread must be created for the first decision
    assert len(client.beta.threads.create_calls) == 1
    first_thread = client.beta.threads.create_calls[0]["id"]
    assert session.thread_id == first_thread

    # Second decision should reuse the same assistant and thread
    result_two = session.decide(state)
    assert result_two == result_one
    assert len(client.beta.assistants.create_calls) == 1
    assert len(client.beta.threads.create_calls) == 1


def test_daybreak_session_reset_creates_new_thread(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    response = json.dumps(
        {
            "order_upstream": 2,
            "ship_to_downstream": 1,
            "rationale": "stub",
        }
    )
    client = DummyClient(response)

    monkeypatch.setattr(daybreak_client, "_CLIENT", None)
    monkeypatch.setattr(daybreak_client, "_ASSISTANT_CACHE", {})
    monkeypatch.setattr(daybreak_client, "_get_client", lambda: client)

    session = daybreak_client.DaybreakStrategistSession(model="gpt-reset")
    state = _minimal_state()

    first = session.decide(state)
    session.reset()
    second = session.decide(state)

    assert first == second
    # Thread creation should have been invoked twice due to reset
    assert len(client.beta.threads.create_calls) == 2
