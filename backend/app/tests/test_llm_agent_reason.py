import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def test_llm_agent_records_rationale(monkeypatch):
    """LLM agent should surface the strategist rationale in its explanation."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=SimpleNamespace))

    from app.services import llm_agent as llm_module

    class DummySession:
        def __init__(self, model: str):
            self.model = model
            self.decide_calls = []

        def decide(self, state):
            self.decide_calls.append(state)
            return {
                "order_upstream": 9,
                "ship_to_downstream": 4,
                "rationale": "Maintain a two-week buffer while clearing backlog.",
            }

        def reset(self):
            self.decide_calls.clear()

    monkeypatch.setattr(llm_module, "DaybreakStrategistSession", DummySession)

    agent = llm_module.LLMAgent(role="retailer", model="stub-model")

    order = agent.make_decision(
        current_round=1,
        current_inventory=10,
        backorders=2,
        incoming_shipments=[2, 2],
        demand_history=[8, 9],
        order_history=[8, 8],
        current_demand=8,
        upstream_data={"llm_payload": {"stub": True}},
    )

    assert order == 9
    assert agent.last_explanation is not None
    assert "Maintain a two-week buffer" in agent.last_explanation
    assert agent.last_decision == {
        "order_upstream": 9,
        "ship_to_downstream": 4,
        "rationale": "Maintain a two-week buffer while clearing backlog.",
    }


def test_llm_agent_records_fallback_reason(monkeypatch):
    """If the payload is missing the agent should document its fallback."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=SimpleNamespace))

    from app.services import llm_agent as llm_module

    class DummySession:
        def __init__(self, model: str):
            self.model = model

        def decide(self, state):  # pragma: no cover - shouldn't be hit
            raise AssertionError("Decide should not be called without payload")

        def reset(self):
            pass

    monkeypatch.setattr(llm_module, "DaybreakStrategistSession", DummySession)

    agent = llm_module.LLMAgent(role="retailer", model="stub-model")

    order = agent.make_decision(
        current_round=1,
        current_inventory=10,
        backorders=2,
        incoming_shipments=[2, 2],
        demand_history=[8, 9],
        order_history=[8, 8],
        current_demand=8,
        upstream_data={"other": "context"},
    )

    assert order >= 0
    assert agent.last_decision is None
    assert agent.last_explanation is not None
    assert "fallback" in agent.last_explanation.lower()


def test_llm_agent_records_error_reason(monkeypatch):
    """Errors surfaced by the strategist should translate to fallback notes."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=SimpleNamespace))

    from app.services import llm_agent as llm_module

    class FailingSession:
        def __init__(self, model: str):
            self.model = model

        def decide(self, state):
            raise RuntimeError("boom")

        def reset(self):
            pass

    monkeypatch.setattr(llm_module, "DaybreakStrategistSession", FailingSession)

    agent = llm_module.LLMAgent(role="retailer", model="stub-model")

    order = agent.make_decision(
        current_round=1,
        current_inventory=10,
        backorders=2,
        incoming_shipments=[2, 2],
        demand_history=[8, 9],
        order_history=[8, 8],
        current_demand=8,
        upstream_data={"llm_payload": {"stub": True}},
    )

    assert order >= 0
    assert agent.last_decision is None
    assert agent.last_explanation is not None
    assert "fallback" in agent.last_explanation.lower()
    assert "boom" in agent.last_explanation
