import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.agents import (
    AgentStrategy,
    AgentType,
    BeerGameAgent,
    DaybreakCoordinator,
    DaybreakGlobalController,
)


@pytest.fixture
def base_local_state():
    return {
        "inventory": 18,
        "backlog": 3,
        "incoming_shipments": [4, 4],
        "node_label": "Retailer",
        "node_key": "retailer",
    }


@pytest.fixture
def base_upstream_data():
    return {
        "previous_orders": [6, 7, 8],
        "previous_orders_by_role": {"retailer": 7, "wholesaler": 8},
    }


def _assert_reason(decision):
    assert isinstance(decision.reason, str)
    assert decision.reason.strip(), "expected a non-empty explanation"
    assert decision.reason.startswith("["), "expected label prefix in explanation"


def test_daybreak_dtce_agent_returns_reason(base_local_state, base_upstream_data):
    agent = BeerGameAgent(
        agent_id=1,
        agent_type=AgentType.RETAILER,
        strategy=AgentStrategy.DAYBREAK_DTCE,
        central_coordinator=DaybreakCoordinator(),
        global_controller=DaybreakGlobalController(),
    )

    decision = agent.make_decision(
        current_round=3,
        current_demand=9,
        upstream_data=base_upstream_data,
        local_state=base_local_state,
    )

    assert decision.quantity >= 0
    _assert_reason(decision)


def test_daybreak_dtce_central_agent_returns_reason(base_local_state, base_upstream_data):
    coordinator = DaybreakCoordinator(default_override=0.1)
    agent = BeerGameAgent(
        agent_id=2,
        agent_type=AgentType.WHOLESALER,
        strategy=AgentStrategy.DAYBREAK_DTCE_CENTRAL,
        central_coordinator=coordinator,
        global_controller=DaybreakGlobalController(),
    )

    decision = agent.make_decision(
        current_round=5,
        current_demand=None,
        upstream_data=base_upstream_data,
        local_state={**base_local_state, "node_label": "Wholesaler", "node_key": "wholesaler"},
    )

    assert decision.quantity >= 0
    _assert_reason(decision)


def test_daybreak_dtce_global_agent_returns_reason(base_local_state, base_upstream_data):
    coordinator = DaybreakCoordinator(default_override=0.1)
    controller = DaybreakGlobalController()
    agent = BeerGameAgent(
        agent_id=3,
        agent_type=AgentType.DISTRIBUTOR,
        strategy=AgentStrategy.DAYBREAK_DTCE_GLOBAL,
        central_coordinator=coordinator,
        global_controller=controller,
    )

    decision = agent.make_decision(
        current_round=7,
        current_demand=None,
        upstream_data=base_upstream_data,
        local_state={**base_local_state, "node_label": "Distributor", "node_key": "distributor"},
    )

    assert decision.quantity >= 0
    _assert_reason(decision)
