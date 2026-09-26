import pytest
from unittest.mock import AsyncMock, MagicMock
from src.controller.agent import Agent
from src.controller.agent_state import AgentStatus


@pytest.mark.asyncio
async def test_agent_abstains_on_insufficient_context(monkeypatch):
    """
    Regression test: kalau generator balikin status INSUFFICIENT_CONTEXT,
    agent.status HARUS jadi ABSTAINED, bukan tetap RUNNING.
    """
    agent = Agent.__new__(Agent)  # skip __init__ berat (no real Qdrant/Groq connection)

    fake_response = MagicMock()
    fake_response.status = "INSUFFICIENT_CONTEXT"
    fake_response.answer = "INSUFFICIENT_CONTEXT"

    agent.tools = MagicMock()
    agent.tools.call_tool = AsyncMock(return_value={"response": fake_response})

    from src.controller.agent_state import AgentState
    state = AgentState(query="dummy query")

    from src.controller.policy_engine import PlanHints
    plan = MagicMock()
    plan.generation_hint = {"max_tokens": 100, "temperature": 0.0}

    result = await agent._execute_generation(state, plan)

    assert result.status == AgentStatus.ABSTAINED, (
        f"Expected ABSTAINED when response.status=INSUFFICIENT_CONTEXT, "
        f"got {result.status}"
    )