from dataclasses import FrozenInstanceError

import pytest

from agent_kernel.core.results import AuthorityEnvelope, ExecutionPlan


def test_execution_plan_is_immutable():
    plan = ExecutionPlan(
        action="chat",
        label="chat",
        capabilities=frozenset(),
        instructions=("reply",),
        tool_call_limit=0,
        reads_history=True,
        envelope=AuthorityEnvelope(principal_id="u1", allowed=frozenset()),
        reason=None,
    )
    with pytest.raises(FrozenInstanceError):
        plan.action = "draw"
