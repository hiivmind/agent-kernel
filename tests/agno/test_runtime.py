from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import pytest

from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.results import (
    AuthorityEnvelope,
    Completed,
    ExecutionPlan,
    Failed,
    Pause,
)
from agent_kernel.integrations.agno import AgnoRunContext, AgnoRuntime


class FakeAgent:
    def __init__(self, *, response, error=None):
        self.response = response
        self.error = error
        self.run_calls = []

    def run(self, message, **kwargs):
        self.run_calls.append((message, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


class FakeAgentFactory:
    def __init__(self, response=None, *, error=None):
        self.response = response or SimpleNamespace(
            content="finished",
            is_paused=False,
        )
        self.error = error
        self.calls = []
        self.agents = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        agent = FakeAgent(response=self.response, error=self.error)
        self.agents.append(agent)
        return agent

    @property
    def last_kwargs(self):
        return self.calls[-1]


@pytest.fixture
def plan():
    return ExecutionPlan(
        action="spin",
        label="Spin",
        capabilities=frozenset({"zeta_tool", "alpha_tool"}),
        instructions=("Follow the spin procedure.",),
        tool_call_limit=2,
        reads_history=False,
        envelope=AuthorityEnvelope(
            principal_id="member-1",
            allowed=frozenset({"zeta_tool", "alpha_tool"}),
        ),
        reason=None,
    )


def test_run_context_has_exact_frozen_shape():
    context = AgnoRunContext()

    assert [field.name for field in fields(context)] == [
        "user_id",
        "session_id",
        "session_state",
        "metadata",
    ]
    assert context == AgnoRunContext(
        user_id=None,
        session_id=None,
        session_state=None,
        metadata=None,
    )
    with pytest.raises(FrozenInstanceError):
        context.user_id = "changed"


def test_runtime_exposes_only_planned_tools_in_deterministic_order(plan):
    alpha_tool = object()
    zeta_tool = object()
    factory = FakeAgentFactory()
    runtime = AgnoRuntime(
        tools={
            "zeta_tool": zeta_tool,
            "delete_world": object(),
            "alpha_tool": alpha_tool,
        },
        agent_factory=factory,
    )

    runtime.execute("spin", plan)

    assert factory.last_kwargs["tools"] == [alpha_tool, zeta_tool]
    assert factory.last_kwargs["tool_call_limit"] == 2


def test_runtime_forwards_model_db_and_plan_instructions(plan):
    model = object()
    db = object()
    factory = FakeAgentFactory()
    runtime = AgnoRuntime(
        model=model,
        tools={"alpha_tool": object(), "zeta_tool": object()},
        db=db,
        agent_factory=factory,
    )

    runtime.execute("spin", plan)

    assert factory.last_kwargs["model"] is model
    assert factory.last_kwargs["db"] is db
    assert factory.last_kwargs["instructions"] == list(plan.instructions)


def test_runtime_forwards_explicit_run_context_fields(plan):
    factory = FakeAgentFactory()
    runtime = AgnoRuntime(
        tools={"alpha_tool": object(), "zeta_tool": object()},
        agent_factory=factory,
    )
    context = AgnoRunContext(
        user_id="member-1",
        session_id="session-7",
        session_state={"recent": ["hello"]},
        metadata={"request_id": "request-9"},
    )

    runtime.execute("spin", plan, context=context)

    assert factory.agents[0].run_calls == [
        (
            "spin",
            {
                "user_id": "member-1",
                "session_id": "session-7",
                "session_state": {"recent": ["hello"]},
                "metadata": {"request_id": "request-9"},
            },
        )
    ]


def test_runtime_omits_context_fields_when_context_is_not_supplied(plan):
    factory = FakeAgentFactory()
    runtime = AgnoRuntime(
        tools={"alpha_tool": object(), "zeta_tool": object()},
        agent_factory=factory,
    )

    runtime.execute("spin", plan)

    assert factory.agents[0].run_calls == [("spin", {})]


def test_runtime_returns_completed_response(plan):
    response = SimpleNamespace(content={"result": 17}, is_paused=False)
    runtime = AgnoRuntime(
        tools={"alpha_tool": object(), "zeta_tool": object()},
        agent_factory=FakeAgentFactory(response),
    )

    outcome = runtime.execute("spin", plan)

    assert outcome == Completed(content={"result": 17}, raw=response)


def test_runtime_minimally_translates_paused_response(plan):
    response = SimpleNamespace(content=None, is_paused=True)
    context = AgnoRunContext(user_id="member-1", session_id="session-7")
    runtime = AgnoRuntime(
        tools={"alpha_tool": object(), "zeta_tool": object()},
        agent_factory=FakeAgentFactory(response),
    )

    outcome = runtime.execute("spin", plan, context=context)

    assert outcome == Pause(
        requirements=(),
        adapter_state=response,
        envelope=plan.envelope,
        runtime_context=context,
    )


def test_runtime_exception_becomes_failed(plan):
    error = RuntimeError("agent failed")
    runtime = AgnoRuntime(
        tools={"alpha_tool": object(), "zeta_tool": object()},
        agent_factory=FakeAgentFactory(error=error),
    )

    outcome = runtime.execute("spin", plan)

    assert outcome == Failed(stage="runtime", cause=error)


def test_missing_tool_binding_raises_before_agent_construction(plan):
    factory = FakeAgentFactory()
    runtime = AgnoRuntime(
        tools={"alpha_tool": object()},
        agent_factory=factory,
    )

    with pytest.raises(ConfigurationError, match="zeta_tool"):
        runtime.execute("spin", plan)

    assert factory.calls == []
