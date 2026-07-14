from dataclasses import replace
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
from agent_kernel.integrations.agno.hitl import UnsupportedRequirement


class FakeField:
    def __init__(
        self,
        name,
        *,
        description=None,
        field_type=str,
    ):
        self.name = name
        self.description = description
        self.field_type = field_type


class FakeRequirement:
    def __init__(
        self,
        fields=(),
        *,
        confirmation=False,
        external_execution=False,
    ):
        self.user_input_schema = list(fields)
        self.needs_user_input = bool(fields)
        self.needs_confirmation = confirmation
        self.needs_external_execution = external_execution
        self.provided = []

    def provide_user_input(self, answers):
        self.provided.append(answers)


def paused_response(*requirements):
    return SimpleNamespace(
        content=None,
        is_paused=True,
        requirements=list(requirements),
    )


def completed_response(content="finished"):
    return SimpleNamespace(
        content=content,
        is_paused=False,
        requirements=[],
    )


class FakeAgent:
    def __init__(
        self,
        *,
        run_response,
        continue_responses=(),
        continue_error=None,
    ):
        self.run_response = run_response
        self.continue_responses = list(continue_responses)
        self.continue_error = continue_error
        self.run_calls = []
        self.continue_calls = []

    def run(self, message, **kwargs):
        self.run_calls.append((message, kwargs))
        return self.run_response

    def continue_run(self, run_response=None, **kwargs):
        self.continue_calls.append((run_response, kwargs))
        if self.continue_error is not None:
            raise self.continue_error
        return self.continue_responses.pop(0)


class FakeAgentFactory:
    def __init__(
        self,
        run_response,
        *,
        continue_responses=(),
        continue_error=None,
    ):
        self.run_response = run_response
        self.continue_responses = continue_responses
        self.continue_error = continue_error
        self.calls = []
        self.agents = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        agent = FakeAgent(
            run_response=self.run_response,
            continue_responses=self.continue_responses,
            continue_error=self.continue_error,
        )
        self.agents.append(agent)
        return agent


@pytest.fixture
def plan():
    return ExecutionPlan(
        action="spin",
        label="Spin",
        capabilities=frozenset({"spin_wheel"}),
        instructions=("Follow the spin procedure.",),
        tool_call_limit=2,
        reads_history=False,
        envelope=AuthorityEnvelope(
            principal_id="member-1",
            allowed=frozenset({"spin_wheel"}),
        ),
        reason=None,
    )


def make_runtime(factory):
    return AgnoRuntime(
        tools={"spin_wheel": object()},
        agent_factory=factory,
    )


def test_pause_exposes_user_input_fields_and_preserves_original_state(
    plan,
):
    requirement = FakeRequirement(
        [
            FakeField("sides", description="Which die?"),
            FakeField("modifier", description="Modifier?"),
        ]
    )
    response = paused_response(requirement)
    context = AgnoRunContext(
        user_id="member-1",
        session_id="session-7",
    )
    runtime = make_runtime(FakeAgentFactory(response))

    outcome = runtime.execute("spin", plan, context=context)

    assert isinstance(outcome, Pause)
    assert outcome.adapter_state is response
    assert outcome.envelope is plan.envelope
    assert outcome.runtime_context is context
    assert tuple(item["field"] for item in outcome.requirements) == (
        "sides",
        "modifier",
    )
    assert tuple(item["description"] for item in outcome.requirements) == (
        "Which die?",
        "Modifier?",
    )


def test_resume_uses_same_agent_and_restamps_original_authority(plan):
    requirement = FakeRequirement(
        [
            FakeField("sides"),
            FakeField("modifier", field_type=int),
        ]
    )
    response = paused_response(requirement)
    finished = completed_response({"roll": 17})
    factory = FakeAgentFactory(
        response,
        continue_responses=[finished],
    )
    runtime = make_runtime(factory)
    context = AgnoRunContext(
        user_id="member-1",
        session_id="session-7",
    )
    pause = runtime.execute("spin", plan, context=context)

    outcome = runtime.resume(
        pause,
        {"sides": "d20", "modifier": "2", "ignored": "value"},
    )

    assert outcome == Completed(content={"roll": 17}, raw=finished)
    assert len(factory.agents) == 1
    assert requirement.provided == [
        {"sides": "d20", "modifier": "2"},
    ]
    assert factory.agents[0].continue_calls == [
        (
            response,
            {
                "dependencies": {
                    "agent_kernel_authority": plan.envelope,
                },
                "user_id": "member-1",
            },
        )
    ]


def test_resume_rejects_tampered_envelope_before_continuing(plan):
    response = paused_response(
        FakeRequirement([FakeField("sides")]),
    )
    factory = FakeAgentFactory(
        response,
        continue_responses=[completed_response()],
    )
    runtime = make_runtime(factory)
    pause = runtime.execute("spin", plan)
    tampered = replace(
        pause,
        envelope=AuthorityEnvelope(
            principal_id="attacker",
            allowed=frozenset({"delete_world"}),
        ),
    )

    with pytest.raises(ConfigurationError, match="envelope"):
        runtime.resume(tampered, {"sides": "d20"})

    assert factory.agents[0].continue_calls == []


def test_resume_rejects_adapter_state_not_created_by_runtime(plan):
    factory = FakeAgentFactory(completed_response())
    runtime = make_runtime(factory)
    foreign_pause = Pause(
        requirements=({"field": "sides"},),
        adapter_state=paused_response(
            FakeRequirement([FakeField("sides")]),
        ),
        envelope=plan.envelope,
    )

    with pytest.raises(ConfigurationError, match="pause"):
        runtime.resume(foreign_pause, {"sides": "d20"})

    assert factory.agents == []


@pytest.mark.parametrize(
    "requirement",
    [
        FakeRequirement(confirmation=True),
        FakeRequirement(external_execution=True),
    ],
)
def test_resume_returns_typed_failure_for_unsupported_requirement(
    plan,
    requirement,
):
    response = paused_response(requirement)
    factory = FakeAgentFactory(
        response,
        continue_responses=[completed_response()],
    )
    runtime = make_runtime(factory)
    pause = runtime.execute("spin", plan)

    outcome = runtime.resume(pause, {})

    assert isinstance(outcome, Failed)
    assert outcome.stage == "resume"
    assert isinstance(outcome.cause, UnsupportedRequirement)
    assert factory.agents[0].continue_calls == []


def test_resume_returns_typed_failure_when_declared_answer_is_missing(
    plan,
):
    response = paused_response(
        FakeRequirement(
            [
                FakeField("sides"),
                FakeField("modifier"),
            ]
        )
    )
    factory = FakeAgentFactory(
        response,
        continue_responses=[completed_response()],
    )
    runtime = make_runtime(factory)
    pause = runtime.execute("spin", plan)

    outcome = runtime.resume(pause, {"sides": "d20"})

    assert isinstance(outcome, Failed)
    assert outcome.stage == "resume"
    assert isinstance(outcome.cause, ValueError)
    assert "modifier" in str(outcome.cause)
    assert factory.agents[0].continue_calls == []


def test_resume_runtime_exception_becomes_resume_failure(plan):
    error = RuntimeError("resume unavailable")
    response = paused_response(
        FakeRequirement([FakeField("sides")]),
    )
    factory = FakeAgentFactory(
        response,
        continue_error=error,
    )
    runtime = make_runtime(factory)
    pause = runtime.execute("spin", plan)

    outcome = runtime.resume(pause, {"sides": "d20"})

    assert outcome == Failed(stage="resume", cause=error)


def test_resume_can_pause_again_and_then_complete_on_same_agent(plan):
    first_requirement = FakeRequirement([FakeField("sides")])
    second_requirement = FakeRequirement([FakeField("color")])
    first_response = paused_response(first_requirement)
    second_response = paused_response(second_requirement)
    finished = completed_response("done")
    factory = FakeAgentFactory(
        first_response,
        continue_responses=[second_response, finished],
    )
    runtime = make_runtime(factory)
    context = AgnoRunContext(user_id="member-1")
    first_pause = runtime.execute("spin", plan, context=context)

    second_pause = runtime.resume(first_pause, {"sides": "d20"})

    assert isinstance(second_pause, Pause)
    assert second_pause.adapter_state is second_response
    assert second_pause.envelope is plan.envelope
    assert second_pause.runtime_context is context
    assert tuple(
        item["field"] for item in second_pause.requirements
    ) == ("color",)

    outcome = runtime.resume(second_pause, {"color": "blue"})

    assert outcome == Completed(content="done", raw=finished)
    assert len(factory.agents) == 1
    assert first_requirement.provided == [{"sides": "d20"}]
    assert second_requirement.provided == [{"color": "blue"}]
    assert [
        call[1]["dependencies"]["agent_kernel_authority"]
        for call in factory.agents[0].continue_calls
    ] == [plan.envelope, plan.envelope]
