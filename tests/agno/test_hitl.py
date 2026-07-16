from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, TypeAdapter

from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.operations import (
    OperationCompletion,
    OperationContract,
    OperationInvocation,
    SubjectContext,
)
from agent_kernel.core.results import (
    AuthorityEnvelope,
    Completed,
    ExecutionPlan,
    Failed,
    Pause,
)
from agent_kernel.integrations.agno import AgnoRunContext, AgnoRuntime
from agent_kernel.integrations.agno.bindings import (
    AgnoCapabilityBinding,
    AgnoInvocationContext,
)
from agent_kernel.integrations.agno.hitl import UnsupportedRequirement
from agent_kernel.integrations.agno.skill_runtime import AgnoSkillRuntime
from agent_kernel.integrations.agno.skills import AgnoSkillSource


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
    assert outcome.adapter_state is not response
    assert not hasattr(outcome.adapter_state, "requirements")
    assert not hasattr(outcome.adapter_state, "content")
    with pytest.raises(AttributeError):
        outcome.adapter_state.requirements = []
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


def test_resume_rejects_replaced_opaque_pause_token(plan):
    response = paused_response(FakeRequirement([FakeField("sides")]))
    factory = FakeAgentFactory(
        response,
        continue_responses=[completed_response()],
    )
    runtime = make_runtime(factory)
    pause = runtime.execute("spin", plan)
    forged = replace(pause, adapter_state=object())

    with pytest.raises(ConfigurationError, match="pause"):
        runtime.resume(forged, {"sides": "d20"})

    assert factory.agents[0].continue_calls == []


def test_public_requirement_snapshot_cannot_mutate_private_response(plan):
    requirement = FakeRequirement([FakeField("sides")])
    response = paused_response(requirement)
    finished = completed_response()
    runtime = make_runtime(
        FakeAgentFactory(response, continue_responses=[finished])
    )
    pause = runtime.execute("spin", plan)
    pause.requirements[0]["field"] = "delete_world"

    outcome = runtime.resume(pause, {"sides": "d20"})

    assert outcome == Completed(content="finished", raw=finished)
    assert requirement.provided == [{"sides": "d20"}]


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


def test_resume_rejects_in_place_mutation_of_paused_requirements(plan):
    field = FakeField("sides")
    requirement = FakeRequirement([field])
    response = paused_response(requirement)
    factory = FakeAgentFactory(
        response,
        continue_responses=[completed_response()],
    )
    runtime = make_runtime(factory)
    pause = runtime.execute("spin", plan)
    field.name = "delete_world"

    with pytest.raises(ConfigurationError, match="state"):
        runtime.resume(pause, {"delete_world": "yes"})

    assert requirement.provided == []
    assert factory.agents[0].continue_calls == []


def test_resume_rejects_mutated_pause_status(plan):
    response = paused_response(FakeRequirement([FakeField("sides")]))
    factory = FakeAgentFactory(
        response,
        continue_responses=[completed_response()],
    )
    runtime = make_runtime(factory)
    pause = runtime.execute("spin", plan)
    response.is_paused = False

    with pytest.raises(ConfigurationError, match="state"):
        runtime.resume(pause, {"sides": "d20"})

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
    assert second_pause.adapter_state is not second_response
    assert second_pause.adapter_state is not first_pause.adapter_state
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


class HitlInput(BaseModel):
    verbose: bool


class HitlOutput(BaseModel):
    summary: str


class RecordingAdapter:
    def __init__(self):
        self.values = []

    def validate_python(self, value):
        self.values.append(value)
        return HitlOutput.model_validate(value)


class StaticSkillProvider:
    def __init__(self, source):
        self.source = source

    def require(self, skill_id):
        return self.source


class StaticBindingProvider:
    def __init__(self, bindings):
        self.bindings = bindings

    def bindings_for(self, invocation, context):
        return self.bindings


def make_skill_runtime_case(tmp_path: Path, first_response, continue_responses):
    source = tmp_path / "status"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        "---\nname: status\ndescription: Test status\n---\nRead status safely.\n",
        encoding="utf-8",
    )
    envelope = AuthorityEnvelope(
        "original-principal",
        frozenset({"resources:read:corpus-agno"}),
    )
    plan = ExecutionPlan(
        action="status",
        label="Status",
        capabilities=envelope.allowed,
        instructions=("Read status.",),
        tool_call_limit=3,
        reads_history=False,
        envelope=envelope,
        reason=None,
    )
    invocation = OperationInvocation(
        invocation_id="original-run",
        target_id="status",
        request="status?",
        subject=SubjectContext("corpus", "agno", {}),
        inputs=HitlInput(verbose=False),
        plan=plan,
    )
    output_adapter = RecordingAdapter()
    contract = OperationContract(
        input_adapter=TypeAdapter(HitlInput),
        output_adapter=output_adapter,
        model_output_type=HitlOutput,
        model_input_builder=lambda item: {"request": item.request},
    )
    handle = object()
    context = AgnoInvocationContext(
        run=AgnoRunContext(
            user_id="original-principal",
            session_id="original-session",
            session_state={"original": True},
            metadata={"trace": "original"},
        ),
        invocation_handle=handle,
    )
    factory = FakeAgentFactory(
        first_response,
        continue_responses=continue_responses,
    )
    runtime = AgnoSkillRuntime(
        skill_provider=StaticSkillProvider(
            AgnoSkillSource("status", str(source))
        ),
        binding_provider=StaticBindingProvider(
            {
                "resources:read:corpus-agno": AgnoCapabilityBinding(
                    "resources:read:corpus-agno"
                )
            }
        ),
        agent_factory=factory,
    )
    return SimpleNamespace(
        runtime=runtime,
        invocation=invocation,
        contract=contract,
        context=context,
        handle=handle,
        factory=factory,
        output_adapter=output_adapter,
    )


def test_skill_resume_preserves_original_trusted_state_across_second_pause(
    tmp_path,
):
    first = paused_response(FakeRequirement([FakeField("scope")]))
    second = paused_response(FakeRequirement([FakeField("format")]))
    finished = completed_response({"summary": "healthy"})
    case = make_skill_runtime_case(tmp_path, first, [second, finished])
    first_pause = case.runtime.execute(
        case.invocation,
        case.contract,
        context=case.context,
    )
    original_agent = case.factory.agents[0]
    original_skill = case.factory.calls[0]["skills"]
    assert isinstance(first_pause, Pause)
    assert first_pause.envelope is case.invocation.plan.envelope
    assert first_pause.runtime_context is case.context.run
    assert original_skill.get_skill_names() == [case.invocation.target_id]
    assert case.factory.calls[0]["instructions"] == list(
        case.invocation.plan.instructions
    )
    assert case.factory.calls[0]["tool_call_limit"] == (
        case.invocation.plan.tool_call_limit
    )
    assert original_agent.run_calls[0][1]["run_id"] == (
        case.invocation.invocation_id
    )

    second_pause = case.runtime.resume(first_pause, {"scope": "workspace"})
    outcome = case.runtime.resume(second_pause, {"format": "summary"})

    assert isinstance(second_pause, Pause)
    assert second_pause.envelope is case.invocation.plan.envelope
    assert second_pause.runtime_context is case.context.run
    assert outcome == Completed(
        content=OperationCompletion(
            invocation_id="original-run",
            output=HitlOutput(summary="healthy"),
            raw=finished,
        ),
        raw=finished,
    )
    assert case.output_adapter.values == [{"summary": "healthy"}]
    assert len(case.factory.agents) == 1
    assert case.factory.agents[0] is original_agent
    assert case.factory.calls[0]["skills"] is original_skill
    for _, kwargs in case.factory.agents[0].continue_calls:
        assert kwargs["run_id"] == "original-run"
        assert kwargs["dependencies"] == {
            "agent_kernel_authority": case.invocation.plan.envelope,
            "agent_kernel_invocation_handle": case.handle,
        }
        assert kwargs["user_id"] == "original-principal"
        assert kwargs["session_id"] == "original-session"
        assert kwargs["session_state"] == {"original": True}
        assert kwargs["metadata"] == {"trace": "original"}


def test_skill_resume_rejects_pause_token_from_another_runtime(tmp_path):
    first_case = make_skill_runtime_case(
        tmp_path / "first",
        paused_response(FakeRequirement([FakeField("scope")])),
        [],
    )
    second_case = make_skill_runtime_case(
        tmp_path / "second",
        paused_response(FakeRequirement([FakeField("scope")])),
        [],
    )
    first_pause = first_case.runtime.execute(
        first_case.invocation,
        first_case.contract,
        context=first_case.context,
    )
    second_pause = second_case.runtime.execute(
        second_case.invocation,
        second_case.contract,
        context=second_case.context,
    )
    assert isinstance(first_pause, Pause)
    assert isinstance(second_pause, Pause)

    with pytest.raises(ConfigurationError, match="pause"):
        first_case.runtime.resume(second_pause, {"scope": "workspace"})

    assert first_case.factory.agents[0].continue_calls == []
    assert second_case.factory.agents[0].continue_calls == []
