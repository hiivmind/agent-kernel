from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from agno.run.base import RunContext, RunStatus
from agno.models.response import ToolExecution
from agno.tools.function import Function, FunctionCall
from pydantic import BaseModel, TypeAdapter, ValidationError

from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.operations import (
    OperationContract,
    OperationInvocation,
    SubjectContext,
)
from agent_kernel.core.results import AuthorityEnvelope, Completed, ExecutionPlan, Failed
from agent_kernel.integrations.agno.bindings import (
    AgnoCapabilityBinding,
    AgnoInvocationContext,
)
from agent_kernel.integrations.agno.context import AgnoRunContext
from agent_kernel.integrations.agno.skill_runtime import AgnoSkillRuntime
from agent_kernel.integrations.agno.skills import AgnoSkillSource


class StatusInput(BaseModel):
    verbose: bool


class StatusOutput(BaseModel):
    summary: str


class FakeSkillProvider:
    def __init__(self, source: AgnoSkillSource, *, error: Exception | None = None):
        self.source = source
        self.error = error
        self.calls: list[str] = []

    def require(self, skill_id: str) -> AgnoSkillSource:
        self.calls.append(skill_id)
        if self.error is not None:
            raise self.error
        return self.source


class FakeBindingProvider:
    def __init__(self, bindings: dict[str, AgnoCapabilityBinding]):
        self.bindings = bindings
        self.calls: list[tuple[object, object]] = []

    def bindings_for(self, invocation, context):
        self.calls.append((invocation, context))
        return self.bindings


class FakeAgent:
    def __init__(self, *, response: object, error: Exception | None = None):
        self.response = response
        self.error = error
        self.run_calls: list[tuple[object, dict[str, object]]] = []

    def run(self, message, **kwargs):
        self.run_calls.append((message, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


class FakeAgentFactory:
    def __init__(self, response: object, *, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []
        self.agents: list[FakeAgent] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        agent = FakeAgent(response=self.response, error=self.error)
        self.agents.append(agent)
        return agent

    @property
    def last_kwargs(self):
        return self.calls[-1]

    @property
    def agent(self):
        return self.agents[-1]


@pytest.fixture
def runtime_case(tmp_path: Path):
    source = tmp_path / "status"
    source.mkdir()
    (source / "SKILL.md").write_text(
        "---\nname: status\ndescription: Test status\n---\nRead status safely.\n",
        encoding="utf-8",
    )
    envelope = AuthorityEnvelope(
        "operator-5",
        frozenset({"resources:read:corpus-agno", "history:read"}),
    )
    plan = ExecutionPlan(
        action="status",
        label="Status",
        capabilities=envelope.allowed,
        instructions=("Load the selected status Skill.",),
        tool_call_limit=8,
        reads_history=True,
        envelope=envelope,
        reason=None,
    )
    invocation = OperationInvocation(
        invocation_id="run-17",
        target_id="status",
        request="what is the Agno corpus status?",
        subject=SubjectContext(
            kind="corpus",
            id="agno",
            attributes={"workspace_role": "corpus-root"},
        ),
        inputs=StatusInput(verbose=False),
        plan=plan,
    )
    contract = OperationContract(
        input_adapter=TypeAdapter(StatusInput),
        output_adapter=TypeAdapter(StatusOutput),
        model_output_type=StatusOutput,
        model_input_builder=lambda item: {
            "request": item.request,
            "subject": {
                "kind": item.subject.kind,
                "id": item.subject.id,
                "attributes": dict(item.subject.attributes),
            },
            "inputs": item.inputs.model_dump(),
        },
    )
    handle = object()
    context = AgnoInvocationContext(
        run=AgnoRunContext(
            user_id="operator-5",
            session_id="session-3",
            session_state={"turn": 2},
            metadata={"trace": "trusted"},
        ),
        invocation_handle=handle,
    )
    read_tool = object()
    bindings = {
        "resources:read:corpus-agno": AgnoCapabilityBinding(
            "resources:read:corpus-agno",
            tools=(read_tool,),
            function_names=frozenset({"read_file"}),
        ),
        "history:read": AgnoCapabilityBinding("history:read"),
    }
    response = SimpleNamespace(
        content={"summary": "healthy"},
        is_paused=False,
        tools=[],
    )
    factory = FakeAgentFactory(response)
    skill_provider = FakeSkillProvider(AgnoSkillSource("status", str(source)))
    binding_provider = FakeBindingProvider(bindings)
    runtime = AgnoSkillRuntime(
        skill_provider=skill_provider,
        binding_provider=binding_provider,
        model="model",
        db="db",
        agent_factory=factory,
    )
    return SimpleNamespace(
        runtime=runtime,
        invocation=invocation,
        contract=contract,
        context=context,
        handle=handle,
        factory=factory,
        skill_provider=skill_provider,
        binding_provider=binding_provider,
        bindings=bindings,
        read_tool=read_tool,
        response=response,
    )


def test_runtime_loads_one_skill_and_stamps_correlation(runtime_case):
    outcome = runtime_case.runtime.execute(
        runtime_case.invocation,
        runtime_case.contract,
        context=runtime_case.context,
    )

    assert isinstance(outcome, Completed)
    assert outcome.content.invocation_id == "run-17"
    assert outcome.content.output.summary == "healthy"
    assert outcome.content.raw is runtime_case.response
    assert outcome.raw is runtime_case.response
    assert "invocation_id" not in runtime_case.factory.last_kwargs["output_schema"].model_fields
    assert runtime_case.factory.last_kwargs["tool_call_limit"] == 8
    assert runtime_case.factory.last_kwargs["skills"].get_skill_names() == ["status"]
    message, kwargs = runtime_case.factory.agent.run_calls[0]
    assert message == {
        "request": "what is the Agno corpus status?",
        "subject": {
            "kind": "corpus",
            "id": "agno",
            "attributes": {"workspace_role": "corpus-root"},
        },
        "inputs": {"verbose": False},
    }
    assert "agent_kernel_authority" not in repr(message)
    assert kwargs["run_id"] == "run-17"


def test_runtime_keeps_authority_and_handle_in_trusted_dependencies(runtime_case):
    runtime_case.runtime.execute(
        runtime_case.invocation,
        runtime_case.contract,
        context=runtime_case.context,
    )

    message, kwargs = runtime_case.factory.agent.run_calls[0]
    dependencies = kwargs["dependencies"]
    assert dependencies["agent_kernel_authority"] is runtime_case.invocation.plan.envelope
    assert dependencies["agent_kernel_invocation_handle"] is runtime_case.handle
    assert runtime_case.handle not in _objects_in(message)
    assert runtime_case.invocation.plan.envelope not in _objects_in(message)


def _objects_in(value: object) -> list[object]:
    if isinstance(value, dict):
        return [value, *[item for child in value.values() for item in _objects_in(child)]]
    if isinstance(value, (list, tuple)):
        return [value, *[item for child in value for item in _objects_in(child)]]
    return [value]


def test_runtime_selects_tools_in_capability_then_declared_order(runtime_case):
    first, second, third = object(), object(), object()
    runtime_case.binding_provider.bindings = {
        "resources:read:corpus-agno": AgnoCapabilityBinding(
            "resources:read:corpus-agno",
            tools=(second, third),
            function_names=frozenset({"read_file", "fetch_url"}),
        ),
        "history:read": AgnoCapabilityBinding(
            "history:read",
            tools=(first,),
            function_names=frozenset({"read_history"}),
        ),
    }

    runtime_case.runtime.execute(runtime_case.invocation, runtime_case.contract)

    assert runtime_case.factory.last_kwargs["tools"] == [first, second, third]


def test_bad_input_fails_before_skill_or_agent_construction(runtime_case):
    invocation = replace(runtime_case.invocation, inputs={"verbose": "not-a-boolean"})

    outcome = runtime_case.runtime.execute(invocation, runtime_case.contract)

    assert isinstance(outcome, Failed)
    assert outcome.stage == "input-validation"
    assert isinstance(outcome.cause, ValidationError)
    assert runtime_case.skill_provider.calls == []
    assert runtime_case.factory.calls == []


def test_missing_skill_fails_before_agent_construction(runtime_case):
    missing_skill_error = KeyError("status")
    runtime_case.skill_provider.error = missing_skill_error

    outcome = runtime_case.runtime.execute(runtime_case.invocation, runtime_case.contract)

    assert outcome == Failed(stage="skill-load", cause=missing_skill_error)
    assert runtime_case.factory.calls == []


def _write_skill(source: Path, name: str) -> None:
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test {name}\n---\nRead safely.\n",
        encoding="utf-8",
    )


def test_zero_loaded_skills_is_skill_load_failure(runtime_case, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    runtime_case.skill_provider.source = AgnoSkillSource("status", str(empty))

    outcome = runtime_case.runtime.execute(runtime_case.invocation, runtime_case.contract)

    assert isinstance(outcome, Failed)
    assert outcome.stage == "skill-load"
    assert isinstance(outcome.cause, ConfigurationError)
    assert "exactly one" in str(outcome.cause)
    assert runtime_case.factory.calls == []


def test_multiple_loaded_skills_is_skill_load_failure(runtime_case, tmp_path):
    collection = tmp_path / "collection"
    _write_skill(collection / "alpha", "alpha")
    _write_skill(collection / "beta", "beta")
    runtime_case.skill_provider.source = AgnoSkillSource("status", str(collection))

    outcome = runtime_case.runtime.execute(runtime_case.invocation, runtime_case.contract)

    assert isinstance(outcome, Failed)
    assert outcome.stage == "skill-load"
    assert isinstance(outcome.cause, ConfigurationError)
    assert "exactly one" in str(outcome.cause)
    assert runtime_case.factory.calls == []


def test_provider_skill_id_mismatch_is_skill_load_failure(runtime_case):
    runtime_case.skill_provider.source = replace(
        runtime_case.skill_provider.source,
        id="different",
    )

    outcome = runtime_case.runtime.execute(runtime_case.invocation, runtime_case.contract)

    assert isinstance(outcome, Failed)
    assert outcome.stage == "skill-load"
    assert isinstance(outcome.cause, ConfigurationError)
    assert "different" in str(outcome.cause)
    assert "status" in str(outcome.cause)
    assert runtime_case.factory.calls == []


def test_loaded_skill_name_mismatch_is_skill_load_failure(runtime_case, tmp_path):
    different = tmp_path / "different"
    _write_skill(different, "different")
    runtime_case.skill_provider.source = AgnoSkillSource("status", str(different))

    outcome = runtime_case.runtime.execute(runtime_case.invocation, runtime_case.contract)

    assert isinstance(outcome, Failed)
    assert outcome.stage == "skill-load"
    assert isinstance(outcome.cause, ConfigurationError)
    assert "different" in str(outcome.cause)
    assert "status" in str(outcome.cause)
    assert runtime_case.factory.calls == []


def test_missing_capability_binding_fails_before_agent_construction(runtime_case):
    runtime_case.binding_provider.bindings = {
        "history:read": AgnoCapabilityBinding("history:read")
    }

    outcome = runtime_case.runtime.execute(runtime_case.invocation, runtime_case.contract)

    assert isinstance(outcome, Failed)
    assert outcome.stage == "configuration"
    assert isinstance(outcome.cause, ConfigurationError)
    assert "resources:read:corpus-agno" in str(outcome.cause)
    assert runtime_case.factory.calls == []


def test_extra_capability_binding_fails_before_agent_construction(runtime_case):
    runtime_case.binding_provider.bindings = {
        **runtime_case.bindings,
        "resources:delete:everything": AgnoCapabilityBinding(
            "resources:delete:everything"
        ),
    }

    outcome = runtime_case.runtime.execute(runtime_case.invocation, runtime_case.contract)

    assert isinstance(outcome, Failed)
    assert outcome.stage == "configuration"
    assert isinstance(outcome.cause, ConfigurationError)
    assert "resources:delete:everything" in str(outcome.cause)
    assert runtime_case.factory.calls == []


def test_duplicate_function_binding_is_configuration_failure(runtime_case):
    runtime_case.binding_provider.bindings = {
        "history:read": AgnoCapabilityBinding(
            "history:read",
            function_names=frozenset({"read_file"}),
        ),
        "resources:read:corpus-agno": runtime_case.bindings[
            "resources:read:corpus-agno"
        ],
    }

    outcome = runtime_case.runtime.execute(runtime_case.invocation, runtime_case.contract)

    assert isinstance(outcome, Failed)
    assert outcome.stage == "configuration"
    assert "duplicate function name" in str(outcome.cause)
    assert runtime_case.factory.calls == []


def test_reserved_reader_binding_is_configuration_failure_before_agent(
    runtime_case,
):
    runtime_case.binding_provider.bindings = {
        "history:read": runtime_case.bindings["history:read"],
        "resources:read:corpus-agno": AgnoCapabilityBinding(
            "resources:read:corpus-agno",
            function_names=frozenset({"get_skill_instructions"}),
        ),
    }

    outcome = runtime_case.runtime.execute(
        runtime_case.invocation,
        runtime_case.contract,
    )

    assert isinstance(outcome, Failed)
    assert outcome.stage == "configuration"
    assert "collide with internal functions" in str(outcome.cause)
    assert runtime_case.factory.calls == []


@pytest.mark.parametrize(
    "malicious_builder",
    [
        lambda item: {"value": item},
        lambda item: {"value": item.plan},
        lambda item: {"value": item.plan.envelope},
        lambda item: {"correlation": item.invocation_id},
        lambda item: {"value": next(iter(item.plan.capabilities))},
        lambda item: {"value": item.plan.envelope.principal_id},
        lambda item: {"path": "/private/tmp/operator-secrets.json"},
        lambda item: {item.plan.envelope.principal_id: "trusted-key"},
        lambda item: {"/private/tmp/operator-secrets.json": "host-path-key"},
        lambda item: {"api_key": "not-a-real-key"},
        lambda item: {"value": object()},
        lambda item: {"value": {1: "not-a-string-key"}},
        lambda item: {"value": math.nan},
    ],
    ids=[
        "invocation-object",
        "plan-object",
        "authority-envelope",
        "invocation-id",
        "capability-string",
        "principal-string",
        "absolute-path",
        "principal-key",
        "absolute-path-key",
        "credential-key",
        "runtime-object",
        "non-string-key",
        "non-json-float",
    ],
)
def test_model_input_builder_cannot_leak_trusted_state(
    runtime_case,
    malicious_builder,
):
    contract = replace(
        runtime_case.contract,
        model_input_builder=malicious_builder,
    )

    outcome = runtime_case.runtime.execute(runtime_case.invocation, contract)

    assert isinstance(outcome, Failed)
    assert outcome.stage == "input-validation"
    assert runtime_case.factory.calls == []


def test_model_input_builder_cannot_leak_invocation_handle(runtime_case):
    contract = replace(
        runtime_case.contract,
        model_input_builder=lambda item: {
            "request": item.request,
            "handle": runtime_case.handle,
        },
    )

    outcome = runtime_case.runtime.execute(runtime_case.invocation, contract)

    assert isinstance(outcome, Failed)
    assert outcome.stage == "input-validation"
    assert runtime_case.factory.calls == []


def test_provider_exception_is_provider_execution_failure(runtime_case):
    provider_error = RuntimeError("provider unavailable")
    runtime_case.factory.error = provider_error

    outcome = runtime_case.runtime.execute(runtime_case.invocation, runtime_case.contract)

    assert outcome == Failed(stage="provider-execution", cause=provider_error)


def test_agno_marked_tool_error_is_tool_execution_failure(runtime_case):
    tool_error = PermissionError("governed tool rejected")
    runtime_case.factory.response = SimpleNamespace(
        content={"summary": "untrusted"},
        is_paused=False,
        tools=[SimpleNamespace(tool_call_error=True, result=tool_error)],
    )

    outcome = runtime_case.runtime.execute(runtime_case.invocation, runtime_case.contract)

    assert outcome == Failed(stage="tool-execution", cause=tool_error)


def test_real_agno_tool_execution_recovers_original_governed_exception(
    runtime_case,
):
    tool_error = PermissionError("governed tool rejected")

    class ToolFailureAgent:
        def __init__(self, kwargs):
            self.kwargs = kwargs

        def run(self, message, **kwargs):
            del message

            def fail() -> None:
                raise tool_error

            function = Function(
                name="read_file",
                entrypoint=fail,
                tool_hooks=self.kwargs["tool_hooks"],
            )
            function._run_context = RunContext(
                run_id="run-17",
                session_id="session-3",
                dependencies=kwargs["dependencies"],
            )
            execution = FunctionCall(function=function).execute()
            assert execution.status == "failure"
            assert execution.error == str(tool_error)
            return SimpleNamespace(
                content={"summary": "untrusted"},
                is_paused=False,
                tools=[
                    ToolExecution(
                        tool_name="read_file",
                        tool_call_error=True,
                        result=execution.error,
                    )
                ],
            )

    def agent_factory(**kwargs):
        runtime_case.factory.calls.append(kwargs)
        return ToolFailureAgent(kwargs)

    runtime_case.runtime._agent_factory = agent_factory

    outcome = runtime_case.runtime.execute(
        runtime_case.invocation,
        runtime_case.contract,
    )

    assert outcome == Failed(stage="tool-execution", cause=tool_error)
    assert outcome.cause is tool_error


def test_unmarked_agno_error_is_provider_execution_failure(runtime_case):
    runtime_case.factory.response = SimpleNamespace(
        content="provider rejected the run",
        is_paused=False,
        status=RunStatus.error,
        tools=[],
    )

    outcome = runtime_case.runtime.execute(runtime_case.invocation, runtime_case.contract)

    assert isinstance(outcome, Failed)
    assert outcome.stage == "provider-execution"
    assert str(outcome.cause) == "provider rejected the run"


def test_invalid_content_is_output_validation_failure(runtime_case):
    runtime_case.factory.response = SimpleNamespace(
        content={"wrong": "shape"},
        is_paused=False,
        tools=[],
    )

    outcome = runtime_case.runtime.execute(runtime_case.invocation, runtime_case.contract)

    assert isinstance(outcome, Failed)
    assert outcome.stage == "output-validation"
    assert isinstance(outcome.cause, ValidationError)


def test_runtime_builds_agent_with_selected_governance(runtime_case):
    runtime_case.runtime.execute(runtime_case.invocation, runtime_case.contract)

    kwargs = runtime_case.factory.last_kwargs
    assert kwargs["model"] == "model"
    assert kwargs["db"] == "db"
    assert kwargs["tools"] == [runtime_case.read_tool]
    assert kwargs["instructions"] == list(runtime_case.invocation.plan.instructions)
    assert len(kwargs["tool_hooks"]) == 2
