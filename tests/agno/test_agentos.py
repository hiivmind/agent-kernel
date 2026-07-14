from types import SimpleNamespace

import pytest
from agno.agent.protocol import AgentProtocol
from agno.models.response import ToolExecution
from agno.run.agent import (
    RunCompletedEvent,
    RunContentEvent,
    RunErrorEvent,
    RunStartedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from agno.run.base import RunStatus

from agent_kernel import (
    AuthorityEnvelope,
    Completed,
    ExecutionPlan,
    Failed,
    Pause,
    Principal,
    Role,
    TurnResult,
)
from agent_kernel.integrations.agno.agentos import KernelAgent
from agent_kernel.integrations.agno.context import AgnoRunContext


PLAN = ExecutionPlan(
    action="chat",
    label="Chat",
    capabilities=frozenset(),
    instructions=("Reply briefly.",),
    tool_call_limit=0,
    reads_history=False,
    envelope=AuthorityEnvelope(principal_id="local", allowed=frozenset()),
    reason=None,
)
PRINCIPAL = Principal(
    id="local",
    role=Role(name="local", capabilities=frozenset({"spin_wheel"})),
)


class RecordingKernel:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def run(self, message, *, principal, context=None, runtime_context=None):
        self.calls.append((message, principal, context, runtime_context))
        return TurnResult(plan=PLAN, outcome=self.outcome)


@pytest.mark.asyncio
async def test_kernel_agent_satisfies_protocol_and_forwards_transport_context():
    kernel = RecordingKernel(Completed(content="hello", raw=None))
    agent = KernelAgent(
        id="kernel-spinner",
        name="Kernel Spinner",
        kernel=kernel,
        principal=PRINCIPAL,
    )

    assert isinstance(agent, AgentProtocol)
    output = await agent.arun(
        "hi",
        stream=False,
        user_id="browser-user",
        session_id="session-7",
    )

    assert output.content == "hello"
    assert output.status is RunStatus.completed
    assert kernel.calls == [
        (
            "hi",
            PRINCIPAL,
            None,
            AgnoRunContext(user_id="browser-user", session_id="session-7"),
        )
    ]


@pytest.mark.asyncio
async def test_non_stream_without_session_id_forwards_generated_output_identity():
    kernel = RecordingKernel(Completed(content="hello", raw=None))
    agent = KernelAgent(
        id="kernel-spinner",
        name="Kernel Spinner",
        kernel=kernel,
        principal=PRINCIPAL,
    )

    output = await agent.arun(
        "hi",
        stream=False,
        user_id="browser-user",
    )

    assert output.session_id is not None
    assert len(kernel.calls) == 1
    runtime_context = kernel.calls[0][3]
    assert runtime_context.session_id == output.session_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (Failed(stage="runtime", cause=RuntimeError("provider down")), "runtime: provider down"),
        (
            Pause(
                requirements=(),
                adapter_state=object(),
                envelope=PLAN.envelope,
            ),
            "AgentOS continuation is not supported",
        ),
    ],
)
async def test_failed_and_paused_outcomes_are_agentos_error_runs(outcome, message):
    agent = KernelAgent(
        id="kernel-spinner",
        name="Kernel Spinner",
        kernel=RecordingKernel(outcome),
        principal=PRINCIPAL,
    )

    output = await agent.arun("go", stream=False)

    assert output.status is RunStatus.error
    assert message in str(output.content)


def test_kernel_and_principal_are_required():
    with pytest.raises(ValueError, match="kernel is required"):
        KernelAgent(id="missing-kernel", name="Missing", principal=PRINCIPAL)
    with pytest.raises(ValueError, match="principal is required"):
        KernelAgent(
            id="missing-principal",
            name="Missing",
            kernel=RecordingKernel(Completed(content="x", raw=SimpleNamespace())),
        )


async def collect_events(agent, message="spin"):
    return [
        event
        async for event in agent.arun(
            message,
            stream=True,
            user_id="browser-user",
            session_id="session-7",
        )
    ]


@pytest.mark.asyncio
async def test_stream_wraps_content_with_agentos_lifecycle_events():
    kernel = RecordingKernel(Completed(content={"colour": "blue"}, raw=None))
    agent = KernelAgent(
        id="kernel-spinner",
        name="Kernel Spinner",
        kernel=kernel,
        principal=PRINCIPAL,
    )

    events = await collect_events(agent)

    assert [type(event) for event in events] == [
        RunStartedEvent,
        RunContentEvent,
        RunCompletedEvent,
    ]
    assert events[1].content == "{'colour': 'blue'}"
    assert events[2].content == "{'colour': 'blue'}"
    assert len(kernel.calls) == 1


@pytest.mark.asyncio
async def test_stream_reconstructs_tool_events_before_final_content():
    tool = ToolExecution(
        tool_call_id="tool-7",
        tool_name="spin_wheel",
        tool_args={},
        result="blue",
    )
    raw = SimpleNamespace(tools=[tool])
    kernel = RecordingKernel(Completed(content="The wheel chose blue.", raw=raw))
    agent = KernelAgent(
        id="kernel-spinner",
        name="Kernel Spinner",
        kernel=kernel,
        principal=PRINCIPAL,
    )

    events = await collect_events(agent)

    assert [type(event) for event in events] == [
        RunStartedEvent,
        ToolCallStartedEvent,
        ToolCallCompletedEvent,
        RunContentEvent,
        RunCompletedEvent,
    ]
    assert events[1].tool.tool_name == "spin_wheel"
    assert events[1].tool.result is None
    assert events[2].tool.tool_call_id == "tool-7"
    assert events[2].tool.result == "blue"
    assert events[3].content == "The wheel chose blue."
    assert len(kernel.calls) == 1


@pytest.mark.asyncio
async def test_stream_without_session_id_uses_one_non_null_event_identity():
    tool = ToolExecution(
        tool_call_id="tool-7",
        tool_name="spin_wheel",
        tool_args={},
        result="blue",
    )
    kernel = RecordingKernel(
        Completed(content="The wheel chose blue.", raw=SimpleNamespace(tools=[tool]))
    )
    agent = KernelAgent(
        id="kernel-spinner",
        name="Kernel Spinner",
        kernel=kernel,
        principal=PRINCIPAL,
    )

    events = [
        event
        async for event in agent.arun(
            "spin",
            stream=True,
            user_id="browser-user",
        )
    ]

    assert len({event.run_id for event in events}) == 1
    assert events[0].run_id is not None
    assert len({event.session_id for event in events}) == 1
    assert events[0].session_id is not None
    assert len(kernel.calls) == 1


@pytest.mark.asyncio
async def test_stream_turn_failure_emits_error_not_content():
    kernel = RecordingKernel(Failed(stage="runtime", cause=RuntimeError("down")))
    agent = KernelAgent(
        id="kernel-spinner",
        name="Kernel Spinner",
        kernel=kernel,
        principal=PRINCIPAL,
    )

    events = await collect_events(agent)

    assert [type(event) for event in events] == [RunStartedEvent, RunErrorEvent]
    assert "runtime: down" in str(events[-1].content)
    assert len(kernel.calls) == 1
