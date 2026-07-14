from types import SimpleNamespace

import pytest
from agno.agent.protocol import AgentProtocol
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
