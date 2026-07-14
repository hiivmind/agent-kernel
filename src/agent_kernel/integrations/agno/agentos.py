import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from agno.agents.base import BaseExternalAgent
from agno.models.response import ToolExecution
from agno.run.agent import (
    RunContentEvent,
    RunOutputEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)

from agent_kernel.core.kernel import Kernel
from agent_kernel.core.results import Completed, Failed, Pause, Principal, TurnResult
from agent_kernel.integrations.agno.context import AgnoRunContext


class KernelAgentError(RuntimeError):
    """Raised when a kernel turn cannot be represented as an AgentOS success."""


class KernelAgentPauseUnsupported(KernelAgentError):
    """Raised when a kernel pause reaches the v1 AgentOS hosting adapter."""


@dataclass
class KernelAgent(BaseExternalAgent):
    kernel: Kernel[Any] | None = None
    principal: Principal | None = None
    framework: str = "agent-kernel"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kernel is None:
            raise ValueError("kernel is required")
        if self.principal is None:
            raise ValueError("principal is required")

    @staticmethod
    def _completed(result: TurnResult) -> Completed:
        outcome = result.outcome
        if isinstance(outcome, Completed):
            return outcome
        if isinstance(outcome, Failed):
            raise KernelAgentError(f"{outcome.stage}: {outcome.cause}") from outcome.cause
        if isinstance(outcome, Pause):
            raise KernelAgentPauseUnsupported(
                "AgentOS continuation is not supported by KernelAgent"
            )
        raise KernelAgentError(f"unsupported kernel outcome: {type(outcome).__name__}")

    async def _execute_kernel(
        self,
        input: Any,
        *,
        user_id: str | None,
        session_id: str | None,
    ) -> Completed:
        assert self.kernel is not None
        assert self.principal is not None
        result = await asyncio.to_thread(
            self.kernel.run,
            str(input),
            principal=self.principal,
            runtime_context=AgnoRunContext(
                user_id=user_id,
                session_id=session_id,
            ),
        )
        return self._completed(result)

    async def _arun_adapter(self, input: Any, **kwargs: Any) -> str:
        completed = await self._execute_kernel(
            input,
            user_id=kwargs.get("user_id"),
            session_id=kwargs.get("session_id"),
        )
        return str(completed.content)

    @staticmethod
    def _tools(completed: Completed) -> tuple[ToolExecution, ...]:
        raw_tools = getattr(completed.raw, "tools", None)
        if not isinstance(raw_tools, (list, tuple)):
            return ()
        return tuple(tool for tool in raw_tools if isinstance(tool, ToolExecution))

    async def _arun_stream(
        self,
        input: Any,
        **kwargs: Any,
    ) -> AsyncIterator[RunOutputEvent]:
        if not kwargs.get("session_id"):
            kwargs["session_id"] = str(uuid4())
        async for event in super()._arun_stream(input, **kwargs):
            yield event

    async def _arun_adapter_stream(
        self,
        input: Any,
        **kwargs: Any,
    ) -> AsyncIterator[RunOutputEvent]:
        completed = await self._execute_kernel(
            input,
            user_id=kwargs.get("user_id"),
            session_id=kwargs.get("session_id"),
        )
        run_id = kwargs.get("run_id")
        session_id = kwargs.get("session_id")
        for recorded in self._tools(completed):
            tool_call_id = recorded.tool_call_id or str(uuid4())
            yield ToolCallStartedEvent(
                run_id=run_id,
                session_id=session_id,
                agent_id=self.get_id(),
                agent_name=self.name or "",
                tool=ToolExecution(
                    tool_call_id=tool_call_id,
                    tool_name=recorded.tool_name,
                    tool_args=recorded.tool_args,
                ),
            )
            yield ToolCallCompletedEvent(
                run_id=run_id,
                session_id=session_id,
                agent_id=self.get_id(),
                agent_name=self.name or "",
                tool=ToolExecution(
                    tool_call_id=tool_call_id,
                    tool_name=recorded.tool_name,
                    tool_args=recorded.tool_args,
                    result=recorded.result,
                ),
            )
        yield RunContentEvent(
            run_id=run_id,
            session_id=session_id,
            agent_id=self.get_id(),
            agent_name=self.name or "",
            content=str(completed.content),
        )
