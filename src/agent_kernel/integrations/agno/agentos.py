import asyncio
from dataclasses import dataclass
from typing import Any

from agno.agents.base import BaseExternalAgent

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
