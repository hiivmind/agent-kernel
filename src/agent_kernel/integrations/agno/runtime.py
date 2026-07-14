from collections.abc import Callable
from typing import Any

from agno.agent import Agent

from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.results import (
    Completed,
    ExecutionPlan,
    Failed,
    Pause,
    RuntimeOutcome,
)
from agent_kernel.integrations.agno.context import AgnoRunContext


class AgnoRuntime:
    def __init__(
        self,
        *,
        model: object | None = None,
        tools: dict[str, object],
        db: object | None = None,
        agent_factory: Callable[..., Any] = Agent,
    ):
        self.model = model
        self.tools = dict(tools)
        self.db = db
        self.agent_factory = agent_factory

    def execute(
        self,
        message: str,
        plan: ExecutionPlan,
        *,
        context: AgnoRunContext | None = None,
    ) -> RuntimeOutcome:
        missing = sorted(plan.capabilities.difference(self.tools))
        if missing:
            raise ConfigurationError(f"missing Agno tool bindings: {missing}")

        try:
            selected_tools = [
                self.tools[capability]
                for capability in sorted(plan.capabilities)
            ]
            agent = self.agent_factory(
                model=self.model,
                tools=selected_tools,
                db=self.db,
                instructions=list(plan.instructions),
                tool_call_limit=plan.tool_call_limit,
            )
            if context is None:
                response = agent.run(message)
            else:
                response = agent.run(
                    message,
                    user_id=context.user_id,
                    session_id=context.session_id,
                    session_state=context.session_state,
                    metadata=context.metadata,
                )
            if getattr(response, "is_paused", False):
                return Pause(
                    requirements=(),
                    adapter_state=response,
                    envelope=plan.envelope,
                    runtime_context=context,
                )
            return Completed(content=response.content, raw=response)
        except Exception as exc:
            return Failed(stage="runtime", cause=exc)
