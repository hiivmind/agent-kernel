from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agno.agent import Agent

from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.results import (
    AuthorityEnvelope,
    Completed,
    ExecutionPlan,
    Failed,
    Pause,
    RuntimeOutcome,
)
from agent_kernel.integrations.agno.context import AgnoRunContext
from agent_kernel.integrations.agno.hitl import (
    UnsupportedRequirement,
    apply_user_input,
    requirement_state,
    translate_requirements,
)
from agent_kernel.integrations.agno.hooks import authority_hook


@dataclass(frozen=True)
class _PauseRecord:
    response: object
    agent: Any
    envelope: AuthorityEnvelope
    context: AgnoRunContext | None
    requirement_state: tuple[object, ...]


class _PauseToken:
    __slots__ = ()


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
        self._pause_records: dict[_PauseToken, _PauseRecord] = {}

    def _pause(
        self,
        response: object,
        *,
        agent: Any,
        envelope: AuthorityEnvelope,
        context: AgnoRunContext | None,
    ) -> Pause:
        requirements = translate_requirements(response)
        token = _PauseToken()
        self._pause_records[token] = _PauseRecord(
            response=response,
            agent=agent,
            envelope=envelope,
            context=context,
            requirement_state=requirement_state(response),
        )
        return Pause(
            requirements=requirements,
            adapter_state=token,
            envelope=envelope,
            runtime_context=context,
        )

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
            selected_tools = [self.tools[capability] for capability in sorted(plan.capabilities)]
            agent = self.agent_factory(
                model=self.model,
                tools=selected_tools,
                db=self.db,
                instructions=list(plan.instructions),
                tool_call_limit=plan.tool_call_limit,
                tool_hooks=[authority_hook],
            )
            run_kwargs: dict[str, object] = {
                "dependencies": {
                    "agent_kernel_authority": plan.envelope,
                }
            }
            if context is None:
                response = agent.run(message, **run_kwargs)
            else:
                run_kwargs.update(
                    {
                        "user_id": context.user_id,
                        "session_id": context.session_id,
                        "session_state": context.session_state,
                        "metadata": context.metadata,
                    }
                )
                response = agent.run(
                    message,
                    **run_kwargs,
                )
            if getattr(response, "is_paused", False):
                return self._pause(
                    response,
                    agent=agent,
                    envelope=plan.envelope,
                    context=context,
                )
            return Completed(content=response.content, raw=response)
        except Exception as exc:
            return Failed(stage="runtime", cause=exc)

    def resume(
        self,
        pause: Pause,
        answers: dict[str, str],
    ) -> RuntimeOutcome:
        record_key = pause.adapter_state
        if not isinstance(record_key, _PauseToken):
            raise ConfigurationError("pause state was not created by this Agno runtime")
        record = self._pause_records.get(record_key)
        if record is None:
            raise ConfigurationError("pause state was not created by this Agno runtime")
        response = record.response
        if pause.envelope != record.envelope:
            raise ConfigurationError("pause authority envelope does not match the original run")
        if requirement_state(response) != record.requirement_state:
            raise ConfigurationError("paused Agno response state does not match the original run")

        try:
            apply_user_input(response, answers)
            resumed = record.agent.continue_run(
                run_response=response,
                dependencies={
                    "agent_kernel_authority": record.envelope,
                },
                user_id=(record.context.user_id if record.context is not None else None),
            )
            if getattr(resumed, "is_paused", False):
                outcome = self._pause(
                    resumed,
                    agent=record.agent,
                    envelope=record.envelope,
                    context=record.context,
                )
                self._pause_records.pop(record_key, None)
                return outcome

            self._pause_records.pop(record_key, None)
            return Completed(content=resumed.content, raw=resumed)
        except UnsupportedRequirement as exc:
            self._pause_records.pop(record_key, None)
            return Failed(stage="resume", cause=exc)
        except Exception as exc:
            return Failed(stage="resume", cause=exc)
