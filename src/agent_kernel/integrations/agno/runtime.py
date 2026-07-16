from collections.abc import Callable, Mapping
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
class _LifecycleState:
    envelope: AuthorityEnvelope
    context: AgnoRunContext | None
    dependencies: Mapping[str, object]
    completion_adapter: Callable[[object, object], Completed | Failed]
    operation_state: object | None


@dataclass(frozen=True)
class _PauseRecord:
    response: object
    agent: Any
    state: _LifecycleState
    requirement_state: tuple[object, ...]


class _PauseToken:
    __slots__ = ()


class _AgnoLifecycle:
    def __init__(self) -> None:
        self._pause_records: dict[_PauseToken, _PauseRecord] = {}

    def _pause(
        self,
        response: object,
        *,
        agent: Any,
        state: _LifecycleState,
    ) -> Pause:
        requirements = translate_requirements(response)
        token = _PauseToken()
        self._pause_records[token] = _PauseRecord(
            response=response,
            agent=agent,
            state=state,
            requirement_state=requirement_state(response),
        )
        return Pause(
            requirements=requirements,
            adapter_state=token,
            envelope=state.envelope,
            runtime_context=state.context,
        )

    def start(
        self,
        agent: Any,
        message: object,
        state: _LifecycleState,
        *,
        run_kwargs: Mapping[str, object] | None = None,
        failure_stage: str,
        persistence_failure_stage: str | None = None,
    ) -> RuntimeOutcome:
        try:
            kwargs = dict(run_kwargs or {})
            kwargs["dependencies"] = dict(state.dependencies)
            if state.context is not None:
                kwargs.update(
                    {
                        "user_id": state.context.user_id,
                        "session_id": state.context.session_id,
                        "session_state": state.context.session_state,
                        "metadata": state.context.metadata,
                    }
                )
            response = agent.run(message, **kwargs)
        except Exception as exc:
            return Failed(stage=failure_stage, cause=exc)
        if getattr(response, "is_paused", False):
            try:
                return self._pause(response, agent=agent, state=state)
            except Exception as exc:
                return Failed(
                    stage=persistence_failure_stage or failure_stage,
                    cause=exc,
                )
        try:
            return state.completion_adapter(response.content, response)
        except Exception as exc:
            return Failed(stage=failure_stage, cause=exc)

    def resume(
        self,
        pause: Pause,
        answers: dict[str, str],
        *,
        failure_stage: str,
        persistence_failure_stage: str | None = None,
    ) -> RuntimeOutcome:
        record_key = pause.adapter_state
        if not isinstance(record_key, _PauseToken):
            raise ConfigurationError(
                "pause state was not created by this Agno runtime"
            )
        record = self._pause_records.get(record_key)
        if record is None:
            raise ConfigurationError(
                "pause state was not created by this Agno runtime"
            )
        response = record.response
        if pause.envelope != record.state.envelope:
            raise ConfigurationError(
                "pause authority envelope does not match the original run"
            )
        if requirement_state(response) != record.requirement_state:
            raise ConfigurationError(
                "paused Agno response state does not match the original run"
            )

        try:
            apply_user_input(response, answers)
            continue_kwargs: dict[str, object] = {
                "dependencies": dict(record.state.dependencies),
                "user_id": (
                    record.state.context.user_id
                    if record.state.context is not None
                    else None
                ),
            }
            operation_resume_kwargs = getattr(
                record.state.operation_state,
                "resume_kwargs",
                None,
            )
            if isinstance(operation_resume_kwargs, Mapping):
                continue_kwargs.update(operation_resume_kwargs)
            resumed = record.agent.continue_run(
                run_response=response,
                **continue_kwargs,
            )
            if getattr(resumed, "is_paused", False):
                try:
                    outcome = self._pause(
                        resumed,
                        agent=record.agent,
                        state=record.state,
                    )
                except Exception as exc:
                    return Failed(
                        stage=persistence_failure_stage or failure_stage,
                        cause=exc,
                    )
                self._pause_records.pop(record_key, None)
                return outcome

            self._pause_records.pop(record_key, None)
            return record.state.completion_adapter(resumed.content, resumed)
        except UnsupportedRequirement as exc:
            self._pause_records.pop(record_key, None)
            return Failed(stage=failure_stage, cause=exc)
        except Exception as exc:
            return Failed(stage=failure_stage, cause=exc)


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
        self._lifecycle = _AgnoLifecycle()

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
                tool_hooks=[authority_hook],
            )
            state = _LifecycleState(
                envelope=plan.envelope,
                context=context,
                dependencies={"agent_kernel_authority": plan.envelope},
                completion_adapter=lambda content, raw: Completed(
                    content=content,
                    raw=raw,
                ),
                operation_state=None,
            )
            return self._lifecycle.start(
                agent,
                message,
                state,
                failure_stage="runtime",
            )
        except Exception as exc:
            return Failed(stage="runtime", cause=exc)

    def resume(
        self,
        pause: Pause,
        answers: dict[str, str],
    ) -> RuntimeOutcome:
        return self._lifecycle.resume(
            pause,
            answers,
            failure_stage="resume",
        )
