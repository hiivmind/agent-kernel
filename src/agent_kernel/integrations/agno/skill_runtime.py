from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, TypeVar, cast

from agno.agent import Agent
from agno.run.base import RunStatus

from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.operations import (
    OperationCompletion,
    OperationContract,
    OperationInvocation,
)
from agent_kernel.core.results import Completed, Failed, Pause, RuntimeOutcome
from agent_kernel.integrations.agno.bindings import (
    AgnoBindingProvider,
    AgnoCapabilityBinding,
    AgnoInvocationContext,
    capability_hook,
    function_capability_map,
)
from agent_kernel.integrations.agno.runtime import _AgnoLifecycle, _LifecycleState
from agent_kernel.integrations.agno.skills import AgnoSkillProvider, SafeSkills

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class _SkillOperationState:
    invocation_id: str
    contract: OperationContract[Any, Any]
    resume_kwargs: Mapping[str, object]


class AgnoSkillRuntime:
    def __init__(
        self,
        *,
        skill_provider: AgnoSkillProvider,
        binding_provider: AgnoBindingProvider,
        model: object | None = None,
        db: object | None = None,
        agent_factory: Callable[..., object] = Agent,
    ) -> None:
        self._skill_provider = skill_provider
        self._binding_provider = binding_provider
        self._model = model
        self._db = db
        self._agent_factory = agent_factory
        self._lifecycle = _AgnoLifecycle()

    def execute(
        self,
        invocation: OperationInvocation[InputT],
        contract: OperationContract[InputT, OutputT],
        *,
        context: AgnoInvocationContext | None = None,
    ) -> RuntimeOutcome:
        try:
            validated_inputs = contract.input_adapter.validate_python(invocation.inputs)
            validated_invocation = replace(invocation, inputs=validated_inputs)
        except Exception as exc:
            return Failed(stage="input-validation", cause=exc)

        try:
            source = self._skill_provider.require(validated_invocation.target_id)
            if source.id != validated_invocation.target_id:
                raise ConfigurationError(
                    "Agno Skill provider returned "
                    f"{source.id!r} for target {validated_invocation.target_id!r}"
                )
            skill = SafeSkills.from_source(source)
            loaded_skill_names = skill.get_skill_names()
            if len(loaded_skill_names) != 1:
                raise ConfigurationError(
                    "Agno Skill source must load exactly one Skill; "
                    f"loaded {loaded_skill_names!r}"
                )
            if loaded_skill_names[0] != source.id:
                raise ConfigurationError(
                    "loaded Agno Skill name "
                    f"{loaded_skill_names[0]!r} does not match selected target "
                    f"{source.id!r}"
                )
        except Exception as exc:
            return Failed(stage="skill-load", cause=exc)

        invocation_context = context or AgnoInvocationContext()
        try:
            available_bindings = self._binding_provider.bindings_for(
                cast(OperationInvocation[object], validated_invocation),
                invocation_context,
            )
            selected_bindings = self._select_bindings(
                validated_invocation,
                available_bindings,
            )
            selected_tools = [
                tool
                for capability in sorted(selected_bindings)
                for tool in selected_bindings[capability].tools
            ]
            function_capabilities = function_capability_map(selected_bindings)
            internal_functions = frozenset(
                tool.name for tool in skill.get_tools()
            )
            hook = capability_hook(
                function_capabilities,
                internal_functions=internal_functions,
            )
        except Exception as exc:
            return Failed(stage="configuration", cause=exc)

        try:
            message = contract.model_input_builder(validated_invocation)
        except Exception as exc:
            return Failed(stage="input-validation", cause=exc)

        operation_state = _SkillOperationState(
            invocation_id=validated_invocation.invocation_id,
            contract=cast(OperationContract[Any, Any], contract),
            resume_kwargs={
                "run_id": validated_invocation.invocation_id,
                "user_id": (
                    invocation_context.run.user_id
                    if invocation_context.run is not None
                    else None
                ),
                "session_id": (
                    invocation_context.run.session_id
                    if invocation_context.run is not None
                    else None
                ),
                "session_state": (
                    invocation_context.run.session_state
                    if invocation_context.run is not None
                    else None
                ),
                "metadata": (
                    invocation_context.run.metadata
                    if invocation_context.run is not None
                    else None
                ),
            },
        )

        def complete(content: object, raw: object) -> Completed | Failed:
            tool_failure = self._tool_failure(raw)
            if tool_failure is not None:
                return Failed(stage="tool-execution", cause=tool_failure)
            if getattr(raw, "status", None) == RunStatus.error:
                return Failed(
                    stage="provider-execution",
                    cause=self._response_error(raw),
                )
            try:
                output = operation_state.contract.output_adapter.validate_python(
                    content
                )
            except Exception as exc:
                return Failed(stage="output-validation", cause=exc)
            completion = OperationCompletion(
                invocation_id=operation_state.invocation_id,
                output=output,
                raw=raw,
            )
            return Completed(content=completion, raw=raw)

        state = _LifecycleState(
            envelope=validated_invocation.plan.envelope,
            context=invocation_context.run,
            dependencies={
                "agent_kernel_authority": validated_invocation.plan.envelope,
                "agent_kernel_invocation_handle": invocation_context.invocation_handle,
            },
            completion_adapter=complete,
            operation_state=operation_state,
        )
        try:
            agent = self._agent_factory(
                model=self._model,
                db=self._db,
                skills=skill,
                tools=selected_tools,
                output_schema=contract.model_output_type,
                instructions=list(validated_invocation.plan.instructions),
                tool_call_limit=validated_invocation.plan.tool_call_limit,
                tool_hooks=[hook],
            )
        except Exception as exc:
            return Failed(stage="provider-execution", cause=exc)

        return self._lifecycle.start(
            agent,
            message,
            state,
            run_kwargs={"run_id": validated_invocation.invocation_id},
            failure_stage="provider-execution",
        )

    def resume(
        self,
        pause: Pause,
        answers: dict[str, str],
    ) -> RuntimeOutcome:
        return self._lifecycle.resume(
            pause,
            answers,
            failure_stage="provider-execution",
        )

    @staticmethod
    def _select_bindings(
        invocation: OperationInvocation[Any],
        bindings: Mapping[str, AgnoCapabilityBinding],
    ) -> dict[str, AgnoCapabilityBinding]:
        missing = sorted(invocation.plan.capabilities.difference(bindings))
        extra = sorted(set(bindings).difference(invocation.plan.capabilities))
        if missing or extra:
            raise ConfigurationError(
                "Agno capability bindings must exactly match the plan; "
                f"missing: {missing}; extra: {extra}"
            )
        selected = {
            capability: bindings[capability]
            for capability in sorted(invocation.plan.capabilities)
        }
        mismatched = sorted(
            capability
            for capability, binding in selected.items()
            if binding.capability != capability
        )
        if mismatched:
            raise ConfigurationError(
                f"Agno capability binding keys do not match: {mismatched}"
            )
        return selected

    @staticmethod
    def _tool_failure(response: object) -> Exception | None:
        for tool in getattr(response, "tools", None) or ():
            if getattr(tool, "tool_call_error", False):
                result = getattr(tool, "result", None)
                if isinstance(result, Exception):
                    return result
                return RuntimeError(str(result or "Agno tool execution failed"))
        return None

    @staticmethod
    def _response_error(response: object) -> Exception:
        content = getattr(response, "content", None)
        if isinstance(content, Exception):
            return content
        return RuntimeError(str(content or "Agno provider execution failed"))
