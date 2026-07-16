from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from agent_kernel.core.operations import OperationInvocation
from agent_kernel.core.results import AuthorityEnvelope
from agent_kernel.integrations.agno.context import AgnoRunContext


@dataclass(frozen=True)
class AgnoCapabilityBinding:
    capability: str
    tools: tuple[object, ...] = ()
    function_names: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if any(not function_name.strip() for function_name in self.function_names):
            raise ValueError("function names must be non-empty")


@dataclass(frozen=True)
class AgnoInvocationContext:
    run: AgnoRunContext | None = None
    invocation_handle: object | None = None


class AgnoBindingProvider(Protocol):
    def bindings_for(
        self,
        invocation: OperationInvocation[object],
        context: AgnoInvocationContext,
    ) -> Mapping[str, AgnoCapabilityBinding]:
        raise NotImplementedError


def capability_hook(
    function_capabilities: Mapping[str, str],
    *,
    internal_functions: frozenset[str] = frozenset(),
) -> Callable[..., Any]:
    if any(not function_name.strip() for function_name in function_capabilities):
        raise ValueError("function names must be non-empty")
    if any(not function_name.strip() for function_name in internal_functions):
        raise ValueError("function names must be non-empty")

    declared_capabilities = dict(function_capabilities)

    def hook(
        *,
        function_name: str,
        run_context: object,
        args: dict[str, Any],
        function_call: Callable[..., Any],
    ) -> Any:
        if function_name in internal_functions:
            return function_call(**args)

        dependencies = getattr(run_context, "dependencies", None) or {}
        envelope = dependencies.get("agent_kernel_authority")
        required_capability = declared_capabilities.get(function_name)
        if (
            not isinstance(envelope, AuthorityEnvelope)
            or required_capability is None
            or required_capability not in envelope.allowed
        ):
            raise PermissionError(f"principal may not call {function_name}")
        return function_call(**args)

    return hook
