from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import iscoroutinefunction
from typing import Any, Protocol

from agno.tools.function import Function
from agno.tools.toolkit import Toolkit

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


def validate_bound_function_names(
    bindings: Mapping[str, AgnoCapabilityBinding],
    *,
    reserved_functions: frozenset[str],
) -> None:
    owners: dict[str, str] = {}
    for capability in sorted(bindings):
        binding = bindings[capability]
        actual: set[str] = set()
        for tool in binding.tools:
            if isinstance(tool, Function):
                if iscoroutinefunction(tool.entrypoint):
                    raise ValueError(
                        "Agno Skill runtime does not support asynchronous "
                        "application tools"
                    )
                names = {tool.name}
            elif isinstance(tool, Toolkit):
                if tool.async_functions:
                    raise ValueError(
                        "Agno Skill runtime does not support asynchronous "
                        "Toolkit functions"
                    )
                names = {
                    function.name
                    for function in tool.functions.values()
                }
            elif callable(tool):
                if iscoroutinefunction(tool):
                    raise ValueError(
                        "Agno Skill runtime does not support asynchronous "
                        "application tools"
                    )
                name = getattr(tool, "__name__", None)
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        f"bound callable for {capability!r} has no exposed name"
                    )
                names = {name}
            else:
                raise TypeError(
                    f"unsupported Agno tool binding for {capability!r}: "
                    f"{type(tool).__name__}"
                )
            duplicates = sorted(actual.intersection(names))
            if duplicates:
                raise ValueError(
                    f"duplicate actual function names for {capability!r}: {duplicates}"
                )
            actual.update(names)

        collisions = sorted(actual.intersection(reserved_functions))
        if collisions:
            raise ValueError(
                "application function names collide with internal functions: "
                f"{collisions}"
            )
        if actual != set(binding.function_names):
            raise ValueError(
                f"declared function names do not match actual functions for "
                f"{capability!r}; declared: {sorted(binding.function_names)}; "
                f"actual: {sorted(actual)}"
            )
        for name in sorted(actual):
            owner = owners.get(name)
            if owner is not None:
                raise ValueError(
                    f"duplicate actual function name {name!r} in "
                    f"{owner!r} and {capability!r}"
                )
            owners[name] = capability
def function_capability_map(
    bindings: Mapping[str, AgnoCapabilityBinding],
) -> dict[str, str]:
    function_capabilities: dict[str, str] = {}
    for selected_capability in sorted(bindings):
        binding = bindings[selected_capability]
        for function_name in sorted(binding.function_names):
            if function_name in function_capabilities:
                raise ValueError(f"duplicate function name: {function_name}")
            function_capabilities[function_name] = binding.capability
    return function_capabilities


def capability_hook(
    function_capabilities: Mapping[str, str],
    *,
    internal_functions: frozenset[str] = frozenset(),
) -> Callable[..., Any]:
    if any(not function_name.strip() for function_name in function_capabilities):
        raise ValueError("function names must be non-empty")
    if any(not function_name.strip() for function_name in internal_functions):
        raise ValueError("function names must be non-empty")
    collisions = sorted(set(function_capabilities).intersection(internal_functions))
    if collisions:
        raise ValueError(
            "application function names collide with internal functions: "
            f"{collisions}"
        )

    declared_capabilities = dict(function_capabilities)

    def authorize(function_name: str, run_context: object) -> None:
        if function_name in internal_functions:
            return
        dependencies = getattr(run_context, "dependencies", None) or {}
        envelope = dependencies.get("agent_kernel_authority")
        required_capability = declared_capabilities.get(function_name)
        if (
            not isinstance(envelope, AuthorityEnvelope)
            or required_capability is None
            or required_capability not in envelope.allowed
        ):
            raise PermissionError(f"principal may not call {function_name}")

    def hook(
        *,
        function_name: str,
        run_context: object,
        args: dict[str, Any],
        function_call: Callable[..., Any],
    ) -> Any:
        authorize(function_name, run_context)
        return function_call(**args)

    return hook
