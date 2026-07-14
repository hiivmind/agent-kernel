from collections.abc import Callable
from typing import Any

from agent_kernel.core.results import AuthorityEnvelope


def authority_hook(
    *,
    function_name: str,
    run_context: object,
    args: dict[str, Any],
    function_call: Callable[..., Any],
) -> Any:
    dependencies = getattr(run_context, "dependencies", None) or {}
    envelope = dependencies.get("agent_kernel_authority")
    if (
        not isinstance(envelope, AuthorityEnvelope)
        or function_name not in envelope.allowed
    ):
        raise PermissionError(f"principal may not call {function_name}")
    return function_call(**args)
