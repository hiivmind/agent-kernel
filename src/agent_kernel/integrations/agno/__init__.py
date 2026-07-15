"""Agno integration for structured classification and bounded execution."""

from agent_kernel.integrations.agno.classifier import AgnoClassifier
from agent_kernel.integrations.agno.context import AgnoRunContext
from agent_kernel.integrations.agno.agentos import (
    KernelAgent,
    KernelAgentError,
    KernelAgentPauseUnsupported,
)
from agent_kernel.integrations.agno.hitl import UnsupportedRequirement
from agent_kernel.integrations.agno.hooks import authority_hook
from agent_kernel.integrations.agno.runtime import AgnoRuntime
from agent_kernel.integrations.agno.trusted import (
    AgentOSPrincipalResolver,
    AgentOSTrustedStateMiddleware,
    TrustedAgentOSIdentity,
)

__all__ = [
    "AgnoClassifier",
    "AgnoRunContext",
    "AgnoRuntime",
    "AgentOSPrincipalResolver",
    "AgentOSTrustedStateMiddleware",
    "KernelAgent",
    "KernelAgentError",
    "KernelAgentPauseUnsupported",
    "TrustedAgentOSIdentity",
    "UnsupportedRequirement",
    "authority_hook",
]
