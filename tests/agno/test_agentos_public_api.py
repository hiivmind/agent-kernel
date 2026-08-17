def test_supported_agno_api_includes_agentos_hosting_adapter():
    from agent_kernel.integrations.agno import (
        KernelAgent,
        KernelAgentError,
        KernelAgentPauseUnsupported,
    )

    assert all((KernelAgent, KernelAgentError, KernelAgentPauseUnsupported))
