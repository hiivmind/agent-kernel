def test_supported_core_api_is_top_level():
    from agent_kernel import (
        ActionSpec,
        Briefing,
        Completed,
        Continuation,
        Failed,
        Intent,
        Kernel,
        KernelConfig,
        Pause,
        Principal,
        Registry,
        Role,
        TurnResult,
    )

    assert all(
        (
            ActionSpec,
            Briefing,
            Completed,
            Continuation,
            Failed,
            Intent,
            Kernel,
            KernelConfig,
            Pause,
            Principal,
            Registry,
            Role,
            TurnResult,
        )
    )
