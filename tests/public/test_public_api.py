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


def test_readme_uses_pypi_safe_absolute_documentation_links():
    from pathlib import Path

    contents = (Path(__file__).parents[2] / "README.md").read_text(encoding="utf-8")

    assert "](docs/" not in contents
    assert "https://github.com/hiivmind/agent-kernel/blob/main/docs/" in contents
