from types import SimpleNamespace

import pytest

from agent_kernel.core.results import AuthorityEnvelope
from agent_kernel.integrations.agno import authority_hook


def test_authority_hook_calls_planned_function_with_supplied_arguments():
    envelope = AuthorityEnvelope(
        principal_id="member-1",
        allowed=frozenset({"spin_wheel"}),
    )
    run_context = SimpleNamespace(
        dependencies={"agent_kernel_authority": envelope},
    )
    calls = []

    result = authority_hook(
        function_name="spin_wheel",
        run_context=run_context,
        args={"turns": 3},
        function_call=lambda **kwargs: calls.append(kwargs) or "spun",
    )

    assert result == "spun"
    assert calls == [{"turns": 3}]


def test_authority_hook_rejects_unplanned_function_before_calling_it():
    envelope = AuthorityEnvelope(
        principal_id="member-1",
        allowed=frozenset({"spin_wheel"}),
    )
    run_context = SimpleNamespace(
        dependencies={"agent_kernel_authority": envelope},
    )
    calls = []

    with pytest.raises(
        PermissionError,
        match="principal may not call delete_world",
    ):
        authority_hook(
            function_name="delete_world",
            run_context=run_context,
            args={},
            function_call=lambda **kwargs: calls.append(kwargs),
        )

    assert calls == []


@pytest.mark.parametrize(
    "run_context",
    [
        SimpleNamespace(),
        SimpleNamespace(dependencies={}),
        SimpleNamespace(
            dependencies={"agent_kernel_authority": object()},
        ),
    ],
)
def test_authority_hook_fails_closed_without_valid_envelope(run_context):
    calls = []

    with pytest.raises(
        PermissionError,
        match="principal may not call spin_wheel",
    ):
        authority_hook(
            function_name="spin_wheel",
            run_context=run_context,
            args={},
            function_call=lambda **kwargs: calls.append(kwargs),
        )

    assert calls == []
