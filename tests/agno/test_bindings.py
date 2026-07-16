import asyncio
from types import SimpleNamespace

import pytest

from agent_kernel.core.results import AuthorityEnvelope
from agent_kernel.integrations.agno.bindings import (
    AgnoCapabilityBinding,
    capability_hook,
    function_capability_map,
)


SAFE_SKILL_READERS = frozenset(
    {
        "get_skill_instructions",
        "get_skill_reference",
        "get_skill_script_source",
    }
)


def test_authorization_only_binding_declares_no_tools_or_functions():
    binding = AgnoCapabilityBinding("history:read")

    assert binding.tools == ()
    assert binding.function_names == frozenset()


def test_capability_binding_can_declare_multiple_tool_functions():
    binding = AgnoCapabilityBinding(
        "resources:read:corpus-agno",
        tools=(object(), object()),
        function_names=frozenset({"read_file", "fetch_url"}),
    )

    assert binding.function_names == frozenset({"read_file", "fetch_url"})


@pytest.mark.parametrize("function_name", ["", " ", "\t\n"])
def test_capability_binding_rejects_empty_function_name(function_name):
    with pytest.raises(ValueError, match="function names must be non-empty"):
        AgnoCapabilityBinding(
            "resources:read:corpus-agno",
            function_names=frozenset({function_name}),
        )


def test_function_capability_map_rejects_duplicate_name_across_bindings():
    bindings = {
        "resources:read:corpus-agno": AgnoCapabilityBinding(
            "resources:read:corpus-agno",
            function_names=frozenset({"read_file"}),
        ),
        "resources:read:corpus-dspy": AgnoCapabilityBinding(
            "resources:read:corpus-dspy",
            function_names=frozenset({"read_file"}),
        ),
    }

    with pytest.raises(ValueError, match="duplicate function name: read_file"):
        function_capability_map(bindings)


def test_capability_hook_rejects_application_collision_with_internal_reader():
    with pytest.raises(
        ValueError,
        match="application function names collide with internal functions",
    ):
        capability_hook(
            {"get_skill_instructions": "resources:delete:everything"},
            internal_functions=SAFE_SKILL_READERS,
        )


def test_capability_hook_maps_function_to_declared_capability():
    hook = capability_hook(
        {
            "read_file": "resources:read:corpus-agno",
            "fetch_url": "resources:read:corpus-agno",
        },
        internal_functions=frozenset({"get_skill_instructions"}),
    )
    context = SimpleNamespace(
        dependencies={
            "agent_kernel_authority": AuthorityEnvelope(
                "operator", frozenset({"resources:read:corpus-agno"})
            )
        }
    )
    assert hook(
        function_name="read_file",
        run_context=context,
        args={"path": "config.yaml"},
        function_call=lambda **args: args["path"],
    ) == "config.yaml"


def test_capability_hook_awaits_async_continuation():
    envelope = AuthorityEnvelope(
        "operator",
        frozenset({"resources:read:corpus-agno"}),
    )
    hook = capability_hook(
        {"read_file": "resources:read:corpus-agno"},
        async_mode=True,
    )

    async def function_call(**args):
        await asyncio.sleep(0)
        return args["path"]

    result = asyncio.run(
        hook(
            function_name="read_file",
            run_context=SimpleNamespace(
                dependencies={"agent_kernel_authority": envelope}
            ),
            args={"path": "config.yaml"},
            function_call=function_call,
        )
    )

    assert result == "config.yaml"


@pytest.mark.parametrize("function_name", sorted(SAFE_SKILL_READERS))
def test_capability_hook_allows_explicit_safe_internal_reader(function_name):
    hook = capability_hook({}, internal_functions=SAFE_SKILL_READERS)

    assert hook(
        function_name=function_name,
        run_context=SimpleNamespace(),
        args={"value": function_name},
        function_call=lambda **args: args["value"],
    ) == function_name


@pytest.mark.parametrize(
    "run_context",
    [
        SimpleNamespace(),
        SimpleNamespace(dependencies={}),
        SimpleNamespace(dependencies={"agent_kernel_authority": object()}),
    ],
)
def test_capability_hook_rejects_declared_function_without_authority(run_context):
    hook = capability_hook({"read_file": "resources:read:corpus-agno"})
    calls = []

    with pytest.raises(PermissionError, match="principal may not call read_file"):
        hook(
            function_name="read_file",
            run_context=run_context,
            args={},
            function_call=lambda: calls.append("called"),
        )

    assert calls == []


def test_capability_hook_rejects_authority_for_different_capability():
    hook = capability_hook({"read_file": "resources:read:corpus-agno"})
    calls = []
    context = SimpleNamespace(
        dependencies={
            "agent_kernel_authority": AuthorityEnvelope(
                "operator", frozenset({"resources:write:corpus-agno"})
            )
        }
    )

    with pytest.raises(PermissionError, match="principal may not call read_file"):
        hook(
            function_name="read_file",
            run_context=context,
            args={},
            function_call=lambda: calls.append("called"),
        )

    assert calls == []


def test_capability_hook_rejects_undeclared_non_internal_function():
    hook = capability_hook({})
    calls = []

    with pytest.raises(PermissionError, match="principal may not call delete_world"):
        hook(
            function_name="delete_world",
            run_context=SimpleNamespace(),
            args={},
            function_call=lambda: calls.append("called"),
        )

    assert calls == []


@pytest.mark.parametrize("function_name", ["", " ", "\t\n"])
def test_capability_hook_rejects_empty_declared_function_name(function_name):
    with pytest.raises(ValueError, match="function names must be non-empty"):
        capability_hook({function_name: "resources:read:corpus-agno"})


@pytest.mark.parametrize("function_name", ["", " ", "\t\n"])
def test_capability_hook_rejects_empty_internal_function_name(function_name):
    with pytest.raises(ValueError, match="function names must be non-empty"):
        capability_hook({}, internal_functions=frozenset({function_name}))
