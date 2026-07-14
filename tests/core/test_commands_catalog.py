# ruff: noqa: F811

from dataclasses import replace

from agent_kernel.core.catalog import describe
from agent_kernel.core.commands import parse_command
from agent_kernel.core.specs import CommandSpec, Registry
from example_registry import ExampleIntent, example, guest, member  # noqa: F401


def test_leading_command_deterministically_emits_registered_intent(example):
    intent = parse_command("/spin    clockwise   ", example.registry)

    assert isinstance(intent, ExampleIntent)
    assert intent.action == "spin"
    assert intent.action in example.registry.actions
    assert intent.confidence == 1.0
    assert intent.brief == "clockwise"
    assert intent.mode == "safe"


def test_command_must_be_leading_and_match_the_declared_name(example):
    assert parse_command("please /spin clockwise", example.registry) is None
    assert parse_command("/spinach clockwise", example.registry) is None
    assert parse_command("/unknown clockwise", example.registry) is None


def test_command_without_arguments_normalizes_to_empty_brief(example):
    intent = parse_command("/spin", example.registry)

    assert isinstance(intent, ExampleIntent)
    assert intent.brief == ""


def test_command_normalizing_to_unknown_action_falls_to_registered_default(example):
    spin = example.registry.get("spin")
    assert spin is not None
    unsafe_spin = replace(
        spin,
        command=CommandSpec(
            name="spin",
            build=lambda argument: {
                "action": "root",
                "confidence": 1.0,
                "brief": argument,
            },
        ),
    )
    registry = Registry(
        tuple(
            unsafe_spin if spec.name == "spin" else spec
            for spec in example.registry.specs
        ),
        default=example.registry.default,
        denied=example.registry.denied,
        intent_type=example.intent_type,
    )

    intent = parse_command("/spin ignore", registry)

    assert isinstance(intent, ExampleIntent)
    assert intent.action == registry.default
    assert intent.action in registry.actions


def test_catalog_does_not_advertise_actions_denied_to_principal(example, guest):
    catalog = describe(example.planner, guest)
    keys = {entry["key"] for entry in catalog}

    assert "chat" in keys
    assert "spin" not in keys
    assert "classic_spin" not in keys


def test_catalog_includes_authorized_action(example, member):
    catalog = describe(example.planner, member)
    keys = {entry["key"] for entry in catalog}

    assert "spin" in keys


def test_catalog_removes_entry_superseded_by_visible_replacement(example, member):
    catalog = describe(example.planner, member)
    keys = {entry["key"] for entry in catalog}

    assert "spin" in keys
    assert "classic_spin" not in keys
