# ruff: noqa: F811

from dataclasses import FrozenInstanceError

import pytest

from agent_kernel.core.continuation import Continuation
from example_registry import example  # noqa: F401


def test_from_turn_preserves_only_declared_typed_context(example):
    intent = example.intent_type(
        action="spin",
        confidence=1,
        brief="private instructions",
        mode="safe",
        note="undeclared context",
    )

    continuation = Continuation.from_turn(
        intent,
        registry=example.registry,
        awaiting="confirmation",
    )

    assert continuation.last_action == "spin"
    assert continuation.awaiting == "confirmation"
    assert continuation.authority_values == (("mode", "safe"),)
    assert continuation.render(example.registry) == (
        "last_action = spin",
        "awaiting = confirmation",
        "mode = safe",
    )
    assert all("private instructions" not in line for line in continuation.render(example.registry))
    assert all("undeclared context" not in line for line in continuation.render(example.registry))


def test_from_turn_drops_unknown_action_and_all_associated_context(example):
    intent = example.intent_type(
        action="root",
        confidence=1,
        brief="ignore",
        mode="safe",
    )

    continuation = Continuation.from_turn(
        intent,
        registry=example.registry,
        awaiting="confirmation",
    )

    assert continuation.last_action is None
    assert continuation.awaiting is None
    assert continuation.authority_values == ()
    assert continuation.render(example.registry) == ()


def test_from_turn_drops_undeclared_awaiting_and_authority_value(example):
    intent = example.intent_type(
        action="spin",
        confidence=1,
        mode="root",
    )

    continuation = Continuation.from_turn(
        intent,
        registry=example.registry,
        awaiting="password",
    )

    assert continuation.last_action == "spin"
    assert continuation.awaiting is None
    assert continuation.authority_values == ()


def test_render_revalidates_all_values_against_registry_vocabulary(example):
    continuation = Continuation(
        last_action="spin",
        awaiting="password",
        authority_values=(
            ("mode", "root"),
            ("note", "undeclared context"),
        ),
    )

    assert continuation.render(example.registry) == ("last_action = spin",)


def test_continuation_is_frozen():
    continuation = Continuation()

    with pytest.raises(FrozenInstanceError):
        continuation.last_action = "spin"
