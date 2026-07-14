from typing import Literal

import pytest
from pydantic import model_validator

from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.intents import Intent
from agent_kernel.core.specs import (
    ActionSpec,
    Briefing,
    ContinuationSpec,
    Registry,
)


def build_chat(intent: Intent, context) -> Briefing:
    return Briefing(instructions=(f"reply to: {intent.brief}",))


def test_registry_rejects_duplicate_actions():
    chat = ActionSpec(name="chat", kind="toolfree", build=build_chat)
    with pytest.raises(ConfigurationError, match="duplicate action: chat"):
        Registry((chat, chat), default="chat", denied="chat", intent_type=Intent)


def test_registry_requires_toolfree_default():
    draw = ActionSpec(
        name="draw",
        kind="privileged",
        build=build_chat,
        capabilities=frozenset({"draw"}),
    )
    denied = ActionSpec(name="denied", kind="toolfree", build=build_chat)
    with pytest.raises(ConfigurationError, match="default action must be toolfree"):
        Registry(
            (draw, denied),
            default="draw",
            denied="denied",
            intent_type=Intent,
        )


def test_continuation_snapshots_authority_vocabularies():
    modes = {"safe"}
    fields = {"mode": modes}
    continuation = ContinuationSpec(
        authority_fields=fields,  # type: ignore[arg-type]
    )

    modes.add("root")
    fields["other"] = {"root"}

    assert continuation.authority_fields == {
        "mode": frozenset({"safe"})
    }
    with pytest.raises(TypeError):
        continuation.authority_fields["mode"] = frozenset({"root"})  # type: ignore[index]


def test_continuation_rejects_invalid_authority_vocabulary():
    with pytest.raises(ConfigurationError, match="continuation authority"):
        ContinuationSpec(
            authority_fields={"mode": frozenset({""})},
        )


@pytest.mark.parametrize(
    ("field_name", "values"),
    [
        ("missing", frozenset({"safe"})),
        ("mode", frozenset({"root"})),
        ("mode", frozenset()),
    ],
)
def test_registry_rejects_unconstructible_authority_vocabulary(
    field_name,
    values,
):
    class ModeIntent(Intent):
        mode: Literal["safe", "fast"]

    spin = ActionSpec(
        name="spin",
        kind="privileged",
        build=build_chat,
        continuation=ContinuationSpec(
            authority_fields={field_name: values}
        ),
    )
    chat = ActionSpec(name="chat", kind="toolfree", build=build_chat)

    with pytest.raises(ConfigurationError, match="authority"):
        Registry(
            (chat, spin),
            default="chat",
            denied="chat",
            intent_type=ModeIntent,
        )


def test_registry_rejects_when_no_authority_combination_builds_safe_default():
    class ActionDependentIntent(Intent):
        mode: Literal["safe", "fast"]

        @model_validator(mode="after")
        def reject_default_action(self):
            if self.action == "chat":
                raise ValueError("chat has no valid mode")
            return self

    chat = ActionSpec(name="chat", kind="toolfree", build=build_chat)
    spin = ActionSpec(
        name="spin",
        kind="privileged",
        build=build_chat,
        continuation=ContinuationSpec(
            authority_fields={
                "mode": frozenset({"safe", "fast"})
            }
        ),
    )

    with pytest.raises(ConfigurationError, match="fallback"):
        Registry(
            (chat, spin),
            default="chat",
            denied="chat",
            intent_type=ActionDependentIntent,
        )
