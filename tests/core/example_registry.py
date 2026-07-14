from dataclasses import dataclass

import pytest

from agent_kernel.core.authority import Planner
from agent_kernel.core.intents import Intent
from agent_kernel.core.results import Principal, Role
from agent_kernel.core.specs import (
    ActionSpec,
    Briefing,
    CatalogEntry,
    CommandSpec,
    ContinuationSpec,
    Registry,
)


class ExampleIntent(Intent):
    mode: str | None = None
    note: str = ""


def build_chat(intent: ExampleIntent, context) -> Briefing:
    return Briefing(
        instructions=(f"reply to: {intent.brief}",),
        label="Chat",
    )


def build_denied(intent: ExampleIntent, context) -> Briefing:
    return Briefing(
        instructions=("explain that the requested action is unavailable",),
        label="Unavailable",
    )


def build_spin(intent: ExampleIntent, context) -> Briefing:
    return Briefing(
        instructions=(f"spin the wheel: {intent.brief}",),
        grants=frozenset({"spin_wheel"}),
        label="Spin",
        tool_call_limit=1,
    )


def build_spin_command(argument: str):
    return {
        "action": "spin",
        "confidence": 1.0,
        "brief": argument,
        "mode": "safe",
    }


CHAT = ActionSpec(
    name="chat",
    kind="toolfree",
    build=build_chat,
    reads_history=True,
    catalog=(
        CatalogEntry(
            key="chat",
            emoji="💬",
            blurb="Have a conversation",
            say="Ask me anything.",
        ),
    ),
)

DENIED = ActionSpec(
    name="denied",
    kind="toolfree",
    build=build_denied,
)

SPIN = ActionSpec(
    name="spin",
    kind="privileged",
    build=build_spin,
    capabilities=frozenset({"spin_wheel"}),
    tool_call_limit=1,
    catalog=(
        CatalogEntry(
            key="classic_spin",
            emoji="🎡",
            blurb="Spin the classic wheel",
            say="Spin the classic wheel.",
            superseded_by="spin",
        ),
        CatalogEntry(
            key="spin",
            emoji="🎯",
            blurb="Spin the wheel",
            say="Spin the wheel.",
            trigger="/spin",
        ),
    ),
    command=CommandSpec(name="spin", build=build_spin_command),
    continuation=ContinuationSpec(
        awaiting=("confirmation",),
        authority_fields={"mode": frozenset({"safe", "fast"})},
    ),
)


@dataclass(frozen=True)
class Example:
    registry: Registry[ExampleIntent]
    planner: Planner[ExampleIntent]
    intent_type: type[ExampleIntent] = ExampleIntent


@pytest.fixture
def example() -> Example:
    registry = Registry(
        (CHAT, DENIED, SPIN),
        default="chat",
        denied="denied",
        intent_type=ExampleIntent,
    )
    return Example(
        registry=registry,
        planner=Planner(registry, confidence_threshold=0.75, identity="Oracle"),
    )


@pytest.fixture
def member() -> Principal:
    return Principal(
        id="member-1",
        role=Role(
            name="member",
            capabilities=frozenset({"spin_wheel", "admin"}),
        ),
    )


@pytest.fixture
def guest() -> Principal:
    return Principal(
        id="guest-1",
        role=Role(name="guest", capabilities=frozenset()),
    )
