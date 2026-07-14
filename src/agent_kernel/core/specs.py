from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.intents import Intent

I = TypeVar("I", bound=Intent)  # noqa: E741


@dataclass(frozen=True)
class Briefing:
    instructions: tuple[str, ...]
    grants: frozenset[str] = frozenset()
    tool_call_limit: int | None = None
    label: str | None = None


@dataclass(frozen=True)
class BuildContext:
    identity: str
    catalog: Callable[[], tuple[dict[str, str | None], ...]]
    summary: Callable[[], str]


@dataclass(frozen=True)
class CatalogEntry:
    key: str
    emoji: str
    blurb: str
    say: str
    trigger: str | None = None
    superseded_by: str | None = None


@dataclass(frozen=True)
class CommandSpec(Generic[I]):
    name: str
    build: Callable[[str], I]


@dataclass(frozen=True)
class ContinuationSpec:
    awaiting: tuple[str, ...] = ()
    authority_fields: dict[str, frozenset[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionSpec(Generic[I]):
    name: str
    kind: Literal["toolfree", "privileged"]
    build: Callable[[I, BuildContext], Briefing]
    capabilities: frozenset[str] = frozenset()  # maximum declared envelope across every build branch
    tool_call_limit: int = 0
    reads_history: bool = False
    catalog: tuple[CatalogEntry, ...] = ()
    command: CommandSpec[I] | None = None
    continuation: ContinuationSpec | None = None


class Registry(Generic[I]):
    def __init__(self, specs: tuple[ActionSpec[I], ...], *, default: str, denied: str, intent_type: type[I]):
        names = [spec.name for spec in specs]
        duplicate = next((name for name in names if names.count(name) > 1), None)
        if duplicate:
            raise ConfigurationError(f"duplicate action: {duplicate}")
        self._specs = {spec.name: spec for spec in specs}
        if default not in self._specs or denied not in self._specs:
            raise ConfigurationError("default and denied actions must exist")
        if self._specs[default].kind != "toolfree":
            raise ConfigurationError("default action must be toolfree")
        if self._specs[denied].kind != "toolfree":
            raise ConfigurationError("denied action must be toolfree")
        self.default = default
        self.denied = denied
        self.intent_type = intent_type

    @property
    def actions(self) -> frozenset[str]:
        return frozenset(self._specs)

    @property
    def specs(self) -> tuple[ActionSpec[I], ...]:
        return tuple(self._specs.values())

    def get(self, action: str) -> ActionSpec[I] | None:
        return self._specs.get(action)
