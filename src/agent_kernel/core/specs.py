from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError
from agent_kernel.core._validation import capability_set, string_tuple
from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.intents import Intent

I = TypeVar("I", bound=Intent)  # noqa: E741


@dataclass(frozen=True)
class Briefing:
    instructions: tuple[str, ...]
    grants: frozenset[str] = frozenset()
    tool_call_limit: int | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instructions",
            string_tuple(self.instructions, field="briefing instructions"),
        )
        object.__setattr__(
            self,
            "grants",
            capability_set(self.grants, field="briefing grants"),
        )


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
    authority_fields: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "awaiting",
            string_tuple(self.awaiting, field="continuation awaiting"),
        )
        snapshot: dict[str, frozenset[str]] = {}
        for field_name, values in self.authority_fields.items():
            if not isinstance(field_name, str) or not field_name.strip():
                raise ConfigurationError(
                    "continuation authority fields must be non-empty strings"
                )
            snapshot[field_name] = capability_set(
                values,
                field="continuation authority values",
            )
        object.__setattr__(self, "authority_fields", MappingProxyType(snapshot))


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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capabilities",
            capability_set(self.capabilities, field="action capabilities"),
        )
        object.__setattr__(self, "catalog", tuple(self.catalog))


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
        authority_values: dict[str, set[str]] = {}
        for spec in self._specs.values():
            if spec.continuation is None:
                continue
            for field_name, values in spec.continuation.authority_fields.items():
                model_field = intent_type.model_fields.get(field_name)
                if model_field is None or not values:
                    raise ConfigurationError(
                        f"invalid authority vocabulary for field {field_name!r}"
                    )
                adapter: TypeAdapter[Any] = TypeAdapter(model_field.annotation)
                try:
                    for value in values:
                        adapter.validate_python(value)
                except ValidationError as exc:
                    raise ConfigurationError(
                        f"invalid authority vocabulary for field {field_name!r}"
                    ) from exc
                authority_values.setdefault(field_name, set()).update(values)
        self._authority_defaults = MappingProxyType(
            {
                field_name: sorted(values)[0]
                for field_name, values in authority_values.items()
            }
        )
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

    def normalize_intent(self, value: object) -> tuple[I, bool]:
        serialized = value.model_dump() if isinstance(value, BaseModel) else value
        try:
            return self.intent_type.model_validate(serialized), True
        except ValidationError as exc:
            if not isinstance(serialized, dict):
                raise ConfigurationError("invalid intent") from exc
            fallback = dict(serialized)
            fallback["action"] = self.default
            fallback["confidence"] = 0.0
            fallback.update(self._authority_defaults)
            try:
                return self.intent_type.model_validate(fallback), False
            except ValidationError as fallback_exc:
                raise ConfigurationError("invalid intent") from fallback_exc
