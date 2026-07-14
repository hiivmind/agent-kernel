from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from agent_kernel.core.intents import Intent
from agent_kernel.core.specs import Registry

I = TypeVar("I", bound=Intent)  # noqa: E741


@dataclass(frozen=True)
class Continuation:
    last_action: str | None = None
    awaiting: str | None = None
    authority_values: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_turn(
        cls,
        intent: I,
        *,
        registry: Registry[I],
        awaiting: str | None = None,
    ) -> Continuation:
        spec = registry.get(intent.action)
        if spec is None:
            return cls()

        continuation_spec = spec.continuation
        if continuation_spec is None:
            return cls(last_action=spec.name)

        bounded_awaiting = (
            awaiting if awaiting in continuation_spec.awaiting else None
        )
        authority_values = []
        for field_name, allowed_values in continuation_spec.authority_fields.items():
            value = getattr(intent, field_name, None)
            if isinstance(value, str) and value in allowed_values:
                authority_values.append((field_name, value))

        return cls(
            last_action=spec.name,
            awaiting=bounded_awaiting,
            authority_values=tuple(authority_values),
        )

    def render(self, registry: Registry[I]) -> tuple[str, ...]:
        if self.last_action is None:
            return ()

        spec = registry.get(self.last_action)
        if spec is None:
            return ()

        rendered = [f"last_action = {spec.name}"]
        continuation_spec = spec.continuation
        if continuation_spec is None:
            return tuple(rendered)

        if self.awaiting in continuation_spec.awaiting:
            rendered.append(f"awaiting = {self.awaiting}")

        for field_name, value in self.authority_values:
            allowed_values = continuation_spec.authority_fields.get(field_name)
            if allowed_values is not None and value in allowed_values:
                rendered.append(f"{field_name} = {value}")

        return tuple(rendered)
