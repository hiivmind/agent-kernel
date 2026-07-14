from typing import TypeVar

from agent_kernel.core.intents import Intent
from agent_kernel.core.specs import CommandSpec, Registry

I = TypeVar("I", bound=Intent)  # noqa: E741


def parse_command(message: str, registry: Registry[I]) -> I | None:
    if not message.startswith("/"):
        return None

    command_text, separator, argument = message.partition(" ")
    name = command_text[1:]
    if not name:
        return None

    matches: list[CommandSpec[I]] = [
        spec.command
        for spec in registry.specs
        if spec.command is not None and spec.command.name == name
    ]
    if len(matches) != 1:
        return None

    normalized_argument = argument.strip() if separator else ""
    built = matches[0].build(normalized_argument)
    normalized = registry.intent_type.model_validate(built)
    if normalized.action not in registry.actions:
        return normalized.model_copy(update={"action": registry.default})
    return normalized
