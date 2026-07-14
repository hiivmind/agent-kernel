from collections.abc import Callable
from typing import Any, Generic, TypeVar

from agno.agent import Agent

from agent_kernel.core.continuation import Continuation
from agent_kernel.core.intents import Intent
from agent_kernel.core.specs import Registry

I = TypeVar("I", bound=Intent)  # noqa: E741


class AgnoClassifier(Generic[I]):
    def __init__(
        self,
        *,
        registry: Registry[I],
        model: object | None,
        instructions: str | list[str],
        agent_factory: Callable[..., Any] = Agent,
    ):
        self.registry = registry
        self.model = model
        self.instructions = instructions
        self.agent = agent_factory(
            model=model,
            instructions=instructions,
            tools=[],
            output_schema=registry.intent_type,
        )

    def classify(
        self,
        message: str,
        *,
        context: Continuation | None = None,
    ) -> I:
        del context
        response = self.agent.run(message)
        return self.registry.intent_type.model_validate(response.content)
