from typing import Protocol, TypeVar

from agent_kernel.core.continuation import Continuation
from agent_kernel.core.intents import Intent
from agent_kernel.core.results import (
    ExecutionPlan,
    Pause,
    RuntimeOutcome,
)

I_co = TypeVar("I_co", bound=Intent, covariant=True)


class Classifier(Protocol[I_co]):
    def classify(
        self,
        message: str,
        *,
        context: Continuation | None = None,
    ) -> I_co: ...


class Runtime(Protocol):
    def execute(
        self,
        message: str,
        plan: ExecutionPlan,
        *,
        context: object | None = None,
    ) -> RuntimeOutcome: ...

    def resume(
        self,
        pause: Pause,
        answers: dict[str, str],
    ) -> RuntimeOutcome: ...
