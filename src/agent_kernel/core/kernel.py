from dataclasses import dataclass
from typing import Generic, TypeVar

from agent_kernel.core.authority import Planner
from agent_kernel.core.catalog import describe
from agent_kernel.core.continuation import Continuation
from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.intents import Intent
from agent_kernel.core.protocols import Classifier, Runtime
from agent_kernel.core.results import (
    ExecutionPlan,
    Failed,
    Pause,
    Principal,
    RuntimeOutcome,
    TurnResult,
)
from agent_kernel.core.specs import Registry

I = TypeVar("I", bound=Intent)  # noqa: E741


@dataclass(frozen=True)
class KernelConfig(Generic[I]):
    registry: Registry[I]
    confidence_threshold: float = 0.55
    identity: str = ""


class Kernel(Generic[I]):
    def __init__(
        self,
        config: KernelConfig[I],
        classifier: Classifier[I],
        runtime: Runtime,
    ):
        self._config = config
        self._classifier = classifier
        self._runtime = runtime
        self._planner = Planner(
            config.registry,
            confidence_threshold=config.confidence_threshold,
            identity=config.identity,
        )

    @property
    def config(self) -> KernelConfig[I]:
        return self._config

    @property
    def classifier(self) -> Classifier[I]:
        return self._classifier

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    def classify(
        self,
        message: str,
        *,
        context: Continuation | None = None,
    ) -> I:
        classified = self.classifier.classify(message, context=context)
        return self.config.registry.intent_type.model_validate(classified)

    def plan(self, intent: I, *, principal: Principal) -> ExecutionPlan:
        return self._planner.plan(intent, principal)

    def _execute(
        self,
        message: str,
        plan: ExecutionPlan,
        *,
        runtime_context: object | None,
    ) -> RuntimeOutcome:
        try:
            return self.runtime.execute(
                message,
                plan,
                context=runtime_context,
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            return Failed(stage="runtime", cause=exc)

    def run(
        self,
        message: str,
        *,
        principal: Principal,
        context: Continuation | None = None,
        runtime_context: object | None = None,
    ) -> TurnResult:
        intent = self.classify(message, context=context)
        plan = self.plan(intent, principal=principal)
        outcome = self._execute(
            message,
            plan,
            runtime_context=runtime_context,
        )
        return TurnResult(plan=plan, outcome=outcome)

    def dispatch(
        self,
        intent: I,
        *,
        principal: Principal,
        runtime_context: object | None = None,
    ) -> TurnResult:
        normalized = self.config.registry.intent_type.model_validate(intent)
        plan = self.plan(normalized, principal=principal)
        outcome = self._execute(
            normalized.brief,
            plan,
            runtime_context=runtime_context,
        )
        return TurnResult(plan=plan, outcome=outcome)

    def resume(
        self,
        pause: Pause,
        answers: dict[str, str],
    ) -> RuntimeOutcome:
        try:
            return self.runtime.resume(pause, answers)
        except ConfigurationError:
            raise
        except Exception as exc:
            return Failed(stage="runtime", cause=exc)

    def describe(
        self,
        principal: Principal,
    ) -> tuple[dict[str, str | None], ...]:
        return describe(self._planner, principal)
