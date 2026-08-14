from typing import Generic, TypeVar

from agent_kernel.core.catalog import describe, summary_line
from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.intents import Intent
from agent_kernel.core.results import (
    AuthorityEnvelope,
    ExecutionPlan,
    Principal,
)
from agent_kernel.core.specs import ActionSpec, Briefing, BuildContext, Registry

I = TypeVar("I", bound=Intent)  # noqa: E741


class Planner(Generic[I]):
    def __init__(
        self,
        registry: Registry[I],
        *,
        confidence_threshold: float,
        identity: str = "",
    ):
        self.registry = registry
        self.confidence_threshold = confidence_threshold
        self.identity = identity

    @staticmethod
    def _validate_grants(spec: ActionSpec[I], briefing: Briefing) -> None:
        undeclared = briefing.grants - spec.capabilities
        if undeclared:
            raise ConfigurationError(f"action {spec.name!r} built undeclared grants: {sorted(undeclared)}")
        if spec.kind == "toolfree" and briefing.grants:
            raise ConfigurationError(f"toolfree action {spec.name!r} built grants: {sorted(briefing.grants)}")

    def plan(self, intent: I, principal: Principal) -> ExecutionPlan:
        intent, valid = self.registry.normalize_intent(intent)
        spec = self.registry.get(intent.action)
        reason = None if valid else "invalid intent"
        if not valid:
            spec = self.registry.get(self.registry.default)
        elif spec is None:
            reason = f"unknown action: {intent.action}"
            spec = self.registry.get(self.registry.default)
        elif spec.kind == "privileged" and intent.confidence < self.confidence_threshold:
            reason = f"confidence {intent.confidence:.2f} < {self.confidence_threshold:.2f}"
            spec = self.registry.get(self.registry.default)
        elif spec.kind == "privileged" and spec.continuation is not None:
            for field_name, allowed_values in spec.continuation.authority_fields.items():
                value = getattr(intent, field_name, None)
                if value is not None and value not in allowed_values:
                    reason = f"invalid authority value: {field_name}={value!r}"
                    spec = self.registry.get(self.registry.default)
                    break

        assert spec is not None
        context = BuildContext(
            identity=self.identity,
            catalog=lambda: describe(self, principal),
            summary=lambda: summary_line(self, principal),
        )
        briefing = spec.build(intent, context)
        self._validate_grants(spec, briefing)

        if not briefing.grants <= principal.role.capabilities:
            missing = sorted(briefing.grants - principal.role.capabilities)
            reason = f"role {principal.role.name!r} lacks {missing}"
            spec = self.registry.get(self.registry.denied)
            assert spec is not None
            briefing = spec.build(intent, context)
            self._validate_grants(spec, briefing)

        if not briefing.grants <= principal.role.capabilities:
            raise ConfigurationError(
                f"final grants exceed principal authority: {sorted(briefing.grants - principal.role.capabilities)}"
            )

        return ExecutionPlan(
            action=spec.name,
            label=briefing.label or spec.name,
            capabilities=briefing.grants,
            instructions=briefing.instructions,
            tool_call_limit=(
                briefing.tool_call_limit if briefing.tool_call_limit is not None else spec.tool_call_limit
            ),
            reads_history=spec.reads_history,
            envelope=AuthorityEnvelope(principal.id, briefing.grants),
            reason=reason,
        )
