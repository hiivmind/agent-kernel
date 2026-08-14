from dataclasses import dataclass
from typing import TypeAlias

from agent_kernel.core._validation import capability_set, string_tuple
from agent_kernel.core.errors import ConfigurationError


@dataclass(frozen=True)
class Role:
    name: str
    capabilities: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capabilities",
            capability_set(self.capabilities, field="role capabilities"),
        )


@dataclass(frozen=True)
class Principal:
    id: str
    role: Role


@dataclass(frozen=True)
class AuthorityEnvelope:
    principal_id: str
    allowed: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed",
            capability_set(self.allowed, field="authority envelope allowed"),
        )


@dataclass(frozen=True)
class ExecutionPlan:
    action: str
    label: str
    capabilities: frozenset[str]
    instructions: tuple[str, ...]
    tool_call_limit: int
    reads_history: bool
    envelope: AuthorityEnvelope
    reason: str | None

    def __post_init__(self) -> None:
        capabilities = capability_set(
            self.capabilities,
            field="execution plan capabilities",
        )
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(
            self,
            "instructions",
            string_tuple(self.instructions, field="execution plan instructions"),
        )
        if capabilities != self.envelope.allowed:
            raise ConfigurationError("execution plan capabilities must equal authority envelope allowed")


@dataclass(frozen=True)
class Completed:
    content: object
    raw: object


@dataclass(frozen=True)
class Pause:
    requirements: tuple[dict[str, object], ...]
    adapter_state: object
    envelope: AuthorityEnvelope
    runtime_context: object | None = None


@dataclass(frozen=True)
class Failed:
    stage: str
    cause: Exception


RuntimeOutcome: TypeAlias = Completed | Pause | Failed


@dataclass(frozen=True)
class TurnResult:
    plan: ExecutionPlan
    outcome: RuntimeOutcome
