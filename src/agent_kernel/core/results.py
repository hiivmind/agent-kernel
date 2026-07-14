from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class Role:
    name: str
    capabilities: frozenset[str]


@dataclass(frozen=True)
class Principal:
    id: str
    role: Role


@dataclass(frozen=True)
class AuthorityEnvelope:
    principal_id: str
    allowed: frozenset[str]


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
