from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar

from agent_kernel.core.results import ExecutionPlan, RuntimeOutcome

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ValueT_co = TypeVar("ValueT_co", covariant=True)


class ValueAdapter(Protocol[ValueT_co]):
    def validate_python(self, value: object) -> ValueT_co:
        raise NotImplementedError


def _freeze_json(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("descriptive attributes must be JSON-compatible")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError("descriptive attributes must be JSON-compatible")


@dataclass(frozen=True)
class SubjectContext:
    kind: str
    id: str
    attributes: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.kind.strip() or not self.id.strip():
            raise ValueError("subject kind and id must be non-empty")
        frozen = _freeze_json(self.attributes)
        if not isinstance(frozen, Mapping):
            raise TypeError("subject attributes must be a mapping")
        object.__setattr__(self, "attributes", frozen)


@dataclass(frozen=True)
class OperationInvocation(Generic[InputT]):
    invocation_id: str
    target_id: str
    request: str
    subject: SubjectContext
    inputs: InputT
    plan: ExecutionPlan

    def __post_init__(self) -> None:
        if not self.invocation_id.strip() or not self.target_id.strip():
            raise ValueError("invocation and target ids must be non-empty")
        if not self.request.strip():
            raise ValueError("operation request must be non-empty")


@dataclass(frozen=True)
class OperationModelInput(Generic[InputT]):
    request: str
    subject: SubjectContext
    inputs: InputT


@dataclass(frozen=True)
class OperationContract(Generic[InputT, OutputT]):
    input_adapter: ValueAdapter[InputT]
    output_adapter: ValueAdapter[OutputT]
    model_output_type: type[OutputT]
    model_input_builder: Callable[[OperationModelInput[InputT]], object]


@dataclass(frozen=True)
class OperationCompletion(Generic[OutputT]):
    invocation_id: str
    output: OutputT
    raw: object


class OperationRuntime(Protocol[InputT, OutputT]):
    def execute(
        self,
        invocation: OperationInvocation[InputT],
        contract: OperationContract[InputT, OutputT],
        *,
        context: object | None = None,
    ) -> RuntimeOutcome:
        raise NotImplementedError
