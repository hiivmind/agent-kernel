from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, cast

import dspy  # type: ignore[import-untyped]

from agent_kernel.core.continuation import Continuation
from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.intents import Intent
from agent_kernel.core.protocols import Classifier
from agent_kernel.core.specs import Registry
from agent_kernel.integrations.dspy.artifacts import load_artifact

I = TypeVar("I", bound=Intent)  # noqa: E741

_MISSING = object()
_RESERVED_FIELDS = frozenset({"message", "action", "confidence", "brief"})


def _authority_vocabularies(registry: Registry[Any]) -> dict[str, tuple[str, ...]]:
    vocabularies: dict[str, set[str]] = {}
    for spec in registry.specs:
        continuation = spec.continuation
        if continuation is None:
            continue
        for field_name, values in continuation.authority_fields.items():
            vocabularies.setdefault(field_name, set()).update(values)
    return {field_name: tuple(sorted(values)) for field_name, values in sorted(vocabularies.items())}


def _literal_type(values: tuple[str, ...]) -> Any:
    return Literal[*values]


def build_signature(
    registry: Registry[Any],
    *,
    instructions: str = (
        "Classify the message into one registered action and return its "
        "confidence, brief, and declared authority fields."
    ),
) -> type[dspy.Signature]:
    action_names = tuple(sorted(action for action in registry.actions if action != registry.denied))
    authority_vocabularies = _authority_vocabularies(registry)
    collisions = sorted(_RESERVED_FIELDS.intersection(authority_vocabularies))
    if collisions:
        raise ConfigurationError(
            "authority fields collide with reserved DSPy projection fields: " + ", ".join(collisions)
        )

    fields: dict[str, tuple[Any, Any]] = {
        "message": (
            str,
            dspy.InputField(desc="The message to classify."),
        ),
        "action": (
            _literal_type(action_names),
            dspy.OutputField(desc="The registered action selected for the message."),
        ),
    }
    for field_name, values in authority_vocabularies.items():
        fields[field_name] = (
            _literal_type(values) | None,
            dspy.OutputField(
                desc=("A declared authority value when applicable to the selected action, otherwise null.")
            ),
        )
    fields["confidence"] = (
        float,
        dspy.OutputField(desc="Classification confidence from 0.0 to 1.0."),
    )
    fields["brief"] = (
        str,
        dspy.OutputField(desc="A concise description of what the message requests."),
    )
    return cast(type[Any], dspy.Signature(fields, instructions))


class DspyClassifier(Generic[I]):
    def __init__(
        self,
        *,
        registry: Registry[I],
        module: Callable[..., Any],
        fallback: Classifier[I] | None = None,
    ):
        self.registry = registry
        self.module = module
        self.fallback = fallback

    @classmethod
    def from_artifact(
        cls,
        *,
        registry: Registry[I],
        path: Path,
        fallback: Classifier[I] | None = None,
    ) -> DspyClassifier[I]:
        module = dspy.Predict(build_signature(registry))
        load_artifact(
            registry=registry,
            module=module,
            path=path,
        )
        return cls(
            registry=registry,
            module=module,
            fallback=fallback,
        )

    def classify(
        self,
        message: str,
        *,
        context: Continuation | None = None,
    ) -> I:
        try:
            prediction = self.module(message=message)
            payload = self._normalization_payload(prediction)
            return self.registry.intent_type.model_validate(payload)
        except Exception:
            if self.fallback is None:
                raise
            return self.fallback.classify(message, context=context)

    def _normalization_payload(self, prediction: Any) -> Any:
        if isinstance(prediction, self.registry.intent_type):
            return prediction

        if isinstance(prediction, Mapping):
            if "intent" in prediction:
                return prediction["intent"]
            return {name: prediction[name] for name in self.registry.intent_type.model_fields if name in prediction}

        structured = getattr(prediction, "intent", _MISSING)
        if structured is not _MISSING:
            return structured

        payload = {}
        for name in self.registry.intent_type.model_fields:
            value = getattr(prediction, name, _MISSING)
            if value is not _MISSING:
                payload[name] = value
        return payload
