# ruff: noqa: F811

from dataclasses import replace
import re
from types import SimpleNamespace

import pytest

from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.specs import ContinuationSpec, Registry
from agent_kernel.integrations.dspy import (
    DspyClassifier,
    build_signature,
    registry_fingerprint,
)
from tests.core.example_registry import (  # noqa: F401
    CHAT,
    DENIED,
    SPIN,
    ExampleIntent,
    example,
)


def _registry(
    *,
    specs=(CHAT, DENIED, SPIN),
    intent_type=ExampleIntent,
):
    return Registry(
        specs,
        default="chat",
        denied="denied",
        intent_type=intent_type,
    )


def _enum_values(schema):
    values = set(schema.get("enum", ()))
    for branch in schema.get("anyOf", ()):
        values.update(_enum_values(branch))
    return values


def test_projected_signature_excludes_denied_and_includes_authority_fields(example):
    signature = build_signature(example.registry)

    assert set(signature.input_fields) == {"message"}
    assert set(signature.output_fields) == {
        "action",
        "mode",
        "confidence",
        "brief",
    }

    schema = signature.model_json_schema()["properties"]
    assert signature.output_fields["action"].annotation is not str
    assert _enum_values(schema["action"]) == {"chat", "spin"}
    assert "denied" not in _enum_values(schema["action"])

    assert signature.output_fields["mode"].annotation is not str
    assert _enum_values(schema["mode"]) == {"safe", "fast"}


@pytest.mark.parametrize(
    "field_name",
    ["message", "action", "confidence", "brief"],
)
def test_projected_signature_rejects_authority_field_colliding_with_reserved_field(
    field_name,
):
    conflicting_spin = replace(
        SPIN,
        continuation=ContinuationSpec(
            authority_fields={field_name: frozenset({"safe", "fast"})},
        ),
    )
    with pytest.raises(ConfigurationError, match=field_name):
        registry = _registry(specs=(CHAT, DENIED, conflicting_spin))
        build_signature(registry)


def test_registry_fingerprint_is_stable_across_registry_declaration_order():
    first = registry_fingerprint(_registry())
    reordered = registry_fingerprint(_registry(specs=(SPIN, CHAT, DENIED)))

    assert first == reordered
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_registry_fingerprint_changes_with_intent_schema():
    class ExtendedIntent(ExampleIntent):
        source: str | None = None

    assert registry_fingerprint(_registry()) != registry_fingerprint(_registry(intent_type=ExtendedIntent))


def test_registry_fingerprint_changes_with_authority_vocabulary():
    changed_spin = replace(
        SPIN,
        continuation=ContinuationSpec(
            awaiting=("confirmation",),
            authority_fields={"mode": frozenset({"safe", "turbo"})},
        ),
    )

    assert registry_fingerprint(_registry()) != registry_fingerprint(_registry(specs=(CHAT, DENIED, changed_spin)))


class RecordingModule:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class RaisingModule:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError("DSPy failed")


class RecordingFallback:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def classify(self, message, *, context=None):
        self.calls.append((message, context))
        return self.result


def test_classifier_normalizes_structured_intent_result(example):
    expected = ExampleIntent(
        action="spin",
        confidence=0.91,
        brief="spin quickly",
        mode="fast",
    )
    module = RecordingModule(SimpleNamespace(intent=expected))

    result = DspyClassifier(
        registry=example.registry,
        module=module,
    ).classify("please spin")

    assert result == expected
    assert type(result) is example.registry.intent_type
    assert module.calls[0]["message"] == "please spin"


def test_classifier_normalizes_projected_prediction_fields(example):
    module = RecordingModule(
        SimpleNamespace(
            action="spin",
            confidence="0.84",
            brief="spin safely",
            mode="safe",
        )
    )

    result = DspyClassifier(
        registry=example.registry,
        module=module,
    ).classify("spin it")

    assert result == ExampleIntent(
        action="spin",
        confidence=0.84,
        brief="spin safely",
        mode="safe",
    )
    assert type(result) is example.registry.intent_type


def test_classifier_does_not_use_fallback_after_success(example):
    module = RecordingModule(
        SimpleNamespace(
            action="chat",
            confidence=0.9,
            brief="say hello",
            mode=None,
        )
    )
    fallback = RecordingFallback(ExampleIntent(action="chat", confidence=0.1, brief="fallback"))

    result = DspyClassifier(
        registry=example.registry,
        module=module,
        fallback=fallback,
    ).classify("hello")

    assert result.brief == "say hello"
    assert fallback.calls == []


@pytest.mark.parametrize(
    "module",
    [
        RecordingModule(SimpleNamespace(action="spin")),
        RaisingModule(),
    ],
    ids=["normalization-failure", "module-failure"],
)
def test_classifier_uses_explicit_fallback_for_module_or_normalization_failure(
    example,
    module,
):
    expected = ExampleIntent(
        action="chat",
        confidence=0.5,
        brief="fallback result",
    )
    fallback = RecordingFallback(expected)

    result = DspyClassifier(
        registry=example.registry,
        module=module,
        fallback=fallback,
    ).classify("hello")

    assert result == expected
    assert fallback.calls == [("hello", None)]
