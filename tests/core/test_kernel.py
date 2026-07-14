# ruff: noqa: F811

from dataclasses import FrozenInstanceError
from typing import Literal

import pytest
from pydantic import field_validator

from agent_kernel.core.continuation import Continuation
from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.kernel import Kernel, KernelConfig
from agent_kernel.core.results import (
    AuthorityEnvelope,
    Completed,
    Failed,
    Pause,
)
from agent_kernel.core.intents import Intent
from agent_kernel.core.specs import (
    ActionSpec,
    Briefing,
    ContinuationSpec,
    Registry,
)
from example_registry import ExampleIntent, example, guest, member  # noqa: F401


class FakeClassifier:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def classify(self, message, *, context=None):
        self.calls.append((message, context))
        return self.result


class FakeRuntime:
    def __init__(self, *, error=None):
        self.error = error
        self.execute_calls = []
        self.resume_calls = []

    def execute(self, message, plan, *, context=None):
        self.execute_calls.append((message, plan, context))
        if self.error is not None:
            raise self.error
        return Completed(content=f"ran {plan.action}", raw=None)

    def resume(self, pause, answers):
        self.resume_calls.append((pause, answers))
        if self.error is not None:
            raise self.error
        return Completed(content=str(answers), raw=None)


def classifier_result(*, brief="spin request"):
    return {
        "action": "spin",
        "confidence": 1.0,
        "brief": brief,
        "mode": "safe",
    }


def make_kernel(example, *, classifier=None, runtime=None, **config_options):
    classifier = classifier or FakeClassifier(classifier_result())
    runtime = runtime or FakeRuntime()
    config = KernelConfig(example.registry, **config_options)
    return Kernel(config, classifier, runtime), classifier, runtime


def make_pause():
    return Pause(
        requirements=({"field": "confirmation"},),
        adapter_state={"run_id": "run-1"},
        envelope=AuthorityEnvelope(
            principal_id="member-1",
            allowed=frozenset({"spin_wheel"}),
        ),
    )


def test_kernel_config_has_defaults_and_is_immutable(example):
    config = KernelConfig(example.registry)

    assert config.registry is example.registry
    assert config.confidence_threshold == 0.55
    assert config.identity == ""
    with pytest.raises(FrozenInstanceError):
        config.identity = "Oracle"


def test_kernel_exposes_configured_dependencies_as_read_only(example):
    kernel, classifier, runtime = make_kernel(example)

    assert kernel.config.registry is example.registry
    assert kernel.classifier is classifier
    assert kernel.runtime is runtime

    for attribute, value in (
        ("config", KernelConfig(example.registry)),
        ("classifier", FakeClassifier(classifier_result())),
        ("runtime", FakeRuntime()),
    ):
        with pytest.raises(AttributeError):
            setattr(kernel, attribute, value)


def test_classify_normalizes_classifier_output_and_forwards_context(example):
    context = Continuation(last_action="chat")
    kernel, classifier, _ = make_kernel(example)

    intent = kernel.classify("please spin", context=context)

    assert intent == ExampleIntent(
        action="spin",
        confidence=1.0,
        brief="spin request",
        mode="safe",
    )
    assert classifier.calls == [("please spin", context)]


def test_classify_revalidates_an_existing_mutated_intent(example):
    classified = ExampleIntent(
        action="spin",
        confidence=1.0,
        brief="spin request",
        mode="safe",
    )
    classified.confidence = float("nan")
    kernel, _, _ = make_kernel(
        example,
        classifier=FakeClassifier(classified),
    )

    intent = kernel.classify("please spin")

    assert intent is not classified
    assert intent.action == "chat"
    assert intent.confidence == 0.0


def test_run_falls_to_toolfree_default_for_invalid_required_authority_literal(
    member,
):
    class RequiredModeIntent(Intent):
        mode: Literal["safe", "fast"]

    def build_chat(intent, context):
        return Briefing(instructions=("reply safely",))

    def build_spin(intent, context):
        return Briefing(
            instructions=("spin",),
            grants=frozenset({"spin_wheel"}),
        )

    registry = Registry(
        (
            ActionSpec(name="chat", kind="toolfree", build=build_chat),
            ActionSpec(name="denied", kind="toolfree", build=build_chat),
            ActionSpec(
                name="spin",
                kind="privileged",
                build=build_spin,
                capabilities=frozenset({"spin_wheel"}),
                continuation=ContinuationSpec(
                    authority_fields={
                        "mode": frozenset({"safe", "fast"})
                    }
                ),
            ),
        ),
        default="chat",
        denied="denied",
        intent_type=RequiredModeIntent,
    )
    runtime = FakeRuntime()
    kernel = Kernel(
        KernelConfig(registry),
        FakeClassifier(
            {
                "action": "spin",
                "confidence": 1.0,
                "brief": "spin",
                "mode": "root",
            }
        ),
        runtime,
    )

    result = kernel.run("spin", principal=member)

    assert result.plan.action == "chat"
    assert result.plan.capabilities == frozenset()
    assert runtime.execute_calls == [("spin", result.plan, None)]


def test_run_uses_fully_valid_later_authority_value_for_safe_fallback(member):
    class ValidatedModeIntent(Intent):
        mode: Literal["fast", "safe"]

        @field_validator("mode")
        @classmethod
        def reject_fast(cls, value):
            if value == "fast":
                raise ValueError("fast is unavailable")
            return value

    def build_chat(intent, context):
        return Briefing(instructions=("reply safely",))

    registry = Registry(
        (
            ActionSpec(name="chat", kind="toolfree", build=build_chat),
            ActionSpec(name="denied", kind="toolfree", build=build_chat),
            ActionSpec(
                name="spin",
                kind="privileged",
                build=lambda intent, context: Briefing(
                    instructions=("spin",),
                    grants=frozenset({"spin_wheel"}),
                ),
                capabilities=frozenset({"spin_wheel"}),
                continuation=ContinuationSpec(
                    authority_fields={
                        "mode": frozenset({"fast", "safe"})
                    }
                ),
            ),
        ),
        default="chat",
        denied="denied",
        intent_type=ValidatedModeIntent,
    )
    runtime = FakeRuntime()
    kernel = Kernel(
        KernelConfig(registry),
        FakeClassifier(
            {
                "action": "spin",
                "confidence": 1.0,
                "brief": "spin",
                "mode": "root",
            }
        ),
        runtime,
    )

    result = kernel.run("spin", principal=member)

    assert result.plan.action == "chat"
    assert result.plan.capabilities == frozenset()
    assert runtime.execute_calls == [("spin", result.plan, None)]


def test_plan_uses_the_configured_threshold_and_preserves_actual_grants(
    example,
    member,
):
    kernel, _, _ = make_kernel(
        example,
        confidence_threshold=0.75,
        identity="Oracle",
    )

    accepted = kernel.plan(
        ExampleIntent(
            action="spin",
            confidence=1.0,
            brief="spin",
            mode="safe",
        ),
        principal=member,
    )
    rejected = kernel.plan(
        ExampleIntent(
            action="spin",
            confidence=0.70,
            brief="spin",
            mode="safe",
        ),
        principal=member,
    )

    assert accepted.action == "spin"
    assert accepted.capabilities == frozenset({"spin_wheel"})
    assert rejected.action == "chat"
    assert rejected.capabilities == frozenset()


def test_run_composes_facade_and_executes_the_original_message(example, member):
    classifier = FakeClassifier(classifier_result(brief="normalized request"))
    runtime = FakeRuntime()
    kernel, _, _ = make_kernel(
        example,
        classifier=classifier,
        runtime=runtime,
    )
    context = Continuation(last_action="chat")
    runtime_context = {"history": "runtime history"}

    result = kernel.run(
        "the original message",
        principal=member,
        context=context,
        runtime_context=runtime_context,
    )

    assert result.plan.action == "spin"
    assert result.outcome == Completed(content="ran spin", raw=None)
    assert classifier.calls == [("the original message", context)]
    assert runtime.execute_calls == [
        ("the original message", result.plan, runtime_context)
    ]


def test_dispatch_normalizes_intent_and_executes_its_brief(example, member):
    kernel, _, runtime = make_kernel(example)
    supplied_intent = classifier_result(brief="deterministic request")

    result = kernel.dispatch(
        supplied_intent,
        principal=member,
        runtime_context="runtime context",
    )

    assert result.plan.action == "spin"
    assert runtime.execute_calls == [
        ("deterministic request", result.plan, "runtime context")
    ]


def test_runtime_execution_exception_becomes_failed_outcome(example, member):
    error = RuntimeError("runtime unavailable")
    kernel, _, _ = make_kernel(example, runtime=FakeRuntime(error=error))

    result = kernel.run("spin", principal=member)

    assert result.plan.action == "spin"
    assert result.outcome == Failed(stage="runtime", cause=error)


def test_runtime_configuration_error_continues_to_raise(example, member):
    error = ConfigurationError("invalid runtime configuration")
    kernel, _, _ = make_kernel(example, runtime=FakeRuntime(error=error))

    with pytest.raises(ConfigurationError) as caught:
        kernel.run("spin", principal=member)

    assert caught.value is error


def test_resume_forwards_pause_and_answers(example):
    kernel, _, runtime = make_kernel(example)
    pause = make_pause()
    answers = {"confirmation": "yes"}

    outcome = kernel.resume(pause, answers)

    assert outcome == Completed(content=str(answers), raw=None)
    assert runtime.resume_calls == [(pause, answers)]


def test_resume_wraps_runtime_exception(example):
    error = RuntimeError("resume failed")
    kernel, _, _ = make_kernel(example, runtime=FakeRuntime(error=error))

    outcome = kernel.resume(make_pause(), {"confirmation": "yes"})

    assert outcome == Failed(stage="runtime", cause=error)


def test_describe_returns_only_actions_visible_to_principal(example, guest):
    kernel, _, _ = make_kernel(example)

    assert kernel.describe(guest) == (
        {
            "key": "chat",
            "emoji": "💬",
            "blurb": "Have a conversation",
            "say": "Ask me anything.",
            "trigger": None,
        },
    )
