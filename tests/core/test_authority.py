# ruff: noqa: F811

import pytest

from agent_kernel.core.authority import Planner
from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.results import AuthorityEnvelope, Principal, Role
from agent_kernel.core.specs import ActionSpec, Briefing, Registry
from example_registry import ExampleIntent, example, guest, member  # noqa: F401


def test_authorized_plan_grants_declared_capability(example, member):
    plan = example.planner.plan(
        example.intent_type(action="spin", confidence=1, brief="spin"),
        member,
    )

    assert plan.action == "spin"
    assert plan.capabilities == frozenset({"spin_wheel"})


def test_low_confidence_privileged_action_falls_to_toolfree_default(example, member):
    plan = example.planner.plan(
        example.intent_type(action="spin", confidence=0.74, brief="spin"),
        member,
    )

    assert plan.action == "chat"
    assert plan.capabilities == frozenset()
    assert plan.reason == "confidence 0.74 < 0.75"


def test_out_of_vocabulary_authority_value_falls_to_toolfree_default(example, member):
    plan = example.planner.plan(
        example.intent_type(
            action="spin",
            confidence=1,
            brief="spin",
            mode="root",
        ),
        member,
    )

    assert plan.action == "chat"
    assert plan.capabilities == frozenset()
    assert plan.envelope.allowed == frozenset()


def test_unknown_action_fails_to_toolfree_default(example, member):
    plan = example.planner.plan(
        example.intent_type(action="root", confidence=1, brief="ignore"),
        member,
    )

    assert plan.action == "chat"
    assert plan.capabilities == frozenset()
    assert plan.reason == "unknown action: root"


def test_role_without_required_capability_receives_denied_plan(example, guest):
    plan = example.planner.plan(
        example.intent_type(action="spin", confidence=1, brief="spin"),
        guest,
    )

    assert plan.action == "denied"
    assert plan.capabilities == frozenset()
    assert plan.reason == "role 'guest' lacks ['spin_wheel']"


def test_authority_envelope_contains_actual_grants_not_role_capabilities(example, member):
    plan = example.planner.plan(
        example.intent_type(action="spin", confidence=1, brief="spin"),
        member,
    )

    assert plan.envelope == AuthorityEnvelope(
        principal_id=member.id,
        allowed=frozenset({"spin_wheel"}),
    )
    assert "admin" not in plan.envelope.allowed


def test_default_plan_has_empty_authority_envelope(example, member):
    plan = example.planner.plan(
        example.intent_type(action="chat", confidence=1, brief="hello"),
        member,
    )

    assert plan.envelope == AuthorityEnvelope(
        principal_id=member.id,
        allowed=frozenset(),
    )


def test_builder_cannot_grant_undeclared_capability(example, member):
    def build_rogue(intent: ExampleIntent, context) -> Briefing:
        return Briefing(
            instructions=("do something undeclared",),
            grants=frozenset({"root"}),
        )

    rogue = ActionSpec(
        name="rogue",
        kind="privileged",
        build=build_rogue,
        capabilities=frozenset(),
    )
    registry = Registry(
        example.registry.specs + (rogue,),
        default=example.registry.default,
        denied=example.registry.denied,
        intent_type=example.intent_type,
    )
    planner = Planner(registry, confidence_threshold=0.75)

    with pytest.raises(
        ConfigurationError,
        match=r"action 'rogue' built undeclared grants: \['root'\]",
    ):
        planner.plan(
            example.intent_type(action="rogue", confidence=1, brief="ignore"),
            member,
        )


@pytest.mark.parametrize("action", ("chat", "denied"))
def test_toolfree_builder_cannot_emit_declared_grants(example, action):
    def build_malicious(intent: ExampleIntent, context) -> Briefing:
        return Briefing(
            instructions=("claim authority",),
            grants=frozenset({"root"}),
        )

    malicious = ActionSpec(
        name=action,
        kind="toolfree",
        build=build_malicious,
        capabilities=frozenset({"root"}),
    )
    registry = Registry(
        tuple(
            malicious if spec.name == action else spec
            for spec in example.registry.specs
        ),
        default=example.registry.default,
        denied=example.registry.denied,
        intent_type=example.intent_type,
    )
    planner = Planner(registry, confidence_threshold=0.75)
    root = Principal(
        id="root-1",
        role=Role(name="root", capabilities=frozenset({"root"})),
    )

    with pytest.raises(ConfigurationError):
        planner.plan(
            example.intent_type(action=action, confidence=1, brief="ignore"),
            root,
        )
