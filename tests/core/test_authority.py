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


def test_plan_snapshots_mutable_capability_collections(example):
    declared = {"spin_wheel"}
    granted = {"spin_wheel"}

    def build_mutable(intent: ExampleIntent, context) -> Briefing:
        return Briefing(
            instructions=["spin once"],  # type: ignore[arg-type]
            grants=granted,  # type: ignore[arg-type]
        )

    mutable = ActionSpec(
        name="mutable",
        kind="privileged",
        build=build_mutable,
        capabilities=declared,  # type: ignore[arg-type]
    )
    registry = Registry(
        example.registry.specs + (mutable,),
        default=example.registry.default,
        denied=example.registry.denied,
        intent_type=example.intent_type,
    )
    role_capabilities = {"spin_wheel"}
    principal = Principal(
        id="member-1",
        role=Role(
            name="member",
            capabilities=role_capabilities,  # type: ignore[arg-type]
        ),
    )

    plan = Planner(registry, confidence_threshold=0.75).plan(
        example.intent_type(action="mutable", confidence=1, brief="spin"),
        principal,
    )
    declared.add("delete_world")
    granted.add("delete_world")
    role_capabilities.add("delete_world")

    assert plan.capabilities == frozenset({"spin_wheel"})
    assert plan.envelope.allowed == plan.capabilities
    assert plan.instructions == ("spin once",)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: Role(name="member", capabilities={""}),  # type: ignore[arg-type]
            "role capabilities",
        ),
        (
            lambda: ActionSpec(
                name="bad",
                kind="privileged",
                build=lambda intent, context: Briefing(instructions=("x",)),
                capabilities={1},  # type: ignore[arg-type]
            ),
            "action capabilities",
        ),
        (
            lambda: ActionSpec(
                name="bad",
                kind="privileged",
                build=lambda intent, context: Briefing(instructions=("x",)),
                capabilities=1,  # type: ignore[arg-type]
            ),
            "action capabilities",
        ),
        (
            lambda: ActionSpec(
                name="bad",
                kind="privileged",
                build=lambda intent, context: Briefing(instructions=("x",)),
                capabilities="root",  # type: ignore[arg-type]
            ),
            "action capabilities",
        ),
        (
            lambda: Briefing(
                instructions=("x",),
                grants={" "},  # type: ignore[arg-type]
            ),
            "briefing grants",
        ),
    ],
)
def test_capability_boundaries_reject_invalid_names(factory, message):
    with pytest.raises(ConfigurationError, match=message):
        factory()


def test_execution_plan_rejects_capabilities_that_differ_from_envelope():
    from agent_kernel.core.results import ExecutionPlan

    with pytest.raises(ConfigurationError, match="envelope"):
        ExecutionPlan(
            action="spin",
            label="Spin",
            capabilities=frozenset({"spin_wheel"}),
            instructions=("spin",),
            tool_call_limit=1,
            reads_history=False,
            envelope=AuthorityEnvelope("member-1", frozenset()),
            reason=None,
        )


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), float("-inf")])
def test_mutated_non_finite_confidence_falls_closed(example, member, confidence):
    intent = example.intent_type(
        action="spin",
        confidence=1,
        brief="spin",
        mode="safe",
    )
    intent.confidence = confidence

    plan = example.planner.plan(intent, member)

    assert plan.action == "chat"
    assert plan.capabilities == frozenset()
    assert plan.envelope.allowed == frozenset()
    assert plan.reason == "invalid intent"


def test_builder_can_use_catalog_and_summary_without_recursive_planning(
    example,
    member,
):
    observed = {}

    def build_visible(intent: ExampleIntent, context) -> Briefing:
        observed["catalog"] = context.catalog()
        observed["summary"] = context.summary()
        return Briefing(instructions=("show the catalog",))

    visible = ActionSpec(
        name="visible",
        kind="toolfree",
        build=build_visible,
        catalog=example.registry.get("chat").catalog,
    )
    registry = Registry(
        example.registry.specs + (visible,),
        default=example.registry.default,
        denied=example.registry.denied,
        intent_type=example.intent_type,
    )
    planner = Planner(registry, confidence_threshold=0.75)

    plan = planner.plan(
        example.intent_type(action="visible", confidence=1, brief="show"),
        member,
    )

    assert plan.action == "visible"
    assert observed["catalog"]
    assert "chat" in observed["summary"]


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
