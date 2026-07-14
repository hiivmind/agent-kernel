# Core quickstart

Install the framework-neutral core:

```bash
python -m pip install agent-kernel
```

An `ActionSpec` is tool-free or privileged. A privileged action declares its
maximum logical capabilities in `capabilities`; its builder selects the grants
for the current turn. The kernel bounds that plan with the principal's role
before the runtime sees it.

```python
from agent_kernel import (
    ActionSpec, Briefing, BuildContext, Completed, Continuation, ExecutionPlan,
    Intent, Kernel, KernelConfig, Pause, Principal, Registry, Role,
)


class SpinnerIntent(Intent):
    pass


def build_chat(intent: SpinnerIntent, context: BuildContext) -> Briefing:
    del context
    return Briefing(instructions=(f"Reply to: {intent.brief}",), label="Chat")


def build_denied(intent: SpinnerIntent, context: BuildContext) -> Briefing:
    del intent, context
    return Briefing(instructions=("Explain that this is not authorized.",))


def build_spin(intent: SpinnerIntent, context: BuildContext) -> Briefing:
    del context
    return Briefing(
        instructions=(f"Spin once for: {intent.brief}",),
        grants=frozenset({"spin_wheel"}),
        tool_call_limit=1,
    )


registry: Registry[SpinnerIntent] = Registry(
    (
        ActionSpec(name="chat", kind="toolfree", build=build_chat),
        ActionSpec(name="denied", kind="toolfree", build=build_denied),
        ActionSpec(
            name="spin", kind="privileged", build=build_spin,
            capabilities=frozenset({"spin_wheel"}), tool_call_limit=1,
        ),
    ),
    default="chat", denied="denied", intent_type=SpinnerIntent,
)


class FakeClassifier:
    def classify(self, message: str, *, context: Continuation | None = None) -> SpinnerIntent:
        del context
        action = "spin" if "spin" in message.lower() else "chat"
        return SpinnerIntent(action=action, confidence=1.0, brief=message)


class FakeRuntime:
    def execute(
        self, message: str, plan: ExecutionPlan, *, context: object | None = None,
    ) -> Completed:
        del message, context
        return Completed(
            content="blue" if "spin_wheel" in plan.capabilities else "hello",
            raw=None,
        )

    def resume(self, pause: Pause, answers: dict[str, str]) -> Completed:
        del pause
        return Completed(content=answers, raw=None)


kernel = Kernel(
    KernelConfig(registry, confidence_threshold=0.75, identity="Spinner"),
    FakeClassifier(),
    FakeRuntime(),
)
principal = Principal(
    id="member-1",
    role=Role(name="member", capabilities=frozenset({"spin_wheel"})),
)
turn = kernel.run("Please spin the wheel", principal=principal)
assert turn.plan.action == "spin"
assert turn.plan.capabilities == frozenset({"spin_wheel"})
print(turn.outcome)
```

The registry's `default` and `denied` actions must both be tool-free. Unknown
actions and low-confidence privileged classifications fall back to `default`.
If selected grants exceed the principal's capabilities, the kernel selects
`denied`. Plans contain logical capability names only; concrete tool binding
belongs to a runtime adapter.

Catalog visibility is conservative: a privileged action is advertised only
when the principal holds all of its maximum declared capabilities. Builders
are not executed while producing a catalog, so a branch that would select
fewer grants does not make the action visible to a less-privileged principal.
