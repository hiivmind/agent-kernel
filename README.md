# agent-kernel

`agent-kernel` is a framework-neutral authority kernel for capability-gated
agents. A typed registry describes logical actions and their maximum
capabilities; a principal supplies the available authority; adapters classify
and execute one bounded turn.

Requires Python 3.12 or newer.

```bash
python -m pip install agent-kernel
```

The core package has no agent-framework dependency. See the complete
network-free example below and the focused guides for Agno and DSPy usage.

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


kernel = Kernel(KernelConfig(registry), FakeClassifier(), FakeRuntime())
principal = Principal(
    id="member-1",
    role=Role(name="member", capabilities=frozenset({"spin_wheel"})),
)
turn = kernel.run("Please spin the wheel", principal=principal)
print(turn.plan.action, turn.plan.capabilities, turn.outcome)
```

- [Core quickstart](https://github.com/hiivmind/agent-kernel/blob/main/docs/core-quickstart.md)
- [Agno quickstart](https://github.com/hiivmind/agent-kernel/blob/main/docs/agno-quickstart.md)
- [DSPy classifiers and artifacts](https://github.com/hiivmind/agent-kernel/blob/main/docs/dspy.md)
