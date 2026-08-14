from agent_kernel import (
    ActionSpec,
    Briefing,
    BuildContext,
    Completed,
    Continuation,
    ExecutionPlan,
    Intent,
    Kernel,
    KernelConfig,
    Pause,
    Principal,
    Registry,
    Role,
)


class SpinnerIntent(Intent):
    pass


def build_chat(intent: SpinnerIntent, context: BuildContext) -> Briefing:
    del context
    return Briefing(instructions=(f"Reply to: {intent.brief}",), label="Chat")


def build_denied(intent: SpinnerIntent, context: BuildContext) -> Briefing:
    del intent, context
    return Briefing(instructions=("Explain that the action is not authorized.",))


def build_spin(intent: SpinnerIntent, context: BuildContext) -> Briefing:
    del context
    return Briefing(
        instructions=(f"Spin once for: {intent.brief}",),
        grants=frozenset({"spin_wheel"}),
        tool_call_limit=1,
    )


REGISTRY: Registry[SpinnerIntent] = Registry(
    (
        ActionSpec(name="chat", kind="toolfree", build=build_chat),
        ActionSpec(name="denied", kind="toolfree", build=build_denied),
        ActionSpec(
            name="spin",
            kind="privileged",
            build=build_spin,
            capabilities=frozenset({"spin_wheel"}),
            tool_call_limit=1,
        ),
    ),
    default="chat",
    denied="denied",
    intent_type=SpinnerIntent,
)


class FakeClassifier:
    def classify(
        self,
        message: str,
        *,
        context: Continuation | None = None,
    ) -> SpinnerIntent:
        del context
        action = "spin" if "spin" in message.lower() else "chat"
        return SpinnerIntent(action=action, confidence=1.0, brief=message)


class SpinnerRuntime:
    def execute(
        self,
        message: str,
        plan: ExecutionPlan,
        *,
        context: object | None = None,
    ) -> Completed:
        del message, context
        if "spin_wheel" not in plan.envelope.allowed:
            raise PermissionError("spin_wheel is not authorized")
        return Completed(content="blue", raw=None)

    def resume(self, pause: Pause, answers: dict[str, str]) -> Completed:
        del pause
        return Completed(content=answers, raw=None)


kernel = Kernel(KernelConfig(REGISTRY, confidence_threshold=0.75), FakeClassifier(), SpinnerRuntime())
principal = Principal(
    id="member-1",
    role=Role(name="member", capabilities=frozenset({"spin_wheel"})),
)
turn = kernel.run("Please spin the wheel", principal=principal)

if isinstance(turn.outcome, Completed):
    print(f"authorized spinner result: {turn.outcome.content}")
else:
    raise RuntimeError(f"spinner did not complete: {turn.outcome}")
