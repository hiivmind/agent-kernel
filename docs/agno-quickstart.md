# Agno quickstart

```bash
python -m pip install "agent-kernel[agno]"
```

Keep the typed `SpinnerIntent`, `registry`, and `principal` from the
[core quickstart](core-quickstart.md). Bind that same registry to Agno:

```python
from agent_kernel import Kernel, KernelConfig
from agent_kernel.integrations.agno import AgnoClassifier, AgnoRunContext, AgnoRuntime
from spinner_domain import SpinnerIntent, principal, registry


def spin_wheel() -> str:
    return "blue"


def make_kernel(model: object) -> Kernel[SpinnerIntent]:
    return Kernel(
        KernelConfig(registry, confidence_threshold=0.75, identity="Spinner"),
        AgnoClassifier(
            registry=registry,
            model=model,
            instructions=["Choose exactly one action from the output schema."],
        ),
        AgnoRuntime(model=model, tools={"spin_wheel": spin_wheel}, db=None),
    )


# Inject an Agno-compatible model configured by your application.
model = application_model
kernel = make_kernel(model)
turn = kernel.run(
    "Please spin the wheel",
    principal=principal,
    runtime_context=AgnoRunContext(
        user_id=principal.id,
        session_id="session-1",
        session_state={"channel": "quickstart"},
        metadata={"request_id": "request-1"},
    ),
)
print(turn.outcome)
```

`application_model` is a dependency-injection placeholder; no provider is
hardcoded. Classification has no tools. Execution exposes only tools named by
the authorized plan, stamps the immutable authority envelope into Agno's run
dependencies, and fails closed when a binding is missing.
