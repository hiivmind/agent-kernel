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

## Host the kernel in AgentOS

```python
from agno.db.sqlite import SqliteDb
from agno.os import AgentOS
from agent_kernel.integrations.agno import KernelAgent

db = SqliteDb(db_file="tmp/spinner-agentos.db")
hosted = KernelAgent(
    id="kernel-spinner",
    name="Kernel Spinner",
    description="A capability-gated spinner",
    kernel=kernel,
    principal=principal,
    db=db,
)
agent_os = AgentOS(id="kernel-spinner-os", agents=[hosted], db=db)
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app=app, port=7777)
```

`KernelAgent` is a hosting adapter: it gives AgentOS access to the configured
kernel without adding a wrapper LLM. The kernel's inner `AgnoRuntime` still
enforces its authorized execution plan, using the application-supplied
`Principal`. Streaming is reconstructed after the kernel finishes rather than
emitted token by token. Native HITL continuation and background or resumable
AgentOS runs are not yet supported.
