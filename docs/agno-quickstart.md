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
`Principal`. Each AgentOS run invokes that configured kernel exactly once.

Passing `principal=principal` configures one fixed principal for every run. It
is supported only for local use, such as development and tests, where the
caller is already trusted. It must not stand in for the identity of a remote
AgentOS request.

For authenticated requests, configure `KernelAgent` with an
`AgentOSPrincipalResolver` instead. Its mapper is application policy: the
application must decide how verified roles, scopes, and claims become a
`Principal` and its capabilities. `agent-kernel` does not map application roles
to capabilities.

```python
from agent_kernel import Principal, Role
from agent_kernel.integrations.agno import (
    AgentOSPrincipalResolver,
    AgentOSTrustedStateMiddleware,
    KernelAgent,
    TrustedAgentOSIdentity,
)


def map_identity(identity: TrustedAgentOSIdentity) -> Principal:
    # Replace this example with the application's reviewed authorization policy.
    capabilities = frozenset(
        {"spin_wheel"} if "spinner" in identity.roles else set()
    )
    return Principal(
        id=identity.user_id,
        role=Role(name="agentos-user", capabilities=capabilities),
    )


hosted = KernelAgent(
    id="kernel-spinner",
    name="Kernel Spinner",
    description="A capability-gated spinner",
    kernel=kernel,
    principal_resolver=AgentOSPrincipalResolver(map_identity),
    db=db,
)


def create_agent_os(*, agents: list[KernelAgent], db: SqliteDb) -> AgentOS:
    return AgentOS(id="kernel-spinner-os", agents=agents, db=db)


app = create_agent_os(agents=[hosted], db=db).get_app()
app = AgentOSTrustedStateMiddleware(app)
```

Independent of the factory's concrete arguments, the supported composition is:

```python
app = create_agent_os(...).get_app()
app = AgentOSTrustedStateMiddleware(app)
```

The wrapper ordering is part of the trust boundary. The trusted-state
middleware must be **outside** AgentOS authentication, as shown above, so it
holds a reference to the same mutable ASGI `state` mapping that AgentOS later
populates with verified authentication state. Do not populate that mapping
from request form fields, headers, or other unverified transport input.

`AgentOSPrincipalResolver` refuses to resolve a principal when authentication
is absent, verified identity state is missing, or the transport user/session
IDs do not match the verified state. It copies the verified identity into an
immutable request-scoped value before invoking the application's mapper.

Streaming is reconstructed only after the kernel completes rather than emitted
token by token. Within AgentOS's base started/completed lifecycle events, the
adapter emits each recorded tool-start/tool-completed pair in its original
order, followed by exactly one final content event. Native HITL continuation
and background or resumable AgentOS runs are not yet supported.
