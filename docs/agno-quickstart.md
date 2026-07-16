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

## Execute a governed Agent Skill operation

An operation selects exactly one Skill and binds logical capabilities to the
concrete tools available for that invocation. The example below uses one local
Skill provider, an authorization-only `skills:invoke:status` binding, and a
resource binding for the `read_status` function.

```python
from collections.abc import Mapping

from pydantic import BaseModel, TypeAdapter

from agent_kernel import (
    AuthorityEnvelope,
    ExecutionPlan,
    OperationContract,
    OperationInvocation,
    SubjectContext,
)
from agent_kernel.integrations.agno import (
    AgnoCapabilityBinding,
    AgnoInvocationContext,
    AgnoSkillRuntime,
    AgnoSkillSource,
)


class StatusRequest(BaseModel):
    resource_id: str


class StatusResult(BaseModel):
    status: str


class LocalStatusSkill:
    def require(self, skill_id: str) -> AgnoSkillSource:
        if skill_id != "status":
            raise KeyError(skill_id)
        return AgnoSkillSource("status", "/srv/agent-skills/status")


def read_status(resource_id: str) -> str:
    return application_status_store.read(resource_id)


class StatusBindings:
    def bindings_for(
        self,
        invocation: OperationInvocation[object],
        context: AgnoInvocationContext,
    ) -> Mapping[str, AgnoCapabilityBinding]:
        del invocation, context
        return {
            # Authorizes selection of the Skill; it exposes no external tool.
            "skills:invoke:status": AgnoCapabilityBinding(
                "skills:invoke:status"
            ),
            "resources:read:status": AgnoCapabilityBinding(
                "resources:read:status",
                tools=(read_status,),
                function_names=frozenset({"read_status"}),
            ),
        }


capabilities = frozenset(
    {"skills:invoke:status", "resources:read:status"}
)
envelope = AuthorityEnvelope("member-1", capabilities)
plan = ExecutionPlan(
    action="status",
    label="Read status",
    capabilities=capabilities,
    instructions=("Load the status Skill and report the resource status.",),
    tool_call_limit=2,
    reads_history=False,
    envelope=envelope,
    reason=None,
)
invocation = OperationInvocation(
    invocation_id="invocation-17",
    target_id="status",
    request="Check the selected resource.",
    subject=SubjectContext(kind="resource", id="service-42", attributes={}),
    inputs=StatusRequest(resource_id="service-42"),
    plan=plan,
)
contract = OperationContract(
    input_adapter=TypeAdapter(StatusRequest),
    output_adapter=TypeAdapter(StatusResult),
    model_output_type=StatusResult,
    model_input_builder=lambda item: item.inputs.model_dump(),
)
runtime = AgnoSkillRuntime(
    skill_provider=LocalStatusSkill(),
    binding_provider=StatusBindings(),
    model=application_model,
)
outcome = runtime.execute(
    invocation,
    contract,
    context=AgnoInvocationContext(),
)
```

`StatusResult` deliberately contains only model-produced business data. The
runtime adds correlation outside the model schema as the
`OperationCompletion.invocation_id`, so the model cannot supply or overwrite
it.

`SafeSkills` exposes Skill instructions, references, and script **source**.
Its `get_skill_script_source` tool can read a bundled script but cannot execute
it. Applications that want script execution must provide their own governed
tool and capability binding.

See [Agno upstream issue reports](agno-upstream-issues.md) for the two
non-blocking compatibility gaps behind that design.

## Human-in-the-loop state

Agno pause tokens are opaque and process-local. The runtime retains the
corresponding Agent, authority, invocation state, and requirement snapshot in
memory until that pause is resumed; a second pause replaces the first token
with a new retained record. Tokens do not survive a process restart and are
not a durable workflow store. Explicit cleanup and expiry for pauses that are
never resumed remains a lifecycle follow-up.
