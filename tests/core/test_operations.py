from dataclasses import fields
from inspect import Parameter, signature
from pathlib import Path

import pytest

from agent_kernel.core import (
    OperationCompletion,
    OperationContract,
    OperationInvocation,
    OperationRuntime,
    SubjectContext,
    ValueAdapter,
)
from agent_kernel.core.results import AuthorityEnvelope, ExecutionPlan


@pytest.fixture
def plan() -> ExecutionPlan:
    return ExecutionPlan(
        action="status",
        label="Corpus status",
        capabilities=frozenset(),
        instructions=("report corpus status",),
        tool_call_limit=0,
        reads_history=False,
        envelope=AuthorityEnvelope(
            principal_id="member-1",
            allowed=frozenset(),
        ),
        reason=None,
    )


def test_subject_context_recursively_freezes_descriptive_json(plan):
    attributes = {"workspace_role": "corpus-root", "tags": ["docs"]}
    subject = SubjectContext(kind="corpus", id="agno", attributes=attributes)
    attributes["workspace_role"] = "changed"
    assert subject.attributes["workspace_role"] == "corpus-root"
    assert subject.attributes["tags"] == ("docs",)
    with pytest.raises(TypeError):
        subject.attributes["new"] = "value"  # type: ignore[index]


def test_subject_context_rejects_runtime_objects():
    with pytest.raises(TypeError, match="JSON-compatible"):
        SubjectContext(kind="corpus", id="agno", attributes={"path": Path("/tmp")})


def test_operation_completion_uses_trusted_invocation_id(plan):
    invocation = OperationInvocation(
        invocation_id="run-17",
        target_id="hiivmind-corpus-status",
        request="what is the Agno corpus status?",
        subject=SubjectContext("corpus", "agno", {"workspace_role": "corpus-root"}),
        inputs={"verbose": False},
        plan=plan,
    )
    completion = OperationCompletion(invocation.invocation_id, {"summary": "healthy"}, object())
    assert completion.invocation_id == "run-17"


def test_operation_contract_and_protocol_shape_is_framework_neutral():
    assert tuple(field.name for field in fields(OperationContract)) == (
        "input_adapter",
        "output_adapter",
        "model_output_type",
        "model_input_builder",
    )
    assert tuple(signature(ValueAdapter.validate_python).parameters) == (
        "self",
        "value",
    )
    runtime_parameters = signature(OperationRuntime.execute).parameters
    assert tuple(runtime_parameters) == (
        "self",
        "invocation",
        "contract",
        "context",
    )
    assert runtime_parameters["context"].kind is Parameter.KEYWORD_ONLY
    assert runtime_parameters["context"].default is None
