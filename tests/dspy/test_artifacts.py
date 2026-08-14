# ruff: noqa: F811

import json
from dataclasses import replace
from pathlib import Path

import dspy
import pytest

from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.specs import Registry
from agent_kernel.core.specs import ContinuationSpec
from agent_kernel.integrations.dspy import (
    DspyClassifier,
    build_signature,
    load_artifact,
    registry_fingerprint,
    save_artifact,
)
from tests.core.example_registry import (  # noqa: F401
    CHAT,
    DENIED,
    SPIN,
    ExampleIntent,
    example,
)


class FakeProgram:
    def __init__(self):
        self.save_calls = []
        self.load_calls = []

    def save(self, path):
        path = Path(path)
        self.save_calls.append(path)
        path.write_text("{}", encoding="utf-8")

    def load(self, path):
        self.load_calls.append(Path(path))


def _metadata_path(program_path):
    candidates = [candidate for candidate in program_path.parent.iterdir() if candidate != program_path]
    assert len(candidates) == 1
    return candidates[0]


def _incompatible_registry():
    class IncompatibleIntent(ExampleIntent):
        source: str | None = None

    return Registry(
        (CHAT, DENIED, SPIN),
        default="chat",
        denied="denied",
        intent_type=IncompatibleIntent,
    )


def test_save_artifact_writes_exact_metadata_beside_program(
    tmp_path,
    example,
):
    program = FakeProgram()
    program_path = tmp_path / "classifier.json"

    save_artifact(
        registry=example.registry,
        module=program,
        path=program_path,
    )

    assert program.save_calls == [program_path]
    metadata_path = _metadata_path(program_path)
    assert metadata_path.parent == program_path.parent
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == {
        "format_version": 1,
        "registry_fingerprint": registry_fingerprint(example.registry),
        "program_path": program_path.name,
    }


def test_load_artifact_rejects_registry_mismatch_before_dspy_load(
    tmp_path,
    example,
):
    program_path = tmp_path / "classifier.json"
    save_artifact(
        registry=example.registry,
        module=FakeProgram(),
        path=program_path,
    )
    target = FakeProgram()

    with pytest.raises(ConfigurationError, match="fingerprint"):
        load_artifact(
            registry=_incompatible_registry(),
            module=target,
            path=program_path,
        )

    assert target.load_calls == []


def test_registry_fingerprint_binds_authority_values_to_each_action(example):
    spin_safe = replace(
        SPIN,
        continuation=ContinuationSpec(authority_fields={"mode": frozenset({"safe"})}),
    )
    review_fast = replace(
        SPIN,
        name="review",
        continuation=ContinuationSpec(authority_fields={"mode": frozenset({"fast"})}),
    )
    first = Registry(
        (CHAT, DENIED, spin_safe, review_fast),
        default="chat",
        denied="denied",
        intent_type=ExampleIntent,
    )
    second = Registry(
        (
            CHAT,
            DENIED,
            replace(
                spin_safe,
                continuation=ContinuationSpec(authority_fields={"mode": frozenset({"fast"})}),
            ),
            replace(
                review_fast,
                continuation=ContinuationSpec(authority_fields={"mode": frozenset({"safe"})}),
            ),
        ),
        default="chat",
        denied="denied",
        intent_type=ExampleIntent,
    )

    assert registry_fingerprint(first) != registry_fingerprint(second)


@pytest.mark.parametrize(
    "tampered_program_path",
    [
        "/tmp/other-classifier.json",
        "../other-classifier.json",
        "other-classifier.json",
    ],
    ids=["absolute", "traversal", "different-name"],
)
def test_load_artifact_rejects_tampered_program_path_before_dspy_load(
    tmp_path,
    example,
    tampered_program_path,
):
    program_path = tmp_path / "classifier.json"
    save_artifact(
        registry=example.registry,
        module=FakeProgram(),
        path=program_path,
    )
    metadata_path = _metadata_path(program_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["program_path"] = tampered_program_path
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    target = FakeProgram()

    with pytest.raises(ConfigurationError, match="program path"):
        load_artifact(
            registry=example.registry,
            module=target,
            path=program_path,
        )

    assert target.load_calls == []


def test_load_artifact_loads_verified_program(tmp_path, example):
    program_path = tmp_path / "classifier.json"
    save_artifact(
        registry=example.registry,
        module=FakeProgram(),
        path=program_path,
    )
    target = FakeProgram()

    loaded = load_artifact(
        registry=example.registry,
        module=target,
        path=program_path,
    )

    assert loaded is target
    assert target.load_calls == [program_path]


def test_classifier_from_artifact_constructs_and_loads_projected_dspy_module(
    tmp_path,
    example,
):
    program_path = tmp_path / "classifier.json"
    source = dspy.Predict(build_signature(example.registry))
    save_artifact(
        registry=example.registry,
        module=source,
        path=program_path,
    )

    classifier = DspyClassifier.from_artifact(
        registry=example.registry,
        path=program_path,
    )

    assert isinstance(classifier.module, dspy.Predict)
    assert set(classifier.module.signature.input_fields) == {"message"}
    assert set(classifier.module.signature.output_fields) == {
        "action",
        "mode",
        "confidence",
        "brief",
    }
