from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, TypeVar

from agent_kernel.core.errors import ConfigurationError
from agent_kernel.core.specs import Registry

_FORMAT_VERSION = 1


class ArtifactModule(Protocol):
    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...


M = TypeVar("M", bound=ArtifactModule)


def _authority_vocabularies(
    registry: Registry[Any],
) -> dict[str, dict[str, list[str]]]:
    vocabularies: dict[str, dict[str, list[str]]] = {}
    for spec in registry.specs:
        continuation = spec.continuation
        if continuation is None:
            continue
        vocabularies[spec.name] = {
            field_name: sorted(values) for field_name, values in sorted(continuation.authority_fields.items())
        }
    return dict(sorted(vocabularies.items()))


def registry_fingerprint(registry: Registry[Any]) -> str:
    canonical = {
        "actions": sorted(registry.actions),
        "continuation_authority": _authority_vocabularies(registry),
        "default": registry.default,
        "denied": registry.denied,
        "intent_schema": registry.intent_type.model_json_schema(),
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_path(program_path: Path) -> Path:
    return program_path.with_suffix(".metadata.json")


def save_artifact(
    *,
    registry: Registry[Any],
    module: ArtifactModule,
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module.save(path)

    metadata = {
        "format_version": _FORMAT_VERSION,
        "registry_fingerprint": registry_fingerprint(registry),
        "program_path": path.name,
    }
    _metadata_path(path).write_text(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def load_artifact(
    *,
    registry: Registry[Any],
    module: M,
    path: Path,
) -> M:
    path = Path(path)
    metadata_path = _metadata_path(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    expected_fingerprint = registry_fingerprint(registry)
    artifact_fingerprint = metadata.get("registry_fingerprint")
    if artifact_fingerprint != expected_fingerprint:
        raise ConfigurationError("DSPy artifact registry fingerprint does not match the configured registry")
    if metadata.get("format_version") != _FORMAT_VERSION:
        raise ConfigurationError("unsupported DSPy artifact format version")

    program_name = metadata.get("program_path")
    if not isinstance(program_name, str) or program_name != path.name:
        raise ConfigurationError("DSPy artifact metadata program path does not match the requested artifact")

    module.load(metadata_path.parent / program_name)
    return module
