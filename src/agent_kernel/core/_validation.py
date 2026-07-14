from collections.abc import Iterable
from typing import cast

from agent_kernel.core.errors import ConfigurationError


def capability_set(values: object, *, field: str) -> frozenset[str]:
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise ConfigurationError(f"{field} must be a collection of non-empty strings")
    snapshot = tuple(values)
    invalid = [value for value in snapshot if not isinstance(value, str) or not value.strip()]
    if invalid:
        raise ConfigurationError(f"{field} must contain non-empty strings")
    return frozenset(cast(tuple[str, ...], snapshot))


def string_tuple(values: Iterable[object], *, field: str) -> tuple[str, ...]:
    snapshot = tuple(values)
    if any(not isinstance(value, str) for value in snapshot):
        raise ConfigurationError(f"{field} must contain strings")
    return cast(tuple[str, ...], snapshot)
