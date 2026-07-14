from typing import Any


class UnsupportedRequirement(ValueError):
    def __init__(self, requirement_type: str):
        self.requirement_type = requirement_type
        super().__init__(
            f"Agno requirement type is not supported: {requirement_type}"
        )


def _active_requirements(response: object) -> tuple[Any, ...]:
    active = getattr(response, "active_requirements", None)
    if active is not None:
        return tuple(active)
    return tuple(getattr(response, "requirements", None) or ())


def _requirement_type(requirement: object) -> str:
    if getattr(requirement, "needs_external_execution", False):
        return "external_execution"
    if getattr(requirement, "needs_confirmation", False):
        return "confirmation"
    if getattr(requirement, "needs_user_feedback", False):
        return "user_feedback"
    if getattr(requirement, "needs_user_input", False):
        return "user_input"
    return "unknown"


def translate_requirements(
    response: object,
) -> tuple[dict[str, object], ...]:
    translated: list[dict[str, object]] = []
    for requirement in _active_requirements(response):
        requirement_type = _requirement_type(requirement)
        if requirement_type != "user_input":
            translated.append({"type": requirement_type})
            continue

        for field in getattr(requirement, "user_input_schema", None) or ():
            field_type = getattr(field, "field_type", None)
            type_name = (
                getattr(field_type, "__name__", str(field_type))
                if field_type is not None
                else None
            )
            translated.append(
                {
                    "field": getattr(field, "name", None),
                    "description": getattr(field, "description", None),
                    "type": type_name,
                }
            )
    return tuple(translated)


def apply_user_input(
    response: object,
    answers: dict[str, str],
) -> None:
    prepared: list[tuple[Any, dict[str, str]]] = []
    for requirement in _active_requirements(response):
        requirement_type = _requirement_type(requirement)
        if requirement_type != "user_input":
            raise UnsupportedRequirement(requirement_type)

        field_names: list[str] = []
        for field in getattr(requirement, "user_input_schema", None) or ():
            field_name = getattr(field, "name", None)
            if not isinstance(field_name, str):
                raise ValueError(
                    "Agno user-input requirement has an invalid field name"
                )
            field_names.append(field_name)

        missing = [name for name in field_names if name not in answers]
        if missing:
            raise ValueError(
                f"missing answers for Agno user-input fields: {missing}"
            )
        prepared.append(
            (
                requirement,
                {name: answers[name] for name in field_names},
            )
        )

    for requirement, values in prepared:
        requirement.provide_user_input(values)
