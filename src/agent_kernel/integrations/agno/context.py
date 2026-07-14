from dataclasses import dataclass


@dataclass(frozen=True)
class AgnoRunContext:
    user_id: str | None = None
    session_id: str | None = None
    session_state: dict[str, object] | None = None
    metadata: dict[str, object] | None = None
