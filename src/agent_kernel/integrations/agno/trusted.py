from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from agent_kernel.core.results import Principal
from agent_kernel.integrations.agno.context import AgnoRunContext


_agentos_state: ContextVar[Mapping[str, object] | None] = ContextVar(
    "agentos_state",
    default=None,
)


def current_agentos_state() -> Mapping[str, object] | None:
    return _agentos_state.get()


def _immutable(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_immutable(item) for item in value)
    return value


@dataclass(frozen=True)
class TrustedAgentOSIdentity:
    user_id: str
    session_id: str | None
    roles: tuple[str, ...]
    scopes: frozenset[str]
    claims: Mapping[str, object]


class AgentOSTrustedStateMiddleware:
    def __init__(
        self,
        app: Callable[
            [MutableMapping[str, Any], object, object],
            Awaitable[object],
        ],
    ) -> None:
        self.app = app

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: object,
        send: object,
    ) -> object:
        state = scope.setdefault("state", {})
        if not isinstance(state, MutableMapping):
            raise TypeError('AgentOS scope "state" must be a mutable mapping')
        token = _agentos_state.set(state)
        try:
            return await self.app(scope, receive, send)
        finally:
            _agentos_state.reset(token)


@dataclass(frozen=True)
class AgentOSPrincipalResolver:
    mapper: Callable[[TrustedAgentOSIdentity], Principal]

    def resolve(self, runtime_context: object) -> Principal:
        state = current_agentos_state()
        if state is None or state.get("authenticated") is not True:
            raise PermissionError("verified AgentOS request context is required")
        if not isinstance(runtime_context, AgnoRunContext):
            raise PermissionError("verified AgentOS request context is required")

        user_id = state.get("user_id")
        session_id = state.get("session_id")
        if not isinstance(user_id, str) or not user_id:
            raise PermissionError("verified AgentOS user ID is required")
        if runtime_context.user_id != user_id:
            raise PermissionError("transport user ID does not match verified identity")
        if state.get("session_id") is not None and runtime_context.session_id != session_id:
            raise PermissionError("transport session ID does not match verified identity")

        raw_scopes = state.get("scopes")
        scopes = (
            frozenset(scope for scope in raw_scopes if isinstance(scope, str))
            if isinstance(raw_scopes, (list, tuple, set, frozenset))
            else frozenset()
        )
        raw_claims = state.get("claims")
        claims = raw_claims if isinstance(raw_claims, Mapping) else {}
        raw_roles = claims.get("roles")
        roles = (
            tuple(role for role in raw_roles if isinstance(role, str))
            if isinstance(raw_roles, list)
            else ()
        )
        immutable_claims = _immutable(claims)
        assert isinstance(immutable_claims, Mapping)
        identity = TrustedAgentOSIdentity(
            user_id=user_id,
            session_id=session_id if isinstance(session_id, str) else None,
            roles=roles,
            scopes=scopes,
            claims=immutable_claims,
        )
        return self.mapper(identity)
