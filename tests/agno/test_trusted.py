import asyncio

import pytest

from agent_kernel import Principal, Role
from agent_kernel.integrations.agno import (
    AgentOSPrincipalResolver,
    AgentOSTrustedStateMiddleware,
    AgnoRunContext,
)


@pytest.mark.asyncio
async def test_verified_state_is_available_to_request_time_resolver():
    resolved = []
    resolver = AgentOSPrincipalResolver(
        lambda identity: resolved.append(identity)
        or Principal(identity.user_id, Role("operator", frozenset(identity.scopes)))
    )

    async def authenticated_app(scope, receive, send):
        scope["state"].update(
            authenticated=True,
            user_id="verified-alice",
            session_id="session-7",
            scopes=["agents:agno-operations:run"],
            claims={"roles": ["operations"]},
        )
        principal = resolver.resolve(
            AgnoRunContext(user_id="verified-alice", session_id="session-7")
        )
        assert principal.id == "verified-alice"

    app = AgentOSTrustedStateMiddleware(authenticated_app)
    await app({"type": "http", "state": {}}, object(), object())
    assert resolved[0].roles == ("operations",)


def test_resolver_rejects_unverified_and_transport_mismatched_identity():
    resolver = AgentOSPrincipalResolver(
        lambda identity: Principal(identity.user_id, Role("member", frozenset()))
    )
    with pytest.raises(PermissionError, match="verified AgentOS request context"):
        resolver.resolve(AgnoRunContext(user_id="form-admin", session_id="s1"))


@pytest.mark.asyncio
async def test_resolver_rejects_transport_identity_mismatches():
    resolver = AgentOSPrincipalResolver(
        lambda identity: Principal(identity.user_id, Role("member", frozenset()))
    )

    async def authenticated_app(scope, receive, send):
        scope["state"].update(
            authenticated=True,
            user_id="verified-alice",
            session_id="session-7",
        )
        with pytest.raises(PermissionError, match="transport user ID"):
            resolver.resolve(AgnoRunContext(user_id="form-admin", session_id="session-7"))
        with pytest.raises(PermissionError, match="transport session ID"):
            resolver.resolve(
                AgnoRunContext(user_id="verified-alice", session_id="session-other")
            )

    app = AgentOSTrustedStateMiddleware(authenticated_app)
    await app({"type": "http", "state": {}}, object(), object())


@pytest.mark.asyncio
async def test_concurrent_requests_have_immutable_isolated_identities():
    identities = []
    resolver = AgentOSPrincipalResolver(
        lambda identity: identities.append(identity)
        or Principal(identity.user_id, Role("member", frozenset(identity.scopes)))
    )
    first_ready = asyncio.Event()
    second_ready = asyncio.Event()

    async def authenticated_app(scope, receive, send):
        user_id = scope["user_id"]
        scope["state"].update(
            authenticated=True,
            user_id=user_id,
            session_id=f"session-{user_id}",
            scopes=[f"scope:{user_id}"],
            claims={"roles": [f"role:{user_id}"], "nested": {"mutable": True}},
        )
        (first_ready if user_id == "alice" else second_ready).set()
        await (second_ready if user_id == "alice" else first_ready).wait()
        resolver.resolve(
            AgnoRunContext(user_id=user_id, session_id=f"session-{user_id}")
        )

    app = AgentOSTrustedStateMiddleware(authenticated_app)
    await asyncio.gather(
        app({"type": "http", "state": {}, "user_id": "alice"}, object(), object()),
        app({"type": "http", "state": {}, "user_id": "bob"}, object(), object()),
    )

    by_user = {identity.user_id: identity for identity in identities}
    assert by_user["alice"].scopes == frozenset({"scope:alice"})
    assert by_user["bob"].roles == ("role:bob",)
    with pytest.raises(TypeError):
        by_user["alice"].claims["extra"] = True


@pytest.mark.asyncio
async def test_request_state_is_reset_after_app_returns():
    resolver = AgentOSPrincipalResolver(
        lambda identity: Principal(identity.user_id, Role("member", frozenset()))
    )

    async def authenticated_app(scope, receive, send):
        scope["state"].update(authenticated=True, user_id="verified-alice")

    app = AgentOSTrustedStateMiddleware(authenticated_app)
    await app({"type": "http", "state": {}}, object(), object())

    with pytest.raises(PermissionError, match="verified AgentOS request context"):
        resolver.resolve(AgnoRunContext(user_id="verified-alice"))
