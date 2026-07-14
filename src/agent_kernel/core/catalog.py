from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from agent_kernel.core.intents import Intent
from agent_kernel.core.results import Principal
from agent_kernel.core.specs import CatalogEntry

if TYPE_CHECKING:
    from agent_kernel.core.authority import Planner

I = TypeVar("I", bound=Intent)  # noqa: E741


def describe(
    planner: Planner[I],
    principal: Principal,
) -> tuple[dict[str, str | None], ...]:
    visible: list[CatalogEntry] = []
    for spec in planner.registry.specs:
        probe = planner.registry.intent_type.model_validate(
            {
                "action": spec.name,
                "confidence": 1.0,
                "brief": "",
            }
        )
        plan = planner.plan(probe, principal)
        if plan.action == planner.registry.denied:
            continue
        visible.extend(spec.catalog)

    visible_keys = {entry.key for entry in visible}
    current = (
        entry
        for entry in visible
        if entry.superseded_by is None or entry.superseded_by not in visible_keys
    )
    return tuple(
        {
            "key": entry.key,
            "emoji": entry.emoji,
            "blurb": entry.blurb,
            "say": entry.say,
            "trigger": entry.trigger,
        }
        for entry in current
    )


def summary_line(planner: Planner[I], principal: Principal) -> str:
    return ", ".join(entry["key"] or "" for entry in describe(planner, principal))
