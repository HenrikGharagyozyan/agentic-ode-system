"""Routing: every conditional edge reads the next node from the state's "route" field."""

from __future__ import annotations

from agentic_ode.core.state import DialogueState, Role

# the three nodes of the hardware turn (HW3)
HW_DECIDE, HW_APPROVE, HW_CALL = "hw_decide", "hw_approve", "hw_call"


def route(state: DialogueState) -> str:
    return state["route"]


def note(text: str) -> dict:
    """A turn written by the graph itself, not by a person or a model."""
    return {"role": Role.SYSTEM.value, "content": text}


def back_to_orchestrator(turns: list[dict], **fields) -> dict:
    """The state update of a node whose step does not run now: add `turns`, give the turn back to the orchestrator.
    `fields`: the node's own fields to reset (e.g. pending_step, hw_call)."""
    return {"dialogue": turns, "route": Role.ORCHESTRATOR.value, **fields}
