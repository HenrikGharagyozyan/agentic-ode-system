"""Routing: every conditional edge reads the next node from the state's "route" field."""

from __future__ import annotations

from agentic_ode.core.state import DialogueState

# the three nodes of the hardware turn (HW3)
HW_DECIDE, HW_APPROVE, HW_CALL = "hw_decide", "hw_approve", "hw_call"


def route(state: DialogueState) -> str:
    return state["route"]
