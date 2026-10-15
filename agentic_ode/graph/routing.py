"""Routing: every conditional edge reads the next node from the state's "route" field."""

from __future__ import annotations

from agentic_ode.core.state import DialogueState


def route(state: DialogueState) -> str:
    return state["route"]
