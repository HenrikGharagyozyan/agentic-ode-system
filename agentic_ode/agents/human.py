"""The human node: pauses the graph with interrupt() and appends the person's reply as a human turn."""

from __future__ import annotations

from langgraph.types import interrupt

from agentic_ode.agents.context import decision_field
from agentic_ode.state import DialogueState, Role


def last_orchestrator_turn(state: DialogueState) -> str:
    for turn in reversed(state["dialogue"]):
        if turn["role"] == Role.ORCHESTRATOR:
            return turn["content"]
    return ""


def question_of(content: str) -> str:
    """All text after `question:`. It is the last field of a human decision, so it may have several lines."""
    _, found, rest = content.partition("\nquestion:")
    return rest.strip() if found else ""


def human(state: DialogueState) -> dict:
    """Pause and ask the person. Nothing before interrupt() calls a model, so the re-run on resume is safe."""
    turn = last_orchestrator_turn(state)
    reply = interrupt({"question": question_of(turn), "kind": decision_field(turn, "kind")})
    return {"dialogue": [{"role": Role.HUMAN.value, "content": str(reply).strip()}]}
