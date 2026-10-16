"""The human node: pauses the graph with interrupt(), appends the person's reply, and routes on it."""

from __future__ import annotations

from langgraph.types import interrupt

from agentic_ode.agents.guards import FINISH, accepted, finish_allowed
from agentic_ode.core.dialogue import decision_field, last_orchestrator_turn
from agentic_ode.core.state import END_ROUTE, DialogueState, Role
from agentic_ode.graph.routing import back_to_orchestrator
from agentic_ode.graph.routing import note as _note

STEPS_NEEDING_APPROVAL = (Role.EXECUTOR.value, FINISH)


def question_of(content: str) -> str:
    """All text after `question:`. It is the last field of a human decision, so it may have several lines."""
    _, found, rest = content.partition("\nquestion:")
    return rest.strip() if found else ""


def human(state: DialogueState) -> dict:
    """Pause and ask the person. Nothing before interrupt() calls a model, so the re-run on resume is safe."""
    turn = last_orchestrator_turn(state)
    kind = decision_field(turn, "kind")
    reply = str(interrupt({"question": question_of(turn), "kind": kind})).strip()

    # Below runs only once, after resume.
    reply_turn = {"role": Role.HUMAN.value, "content": reply}
    step = state["pending_step"]
    if kind != "hard" or step not in STEPS_NEEDING_APPROVAL:
        return _back([reply_turn])  # ambiguous: the orchestrator reads the answer and decides
    if not accepted(reply):
        return _back([reply_turn, _note(f"The human refused the step '{step}'. It did NOT run.")], refused=step)
    if step == FINISH and not finish_allowed(state):
        return _back([reply_turn, _note("Finish was accepted, but the finish guard refused it: "
                                        "no RMSE from a successful run. It did NOT run.")])
    if step == FINISH:
        return {"dialogue": [reply_turn], "route": END_ROUTE, "stop_reason": "finish", "pending_step": ""}
    return {"dialogue": [reply_turn], "route": step, "pending_step": ""}


def _back(turns: list[dict], refused: str = "") -> dict:
    """The pending step does not run: give the turn back to the orchestrator."""
    return back_to_orchestrator(turns, pending_step="", refused_step=refused)
