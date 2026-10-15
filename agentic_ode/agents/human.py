"""The human node: pauses the graph with interrupt(), appends the person's reply, and routes on it."""

from __future__ import annotations

from langgraph.types import interrupt

from agentic_ode.agents.context import decision_field
from agentic_ode.agents.orchestrator import FINISH, finish_allowed
from agentic_ode.core.state import END_ROUTE, DialogueState, Role

STEPS_NEEDING_APPROVAL = (Role.EXECUTOR.value, FINISH)
YES_WORDS = {"yes", "y"}


def last_orchestrator_turn(state: DialogueState) -> str:
    for turn in reversed(state["dialogue"]):
        if turn["role"] == Role.ORCHESTRATOR:
            return turn["content"]
    return ""


def question_of(content: str) -> str:
    """All text after `question:`. It is the last field of a human decision, so it may have several lines."""
    _, found, rest = content.partition("\nquestion:")
    return rest.strip() if found else ""


def accepted(reply: str) -> bool:
    """Accept only if the reply starts with yes / y. Anything else is a refuse, so the safe choice wins."""
    words = reply.lower().split()
    return bool(words) and words[0].strip(".,!:;") in YES_WORDS


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


def _note(text: str) -> dict:
    """A turn written by the graph itself, not by a person or a model."""
    return {"role": Role.SYSTEM.value, "content": text}


def _back(turns: list[dict], refused: str = "") -> dict:
    """The pending step does not run: give the turn back to the orchestrator."""
    return {"dialogue": turns, "route": Role.ORCHESTRATOR.value, "pending_step": "", "refused_step": refused}
