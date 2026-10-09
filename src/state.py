"""Shared graph state: the dialogue plus bookkeeping for the orchestrator."""

import operator
from typing import Annotated, TypedDict

ROLES = ("orchestrator", "scientist", "general", "executor")
MAX_DECISIONS = 8


class Turn(TypedDict):
    role: str  # "user" or a node name
    content: str


class RunRecord(TypedDict):
    """One real call of run_file, kept outside the text so the finish guard can trust it."""

    turn_index: int  # index in the dialogue of the executor turn that made this run
    filename: str
    exit_code: int | None  # None when the run hit the time limit
    stdout: str
    stderr: str
    test_rmse: float | None  # parsed from stdout, None if not printed
    train_rmse: float | None


class DialogueState(TypedDict):
    dialogue: Annotated[list[Turn], operator.add]
    decisions: Annotated[int, operator.add]  # number of orchestrator decisions so far
    runs: Annotated[list[RunRecord], operator.add]
    route: str  # next node chosen by the orchestrator, or "__end__"
    stop_reason: str


def initial_state(user_message: str) -> DialogueState:
    return {
        "dialogue": [{"role": "user", "content": user_message}],
        "decisions": 0,
        "runs": [],
        "route": "",
        "stop_reason": "",
    }


def render_dialogue(dialogue: list[Turn]) -> str:
    """The dialogue as plain text, the way every node reads it."""
    return "\n\n".join(
        f"### Turn {i} [{t['role']}]\n{t['content']}" for i, t in enumerate(dialogue)
    )
