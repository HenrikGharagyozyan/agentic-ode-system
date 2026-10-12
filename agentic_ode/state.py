"""Shared graph state: the dialogue plus bookkeeping for the orchestrator."""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, TypedDict

END_ROUTE = "__end__"


class Role(StrEnum):
    USER = "user"
    ORCHESTRATOR = "orchestrator"
    SCIENTIST = "scientist"
    GENERAL = "general"
    EXECUTOR = "executor"
    HUMAN = "human"
    SYSTEM = "system"  # notes written by the graph itself, e.g. "the step did NOT run"


SPECIALISTS = (Role.SCIENTIST, Role.GENERAL, Role.EXECUTOR)


class Turn(TypedDict):
    role: str  # a Role value
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


class ExperimentRecord(TypedDict):
    """One real call of run_focus_search, kept outside the text so the finish guard can trust it."""

    turn_index: int
    settings: dict
    n_macropixels: int
    initial_enhancement: float
    final_enhancement: float
    result_line: str  # the FOCUS_ENHANCEMENT line exactly as the tool printed it
    images: dict


class DialogueState(TypedDict):
    dialogue: Annotated[list[Turn], operator.add]
    decisions: Annotated[int, operator.add]  # number of orchestrator decisions so far
    runs: Annotated[list[RunRecord], operator.add]
    experiments: Annotated[list[ExperimentRecord], operator.add]  # focusing-tool calls
    route: str  # next node chosen by the orchestrator, or END_ROUTE
    stop_reason: str
    pending_step: str  # step that waits for the human's approval ("executor", "finish"), or ""
    refused_step: str  # step the human refused just now; the next orchestrator decision may not choose it


def initial_state(user_message: str) -> DialogueState:
    return {
        "dialogue": [{"role": Role.USER.value, "content": user_message}],
        "decisions": 0,
        "runs": [],
        "experiments": [],
        "route": "",
        "stop_reason": "",
        "pending_step": "",
        "refused_step": "",
    }


def render_dialogue(dialogue: list[Turn]) -> str:
    """The dialogue as plain text, the way every node reads it."""
    return "\n\n".join(
        f"### Turn {i} [{t['role']}]\n{t['content']}" for i, t in enumerate(dialogue)
    )
