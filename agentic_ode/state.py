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
    HARDWARE = "hardware"  # the executor's camera turn: one MCP tool call and its result (HW3)


SPECIALISTS = (Role.SCIENTIST, Role.GENERAL, Role.EXECUTOR)
ROLE_LABELS = {Role.HARDWARE: "executor (hardware)"}  # how a role is shown in transcripts and the report

# camera tools (HW3): setting changes need the human's approval in the hardware turn; measurements run directly
# and a successful one allows finish; the other tools (camera_info, get_settings) only read and run directly
CAMERA_SETTING_TOOLS = ("set_bias", "mask_pixels", "clear_mask", "set_roi", "clear_roi")
CAMERA_MEASUREMENT_TOOLS = ("record", "spectrum", "find_hot_pixels")


def role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role)


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


class CameraRecord(TypedDict):
    """One real camera tool call of a hardware turn, kept outside the text so the finish guard can trust it."""

    turn_index: int  # index in the dialogue of the hardware turn that shows this call
    tool: str
    args: dict
    ok: bool  # the tool's own "ok" field
    result_line: str  # the result JSON exactly as written in the hardware turn


class DialogueState(TypedDict):
    dialogue: Annotated[list[Turn], operator.add]
    decisions: Annotated[int, operator.add]  # number of orchestrator decisions so far
    runs: Annotated[list[RunRecord], operator.add]
    experiments: Annotated[list[ExperimentRecord], operator.add]  # focusing-tool calls
    camera: Annotated[list[CameraRecord], operator.add]  # camera tool calls of hardware turns
    route: str  # next node chosen by the orchestrator, or END_ROUTE
    stop_reason: str
    pending_step: str  # step that waits for the human's approval ("executor", "finish"), or ""
    refused_step: str  # step the human refused just now; the next orchestrator decision may not choose it
    hw_call: dict  # camera call {"name", "args"} chosen by hw_decide, waiting to run; {} when there is none
    refused_calls: Annotated[list[dict], operator.add]  # camera calls the human refused; never asked again


def initial_state(user_message: str) -> DialogueState:
    return {
        "dialogue": [{"role": Role.USER.value, "content": user_message}],
        "decisions": 0,
        "runs": [],
        "experiments": [],
        "camera": [],
        "route": "",
        "stop_reason": "",
        "pending_step": "",
        "refused_step": "",
        "hw_call": {},
        "refused_calls": [],
    }


def render_dialogue(dialogue: list[Turn]) -> str:
    """The dialogue as plain text, the way every node reads it."""
    return "\n\n".join(
        f"### Turn {i} [{role_label(t['role'])}]\n{t['content']}" for i, t in enumerate(dialogue)
    )
