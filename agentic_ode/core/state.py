"""Shared graph state: the dialogue plus bookkeeping for the orchestrator."""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, TypedDict

from agentic_ode.core.records import CameraRecord, ExperimentRecord, RunRecord

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

# camera tools (HW3): setting changes need the human's approval in the hardware turn; measurements run directly
# and a successful one allows finish; the other tools (camera_info, get_settings) only read and run directly
CAMERA_SETTING_TOOLS = ("set_bias", "mask_pixels", "clear_mask", "set_roi", "clear_roi")
CAMERA_MEASUREMENT_TOOLS = ("record", "spectrum", "find_hot_pixels")


class Turn(TypedDict):
    role: str  # a Role value
    content: str


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
