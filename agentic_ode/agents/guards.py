"""The guards that decide whether a step may happen: the finish guard and the yes rule for the human's reply.

Used by the orchestrator, the human node and the hardware turn; kept here so that no node imports another node.
"""

from __future__ import annotations

from agentic_ode.core.metrics import last_rmse_line
from agentic_ode.core.state import CAMERA_MEASUREMENT_TOOLS, DialogueState, Role

FINISH = "finish"
YES_WORDS = {"yes", "y"}


def finish_allowed(state: DialogueState) -> bool:
    """True if some executor turn carries a test RMSE parsed from the stdout of a run with exit code 0,
    or the FOCUS_ENHANCEMENT line of a real focusing-tool call, or some hardware turn carries the result of a
    successful camera measurement (record, spectrum, find_hot_pixels with ok: true; a spectrum without a peak
    counts too: "no flicker" is a valid result)."""
    dialogue = state["dialogue"]
    for rec in state.get("camera", []):
        i = rec["turn_index"]
        if (
            rec["ok"]
            and rec["tool"] in CAMERA_MEASUREMENT_TOOLS
            and 0 <= i < len(dialogue)
            and dialogue[i]["role"] == Role.HARDWARE
            and rec["result_line"] in dialogue[i]["content"]
        ):
            return True
    for exp in state.get("experiments", []):
        i = exp["turn_index"]
        in_turn = 0 <= i < len(dialogue) and dialogue[i]["role"] == Role.EXECUTOR
        if in_turn and exp["result_line"] in dialogue[i]["content"]:
            return True
    for run in state.get("runs", []):
        i = run["turn_index"]
        if (
            run["exit_code"] == 0
            and run["test_rmse"] is not None
            and 0 <= i < len(dialogue)
            and dialogue[i]["role"] == Role.EXECUTOR
            and last_rmse_line(run["stdout"]) in dialogue[i]["content"]
        ):
            return True
    return False


def accepted(reply: str) -> bool:
    """Accept only if the reply starts with yes / y. Anything else is a refuse, so the safe choice wins."""
    words = reply.lower().split()
    return bool(words) and words[0].strip(".,!:;") in YES_WORDS
