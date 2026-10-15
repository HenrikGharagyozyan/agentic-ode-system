"""Records of real tool calls, kept in the graph state outside the dialogue text so the finish guard can trust them."""

from __future__ import annotations

from typing import TypedDict


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
