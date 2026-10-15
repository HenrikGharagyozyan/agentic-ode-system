"""The hardware turn (HW3 d): the executor's model picks ONE camera tool call, a setting change waits for the human's
approval, then the call runs and its result is written verbatim into the dialogue. The orchestrator reads it next.

  hw_decide   one model call (executor.camera_llm)  -> hw_approve (setting change) | hw_call | orchestrator (note)
  hw_approve  interrupt(), no model, no camera       -> hw_call (yes) | orchestrator (refused)
  hw_call     the camera call, no model              -> orchestrator
"""

from __future__ import annotations

import json
import logging

from langgraph.types import interrupt

from agentic_ode.agents import prompts
from agentic_ode.agents.context import specialist_messages
from agentic_ode.agents.executor import Executor
from agentic_ode.agents.human import accepted
from agentic_ode.config import Settings
from agentic_ode.llm import text_of
from agentic_ode.state import CAMERA_SETTING_TOOLS, DialogueState, Role

log = logging.getLogger(__name__)

HW_DECIDE, HW_APPROVE, HW_CALL = "hw_decide", "hw_approve", "hw_call"


def call_text(call: dict) -> str:
    """e.g. mask_pixels({"pixels": [[1053, 704]]})"""
    return f"{call['name']}({json.dumps(call['args'])})"


def _note(text: str) -> dict:
    """A turn written by the graph itself, not by a person or a model."""
    return {"role": Role.SYSTEM.value, "content": text}


def _back(turns: list[dict], **extra) -> dict:
    """No camera call runs now: give the turn back to the orchestrator."""
    return {"dialogue": turns, "route": Role.ORCHESTRATOR.value, "hw_call": {}, **extra}


class HardwareDecide:
    """hw_decide: the executor's model reads the dialogue and the orchestrator's instruction and picks ONE camera
    tool call. The call is only stored here; it runs in hw_call."""

    def __init__(self, executor: Executor, settings: Settings):
        self.executor = executor
        self.prompt = prompts.hardware(settings)

    def __call__(self, state: DialogueState) -> dict:
        if self.executor.camera_llm is None:
            return _back([_note("There is no camera in this run (start the run with --camera). "
                                "No camera call was made.")])
        reply = self.executor.camera_reply(specialist_messages(self.prompt, state, Role.HARDWARE))
        calls = reply.tool_calls
        if not calls:
            text = text_of(reply).strip()
            return _back([_note("The hardware turn made no camera call" + (f"; the model answered: {text}" if text
                                                                             else "."))])
        call = {"name": calls[0]["name"], "args": calls[0]["args"]}
        notes = []
        if len(calls) > 1:
            dropped = ", ".join(call_text({"name": c["name"], "args": c["args"]}) for c in calls[1:])
            notes.append(_note(f"The hardware turn asked for {len(calls)} camera calls; only the first one runs. "
                               f"Dropped: {dropped}"))
        if call in state["refused_calls"]:
            return _back(notes + [_note(f"The human refused the camera call {call_text(call)} earlier. "
                                        "It did NOT run and is not asked again; choose something else.")])
        log.info("hardware turn chose %s", call_text(call))
        return {"dialogue": notes, "hw_call": call,
                "route": HW_APPROVE if call["name"] in CAMERA_SETTING_TOOLS else HW_CALL}


def hw_approve(state: DialogueState) -> dict:
    """Ask the human before a camera setting change. Nothing before interrupt() calls a model or the camera,
    so the re-run on resume is safe. kind "hard", so --auto-approve answers yes."""
    call = state["hw_call"]
    reply = str(interrupt({"question": f"The executor wants to change a camera setting: {call_text(call)}. "
                                       "Approve this camera call?", "kind": "hard"})).strip()

    # Below runs only once, after resume.
    reply_turn = {"role": Role.HUMAN.value, "content": reply}
    if accepted(reply):
        return {"dialogue": [reply_turn], "route": HW_CALL}
    return _back([reply_turn, _note(f"The human refused the camera call {call_text(call)}. It did NOT run.")],
                 refused_calls=[call])


class HardwareCall:
    """hw_call: run the stored call through the MCP session and write call + result verbatim into the dialogue."""

    def __init__(self, executor: Executor):
        self.tools = {t.name: t for t in executor.camera_tools}

    def __call__(self, state: DialogueState) -> dict:
        call = state["hw_call"]
        tool = self.tools.get(call["name"])
        if tool is None:
            return _back([_note(f"{call['name']} is not a camera tool. No camera call was made.")])
        try:
            result = tool.invoke(call["args"])  # the tool's result dict as JSON text (agentic_ode.mcp_client)
            ok = bool(json.loads(result).get("ok"))
        except Exception as e:  # e.g. the server stopped or the call timed out: report it, never crash the run
            result, ok = json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}), False
        log.info("camera call %s: ok=%s", call_text(call), ok)
        index = len(state["dialogue"])  # the hardware turn below gets this index
        return {
            "dialogue": [{"role": Role.HARDWARE.value, "content": f"camera call: {call_text(call)}\nresult: {result}"}],
            "camera": [{"turn_index": index, "tool": call["name"], "args": call["args"], "ok": ok,
                        "result_line": result}],
            "route": Role.ORCHESTRATOR.value,
            "hw_call": {},
        }
