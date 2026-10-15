"""The hardware turn (HW3 d): the executor's model picks ONE camera tool call, a setting change waits for the human's
approval, then the call runs and its result is written verbatim into the dialogue. The orchestrator reads it next.

  hw_decide   one model call (executor.camera_llm)  -> hw_approve (setting change) | hw_call | orchestrator (note)
  hw_approve  interrupt(), no model, no camera       -> hw_call (yes) | orchestrator (refused)
  hw_call     the camera call, no model              -> orchestrator
"""

from __future__ import annotations

import json
import logging

import jsonschema
from langgraph.types import interrupt

from agentic_ode.agents import prompts
from agentic_ode.agents.context import specialist_messages
from agentic_ode.agents.executor import Executor
from agentic_ode.agents.guards import accepted
from agentic_ode.core.config import Settings
from agentic_ode.core.llm import text_of
from agentic_ode.core.state import CAMERA_SETTING_TOOLS, DialogueState, Role
from agentic_ode.graph.routing import HW_APPROVE, HW_CALL

log = logging.getLogger(__name__)

def call_text(call: dict) -> str:
    """e.g. mask_pixels({"pixels": [[1053, 704]]})"""
    return f"{call['name']}({json.dumps(call['args'])})"


def same_call(a: dict, b: dict) -> bool:
    """Same tool and same arguments (the order of the arguments does not matter)."""
    return a["name"] == b["name"] and json.dumps(a["args"], sort_keys=True) == json.dumps(b["args"], sort_keys=True)


def schema_of(tool) -> dict:
    """The tool's JSON schema: a dict for MCP tools; a tool with a pydantic schema (e.g. in tests) is converted."""
    return tool.args_schema if isinstance(tool.args_schema, dict) else tool.args_schema.model_json_schema()


def schema_errors(schema: dict, args: dict) -> list[str]:
    """What is wrong with args for this JSON schema ([] = valid), e.g. "pixels/0: {'x': 926} is not of type
    'array'". A little stricter than the server: a number sent as a string ("20") is refused here."""
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.absolute_path)) or 'arguments'}: {e.message}" for e in validator.iter_errors(args)]


def error_of(result_line: str) -> str:
    try:
        return str(json.loads(result_line).get("error", result_line))
    except (ValueError, AttributeError):
        return result_line


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
        self.schemas = {t.name: schema_of(t) for t in executor.camera_tools}
        self.params = {name: set(s.get("properties", {})) for name, s in self.schemas.items()}  # accepted arguments

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
        name, raw_args = calls[0]["name"], calls[0]["args"]
        known = self.params.get(name)  # None for an unknown tool: hw_call reports it
        args = {k: v for k, v in raw_args.items() if known is None or k in known}
        # ignored: arguments the tool does not have (e.g. camera_info({"serial": ...})); hw_call names them
        call = {"name": name, "args": args, "ignored": sorted(set(raw_args) - set(args))}
        notes = []
        if len(calls) > 1:
            dropped = ", ".join(call_text({"name": c["name"], "args": c["args"]}) for c in calls[1:])
            notes.append(_note(f"The hardware turn asked for {len(calls)} camera calls; only the first one runs. "
                               f"Dropped: {dropped}"))
        errors = schema_errors(self.schemas[name], args) if name in self.schemas else []
        if errors:  # it would fail the format check: never ask the human to approve it, never send it
            return _back(notes + [_note(f"The camera call {call_text(call)} did NOT run: its arguments do not match "
                                        f"the tool's schema: {'; '.join(errors)}. Fix the arguments and try again.")])
        if any(same_call(call, r) for r in state["refused_calls"]):
            return _back(notes + [_note(f"The human refused the camera call {call_text(call)} earlier. "
                                        "It did NOT run and is not asked again; choose something else.")])
        failed = [r for r in state["camera"] if not r["ok"] and same_call(call, {"name": r["tool"], "args": r["args"]})]
        if failed:  # repeat guard: the same call would fail the same way
            return _back(notes + [_note(f"The camera call {call_text(call)} did NOT run: this exact call already "
                                        f"failed with: {error_of(failed[-1]['result_line'])}; change the call or "
                                        "choose another step.")])
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
                 refused_calls=[{"name": call["name"], "args": call["args"]}])


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
        ignored = call.get("ignored", [])
        ignored_line = f"ignored unknown argument(s): {', '.join(ignored)}\n" if ignored else ""
        return {
            "dialogue": [{"role": Role.HARDWARE.value,
                          "content": f"camera call: {call_text(call)}\n{ignored_line}result: {result}"}],
            "camera": [{"turn_index": index, "tool": call["name"], "args": call["args"], "ok": ok,
                        "result_line": result}],
            "route": Role.ORCHESTRATOR.value,
            "hw_call": {},
        }
