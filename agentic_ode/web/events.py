"""From graph state to browser events. Pure functions: no threads, no I/O, so they are easy to test.

The browser never parses the graph's text. Each new piece of state becomes one typed event:

- turn.added    a new dialogue turn {index, role, label, content}; orchestrator turns also get `decision`
                (next, instruction, reason, and kind/question for a human question), read from Decision.as_turn()
- tool.pending  hw_decide chose a camera call that has not run yet {name, args, ignored, needs_approval}
- tool.result   a camera call ran: the CameraRecord, its result JSON parsed, warnings as a list, the tool's kind

The graph's own records are the source: CameraRecords are what the finish guard trusts, so the browser shows the
same evidence.
"""

from __future__ import annotations

import json

from agentic_ode.core.dialogue import decision_field, role_label
from agentic_ode.core.state import CAMERA_MEASUREMENT_TOOLS, CAMERA_SETTING_TOOLS, Role

DECISION_FIELDS = ("next", "instruction", "reason", "kind", "proposed_next")


def tool_kind(name: str) -> str:
    """setting (needs approval), measurement (can finish a task) or read; the same groups the graph uses."""
    if name in CAMERA_SETTING_TOOLS:
        return "setting"
    if name in CAMERA_MEASUREMENT_TOOLS:
        return "measurement"
    return "read"


def decision_of(content: str) -> dict:
    """The fields of an orchestrator turn. `question` is the last field and may have several lines."""
    fields = {k: decision_field(content, k) for k in DECISION_FIELDS}
    _, found, question = content.partition("\nquestion:")
    fields["question"] = question.strip() if found else ""
    return {k: v for k, v in fields.items() if v}


def turn_event(index: int, turn: dict) -> dict:
    data = {"index": index, "role": turn["role"], "label": role_label(turn["role"]), "content": turn["content"]}
    if turn["role"] == Role.ORCHESTRATOR:
        data["decision"] = decision_of(turn["content"])
    return data


def parse_result(result_line: str) -> dict | None:
    """The tool's result dict, or None if the line is not a JSON object (then the browser shows the raw line)."""
    try:
        value = json.loads(result_line)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def tool_result_event(record: dict) -> dict:
    result = parse_result(record["result_line"])
    warning = result.get("warning") if result else None
    return {
        "turn_index": record["turn_index"],
        "tool": record["tool"],
        "kind": tool_kind(record["tool"]),
        "args": record["args"],
        "ok": record["ok"],
        "result": result,
        "result_line": None if result is not None else record["result_line"],
        "warnings": [w.strip() for w in str(warning).split(" | ")] if warning else [],  # Result.to_wire() joins
        "error": result.get("error") if result and not record["ok"] else None,
    }


def state_events(prev: dict | None, state: dict) -> list[tuple[str, dict]]:
    """The (type, data) events for everything that is new in `state` compared with `prev` (None = the start)."""
    prev = prev or {"dialogue": [], "camera": [], "hw_call": {}}
    events: list[tuple[str, dict]] = []
    for i in range(len(prev["dialogue"]), len(state["dialogue"])):
        events.append(("turn.added", turn_event(i, state["dialogue"][i])))
    call = state.get("hw_call") or {}
    if call and call != (prev.get("hw_call") or {}):
        events.append(("tool.pending", {"name": call["name"], "args": call["args"],
                                        "ignored": call.get("ignored", []),
                                        "kind": tool_kind(call["name"]),
                                        "needs_approval": call["name"] in CAMERA_SETTING_TOOLS}))
    for record in state.get("camera", [])[len(prev.get("camera", [])):]:
        events.append(("tool.result", tool_result_event(record)))
    return events


def input_request(payload: dict, state: dict) -> dict:
    """What the browser shows when the graph pauses. hw_approve pauses with route "hw_approve" and the exact call in
    hw_call; the human node pauses after an orchestrator question (route "human")."""
    if state.get("route") == "hw_approve" and state.get("hw_call"):
        call = state["hw_call"]
        return {"source": "camera_setting", "kind": "hard", "question": payload.get("question", ""),
                "call": {"name": call["name"], "args": call["args"]}}
    return {"source": "orchestrator", "kind": payload.get("kind") or "none", "question": payload.get("question", ""),
            "proposed_step": state.get("pending_step") or None}
