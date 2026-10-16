"""build_server(): one generic MCP server for any Instrument (Model Hardware Standard).

- Every ActionSpec becomes one MCP tool: same name, description and JSON schema (from the handler's type hints,
  so fastmcp checks argument types before the action runs).
- One lock per server: only one tool uses the instrument at a time (FastMCP runs each tool call in a worker thread).
- Every tool returns Result.to_wire(): {"ok": true, ...} or {"ok": false, "error": "..."}; the server never crashes.
- Every call is appended to the ToolLog: time, tool, arguments (defaults included), result, duration.
- When the server stops, instrument.close() releases the hardware.
"""

from __future__ import annotations

import functools
import inspect
import json
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.tools import Tool

from agentic_ode.hardware.standard import ActionSpec, Instrument

OUTPUT_SCHEMA = {"additionalProperties": True, "type": "object"}  # every tool returns one JSON object


class ToolLog:
    """One JSON line per tool call. `path` may be changed at any time (the tests point it to a temporary folder)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, tool: str, args: dict, result: dict, seconds: float) -> None:
        record = {"time": datetime.now().astimezone().isoformat(timespec="milliseconds"), "tool": tool,
                  "args": args, "ok": result["ok"], "result": result, "duration_s": round(seconds, 3)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")  # default=str: never fail on an unusual value


def build_server(instrument: Instrument, log: ToolLog) -> FastMCP:
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(server):
        try:
            yield {}
        finally:
            with lock:
                instrument.close()  # release the hardware so another process can open it

    mcp = FastMCP(instrument.name, instructions=instrument.describe()["instructions"], lifespan=lifespan)
    for spec in instrument.actions():
        mcp.add_tool(Tool.from_function(_tool_function(instrument, spec, lock, log), name=spec.name,
                                        description=spec.description, output_schema=OUTPUT_SCHEMA,
                                        run_in_thread=True))
    return mcp


def _tool_function(instrument: Instrument, spec: ActionSpec, lock: threading.Lock, log: ToolLog):
    """The function fastmcp calls for one tool. It shows fastmcp the handler's parameters (functools.wraps), so the
    tool has the handler's schema; it returns the wire dict, so its return type is dict, not Result."""
    signature = inspect.signature(spec.handler)

    @functools.wraps(spec.handler)
    def tool(*args, **kwargs) -> dict:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()  # the log shows every argument, defaults included
        arguments = dict(bound.arguments)
        t0 = time.monotonic()
        with lock:
            result = instrument.call(spec.name, arguments).to_wire()  # call() never raises
        log.write(spec.name, arguments, result, time.monotonic() - t0)
        return result

    tool.__signature__ = signature.replace(return_annotation=dict)
    tool.__annotations__ = {**inspect.get_annotations(spec.handler), "return": dict}  # a new dict, not the handler's
    return tool
