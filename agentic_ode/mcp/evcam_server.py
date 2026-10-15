"""MCP server for the IDS event camera (FastMCP, stdio): the generic server (server_factory.py) for EvcamInstrument.

Run:   python -m agentic_ode.mcp_servers.evcam_server                      (real camera: run `evcam` first)
       EVCAM_FAKE=1 python -m agentic_ode.mcp_servers.evcam_server         (FakeEventCamera, no hardware)
       EVCAM_FAKE_SCENE=dark|pwm|steady chooses the fake scene (default pwm)

- The 10 tools are the actions of EvcamInstrument (hardware/devices/evcam/instrument.py).
- One camera object for the whole server: opened on the first tool call that needs it, kept open, closed when the
  server stops. A lock lets only one tool use it at a time (FastMCP runs each tool call in a worker thread).
- Every tool returns {"ok": true, ...} or {"ok": false, "error": "..."}; the server never crashes on a bad call.
  (Arguments of the wrong type, e.g. a string for an int, are refused by FastMCP before the tool runs.)
- record() saves the events with store.save() and returns only a recording_id and a summary, never raw events.
- Every call is appended to runs/evcam/tool_log.jsonl: time, tool, arguments, result, duration.
"""

from __future__ import annotations

import logging
import sys

from agentic_ode.hardware.devices.evcam.instrument import EvcamInstrument
from agentic_ode.mcp.server_factory import ToolLog, build_server

instrument = EvcamInstrument()  # recordings in runs/evcam/ (git-ignored); the tests point it to a temporary folder
tool_log = ToolLog(instrument.recordings_dir / "tool_log.jsonl")
mcp = build_server(instrument, tool_log)


def main() -> None:
    """Run the server over stdio (also called by the alias agentic_ode/mcp_servers/evcam_server.py)."""
    # stdout carries the MCP messages, so all logging goes to stderr
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    mcp.run(show_banner=False)  # default transport: stdio; no banner on stderr


if __name__ == "__main__":
    main()
