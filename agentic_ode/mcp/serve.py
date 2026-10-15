"""Run any registered instrument as an MCP server over stdio.

    python -m agentic_ode.mcp.serve evcam            (EVCAM_FAKE=1: fake camera)

The instruments are registered in agentic_ode/hardware/devices/__init__.py; a new device needs no change here.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import agentic_ode.hardware.devices  # noqa: F401 - registers the known instruments
from agentic_ode.core.config import PROJECT_ROOT
from agentic_ode.hardware.standard import registry
from agentic_ode.mcp.server_factory import ToolLog, build_server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m agentic_ode.mcp.serve", description=__doc__.splitlines()[0])
    parser.add_argument("instrument", choices=registry.names())
    parser.add_argument("--log", type=Path, help="tool log file (default: runs/<instrument>/tool_log.jsonl)")
    args = parser.parse_args(argv)
    instrument = registry.create(args.instrument)
    log = ToolLog(args.log or PROJECT_ROOT / "runs" / args.instrument / "tool_log.jsonl")
    # stdout carries the MCP messages, so all logging goes to stderr
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    build_server(instrument, log).run(show_banner=False)  # default transport: stdio; no banner on stderr


if __name__ == "__main__":
    main()
