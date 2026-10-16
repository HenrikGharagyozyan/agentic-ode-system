"""python -m agentic_ode.web [--port 8000]: run the EventLab API on 127.0.0.1 only.

There is no --host option on purpose: the API can control the camera and has no login yet, so it must not be
reachable from other machines.
"""

from __future__ import annotations

import argparse

import uvicorn

from agentic_ode.web.app import create_app


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m agentic_ode.web", description=__doc__)
    p.add_argument("--port", type=int, default=8000, help="port on 127.0.0.1 (default 8000)")
    args = p.parse_args(argv)
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
