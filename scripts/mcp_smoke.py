"""Smoke test of the evcam MCP server through a real MCP client.

Default (in memory, no separate process): lists the tools, then calls camera_info -> find_hot_pixels ->
mask_pixels (top 2) -> record(2.0) -> spectrum -> get_settings, and prints each call, its result and how long it
took. Stops at the first {"ok": false}.

--stdio: the server runs as a subprocess, reached through agentic_ode.mcp.client (the way the agent will use it).
1. ONE persistent session: mask_pixels, then get_settings -> the mask is still there.
2. Contrast, a NEW session for each call: mask_pixels, then get_settings -> the mask is lost (new server process,
   new camera object).

Run:  EVCAM_FAKE=1 EVCAM_FAKE_SCENE=pwm python scripts/mcp_smoke.py [--stdio]  (fake camera)
      evcam; python scripts/mcp_smoke.py 2> >(grep -v InvalidVectBase >&2)     (real camera; the first call takes
                                                                                 ~9 s: open + settle)
Recordings and the tool log go to runs/evcam/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time

from fastmcp import Client

from agentic_ode.mcp import evcam_server
from agentic_ode.mcp.client import evcam_session

PIXELS = [[1053, 704], [926, 151]]  # the 2 known hot pixels (--stdio only needs some pixels to mask)


class SmokeFailed(Exception):
    pass


def tool_signature(tool) -> str:
    """e.g. find_hot_pixels(seconds=0.5, top=64, min_rate_hz=20000), from the tool's input schema."""
    args = []
    for name, schema in tool.input_schema.get("properties", {}).items():
        args.append(f"{name}={schema['default']!r}" if "default" in schema else name)
    return f"{tool.name}({', '.join(args)})"


async def main() -> int:
    if os.environ.get("EVCAM_FAKE") == "1":
        print(f"camera: FakeEventCamera, scene {os.environ.get('EVCAM_FAKE_SCENE', 'pwm')}")
    else:
        print("camera: EventCamera (real hardware)")
    n_ok = 0
    t_start = time.monotonic()

    async with Client(evcam_server.mcp) as client:

        async def call(name: str, args: dict) -> dict:
            """Call one tool, print the call, the result and the time; stop the test if it failed."""
            nonlocal n_ok
            print(f"\n-> {name}({json.dumps(args)})")
            t0 = time.monotonic()
            result = await client.call_tool(name, args, raise_on_error=False)
            seconds = time.monotonic() - t0
            if result.is_error:  # refused by FastMCP before the tool ran (e.g. an argument of the wrong type)
                raise SmokeFailed(f"{name} was refused by the MCP server: {result.content[0].text}")
            data = result.structured_content
            print(json.dumps(data, indent=2))
            print(f"<- {name}: {'ok' if data['ok'] else 'FAILED'} in {seconds:.2f} s")
            if not data["ok"]:
                raise SmokeFailed(f"{name}: {data['error']}")
            n_ok += 1
            return data

        tools = await client.list_tools()
        print(f"server '{evcam_server.mcp.name}': {len(tools)} tools")
        for tool in tools:
            print(f"  {tool_signature(tool)}\n      {tool.description}")

        try:
            await call("camera_info", {})
            hot = await call("find_hot_pixels", {})
            top2 = [[p["x"], p["y"]] for p in hot["hot_pixels"][:2]]  # fewer than 2 if fewer were found
            await call("mask_pixels", {"pixels": top2})
            rec = await call("record", {"seconds": 2.0, "note": "mcp smoke test"})
            await call("spectrum", {"recording_id": rec["recording_id"]})
            await call("get_settings", {})
        except SmokeFailed as e:
            print(f"\nsmoke test FAILED after {n_ok} ok calls: {e}")
            return 1

    print(f"\nsmoke test passed: {n_ok}/6 calls ok in {time.monotonic() - t_start:.2f} s")
    return 0


# ---- --stdio: through agentic_ode.mcp.client, sync like the graph ----

def call_tool(tools: dict, name: str, args: dict, use_async: bool = False) -> dict:
    """Call one LangChain tool (sync invoke, or async ainvoke from a new event loop) and print call, result, time."""
    print(f"-> {name}({json.dumps(args)})  [{'ainvoke' if use_async else 'invoke'}]")
    t0 = time.monotonic()
    text = asyncio.run(tools[name].ainvoke(args)) if use_async else tools[name].invoke(args)
    data = json.loads(text)
    print("   " + json.dumps({k: v for k, v in data.items() if k != "bias_ranges"}))  # bias_ranges: long, not needed
    print(f"<- {name}: {'ok' if data['ok'] else 'FAILED'} in {time.monotonic() - t0:.2f} s")
    return data


def stdio_main() -> int:
    print(f"camera: {'FakeEventCamera' if os.environ.get('EVCAM_FAKE') == '1' else 'EventCamera (real hardware)'}")

    print("\n== 1. ONE persistent session: mask_pixels, then get_settings")
    t0 = time.monotonic()
    with evcam_session() as session:
        print(f"session open in {time.monotonic() - t0:.2f} s (server subprocess started)")
        tools = {t.name: t for t in session.tools()}
        print(f"{len(tools)} LangChain tools: {', '.join(tools)}")
        call_tool(tools, "mask_pixels", {"pixels": PIXELS})
        kept = call_tool(tools, "get_settings", {}, use_async=True)["mask"]
    print(f"session closed (server subprocess stopped); {time.monotonic() - t0:.2f} s in total")

    print("\n== 2. contrast: a NEW session for each call")
    with evcam_session() as session:
        call_tool({t.name: t for t in session.tools()}, "mask_pixels", {"pixels": PIXELS})
    with evcam_session() as session:
        lost = call_tool({t.name: t for t in session.tools()}, "get_settings", {})["mask"]

    ok_kept, ok_lost = kept == PIXELS, lost == []
    print(f"\none session:          mask after get_settings = {kept} -> {'KEPT' if ok_kept else 'NOT KEPT (bug)'}")
    print(f"new session per call: mask after get_settings = {lost} -> {'LOST, as expected' if ok_lost else '??'}")
    print(f"smoke test --stdio {'passed' if ok_kept and ok_lost else 'FAILED'}")
    return 0 if ok_kept and ok_lost else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test of the evcam MCP server.")
    parser.add_argument("--stdio", action="store_true",
                        help="server as a subprocess through agentic_ode.mcp.client: persistent vs new sessions")
    args = parser.parse_args()
    # the camera's log lines (settle check, recordings) go to stderr; the results go to stdout
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(stdio_main() if args.stdio else asyncio.run(main()))
