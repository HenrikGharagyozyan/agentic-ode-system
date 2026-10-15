"""Smoke test of the evcam MCP server through a real MCP client (in memory: no separate process).

Lists the tools, then calls camera_info -> find_hot_pixels -> mask_pixels (top 2) -> record(2.0) -> spectrum ->
get_settings, and prints each call, its result and how long it took. Stops at the first {"ok": false}.

Run:  EVCAM_FAKE=1 EVCAM_FAKE_SCENE=pwm python scripts/mcp_smoke.py          (fake camera)
      evcam; python scripts/mcp_smoke.py 2> >(grep -v InvalidVectBase >&2)   (real camera; the first call takes
                                                                               ~9 s: open + settle)
Recordings and the tool log go to runs/evcam/.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

from fastmcp import Client

from agentic_ode.mcp_servers import evcam_server


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


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
