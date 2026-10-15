"""A tiny MCP server for the tests of agentic_ode.mcp_client: it keeps a counter in the server process, so a test
can see whether two calls went to the SAME server process (persistent session). No camera, no files.

Run:  python -m tests.mcp_counter_server
"""

from fastmcp import FastMCP

mcp = FastMCP("counter")
_total = 0


@mcp.tool
def add(n: int) -> dict:
    """Add n to the counter of this server process and return the new total."""
    global _total
    _total += n
    return {"ok": True, "total": _total}


@mcp.tool
def total() -> dict:
    """The counter of this server process (0 in a new process)."""
    return {"ok": True, "total": _total}


if __name__ == "__main__":
    mcp.run(show_banner=False)
