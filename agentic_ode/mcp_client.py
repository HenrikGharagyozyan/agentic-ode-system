"""A small MCP client for our SYNC graph: one persistent stdio session, its tools as LangChain StructuredTools.

Why not langchain-mcp-adapters: no released version works with mcp 2.x, which fastmcp 4 needs (2026-10-15).

The fastmcp Client is async, our graph is sync. So the session lives in ONE background thread with its own event
loop for the whole run. A sync call submits its coroutine to that loop and waits for the result; an async call does
the same and awaits it. One session = one server process = one camera object, so settings like the pixel mask stay.

    with evcam_session() as session:
        tools = session.tools()                          # LangChain tools, ready for llm.bind_tools(tools)
        session.call("mask_pixels", {"pixels": [[1053, 704]]})
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from langchain_core.tools import StructuredTool


class McpSession:
    """One MCP server started as a subprocess (stdio), with one session kept open until close()."""

    def __init__(self, command: str, args: list[str], env: dict[str, str] | None = None, timeout_s: float = 120):
        self.command, self.args, self.env, self.timeout_s = command, args, env, timeout_s
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None  # the background thread's event loop
        self._client: Client | None = None
        self._stop: asyncio.Event | None = None  # set by close(): ends the session
        self._ready = threading.Event()  # set by _main() when the session is open, or when opening failed
        self._error: BaseException | None = None

    # ---- open / close ----

    def open(self) -> McpSession:
        """Start the background thread and wait until the session is open (raises if it could not open)."""
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=asyncio.run, args=(self._main(),), name="mcp-session", daemon=True)
        self._thread.start()
        if not self._ready.wait(self.timeout_s):
            raise TimeoutError(f"MCP server did not start within {self.timeout_s} s")
        if self._error is not None:
            self._thread.join()
            raise RuntimeError(f"could not open the MCP session: {self._error}") from self._error
        return self

    async def _main(self) -> None:
        """Runs in the background thread for the whole session. The session is opened and closed in this one
        task: anyio (inside fastmcp) does not allow closing it from another task."""
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        try:
            transport = StdioTransport(command=self.command, args=self.args, env=self.env)
            async with Client(transport, timeout=self.timeout_s) as client:
                self._client = client
                self._ready.set()
                await self._stop.wait()  # keep the session (and the server process) alive until close()
        except BaseException as e:
            self._error = e
        finally:
            self._client = None
            self._ready.set()  # never leave open() waiting

    def close(self) -> None:
        """End the session (the server process stops, and with it the camera) and the background thread."""
        if self._thread is None:
            return
        if self._loop is not None and self._stop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(self.timeout_s)
        self._thread = None

    def __enter__(self) -> McpSession:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- calls ----

    def _submit(self, coro) -> asyncio.Future:
        """Run coro in the background loop; returns a concurrent.futures.Future for its result."""
        if self._client is None:
            coro.close()  # never started: close it so Python does not warn "coroutine was never awaited"
            raise RuntimeError("MCP session is not open; call open() first")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _call_tool(self, name: str, args: dict) -> dict:
        result = await self._client.call_tool(name, args, raise_on_error=False)
        if result.is_error:  # refused by the MCP server before the tool ran, e.g. an argument of the wrong type
            return {"ok": False, "error": result.content[0].text if result.content else "MCP error"}
        return result.structured_content

    def call(self, name: str, args: dict | None = None) -> dict:
        """Call one tool and wait for its result dict (sync: from the graph)."""
        return self._submit(self._call_tool(name, args or {})).result(self.timeout_s)

    async def acall(self, name: str, args: dict | None = None) -> dict:
        """Call one tool from any event loop (async)."""
        return await asyncio.wrap_future(self._submit(self._call_tool(name, args or {})))

    def tools(self) -> list[StructuredTool]:
        """Every tool of the server as a LangChain StructuredTool. The model sees the server's own name,
        description and JSON schema; the tool returns the result dict as a JSON string."""
        mcp_tools = self._submit(self._client.list_tools()).result(self.timeout_s)
        return [self._make_tool(t.name, t.description or "", t.input_schema) for t in mcp_tools]

    def _make_tool(self, name: str, description: str, schema: dict) -> StructuredTool:
        def func(**kwargs) -> str:
            return json.dumps(self.call(name, kwargs))

        async def coroutine(**kwargs) -> str:
            return json.dumps(await self.acall(name, kwargs))

        return StructuredTool(name=name, description=description, args_schema=schema, func=func, coroutine=coroutine)


def evcam_session(env: dict[str, str] | None = None, timeout_s: float = 120) -> McpSession:
    """A session with the event camera MCP server. env defaults to this process's environment (e.g. EVCAM_FAKE=1).
    timeout_s: the real camera needs ~12 s for its first call (open + settle)."""
    return McpSession(sys.executable, ["-m", "agentic_ode.mcp_servers.evcam_server"],
                      env=dict(os.environ) if env is None else env, timeout_s=timeout_s)
