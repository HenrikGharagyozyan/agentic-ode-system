"""agentic_ode.mcp_client against a real MCP server subprocess (tests/mcp_counter_server.py). No camera needed."""

import asyncio
import json
import sys
import threading

import pytest
from langchain_core.tools import StructuredTool

from agentic_ode.config import PROJECT_ROOT
from agentic_ode.mcp_client import McpSession, evcam_session


def counter_session() -> McpSession:
    env = {"PYTHONPATH": str(PROJECT_ROOT)}  # so `python -m tests.mcp_counter_server` finds the tests package
    return McpSession(sys.executable, ["-m", "tests.mcp_counter_server"], env=env, timeout_s=30)


def mcp_threads():
    return [t for t in threading.enumerate() if t.name == "mcp-session"]


def test_one_session_keeps_the_server_state():
    with counter_session() as s:
        assert s.call("add", {"n": 2}) == {"ok": True, "total": 2}
        assert s.call("add", {"n": 3}) == {"ok": True, "total": 5}
        assert s.call("total") == {"ok": True, "total": 5}  # same server process: the state is still there


def test_a_new_session_is_a_new_server_process():
    with counter_session() as s:
        s.call("add", {"n": 7})
    with counter_session() as s:
        assert s.call("total") == {"ok": True, "total": 0}  # new process: the state is gone


def test_tools_are_structured_tools_with_the_server_schema():
    with counter_session() as s:
        tools = {t.name: t for t in s.tools()}
        assert set(tools) == {"add", "total"} and all(isinstance(t, StructuredTool) for t in tools.values())
        assert "Add n to the counter" in tools["add"].description
        assert tools["add"].args_schema["properties"]["n"]["type"] == "integer"
        # sync (the graph) and async calls both go through the same session and return JSON text
        assert json.loads(tools["add"].invoke({"n": 4})) == {"ok": True, "total": 4}
        assert json.loads(asyncio.run(tools["add"].ainvoke({"n": 1}))) == {"ok": True, "total": 5}
        assert json.loads(tools["total"].invoke({})) == {"ok": True, "total": 5}


def test_async_call_from_another_event_loop():
    with counter_session() as s:
        async def two_calls():
            await s.acall("add", {"n": 10})
            return await s.acall("total")
        assert asyncio.run(two_calls()) == {"ok": True, "total": 10}


def test_mcp_error_is_returned_as_ok_false():
    with counter_session() as s:
        r = s.call("add", {"n": "abc"})  # refused by the server before the tool runs
        assert r["ok"] is False and "valid integer" in r["error"]
        r = s.call("no_such_tool")
        assert r["ok"] is False and "no_such_tool" in r["error"]


def test_close_ends_the_thread_and_calls_fail_after_close():
    s = counter_session().open()
    assert len(mcp_threads()) == 1
    s.close()
    assert mcp_threads() == []
    with pytest.raises(RuntimeError, match="not open"):
        s.call("total")
    s.close()  # closing twice is fine


def test_a_server_that_cannot_start_raises_on_open():
    s = McpSession(sys.executable, ["-c", "import sys; sys.exit(3)"], timeout_s=30)
    with pytest.raises(RuntimeError, match="could not open the MCP session"):
        s.open()
    assert mcp_threads() == []


def test_evcam_session_command():
    s = evcam_session(env={"EVCAM_FAKE": "1"})
    assert s.command == sys.executable and s.args == ["-m", "agentic_ode.mcp_servers.evcam_server"]
    assert s.env == {"EVCAM_FAKE": "1"}
