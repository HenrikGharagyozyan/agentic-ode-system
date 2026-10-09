"""Offline tests with a scripted fake model: tool timeout, finish guard, decision limit, tool ownership."""

import time

import pytest
from langchain_core.messages import AIMessage

from src.agents import Decision, finish_allowed
from src.graph import build_graph
from src.state import MAX_DECISIONS, initial_state
from src.tools import parse_rmse, run_file, write_file

GOOD_SCRIPT = "print('FREE_RUN_TRAIN_RMSE: 0.01')\nprint('FREE_RUN_TEST_RMSE: 0.02')\n"


class FakeLLM:
    """Scripted stand-in for a chat model. Records which nodes asked for tools."""

    def __init__(self, decisions, executor_script=GOOD_SCRIPT):
        self.decisions = list(decisions)
        self.decision_calls = 0
        self.bound_tools = []  # tool lists passed to bind_tools
        self.executor_script = executor_script
        self.scripts = 0

    # orchestrator
    def with_structured_output(self, schema, **kw):
        outer = self

        class _S:
            def invoke(self, messages):
                outer.decision_calls += 1
                nxt = outer.decisions.pop(0) if len(outer.decisions) > 1 else outer.decisions[0]
                return Decision(next=nxt, instruction="do it", reason="test")

        return _S()

    # scientist / general
    def invoke(self, messages):
        return AIMessage(content="plan text")

    # executor
    def bind_tools(self, tools):
        self.bound_tools.append([t.name for t in tools])
        outer = self

        class _B:
            def invoke(self, messages):
                n = sum(1 for m in messages if getattr(m, "type", "") == "tool")
                if n == 0:
                    outer.scripts += 1
                    self.name = f"fit_{outer.scripts:02d}.py"
                    return AIMessage(content="", tool_calls=[{"name": "write_file", "id": "a",
                                                              "args": {"filename": self.name, "content": outer.executor_script}}])
                if n == 1:
                    return AIMessage(content="", tool_calls=[{"name": "run_file", "id": "b", "args": {"filename": self.name}}])
                return AIMessage(content="ran it")

        return _B()


@pytest.fixture
def workdir(tmp_path):
    return tmp_path


def run_graph(llm, workdir):
    graph = build_graph(llm, workdir, timeout=20)
    return graph.invoke(initial_state("task"), config={"recursion_limit": 60})


def test_tool_timeout(workdir):
    write_file(workdir, "slow.py", "import time\ntime.sleep(30)\n")
    t0 = time.time()
    res = run_file(workdir, "slow.py", timeout=1)
    assert time.time() - t0 < 10
    assert res["timed_out"] and res["exit_code"] is None
    assert "TIMEOUT" in res["stderr"]


def test_write_file_never_overwrites(workdir):
    assert not write_file(workdir, "a.py", "x=1").startswith("ERROR")
    assert write_file(workdir, "a.py", "x=2").startswith("ERROR")
    assert (workdir / "a.py").read_text() == "x=1"


def test_finish_guard_blocks_early_finish(workdir):
    llm = FakeLLM(["finish"])  # always finish, no executor turn yet
    out = run_graph(llm, workdir)
    assert out["decisions"] == 1
    assert llm.decision_calls == 4  # first call + 3 re-asks with the "not yet allowed" note
    assert out["stop_reason"].startswith("finish refused")
    assert not finish_allowed(out)


def test_finish_guard_allows_after_real_rmse(workdir):
    llm = FakeLLM(["executor", "finish"])
    out = run_graph(llm, workdir)
    assert out["stop_reason"] == "finish"
    assert out["runs"][0]["exit_code"] == 0 and out["runs"][0]["test_rmse"] == 0.02
    assert finish_allowed(out)


def test_guard_rejects_failed_script(workdir):
    llm = FakeLLM(["executor", "finish"], executor_script="print('FREE_RUN_TEST_RMSE: 0.02')\nraise SystemExit(1)\n")
    out = run_graph(llm, workdir)
    assert not finish_allowed(out)
    assert out["stop_reason"] != "finish"


def test_rmse_typed_by_model_does_not_count(workdir):
    state = initial_state("task")
    state["dialogue"].append({"role": "executor", "content": "FREE_RUN_TEST_RMSE: 0.001"})
    assert not finish_allowed(state)  # no real run record behind it


def test_decision_limit(workdir):
    llm = FakeLLM(["general"])  # never finishes
    out = run_graph(llm, workdir)
    assert out["decisions"] == MAX_DECISIONS
    assert sum(t["role"] == "orchestrator" for t in out["dialogue"]) == MAX_DECISIONS
    assert llm.decision_calls == MAX_DECISIONS
    assert "decision limit" in out["stop_reason"]


def test_only_executor_has_tools(workdir):
    llm = FakeLLM(["finish"])
    build_graph(llm, workdir)
    assert llm.bound_tools == [["write_file", "run_file"]]  # bound exactly once, by the executor


def test_parse_rmse():
    assert parse_rmse("loss 1\nFREE_RUN_TEST_RMSE: 2.5e-3\n") == 2.5e-3
    assert parse_rmse("test rmse is about 0.002") is None
