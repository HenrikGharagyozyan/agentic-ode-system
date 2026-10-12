"""A scripted fake chat model, so the graph can be tested without an API."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agentic_ode.agents import Decision
from agentic_ode.config import Settings
from agentic_ode.graph import build_graph
from agentic_ode.state import initial_state

GOOD_SCRIPT = "print('FREE_RUN_TRAIN_RMSE: 0.01')\nprint('FREE_RUN_TEST_RMSE: 0.02')\n"


class FakeLLM:
    """Orchestrator decisions come from a list (the last one repeats). The executor writes one
    script and runs it. Records every bind_tools call to check which nodes receive tools."""

    def __init__(self, decisions, executor_script=GOOD_SCRIPT):
        self.decisions = list(decisions)
        self.decision_calls = 0
        self.bound_tools: list[list[str]] = []
        self.executor_script = executor_script
        self.scripts = 0

    # orchestrator
    def with_structured_output(self, schema, **kw):
        outer = self

        class _Structured:
            def invoke(self, messages):
                outer.decision_calls += 1
                nxt = outer.decisions.pop(0) if len(outer.decisions) > 1 else outer.decisions[0]
                return Decision(next=nxt, instruction="do it", reason="test")

        return _Structured()

    # scientist / general
    def invoke(self, messages):
        return AIMessage(content="plan text")

    # executor
    def bind_tools(self, tools):
        self.bound_tools.append([t.name for t in tools])
        outer = self

        class _Bound:
            def invoke(self, messages):
                n = sum(1 for m in messages if getattr(m, "type", "") == "tool")
                if n == 0:
                    outer.scripts += 1
                    self.name = f"fit_{outer.scripts:02d}.py"
                    return AIMessage(content="", tool_calls=[{
                        "name": "write_file", "id": "a",
                        "args": {"filename": self.name, "content": outer.executor_script},
                    }])
                if n == 1:
                    return AIMessage(content="", tool_calls=[{
                        "name": "run_file", "id": "b", "args": {"filename": self.name},
                    }])
                return AIMessage(content="ran it")

        return _Bound()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(model="fake", run_timeout_s=20, runs_dir=tmp_path / "runs")


@pytest.fixture
def run_graph(tmp_path, settings):
    def _run(llm):
        graph = build_graph(llm, tmp_path, settings)
        return graph.invoke(initial_state("task"), config={"recursion_limit": 60})

    return _run
