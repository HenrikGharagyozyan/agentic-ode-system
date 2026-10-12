"""StateGraph: orchestrator routes to scientist / general / executor / END; specialists return to it."""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, StateGraph

from agentic_ode.agents import Executor, Orchestrator, make_general, make_scientist
from agentic_ode.config import Settings
from agentic_ode.state import END_ROUTE, SPECIALISTS, DialogueState, Role


def route(state: DialogueState) -> str:
    return state["route"]


def build_graph(llm, workdir: Path, settings: Settings | None = None):
    """Compile the four-node graph. `workdir` is where the executor writes and runs scripts."""
    settings = settings or Settings()
    g = StateGraph(DialogueState)
    g.add_node(Role.ORCHESTRATOR.value, Orchestrator(llm, settings))
    g.add_node(Role.SCIENTIST.value, make_scientist(llm, settings))
    g.add_node(Role.GENERAL.value, make_general(llm, settings))
    g.add_node(Role.EXECUTOR.value, Executor(llm, settings, workdir))

    g.set_entry_point(Role.ORCHESTRATOR.value)
    g.add_conditional_edges(
        Role.ORCHESTRATOR.value,
        route,
        {**{r.value: r.value for r in SPECIALISTS}, END_ROUTE: END},
    )
    for role in SPECIALISTS:
        g.add_edge(role.value, Role.ORCHESTRATOR.value)
    return g.compile()
