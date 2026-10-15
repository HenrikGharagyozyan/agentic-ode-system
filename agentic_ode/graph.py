"""StateGraph: orchestrator routes to scientist / general / executor / human / END; specialists return to it.

After the person's reply, human runs the accepted step (executor or END) or returns to the orchestrator.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from agentic_ode.agents import Executor, Orchestrator, human, make_general, make_scientist
from agentic_ode.config import Settings
from agentic_ode.state import END_ROUTE, SPECIALISTS, DialogueState, Role


def route(state: DialogueState) -> str:
    return state["route"]


def build_graph(llm, workdir: Path, settings: Settings | None = None, camera_tools: list | None = None):
    """Compile the five-node graph. `workdir` is where the executor writes and runs scripts.
    `camera_tools` (the event camera's MCP tools) go to the executor only, never to another role.

    The graph has an in-memory checkpointer, so every invoke/stream needs a thread_id in its config.
    """
    settings = settings or Settings()
    g = StateGraph(DialogueState)
    g.add_node(Role.ORCHESTRATOR.value, Orchestrator(llm, settings))
    g.add_node(Role.SCIENTIST.value, make_scientist(llm, settings))
    g.add_node(Role.GENERAL.value, make_general(llm, settings))
    g.add_node(Role.EXECUTOR.value, Executor(llm, settings, workdir, camera_tools))
    g.add_node(Role.HUMAN.value, human)

    g.set_entry_point(Role.ORCHESTRATOR.value)
    g.add_conditional_edges(
        Role.ORCHESTRATOR.value,
        route,
        {**{r.value: r.value for r in SPECIALISTS}, Role.HUMAN.value: Role.HUMAN.value, END_ROUTE: END},
    )
    for role in SPECIALISTS:
        g.add_edge(role.value, Role.ORCHESTRATOR.value)
    g.add_conditional_edges(
        Role.HUMAN.value,
        route,
        {Role.ORCHESTRATOR.value: Role.ORCHESTRATOR.value, Role.EXECUTOR.value: Role.EXECUTOR.value, END_ROUTE: END},
    )
    return g.compile(checkpointer=InMemorySaver())
