"""StateGraph: orchestrator routes to scientist / general / executor / human / hardware / END; specialists return to it.

After the person's reply, human runs the accepted step (executor or END) or returns to the orchestrator.
The hardware turn (HW3): hw_decide (the executor's model picks ONE camera call) -> hw_approve (only for setting
changes) -> hw_call (runs it, writes the result) -> orchestrator.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from agentic_ode.agents import Executor, Orchestrator, make_general, make_scientist
from agentic_ode.core.config import Settings
from agentic_ode.core.state import END_ROUTE, SPECIALISTS, DialogueState, Role
from agentic_ode.graph.nodes.hardware_turn import HardwareCall, HardwareDecide, hw_approve
from agentic_ode.graph.nodes.human_turn import human
from agentic_ode.graph.routing import HW_APPROVE, HW_CALL, HW_DECIDE, route


def build_graph(llm, workdir: Path, settings: Settings | None = None, camera_tools: list | None = None):
    """Compile the graph. `workdir` is where the executor writes and runs scripts.
    `camera_tools` (the event camera's MCP tools) go to the executor only, never to another role.

    The graph has an in-memory checkpointer, so every invoke/stream needs a thread_id in its config.
    """
    settings = settings or Settings()
    camera = bool(camera_tools)
    executor = Executor(llm, settings, workdir, camera_tools)
    orch = Role.ORCHESTRATOR.value
    g = StateGraph(DialogueState)
    g.add_node(orch, Orchestrator(llm, settings, camera))
    g.add_node(Role.SCIENTIST.value, make_scientist(llm, settings, camera))
    g.add_node(Role.GENERAL.value, make_general(llm, settings))
    g.add_node(Role.EXECUTOR.value, executor)
    g.add_node(Role.HUMAN.value, human)
    g.add_node(HW_DECIDE, HardwareDecide(executor, settings))
    g.add_node(HW_APPROVE, hw_approve)
    g.add_node(HW_CALL, HardwareCall(executor))

    g.set_entry_point(orch)
    g.add_conditional_edges(
        orch,
        route,
        {**{r.value: r.value for r in SPECIALISTS}, Role.HUMAN.value: Role.HUMAN.value,
         Role.HARDWARE.value: HW_DECIDE, END_ROUTE: END},
    )
    for role in SPECIALISTS:
        g.add_edge(role.value, orch)
    g.add_conditional_edges(
        Role.HUMAN.value,
        route,
        {orch: orch, Role.EXECUTOR.value: Role.EXECUTOR.value, END_ROUTE: END},
    )
    g.add_conditional_edges(HW_DECIDE, route, {HW_APPROVE: HW_APPROVE, HW_CALL: HW_CALL, orch: orch})
    g.add_conditional_edges(HW_APPROVE, route, {HW_CALL: HW_CALL, orch: orch})
    g.add_edge(HW_CALL, orch)
    return g.compile(checkpointer=InMemorySaver())
