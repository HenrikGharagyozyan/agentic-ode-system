"""StateGraph: orchestrator routes to scientist / general / executor / END; specialists return to it."""

from pathlib import Path

from langgraph.graph import END, StateGraph

from src.agents import Executor, make_general, make_orchestrator, make_scientist
from src.state import DialogueState
from src.tools import RUN_TIMEOUT_S


def route(state: DialogueState) -> str:
    return state["route"]


def build_graph(llm, workdir: Path, timeout: float = RUN_TIMEOUT_S):
    g = StateGraph(DialogueState)
    g.add_node("orchestrator", make_orchestrator(llm))
    g.add_node("scientist", make_scientist(llm))
    g.add_node("general", make_general(llm))
    g.add_node("executor", Executor(llm, workdir, timeout))

    g.set_entry_point("orchestrator")
    g.add_conditional_edges(
        "orchestrator",
        route,
        {"scientist": "scientist", "general": "general", "executor": "executor", "__end__": END},
    )
    for node in ("scientist", "general", "executor"):
        g.add_edge(node, "orchestrator")
    return g.compile()
