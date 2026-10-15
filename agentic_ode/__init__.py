"""A LangGraph system of four model roles (orchestrator, scientist, general, executor), a human pause and a hardware
turn: it fits and tests ODE models, runs focusing experiments on a simulated bench and drives an event camera."""

from agentic_ode.core.config import Settings

__version__ = "0.2.0"

__all__ = ["RunOutcome", "Settings", "build_graph", "run_task", "__version__"]


def __getattr__(name: str):
    """build_graph, run_task and RunOutcome are imported on first use (PEP 562). So importing one small part of the
    package, e.g. the MCP server or a camera driver, does not load LangGraph and LangChain."""
    if name == "build_graph":
        from agentic_ode.graph import build_graph

        return build_graph
    if name in ("run_task", "RunOutcome"):
        from agentic_ode.runtime import runner

        return getattr(runner, name)
    raise AttributeError(f"module 'agentic_ode' has no attribute {name!r}")
