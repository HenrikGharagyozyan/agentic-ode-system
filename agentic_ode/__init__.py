"""A four-role LangGraph system (orchestrator, scientist, general, executor) that fits and tests ODE models."""

from agentic_ode.config import Settings
from agentic_ode.graph import build_graph
from agentic_ode.runner import RunOutcome, run_task

__version__ = "0.2.0"

__all__ = ["RunOutcome", "Settings", "build_graph", "run_task", "__version__"]
