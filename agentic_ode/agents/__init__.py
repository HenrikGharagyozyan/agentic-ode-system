"""The four roles. Each node reads the whole dialogue and appends one turn."""

from agentic_ode.agents.executor import Executor
from agentic_ode.agents.orchestrator import Decision, Orchestrator, finish_allowed
from agentic_ode.agents.specialists import TextSpecialist, make_general, make_scientist

__all__ = [
    "Decision",
    "Executor",
    "Orchestrator",
    "TextSpecialist",
    "finish_allowed",
    "make_general",
    "make_scientist",
]
