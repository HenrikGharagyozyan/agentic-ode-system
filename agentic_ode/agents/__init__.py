"""The four roles. Each node reads the whole dialogue and appends one turn."""

from agentic_ode.agents.executor import Executor
from agentic_ode.agents.guards import finish_allowed
from agentic_ode.agents.human import human
from agentic_ode.agents.orchestrator import Decision, Orchestrator
from agentic_ode.agents.specialists import TextSpecialist, make_general, make_scientist

__all__ = [
    "Decision",
    "Executor",
    "Orchestrator",
    "TextSpecialist",
    "finish_allowed",
    "human",
    "make_general",
    "make_scientist",
]
