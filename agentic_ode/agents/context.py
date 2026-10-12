"""What a specialist sees when it gets the turn: the whole dialogue plus the orchestrator's instruction."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agentic_ode.state import DialogueState, Role, render_dialogue


def decision_field(content: str, key: str) -> str:
    """A `key: value` line of an orchestrator turn, or ''."""
    for line in content.splitlines():
        if line.startswith(key + ":"):
            return line[len(key) + 1:].strip()
    return ""


def latest_instruction(state: DialogueState) -> str:
    for turn in reversed(state["dialogue"]):
        if turn["role"] == Role.ORCHESTRATOR:
            return decision_field(turn["content"], "instruction")
    return ""


def specialist_messages(system_prompt: str, state: DialogueState, role: Role) -> list[BaseMessage]:
    return [
        SystemMessage(system_prompt),
        HumanMessage(
            f"Dialogue so far:\n\n{render_dialogue(state['dialogue'])}\n\n"
            f"The orchestrator gave you ({role}) the turn with this instruction:\n{latest_instruction(state)}"
        ),
    ]
