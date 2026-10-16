"""The dialogue as text: how a role is shown, and the whole dialogue the way every node reads it."""

from __future__ import annotations

from agentic_ode.core.state import DialogueState, Role, Turn

ROLE_LABELS = {Role.HARDWARE: "executor (hardware)"}  # how a role is shown in transcripts and the report


def role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role)


def last_orchestrator_turn(state: DialogueState) -> str:
    """The text of the newest orchestrator turn, or ''."""
    for turn in reversed(state["dialogue"]):
        if turn["role"] == Role.ORCHESTRATOR:
            return turn["content"]
    return ""


def decision_field(content: str, key: str) -> str:
    """A `key: value` line of an orchestrator turn, or ''."""
    for line in content.splitlines():
        if line.startswith(key + ":"):
            return line[len(key) + 1:].strip()
    return ""


def render_dialogue(dialogue: list[Turn]) -> str:
    """The dialogue as plain text, the way every node reads it."""
    return "\n\n".join(
        f"### Turn {i} [{role_label(t['role'])}]\n{t['content']}" for i, t in enumerate(dialogue)
    )
