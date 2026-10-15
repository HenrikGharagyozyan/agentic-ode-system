"""The dialogue as text: how a role is shown, and the whole dialogue the way every node reads it."""

from __future__ import annotations

from agentic_ode.core.state import Role, Turn

ROLE_LABELS = {Role.HARDWARE: "executor (hardware)"}  # how a role is shown in transcripts and the report


def role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role)


def render_dialogue(dialogue: list[Turn]) -> str:
    """The dialogue as plain text, the way every node reads it."""
    return "\n\n".join(
        f"### Turn {i} [{role_label(t['role'])}]\n{t['content']}" for i, t in enumerate(dialogue)
    )
