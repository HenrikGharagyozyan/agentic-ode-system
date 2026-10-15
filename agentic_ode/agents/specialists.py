"""The tool-less specialists: scientist (theory and plan) and general agent (side questions)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from agentic_ode.agents import prompts
from agentic_ode.agents.context import specialist_messages
from agentic_ode.core.config import Settings
from agentic_ode.core.llm import text_of, with_retry
from agentic_ode.core.state import DialogueState, Role


class TextSpecialist:
    """One plain model call over the dialogue; appends one turn. Never receives tools."""

    def __init__(self, llm, settings: Settings, role: Role, prompt: Callable[[Settings], str]):
        self.role = role
        self.prompt = prompt(settings)
        self.llm = with_retry(llm, settings.provider_retries)

    def __call__(self, state: DialogueState) -> dict:
        reply = self.llm.invoke(specialist_messages(self.prompt, state, self.role))
        return {"dialogue": [{"role": self.role.value, "content": text_of(reply)}]}


def make_scientist(llm, settings: Settings, camera: bool = False) -> TextSpecialist:
    """camera=True: the prompt also asks for camera measurement plans (in text; the scientist never calls tools)."""
    return TextSpecialist(llm, settings, Role.SCIENTIST, partial(prompts.scientist, camera=camera))


def make_general(llm, settings: Settings) -> TextSpecialist:
    return TextSpecialist(llm, settings, Role.GENERAL, prompts.general)
