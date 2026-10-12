"""The orchestrator: one structured model decision per turn, plus the finish guard."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic_ode.agents import prompts
from agentic_ode.config import Settings
from agentic_ode.llm import with_retry
from agentic_ode.state import END_ROUTE, DialogueState, Role, render_dialogue
from agentic_ode.tools import last_rmse_line

log = logging.getLogger(__name__)

FINISH = "finish"
HUMAN = "human"


class Decision(BaseModel):
    next: Literal["scientist", "general", "executor", "human", "finish"] = Field(
        description="The role that gets the next turn, human to ask the person, or finish."
    )
    instruction: str = Field(
        description="What that role should do now. For human with kind hard: what the proposed step will do after a yes."
    )
    reason: str = Field(description="One sentence: why this role now.")
    kind: Literal["ambiguous", "hard", "none"] = Field(
        default="none",
        description="ambiguous: the request has more than one reading; hard: the next step needs approval; "
        "none: next is not human.",
    )
    question: str = Field(default="", description="The question for the human. Empty unless next is human.")
    proposed_next: Literal["executor", "finish", "none"] = Field(
        default="none",
        description="When kind is hard: the step that waits for approval (executor or finish). Otherwise none.",
    )

    def as_turn(self) -> str:
        text = f"next: {self.next}\ninstruction: {self.instruction}\nreason: {self.reason}"
        if self.next == HUMAN:
            text += f"\nkind: {self.kind}\nproposed_next: {self.proposed_next}\nquestion: {self.question}"
        return text


def finish_allowed(state: DialogueState) -> bool:
    """True if some executor turn carries a test RMSE parsed from the stdout of a run with exit code 0."""
    dialogue = state["dialogue"]
    for run in state.get("runs", []):
        i = run["turn_index"]
        if (
            run["exit_code"] == 0
            and run["test_rmse"] is not None
            and 0 <= i < len(dialogue)
            and dialogue[i]["role"] == Role.EXECUTOR
            and last_rmse_line(run["stdout"]) in dialogue[i]["content"]
        ):
            return True
    return False


class Orchestrator:
    """Reads the whole dialogue and picks the next role. Writes no theory or code and has no tools."""

    def __init__(self, llm, settings: Settings):
        self.settings = settings
        self.prompt = prompts.orchestrator(settings)
        self.decide = with_retry(
            llm.with_structured_output(Decision, method="function_calling"), settings.provider_retries
        )

    def __call__(self, state: DialogueState) -> dict:
        used, limit = state["decisions"], self.settings.max_decisions
        if used >= limit:
            return {"route": END_ROUTE, "stop_reason": f"decision limit ({limit}) reached"}

        decision = self._decide_respecting_guard(state)
        allowed = finish_allowed(state)
        if decision.next == FINISH and allowed:
            route, stop = END_ROUTE, "finish"
        elif decision.next == FINISH:
            # The model kept choosing finish against the guard; stop without claiming a result.
            route, stop = END_ROUTE, "finish refused by guard (no RMSE from a successful run)"
        elif decision.next == HUMAN:
            # Temporary: there is no human node yet, so stop and keep the question in the dialogue.
            route, stop = END_ROUTE, "human input needed (no human node yet)"
        else:
            route, stop = decision.next, ""
        log.info("decision %d: %s (%s)", used + 1, decision.next, decision.reason)
        return {
            "dialogue": [{"role": Role.ORCHESTRATOR.value, "content": decision.as_turn()}],
            "decisions": 1,
            "route": route,
            "stop_reason": stop,
        }

    def _decide_respecting_guard(self, state: DialogueState) -> Decision:
        """Ask the model; if it finishes too early, ask again with a note (the next role is never hard-coded)."""
        base = [
            SystemMessage(self.prompt),
            HumanMessage(
                f"Dialogue so far:\n\n{render_dialogue(state['dialogue'])}\n\n"
                f"You have used {state['decisions']} of {self.settings.max_decisions} decisions. "
                "Decide the next step."
            ),
        ]
        decision: Decision = self.decide.invoke(base)
        retries = 0
        while decision.next == FINISH and not finish_allowed(state) and retries < self.settings.max_finish_retries:
            retries += 1
            log.info("finish blocked by guard, asking again (%d/%d)", retries, self.settings.max_finish_retries)
            decision = self.decide.invoke(
                base + [AIMessage(decision.as_turn()), HumanMessage(prompts.FINISH_NOT_ALLOWED)]
            )
        return decision
