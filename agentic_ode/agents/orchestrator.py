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
        description="What that role should do now. "
        "For human with kind hard: what the proposed step will do after a yes."
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
    """True if some executor turn carries a test RMSE parsed from the stdout of a run with exit code 0,
    or the FOCUS_ENHANCEMENT line of a real focusing-tool call."""
    dialogue = state["dialogue"]
    for exp in state.get("experiments", []):
        i = exp["turn_index"]
        in_turn = 0 <= i < len(dialogue) and dialogue[i]["role"] == Role.EXECUTOR
        if in_turn and exp["result_line"] in dialogue[i]["content"]:
            return True
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
        refused = state.get("refused_step", "")
        if refused and decision.next == refused:
            # The model kept choosing the step the human just refused; stop so that step never runs.
            route, stop = END_ROUTE, f"guard: the model chose the refused step '{refused}' again"
        elif decision.next == FINISH and allowed:
            route, stop = END_ROUTE, "finish"
        elif decision.next == FINISH:
            # The model kept choosing finish against the guard; stop without claiming a result.
            route, stop = END_ROUTE, "finish refused by guard (no RMSE from a successful run)"
        else:
            route, stop = decision.next, ""
        log.info("decision %d: %s (%s)", used + 1, decision.next, decision.reason)
        return {
            "dialogue": [{"role": Role.ORCHESTRATOR.value, "content": decision.as_turn()}],
            "decisions": 1,
            "route": route,
            "stop_reason": stop,
            "pending_step": decision.proposed_next if decision.next == HUMAN and decision.kind == "hard" else "",
            "refused_step": "",  # the block lasts for this one decision only
        }

    def _decide_respecting_guard(self, state: DialogueState) -> Decision:
        """Ask the model; if a guard blocks its choice, ask again with a note (the next role is never hard-coded)."""
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
        # max_finish_retries limits the re-asks of both guards (finish guard and refused-step guard).
        while (note := _guard_note(decision, state)) and retries < self.settings.max_finish_retries:
            retries += 1
            log.info("%s blocked by guard, asking again (%d/%d)", decision.next, retries,
                     self.settings.max_finish_retries)
            decision = self.decide.invoke(base + [AIMessage(decision.as_turn()), HumanMessage(note)])
        return decision


def _guard_note(decision: Decision, state: DialogueState) -> str:
    """The note to send back if a guard blocks this decision, or "" if the decision is allowed."""
    refused = state.get("refused_step", "")
    if refused and decision.next == refused:
        return prompts.step_refused(refused)  # checked first: a refused finish stays blocked
    if decision.next == FINISH and not finish_allowed(state):
        return prompts.FINISH_NOT_ALLOWED
    return ""
