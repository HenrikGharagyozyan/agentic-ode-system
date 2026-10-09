"""The four roles. Each node reads the whole dialogue and appends one turn."""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from src.state import MAX_DECISIONS, DialogueState, RunRecord, render_dialogue
from src.tools import (
    RUN_TIMEOUT_S,
    TEST_RMSE_RE,
    TRAIN_RMSE_RE,
    format_run,
    make_executor_tools,
    parse_rmse,
    run_file,
    write_file,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "SNLS80mV.mat"

MAX_FINISH_RETRIES = 3
MAX_TOOL_ROUNDS = 10
MAX_RUNS_PER_TURN = 2
MAX_OUTPUT_CHARS = 12000  # per stream in the dialogue; the full output stays in the run record


def make_llm():
    """Chat model named by env var MODEL. Keys come from .env (OpenRouter by default)."""
    load_dotenv(ROOT / ".env")
    model = os.environ["MODEL"]
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
    if key:
        return init_chat_model(
            model,
            model_provider="openai",
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=key,
            temperature=0,
            max_tokens=16000,
            timeout=600,
            max_retries=3,
        )
    return init_chat_model(model, temperature=0, max_tokens=16000)


def with_retry(runnable):
    """Retry transient provider errors (free endpoints report overload as errors in the body)."""
    if hasattr(runnable, "with_retry"):
        return runnable.with_retry(stop_after_attempt=5, wait_exponential_jitter=True)
    return runnable


def text_of(msg) -> str:
    """Message content as a plain string (some providers return a list of blocks)."""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


ENVIRONMENT = f"""Environment facts (not results):
- The data file is at {DATA_FILE} (MATLAB .mat, load with scipy.io.loadmat; channels V1 and V2).
- Scripts run with Python 3, numpy, scipy, torch (CPU only, 4 cores) and matplotlib.
- Each script run is killed after {RUN_TIMEOUT_S} s, so one script must finish all its training and
  the free-run test well inside that limit (aim for under 200 s)."""


# ---------------------------------------------------------------- orchestrator

class Decision(BaseModel):
    next: Literal["scientist", "general", "executor", "finish"] = Field(
        description="The role that gets the next turn, or finish."
    )
    instruction: str = Field(description="What that role should do now.")
    reason: str = Field(description="One sentence: why this role now.")


ORCHESTRATOR_PROMPT = f"""You are the orchestrator of a team of three specialists working on the user's task.
You read the whole dialogue and decide which specialist speaks next, with an instruction and a one-sentence reason.
You write no theory, no equations and no code, and you have no tools.

The specialists:
- scientist: owns the theory and the plan (differential equation, network, state equation, training, free-run test, numbered steps). No tools.
- general: answers conceptual side questions. No tools, never writes the fitting plan.
- executor: the only role with a coding tool; writes a new Python script from the scientist's latest plan, runs it, and reports the printed free-run RMSE or the traceback.

Routing rules:
- Plan / ODE questions (the differential equation, the network, training, the free-run test, revising the plan) -> scientist.
- Tracebacks, failed or timed-out runs, and implementing or running the plan -> executor.
- Conceptual side questions (what a record in the file is, what a term means, what a reported error measures) -> general.
- finish only when an executor turn contains a free-run test RMSE printed by a script that ran with exit code 0, and the task is answered.

You have at most {MAX_DECISIONS} decisions in total. Make the instruction concrete for the current state of the dialogue."""


def _decision_turn(d: Decision) -> str:
    return f"next: {d.next}\ninstruction: {d.instruction}\nreason: {d.reason}"


def finish_allowed(state: DialogueState) -> bool:
    """True if some executor turn carries a test RMSE parsed from the stdout of a run with exit code 0."""
    dialogue = state["dialogue"]
    for r in state.get("runs", []):
        i = r["turn_index"]
        if (
            r["exit_code"] == 0
            and r["test_rmse"] is not None
            and 0 <= i < len(dialogue)
            and dialogue[i]["role"] == "executor"
            and _last_rmse_line(r["stdout"]) in dialogue[i]["content"]
        ):
            return True
    return False


def _last_rmse_line(stdout: str) -> str:
    return [m.group(0) for m in TEST_RMSE_RE.finditer(stdout)][-1]


def make_orchestrator(llm):
    decide = with_retry(llm.with_structured_output(Decision, method="function_calling"))

    def orchestrator(state: DialogueState) -> dict:
        n = state["decisions"]
        if n >= MAX_DECISIONS:
            return {"route": "__end__", "stop_reason": f"decision limit ({MAX_DECISIONS}) reached"}

        base = [
            SystemMessage(ORCHESTRATOR_PROMPT),
            HumanMessage(
                f"Dialogue so far:\n\n{render_dialogue(state['dialogue'])}\n\n"
                f"You have used {n} of {MAX_DECISIONS} decisions. Decide the next step."
            ),
        ]
        messages = list(base)
        decision: Decision = decide.invoke(messages)
        retries = 0
        while decision.next == "finish" and not finish_allowed(state) and retries < MAX_FINISH_RETRIES:
            retries += 1
            messages = base + [
                AIMessage(_decision_turn(decision)),
                HumanMessage(
                    "NOTE: finish is not yet allowed. No executor turn contains a free-run test RMSE "
                    "printed by a script that ran with exit code 0. Choose scientist, general or executor."
                ),
            ]
            decision = decide.invoke(messages)

        if decision.next == "finish" and finish_allowed(state):
            route, stop = "__end__", "finish"
        elif decision.next == "finish":
            # The model kept choosing finish against the guard; stop without claiming a result.
            route, stop = "__end__", "finish refused by guard (no RMSE from a successful run)"
        else:
            route, stop = decision.next, ""
        return {
            "dialogue": [{"role": "orchestrator", "content": _decision_turn(decision)}],
            "decisions": 1,
            "route": route,
            "stop_reason": stop,
        }

    return orchestrator


def latest_instruction(state: DialogueState) -> str:
    for t in reversed(state["dialogue"]):
        if t["role"] == "orchestrator":
            for line in t["content"].splitlines():
                if line.startswith("instruction:"):
                    return line[len("instruction:"):].strip()
    return ""


def _specialist_messages(system: str, state: DialogueState, role: str) -> list:
    return [
        SystemMessage(system),
        HumanMessage(
            f"Dialogue so far:\n\n{render_dialogue(state['dialogue'])}\n\n"
            f"The orchestrator gave you ({role}) the turn with this instruction:\n{latest_instruction(state)}"
        ),
    ]


# ---------------------------------------------------------------- scientist

SCIENTIST_PROMPT = f"""You are the scientist. You own the theory and the plan. You have no tools and write no code.

From the problem in the dialogue, write:
1. The differential equation of the system.
2. The network you propose (inputs, depth, widths, activation, input/output scaling).
3. The state equation to integrate and the integrator.
4. The training procedure (loss, optimizer, learning rate, steps, rollout length), sized to the time limit below.
5. The free-run test, exactly as the problem defines it.
6. A numbered list of implementation steps specific enough to code directly (indices, shapes, constants).
   The script must print the two results on their own lines as
   FREE_RUN_TRAIN_RMSE: <value in volts>
   FREE_RUN_TEST_RMSE: <value in volts>

If an earlier plan exists and the executor reported results or problems, revise the network or the training,
but keep the same differential equation and the same free-run test.
Never claim a measured number (RMSE, coefficient, run time) unless an executor turn in the dialogue already printed it;
when you cite one, say which executor turn printed it.

{ENVIRONMENT}"""


def make_scientist(llm):
    llm = with_retry(llm)

    def scientist(state: DialogueState) -> dict:
        reply = llm.invoke(_specialist_messages(SCIENTIST_PROMPT, state, "scientist"))
        return {"dialogue": [{"role": "scientist", "content": text_of(reply)}]}

    return scientist


# ---------------------------------------------------------------- general

GENERAL_PROMPT = """You are the general agent. You answer high-level side questions that sit beside the fit:
what a record in the data file is, what a term in the problem means, what a reported error measures.
You have no tools. You do not write the fitting plan, equations to fit, training steps or code;
if asked for those, say that they belong to the scientist. Never invent measured numbers.
Answer briefly and precisely."""


def make_general(llm):
    llm = with_retry(llm)

    def general(state: DialogueState) -> dict:
        reply = llm.invoke(_specialist_messages(GENERAL_PROMPT, state, "general"))
        return {"dialogue": [{"role": "general", "content": text_of(reply)}]}

    return general


# ---------------------------------------------------------------- executor

EXECUTOR_PROMPT = f"""You are the executor, the only role with a coding tool. Tools:
- write_file(filename, content): writes a NEW Python file into this run's working directory (never overwrites).
- run_file(filename): runs it and returns exit code, stdout and stderr.

Implement the scientist's latest plan from the dialogue in a new file (a name not used before, e.g. fit_01.py, fit_02.py),
run it, and report. If an earlier script failed, write a repaired script in a new file; you may fix bugs, speed or
numerics, but keep the scientist's differential equation and free-run test unchanged.

Script requirements:
- Self-contained; use the absolute data path below; fix random seeds.
- Print a short progress log (at most ~40 lines) and at the end exactly these two lines:
  FREE_RUN_TRAIN_RMSE: <float, volts>
  FREE_RUN_TEST_RMSE: <float, volts>
- Optionally save a plot of measured vs free-run predicted output on the test segment as test_plot.png.
- Finish well inside the time limit.
You may call run_file at most {MAX_RUNS_PER_TURN} times in this turn.
Finish with a short report: file name, exit code, and the RMSE lines exactly as printed, or the traceback.
Never state a number that is not in the tool output.

{ENVIRONMENT}"""


def _clip(s: str) -> str:
    if len(s) <= MAX_OUTPUT_CHARS:
        return s
    half = MAX_OUTPUT_CHARS // 2
    return f"{s[:half]}\n[... {len(s) - MAX_OUTPUT_CHARS} characters omitted ...]\n{s[-half:]}"


class Executor:
    def __init__(self, llm, workdir: Path, timeout: float = RUN_TIMEOUT_S):
        self.workdir = workdir
        self.timeout = timeout
        self.tools = make_executor_tools(workdir, timeout)
        self.llm = with_retry(llm.bind_tools(self.tools))

    def __call__(self, state: DialogueState) -> dict:
        messages = _specialist_messages(EXECUTOR_PROMPT, state, "executor")
        turn_index = len(state["dialogue"])
        runs: list[RunRecord] = []
        outputs: list[str] = []
        written: list[str] = []
        final = ""
        for _ in range(MAX_TOOL_ROUNDS):
            reply = self.llm.invoke(messages)
            messages.append(reply)
            if not reply.tool_calls:
                final = text_of(reply)
                break
            for call in reply.tool_calls:
                args = call["args"]
                try:
                    if call["name"] == "write_file":
                        out = write_file(self.workdir, args["filename"], args["content"])
                        if not out.startswith("ERROR"):
                            written.append(args["filename"])
                    elif call["name"] == "run_file":
                        if len(runs) >= MAX_RUNS_PER_TURN:
                            out = f"ERROR: run limit of {MAX_RUNS_PER_TURN} runs in this turn reached; report now."
                        else:
                            res = run_file(self.workdir, args["filename"], self.timeout)
                            out = format_run(
                                args["filename"], {**res, "stdout": _clip(res["stdout"]), "stderr": _clip(res["stderr"])}
                            )
                            outputs.append(out)
                            runs.append({
                                "turn_index": turn_index,
                                "filename": args["filename"],
                                "exit_code": res["exit_code"],
                                "stdout": res["stdout"],
                                "stderr": res["stderr"],
                                "test_rmse": parse_rmse(res["stdout"], TEST_RMSE_RE),
                                "train_rmse": parse_rmse(res["stdout"], TRAIN_RMSE_RE),
                            })
                    else:
                        out = f"ERROR: unknown tool {call['name']}"
                except (KeyError, ValueError) as e:
                    out = f"ERROR: {e}"
                messages.append(ToolMessage(content=out, tool_call_id=call["id"]))
        else:
            final = "(tool-call limit reached before a final report)"

        content = final.strip() or "(no report text)"
        if written:
            content += "\n\nFiles written: " + ", ".join(written)
        if outputs:
            content += "\n\n=== tool output (verbatim) ===\n" + "\n\n".join(outputs)
        else:
            content += "\n\n=== tool output ===\n(no script was run in this turn)"
        return {"dialogue": [{"role": "executor", "content": content}], "runs": runs}
