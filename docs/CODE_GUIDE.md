# Code guide: every file and every function

This guide explains what the project does, how one run flows through the code, and what each file, class and function is for. Line numbers refer to the code as of version 0.2.0. If you edit a file, the numbers shift, but the names stay valid.

---

## 1. What the project does

The project is a **multi-agent system built on LangGraph**. You give it a task in plain text (here [task5.md](../task5.md): fit a neural ODE to the Silverbox oscillator data). Four roles then work on it together:

| Role | Job | Tools |
|---|---|---|
| **orchestrator** | Reads the whole conversation and decides who speaks next, with an instruction and a one-sentence reason. Can also decide `finish`. | none |
| **scientist** | Writes the theory and the plan: the differential equation, the network, the integrator, the training, the free-run test, and numbered steps. | none |
| **general** | Answers side questions, such as "what is channel V1?" or "what does RMSE measure?". | none |
| **executor** | Writes a Python script from the plan, runs it, and reports the real output. | `write_file`, `run_file` |

Four ideas carry the design. Keep them in mind when you change things.

1. **The state is the dialogue.** Every role reads the full list of turns and appends one new turn. Roles never call each other directly.
2. **The routing is decided by a model**, not by code. The orchestrator is an LLM call with structured output, so no keyword table or fixed order exists anywhere.
3. **Only the executor can touch code.** It is the only node that calls `bind_tools`.
4. **Results must be real.** The run may only `finish` after a script actually ran with exit code 0 and printed `FREE_RUN_TEST_RMSE: <number>`. The check (the *finish guard*) reads structured run records taken from the real subprocess output, never text the model typed.

---

## 2. One run, step by step

This is what happens when you type `agentic-ode run` (or `python main.py`):

```
cli.main()                              parse args, set up logging
 └─ cli.cmd_run()
     ├─ Settings.from_env()             read .env: MODEL, key, paths
     └─ runner.run_task(task5.md, settings, on_turn=_print_turn)
         ├─ llm.make_llm(settings)      create the chat model (OpenRouter)
         ├─ runner.new_run_dir()        create runs/<timestamp>/
         ├─ graph.build_graph(llm, run_dir, settings)
         │     creates Orchestrator, scientist, general, Executor and wires the edges
         ├─ state.initial_state(task text)   dialogue = [{"role": "user", ...}]
         └─ graph.stream(...)           LangGraph runs the nodes:
               orchestrator → (scientist | general | executor) → orchestrator → ... → END
               after every step: print the new turns, save transcript.json/.md
```

Inside the loop:

1. **Orchestrator node** (`Orchestrator.__call__`):
   - If 8 decisions are already used, it stops (`route = "__end__"`).
   - Otherwise it asks the model for a `Decision`. If the model says `finish` too early, it asks again with a note, up to 3 times.
   - It appends its decision as a turn and sets `route`.
2. **LangGraph** reads `state["route"]` through `graph.route()` and jumps to that node, or to END.
3. **Specialist node:** the scientist or general agent makes one model call. The executor runs its internal tool loop. Either way, the node appends one turn.
4. A plain edge goes back to the orchestrator, and the loop repeats.

When the graph ends, `run_task` saves the final transcript and returns a `RunOutcome`.

Later, `agentic-ode report runs/<timestamp>` reads `transcript.json` and builds `report/solution.pdf`.

---

## 3. Package map

```
agentic_ode/
  __init__.py          public API
  __main__.py          python -m agentic_ode
  cli.py               command line
  config.py            Settings
  llm.py               model creation and helpers
  state.py             the shared graph state
  graph.py             the StateGraph wiring
  runner.py            run_task(): run one task end to end
  transcript.py        save/load transcript files
  agents/
    __init__.py        re-exports
    prompts.py         all system prompts
    context.py         what a specialist sees
    orchestrator.py    Decision, finish guard, Orchestrator
    specialists.py     scientist and general agent
    executor.py        Executor and its tool loop
  tools/
    __init__.py        LangChain tool wrappers + re-exports
    workspace.py       write_file / run_file (the sandbox)
    metrics.py         the RMSE output contract
  reporting/
    __init__.py        re-exports
    builder.py         HTML + PDF report
    diagram.py         graph picture + Mermaid
main.py                thin entry point
tests/                 offline tests with a fake model
pyproject.toml         package metadata, dependencies, pytest/ruff config
```

Dependency direction (who imports whom), from bottom to top:

```
config  ←  llm, state  ←  tools  ←  agents  ←  graph  ←  runner  ←  cli
                                                  ↑
                                     reporting (uses graph, tools, transcript, agents.context)
```

Nothing lower in the stack imports anything higher, so you can change an upper layer without touching the lower ones.

---

## 4. File by file

### 4.1 [agentic_ode/config.py](../agentic_ode/config.py)

**Purpose:** one place for every setting: model access, file paths and all numeric limits.

**`PROJECT_ROOT`** (line 11) is the repository root, computed from the file location. It is used for the default `data/` and `runs/` paths, and for the `.env` location.

**`class Settings`** (line 15) is a frozen dataclass, so it cannot be changed after creation. Use `dataclasses.replace(s, field=value)` to make a modified copy. Fields:

| Field | Default | Used by | Meaning |
|---|---|---|---|
| `model` | `""` | `make_llm` | Model name, e.g. `nvidia/nemotron-3-super-120b-a12b:free` |
| `api_key` | `None` | `make_llm` | OpenRouter key; if `None`, the model's own provider is used |
| `base_url` | OpenRouter URL | `make_llm` | OpenAI-compatible endpoint |
| `temperature` | `0.0` | `make_llm` | Sampling temperature |
| `max_tokens` | `16000` | `make_llm` | Max output tokens per call |
| `request_timeout_s` | `600` | `make_llm` | HTTP timeout per request |
| `request_retries` | `3` | `make_llm` | HTTP-level retries inside the OpenAI client |
| `provider_retries` | `5` | `with_retry` in every agent | Retries of a whole model call when a provider fails (e.g. 503 "overloaded") |
| `data_file` | `data/SNLS80mV.mat` | `prompts.environment` | Path given to the scientist and executor |
| `runs_dir` | `runs/` | `runner.new_run_dir` | Where run folders are created |
| `max_decisions` | `8` | `Orchestrator`, orchestrator prompt | Hard limit of orchestrator decisions |
| `max_finish_retries` | `3` | `Orchestrator._decide_respecting_guard` | How often to re-ask after an early `finish` |
| `run_timeout_s` | `300` | `run_file`, executor prompt | Kill a script after this many seconds |
| `max_tool_rounds` | `10` | `Executor.__call__` | Max model↔tool round trips in one executor turn |
| `max_runs_per_turn` | `2` | `Executor._run`, executor prompt | Max `run_file` calls in one executor turn |
| `max_output_chars` | `12000` | `Executor._clip` | stdout/stderr longer than this is cut in the middle (in the dialogue only) |

**`Settings.from_env(env_file)`** (line 41) loads `.env` (via `python-dotenv`) and builds `Settings` from environment variables:
- `MODEL` → `model`
- `OPENROUTER_API_KEY` or `OPEN_ROUTER_API_KEY` → `api_key`
- `OPENROUTER_BASE_URL` → `base_url`
- `AGENTIC_ODE_DATA_FILE` → `data_file`
- `AGENTIC_ODE_RUNS_DIR` → `runs_dir`

Every other field keeps its default. **To make another limit configurable from the environment, add a line here.**

---

### 4.2 [agentic_ode/llm.py](../agentic_ode/llm.py)

**Purpose:** create the chat model, plus small helpers every agent uses.

- **`class ConfigurationError`** (line 12): raised when the configuration is unusable, e.g. no `MODEL`. `cli.cmd_run` catches it and exits with code 2 and a clear message instead of a traceback.
- **`make_llm(settings)`** (line 16): returns a LangChain chat model.
  - If `MODEL` is empty, it raises `ConfigurationError`.
  - If an OpenRouter key is set, it calls `init_chat_model(model, model_provider="openai", base_url=OpenRouter, api_key=key, ...)`. OpenRouter speaks the OpenAI protocol, so the OpenAI client is used for every model name.
  - Without a key, it calls `init_chat_model(model)`, and LangChain picks the provider from the name (it would then need e.g. `ANTHROPIC_API_KEY`).
- **`with_retry(runnable, attempts)`** (line 37): wraps a LangChain runnable in `.with_retry(...)` with exponential back-off. Free endpoints often return overload errors inside an HTTP 200 body, which the OpenAI client's own retries don't catch. The `hasattr` check lets the test `FakeLLM` (which has no `with_retry`) pass through unchanged.
- **`text_of(msg)`** (line 44): returns a message's content as a plain string. Some providers return content as a list of blocks (`[{"type": "text", "text": ...}]`); this joins them.

---

### 4.3 [agentic_ode/state.py](../agentic_ode/state.py)

**Purpose:** defines the shared state that LangGraph passes between nodes. This is the "memory" of the system.

- **`END_ROUTE = "__end__"`**: the value of `route` that means "go to END".
- **`class Role(StrEnum)`** (line 12): the role names `user`, `orchestrator`, `scientist`, `general`, `executor`. A `StrEnum` compares equal to plain strings, so `turn["role"] == Role.EXECUTOR` works on JSON-loaded data. Node names in the graph are exactly these values.
- **`SPECIALISTS`**: the three roles the orchestrator can hand the turn to. `graph.py` loops over it to build edges.
- **`class Turn(TypedDict)`** (line 23): one dialogue entry, `{"role": str, "content": str}`.
- **`class RunRecord(TypedDict)`** (line 28): one real script execution, written by the executor:
  - `turn_index`: the position in the dialogue of the executor turn that ran it;
  - `filename`, `exit_code` (`None` = timeout), and full `stdout` and `stderr`;
  - `test_rmse` and `train_rmse`: parsed from stdout, `None` if not printed.

  The finish guard trusts these records, not the dialogue text.
- **`class DialogueState(TypedDict)`** (line 40): the graph state.
  - `dialogue: Annotated[list[Turn], operator.add]`: the **reducer** `operator.add` means a node returns only its *new* turns, and LangGraph concatenates them onto the existing list.
  - `decisions: Annotated[int, operator.add]`: the orchestrator returns `1` per decision, and LangGraph sums them.
  - `runs: Annotated[list[RunRecord], operator.add]`: appended the same way.
  - `route: str` and `stop_reason: str` have no reducer, so the last value written wins.
- **`initial_state(user_message)`** (line 48): the starting state, holding a one-turn dialogue with the user message, 0 decisions and no runs.
- **`render_dialogue(dialogue)`** (line 58): turns the dialogue into text like `### Turn 3 [executor]\n...`. **This is exactly what every model sees.** Change it to change how agents perceive the conversation, for example to summarize old turns.

---

### 4.4 [agentic_ode/tools/workspace.py](../agentic_ode/tools/workspace.py)

**Purpose:** the sandbox in which the executor writes and runs scripts. Pure Python, with no LangChain.

- **`class RunResult`** (line 15): the result of one run: `exit_code` (`None` on timeout or a missing file), `stdout`, `stderr`, and `timed_out`.
- **`check_filename(filename)`** (line 22): allows only names like `fit_01.py` (letters, digits, `_`, `-`, then `.py`). This blocks `../x.py`, sub-folders and non-Python files, so the model can't write outside the run folder. It raises `ValueError` otherwise.
- **`write_file(workdir, filename, content)`** (line 28): writes a **new** file. If the file already exists, it returns `"ERROR: ... already exists"` and does not overwrite. This enforces "each script in a new file". It returns a status string, which the model sees as the tool result.
- **`run_file(workdir, filename, timeout)`** (line 38): runs `python <filename>` with `subprocess.run`.
  - The working directory is `workdir`, and the interpreter is the same Python (`sys.executable`), so the script sees the venv's numpy and torch.
  - stdout and stderr are captured.
  - On timeout it returns `exit_code=None`, `timed_out=True`, any partial output, and `TIMEOUT: killed after N s` appended to stderr.
  - If the file doesn't exist, it returns an error result without running anything.
- **`format_run(filename, result)`** (line 57): renders a `RunResult` as the text the model and the dialogue see:
  ```
  run_file('fit_05.py')
  exit_code: 0
  --- stdout ---
  ...
  --- stderr ---
  ...
  ```
- **`_decode(x)`** (line 66): on timeout, `subprocess` may hand back bytes or `None`; this turns either into a string.

---

### 4.5 [agentic_ode/tools/metrics.py](../agentic_ode/tools/metrics.py)

**Purpose:** the **output contract** between generated scripts and the system. A script reports its result by printing these two lines:

```
FREE_RUN_TRAIN_RMSE: <float>
FREE_RUN_TEST_RMSE: <float>
```

- **`TEST_RMSE_RE`, `TRAIN_RMSE_RE`** (lines 14–15): regexes that match one whole line. Spaces and tabs are allowed around the number, but not newlines.
- **`parse_rmse(stdout, pattern)`** (line 18): the **last** matching value as a `float`, or `None`. "Last" matters because scripts may print progress lines before the final one.
- **`last_rmse_line(stdout, pattern)`** (line 24): the last matching line **exactly as printed**, or `None`. The finish guard uses it to check that this exact line appears in the executor's turn. The report uses it too.
- **`rmse_lines(stdout)`** (line 30): all train and test RMSE lines in print order. The report uses this to show both lines verbatim.

**To use this system for a different task with a different metric, this file and `finish_allowed` are the two places to change.**

---

### 4.6 [agentic_ode/tools/\_\_init\_\_.py](../agentic_ode/tools/__init__.py)

**Purpose:** the LangChain side of the tools, plus re-exports so other modules can write `from agentic_ode.tools import run_file`.

- **`WRITE_FILE`, `RUN_FILE`**: the tool names the model calls. `Executor._dispatch` compares against them.
- **`make_executor_tools(workdir, timeout)`** (line 16): builds two `StructuredTool`s from the inner functions `_write` (line 23) and `_run` (line 27). LangChain reads their **type hints and docstrings** to build the JSON schema the model sees, so the docstrings are effectively part of the prompt.
  - The executor binds these tools so the model knows they exist.
  - The executor then **dispatches calls itself** (see `Executor._dispatch`) rather than calling these wrappers, because it must record each run as a `RunRecord`.
  - The wrappers still work on their own, if you ever want a standard LangGraph `ToolNode`.

---

### 4.7 [agentic_ode/agents/prompts.py](../agentic_ode/agents/prompts.py)

**Purpose:** all system prompts in one place. Each is a function of `Settings`, so limits and paths are filled in automatically.

- **`environment(s)`** (line 8): facts given to the scientist and executor: the data path, the available libraries (numpy, scipy, torch on CPU, matplotlib) and the time limit per script.
- **`orchestrator(s)`** (line 16): the orchestrator's role, the three specialists, the **routing rules** (plan/ODE → scientist, tracebacks → executor, side questions → general, finish only after a real RMSE) and the decision budget. **This is where routing behavior lives.**
- **`FINISH_NOT_ALLOWED`** (line 35): the note sent back when the model picks `finish` before the guard allows it.
- **`scientist(s)`** (line 41): the six required sections of a plan, the required print format for the RMSE lines, the rule "keep the same ODE and free-run test when revising", and the rule "never claim a number an executor turn didn't print".
- **`general(s)`** (line 63): side questions only; no plan, no code, no invented numbers.
- **`executor(s)`** (line 71): the tool descriptions, "new file each time", "repairs keep the ODE and test", script requirements (data path, seeds, short log, the two RMSE lines, optional `test_plot.png`), the run limit per turn and the report format.

---

### 4.8 [agentic_ode/agents/context.py](../agentic_ode/agents/context.py)

**Purpose:** builds what a specialist receives when it gets the turn.

- **`decision_field(content, key)`** (line 10): orchestrator turns are stored as text lines `next: ...`, `instruction: ...`, `reason: ...`; this reads one of those lines. The report also uses it for `Files written:` in executor turns.
- **`latest_instruction(state)`** (line 18): finds the most recent orchestrator turn and returns its `instruction`.
- **`specialist_messages(system_prompt, state, role)`** (line 25): returns two messages:
  - `SystemMessage(system_prompt)`;
  - `HumanMessage("Dialogue so far: <the rendered dialogue> ... The orchestrator gave you (<role>) the turn with this instruction: <instruction>")`.

  Every specialist call starts from this.

---

### 4.9 [agentic_ode/agents/orchestrator.py](../agentic_ode/agents/orchestrator.py)

**Purpose:** the routing brain and the finish guard.

- **`FINISH = "finish"`**: the decision value that means stop.
- **`class Decision(BaseModel)`** (line 22): the **structured output** schema, so the model must return exactly:
  - `next`: one of `"scientist" | "general" | "executor" | "finish"` (a `Literal`, so anything else fails validation);
  - `instruction`: what the chosen role should do;
  - `reason`: one sentence.

  `Decision.as_turn()` (line 29) formats it as the three-line text appended to the dialogue.
- **`finish_allowed(state)`** (line 33): **the finish guard.** It returns `True` only if some `RunRecord` has all of these:
  1. `exit_code == 0`;
  2. a parsed `test_rmse`;
  3. a `turn_index` that points at a real turn in the dialogue, which is an executor turn;
  4. its exact last `FREE_RUN_TEST_RMSE:` line appearing in that executor turn's text.

  Because the records are built from real subprocess output, a model that just *writes* "FREE_RUN_TEST_RMSE: 0.001" in a message can't satisfy the guard. The test `test_rmse_typed_by_model_does_not_count` checks exactly this.

  **To require a quality threshold (e.g. RMSE below the 0.06 V baseline), add the condition here.**
- **`class Orchestrator`** (line 49): the node.
  - **`__init__(llm, settings)`** (line 52): stores the settings, builds the prompt, and creates `self.decide = llm.with_structured_output(Decision, method="function_calling")` wrapped in retries. `function_calling` makes the model answer through a tool call whose arguments match `Decision`. It works with most providers on OpenRouter.
  - **`__call__(state)`** (line 59): LangGraph calls this when the orchestrator has the turn.
    1. If `decisions >= max_decisions`, it returns `route="__end__"` and `stop_reason="decision limit (8) reached"` **without** calling the model and without appending a turn. That is why the specialist chosen by the 8th decision still gets to speak.
    2. Otherwise it gets a decision via `_decide_respecting_guard`.
    3. It maps the decision to a route:
       - `finish` allowed → END, `stop_reason="finish"`;
       - `finish` still not allowed after the retries → END, `"finish refused by guard ..."` (the run stops without claiming a result);
       - anything else → that role.
    4. It returns the new orchestrator turn, `decisions: 1` (summed by the reducer), `route` and `stop_reason`.
  - **`_decide_respecting_guard(state)`** (line 81): builds the messages (system prompt + rendered dialogue + "You have used N of 8 decisions") and asks the model. While the answer is `finish` and the guard says no (at most `max_finish_retries` times), it asks again, adding the previous answer and the `FINISH_NOT_ALLOWED` note. It **never picks the next role itself**: the model still decides.

---

### 4.10 [agentic_ode/agents/specialists.py](../agentic_ode/agents/specialists.py)

**Purpose:** the two tool-less roles. They differ only in the prompt and role name, so one class serves both.

- **`class TextSpecialist`** (line 14):
  - `__init__(llm, settings, role, prompt)` (line 17): stores the role, builds the prompt from settings, and wraps the model in retries. It never calls `bind_tools`.
  - `__call__(state)` (line 22): one model call with `specialist_messages(...)`; it returns `{"dialogue": [{"role": ..., "content": reply text}]}`.
- **`make_scientist(llm, settings)`** (line 27): a `TextSpecialist` with `Role.SCIENTIST` and `prompts.scientist`.
- **`make_general(llm, settings)`** (line 31): a `TextSpecialist` with `Role.GENERAL` and `prompts.general`.

**To add another tool-less role** (e.g. a "reviewer"), add a prompt function and a `make_reviewer`, then wire it in `graph.py` (see section 6).

---

### 4.11 [agentic_ode/agents/executor.py](../agentic_ode/agents/executor.py)

**Purpose:** the only node with tools. Inside one graph turn it runs a small agent loop (model → tool calls → results → model ...), then appends **one** turn.

- **`ROUND_LIMIT_NOTE`**: the text used as the report when the model never stops calling tools.
- **`class _TurnLog`** (line 35): scratch data for one executor turn:
  - `turn_index`: where this turn will land in the dialogue;
  - `runs`: `RunRecord`s;
  - `outputs`: the formatted run outputs that go into the turn;
  - `written`: the files created.
- **`class Executor`** (line 44):
  - **`__init__(llm, settings, workdir)`** (line 45): builds the prompt and the tools for this run folder, and `self.llm = llm.bind_tools(tools)` wrapped in retries. **This is the only `bind_tools` call in the whole system** (the test `test_only_executor_has_tools` checks it).
  - **`__call__(state)`** (line 52): the tool loop.
    1. It starts the messages from `specialist_messages(...)`.
    2. Up to `max_tool_rounds` times, it calls the model:
       - if the reply has no tool calls, the reply text becomes the report and the loop stops;
       - otherwise it executes each tool call through `_dispatch` and appends a `ToolMessage` with the result, so the model sees it next round.
    3. If the loop runs out, the report is `ROUND_LIMIT_NOTE`.
    4. It returns one executor turn (`_compose_turn`) and the new `runs`.

    The internal messages (tool calls and so on) are **not** stored in the graph state. Only the final turn is.
  - **`_dispatch(call, turn)`** (line 71): routes a tool call by name to `_write` or `_run`. Unknown tools, missing arguments (`KeyError`) and bad filenames (`ValueError`) become `"ERROR: ..."` strings that the model sees and can fix, instead of crashing the graph.
  - **`_write(filename, content, turn)`** (line 83): calls `workspace.write_file` and records the filename if it succeeded.
  - **`_run(filename, turn)`** (line 89):
    1. It enforces `max_runs_per_turn`.
    2. It calls `workspace.run_file` with `run_timeout_s`.
    3. It builds the text shown to the model, with stdout and stderr clipped by `_clip`.
    4. It stores it in `turn.outputs`.
    5. It appends a `RunRecord` with the **full**, unclipped stdout and stderr and the parsed RMSE values.
    6. It returns the shown text as the tool result.
  - **`_compose_turn(report, turn)`** (line 114): builds the executor's dialogue turn as the model's report, then `Files written: ...`, then `=== tool output (verbatim) ===` followed by each run's output. This is why the dialogue always contains the real tool output, not just the model's summary.
  - **`_clip(s)`** (line 124): if the text is longer than `max_output_chars`, it keeps the first and last half and puts `[... N characters omitted ...]` in between. The tail is kept, so the final RMSE lines survive.

---

### 4.12 [agentic_ode/agents/\_\_init\_\_.py](../agentic_ode/agents/__init__.py)

Re-exports `Decision`, `Executor`, `Orchestrator`, `TextSpecialist`, `finish_allowed`, `make_general` and `make_scientist`, so other code imports from `agentic_ode.agents`.

---

### 4.13 [agentic_ode/graph.py](../agentic_ode/graph.py)

**Purpose:** wires the nodes into a LangGraph `StateGraph`.

- **`route(state)`** (line 14): the function LangGraph calls after the orchestrator node to choose the next node. It returns `state["route"]`, which the orchestrator just set.
- **`build_graph(llm, workdir, settings=None)`** (line 18):
  1. It creates `StateGraph(DialogueState)`.
  2. It adds four nodes: `Orchestrator`, `make_scientist`, `make_general`, and `Executor(llm, settings, workdir)`. The node names are the `Role` values.
  3. It sets the entry point to the orchestrator, so the user message goes only to it.
  4. It adds **conditional edges** from the orchestrator through `route`, with the mapping `{"scientist", "general", "executor", "__end__" → END}`.
  5. It adds **plain edges** from each specialist back to the orchestrator.
  6. It returns `g.compile()`, a runnable graph with `.invoke()` and `.stream()`.

  The model is passed in from outside (dependency injection), which is how the tests plug in `FakeLLM`.

---

### 4.14 [agentic_ode/runner.py](../agentic_ode/runner.py)

**Purpose:** run one task end to end. This is the programmatic API; the CLI is a thin layer on top of it.

- **`RECURSION_LIMIT = 60`**: LangGraph's safety limit on steps. 8 decisions × 2 nodes + 1 = 17 steps, so 60 leaves plenty of room.
- **`class RunOutcome`** (line 20): what `run_task` returns, `run_dir` and the final `state`.
- **`new_run_dir(runs_dir)`** (line 25): creates `runs/YYYYMMDD-HHMMSS/`. The executor writes its scripts here, and the transcript is saved here.
- **`run_task(task_file, settings, on_turn=None, llm=None)`** (line 31):
  1. It creates the model, unless `llm` is given (the tests pass `FakeLLM`).
  2. It creates the run folder and builds the graph.
  3. It prepares metadata (model, task file name, run folder, start time).
  4. It streams the graph with `stream_mode="values"`, so each step yields the **full** state. For each step it:
     - calls `on_turn` for every turn not seen yet (the CLI prints them);
     - saves the transcript, so a crash mid-run still leaves a transcript on disk.
  5. At the end it saves the transcript once more with a `finished` time, and returns a `RunOutcome`.
- **`_now()`** (line 62): the current time as an ISO string.
- **`_display_path(p)`** (line 66): the path relative to the project root if possible (so transcripts don't contain `/home/...`), otherwise absolute.

---

### 4.15 [agentic_ode/transcript.py](../agentic_ode/transcript.py)

**Purpose:** the transcript files that make a run inspectable and reproducible.

- **`save_transcript(run_dir, state, meta)`** (line 12) writes:
  - `transcript.json`: the metadata, `stop_reason`, `decisions`, the full `dialogue`, and all `runs` with full stdout. **This file is the single source of truth for the report.**
  - `transcript.md`: the same dialogue, readable, one section per turn.
- **`load_transcript(run_dir)`** (line 24): reads `transcript.json` back.
- **`_markdown(name, data, meta)`** (line 28): builds the Markdown text. Each turn's content sits in a four-backtick fence, so scripts with triple backticks inside don't break it.

---

### 4.16 [agentic_ode/cli.py](../agentic_ode/cli.py), [\_\_main\_\_.py](../agentic_ode/__main__.py), [main.py](../main.py)

**Purpose:** the command line.

- **`DEFAULT_TASK`**: `task5.md` in the project root.
- **`_print_turn(turn)`** (line 17): prints `[role] text` (the format the spec asks for).
- **`cmd_run(args)`** (line 21): `Settings.from_env()` → `run_task(...)` → prints the stop reason, the decision count and the transcript path. It turns a `ConfigurationError` into exit code 2. The import of `runner` happens inside the function, so `agentic-ode --help` stays fast.
- **`cmd_report(args)`** (line 34): calls `build_report(run_dir, out)` and prints the PDF path.
- **`build_parser()`** (line 41) defines:
  - `agentic-ode [-v|-vv] run [TASK]`;
  - `agentic-ode report RUN_DIR [--out DIR]`.
- **`main(argv=None)`** (line 57): parses the arguments and configures logging:
  - default: warnings only;
  - `-v`: the orchestrator's decisions and script exit codes;
  - `-vv`: also every tool call.

  HTTP libraries stay quiet unless you use `-vv`. It returns the exit code.
- **`__main__.py`**: enables `python -m agentic_ode ...`.
- **`main.py`**: `python main.py [TASK]` = `agentic-ode run [TASK]`. It is kept because the original spec names `main.py`.
- The `agentic-ode` command itself is created by `[project.scripts]` in `pyproject.toml`.

---

### 4.17 [agentic_ode/reporting/diagram.py](../agentic_ode/reporting/diagram.py)

**Purpose:** a picture of the graph for the PDF.

- **`class _NoLLM`** (line 24): a dummy model with `bind_tools` and `with_structured_output`, just enough to compile the graph. It is never called.
- **`graph_png(workdir)`** (line 34): compiles the real graph, reads its nodes and edges via `get_graph()`, and draws them with matplotlib at fixed positions. Conditional edges are dashed and blue; return edges are solid, grey and curved. It returns `(png_bytes, mermaid_text)`. The Mermaid text comes from LangGraph's own `draw_mermaid()`. Because the drawing reads the compiled graph's edges, a new edge you add shows up automatically. **A new node needs a position in `_POS`.**

---

### 4.18 [agentic_ode/reporting/builder.py](../agentic_ode/reporting/builder.py)

**Purpose:** `report/solution.pdf` from `transcript.json`. **No number is typed by hand.** Everything comes from the transcript or the script files in the run folder.

- **`CHROMES`**: the browser binaries tried for HTML→PDF.
- **`class ReportError`** (line 25): raised when no Chrome is found.
- **`repo_url()`** (line 29): reads `git remote get-url origin` and turns `git@github.com:user/repo.git` into `https://github.com/user/repo`. It returns `<URL>` if there is no remote.
- **`_clip(s, n=220)`** (line 39): collapses whitespace and shortens text for table cells.
- **`describe(i, turn, runs)`** (line 44): the "what it did" and "orchestrator's reason" cells of the turn table, derived deterministically:
  - user → a fixed sentence;
  - orchestrator → "Gives the turn to X: instruction", with the reason in the second cell;
  - executor → "Wrote <files>. Ran <file>: exit code N, printed `<RMSE line>`" from the run records;
  - other roles → the first ~220 characters of the text.
- **`build_report(run_dir, out_dir)`** (line 67): loads the transcript, renders the HTML, and writes `solution.html` and `graph.mmd`, then the PDF. It returns the PDF path.
- **`render_html(run_dir, data)`** (line 78): builds the page.
  - **Final run:** the last run with exit code 0 and a test RMSE. If there is none, it uses the last run and says "The run did not reach an RMSE".
  - **Plan:** the last scientist turn *before* that run.
  - **Sections:**
    1. the graph picture;
    2. the turn table;
    3. the RMSE lines copied verbatim with `rmse_lines`, a note on the 0.06 V baseline from Task 5, and a table of all runs;
    4. the scientist's plan as raw text;
    5. the final script's source, plus `test_plot.png` if the final script saved one.

  It fills `_TEMPLATE` with `str.format`, which is why literal braces in the CSS are doubled (`{{ }}`).
- **`_print_pdf(html_file, pdf)`** (line 142): runs `chrome --headless --print-to-pdf`.
- **`_TEMPLATE`**: the HTML and CSS of the report.

---

### 4.19 [agentic_ode/\_\_init\_\_.py](../agentic_ode/__init__.py)

The public API: `Settings`, `build_graph`, `run_task`, `RunOutcome` and `__version__`. Code that uses the package should import from here.

---

## 5. Tests ([tests/](../tests/))

All tests are **offline**: no API key and no network. Run them with `.venv/bin/pytest -q`.

### [conftest.py](../tests/conftest.py)
- **`GOOD_SCRIPT`**: a tiny script that prints both RMSE lines.
- **`class FakeLLM`**: a scripted stand-in for a chat model.
  - `__init__(decisions, executor_script)` (line 20): the list of orchestrator decisions to return (the last one repeats), and the script the executor will write.
  - `with_structured_output(...)` (line 28): returns an object whose `invoke` gives the next `Decision` and counts the calls (`decision_calls`).
  - `invoke(...)` (line 40): the scientist and general answer `"plan text"`.
  - `bind_tools(tools)` (line 44): records the tool names (`bound_tools`) and returns an object that plays the executor:
    1. first, a `write_file` call to `fit_NN.py`;
    2. then a `run_file` call;
    3. then a final text "ran it".

    The real tools run, so a real subprocess executes the script.
- **`settings` fixture** (line 68): `Settings` with a 20 s timeout and `runs_dir` inside pytest's temporary folder.
- **`run_graph` fixture** (line 73): builds the graph with the fake model and runs it to the end.

### [test_tools.py](../tests/test_tools.py)
| Test | Checks |
|---|---|
| `test_run_file_timeout` | A `sleep(30)` script with a 1 s limit returns quickly, with `timed_out` and `TIMEOUT` in stderr |
| `test_run_file_returns_exit_code_and_streams` | The exit code and stdout are captured exactly |
| `test_write_file_never_overwrites` | A second write to the same name is refused and the content is unchanged |
| `test_filenames_stay_inside_workdir` | `../evil.py`, `sub/a.py`, `a.sh` and `a py.py` are rejected |
| `test_parse_rmse` | The last value is parsed, free text is ignored, and the line is returned exactly as printed |

### [test_graph.py](../tests/test_graph.py)
| Test | Checks |
|---|---|
| `test_finish_guard_blocks_early_finish` | A model that always says finish is asked 1 + 3 times, and the run stops as "finish refused" |
| `test_finish_guard_allows_after_real_rmse` | executor → finish works after a real exit-0 run |
| `test_guard_rejects_failed_script` | An RMSE printed by a script that then exits with code 1 does not count |
| `test_rmse_typed_by_model_does_not_count` | RMSE text in a turn without a run record does not count |
| `test_decision_limit` | A model that never finishes stops after exactly 8 decisions |
| `test_only_executor_has_tools` | `bind_tools` is called exactly once, with `write_file` and `run_file` |
| `test_executor_turn_contains_verbatim_tool_output` | The executor turn holds the real output and the files written |

### [test_runner_and_report.py](../tests/test_runner_and_report.py)
| Test | Checks |
|---|---|
| `test_run_task_saves_transcript` | `run_task` writes `transcript.json`, `transcript.md` and the script; `on_turn` sees every turn |
| `test_report_copies_rmse_from_committed_transcript` | The report HTML contains the RMSE line from the committed run, character for character |

---

## 6. How to make common changes

**Change routing behavior** → edit `prompts.orchestrator()`. Keep the routing rules there, not in code, so the routing stays model-driven.

**Change what counts as "done"** → edit `finish_allowed()` in `agents/orchestrator.py`. For example, add `and run["test_rmse"] < 0.06` to require beating the zero baseline. Then update `prompts.FINISH_NOT_ALLOWED` so the model knows why it was refused, and add a test in `test_graph.py`.

**Change a limit** (decisions, timeout, rounds) → edit the default in `Settings`. To make it configurable from `.env`, add it to `Settings.from_env()`.

**Add a new role** (example: `reviewer`, no tools):
1. Add `REVIEWER = "reviewer"` to `Role` in `state.py`, and add it to `SPECIALISTS`.
2. Add `def reviewer(s)` to `prompts.py`.
3. Add `make_reviewer()` to `specialists.py` and export it in `agents/__init__.py`.
4. Add `"reviewer"` to the `Literal` in `Decision.next`.
5. In `graph.py`, add `g.add_node(Role.REVIEWER.value, make_reviewer(llm, settings))`. Its edges come from the `SPECIALISTS` loop.
6. Describe the role and when to route to it in `prompts.orchestrator()`.
7. Add a position for it in `_POS` in `reporting/diagram.py`.

**Add a tool to the executor:**
1. Add the function to `tools/workspace.py`, plus a wrapper and a name constant in `tools/__init__.py`.
2. Add a branch in `Executor._dispatch`.
3. Describe it in `prompts.executor()`.
4. Update `test_only_executor_has_tools`.

**Give a different role its own tools** → follow the `Executor` pattern: call `bind_tools` in its `__init__` and run a loop in `__call__`. Then update `test_only_executor_has_tools`, which currently asserts that only the executor has tools.

**Use another task** → `agentic-ode run my_task.md`. If the task isn't the Silverbox fit, also change:
- `prompts.environment()` (data, libraries);
- the RMSE contract in `tools/metrics.py` and `finish_allowed`;
- the Task-5-specific text in `reporting/builder.py`.

**Use another model** → set `MODEL` in `.env`. The model must support tool calling (the executor) and function-calling structured output (the orchestrator).

**Shorten what models read** (long runs) → change `render_dialogue()` in `state.py`, for example to show old executor outputs as one-line summaries. Every agent reads the dialogue through it.

---

## 7. Known weaknesses (seen in the committed run)

1. **The executor can waste its rounds.** In turn 6 of the committed run it used all 10 tool rounds without running a script. It doesn't know which files already exist, so name collisions cost rounds, and its internal tool calls aren't visible in the transcript. Fix: put the list of existing files into its message in `Executor.__call__`, and add a short log of its tool calls to `_compose_turn`.
2. **Finishing on any RMSE.** The orchestrator finished as soon as one RMSE existed, even though 0.061 V is no better than predicting zero. Fix: a quality condition in `finish_allowed`, or a stronger rule in the orchestrator prompt.
3. **The context grows.** Every model call receives the whole dialogue, including long outputs. Fix: summarization in `render_dialogue`.
4. **Model quality.** The committed run used a free model; a stronger model is the biggest single improvement. The free tier also limits you to 50 requests a day, and a run uses about 30.
