# agentic-ode-system

A four-role LangGraph system (orchestrator, scientist, general agent, executor) that is given Task 5, the neural-ODE fit of the Silverbox oscillator, and tries to solve it. The system spec is in [task6.md](task6.md), and the problem the system receives is in [task5.md](task5.md).

## Graph

```mermaid
graph TD
    START([user message]) --> orchestrator
    orchestrator -.->|next = scientist| scientist
    orchestrator -.->|next = general| general
    orchestrator -.->|next = executor| executor
    orchestrator -.->|finish, guard passed / 8 decisions| END([END])
    scientist --> orchestrator
    general --> orchestrator
    executor --> orchestrator
```

- **State** ([src/state.py](src/state.py)): the dialogue is a list of turns `{"role", "content"}`, appended with `Annotated[list, operator.add]`. The state also holds a counter of orchestrator decisions, plus the records of real script runs (exit code, stdout, the parsed RMSE) that the finish guard checks.
- **Orchestrator**: one model call per turn, with structured output `Decision(next, instruction, reason)`. The decision is appended to the dialogue as its turn. It has no tools and follows no keyword table or fixed order. The routing prompt sends plan and ODE questions to the scientist, tracebacks to the executor, and conceptual side questions to the general agent.
- **Finish guard**: `finish` is accepted only when an executor turn contains a `FREE_RUN_TEST_RMSE:` line from the stdout of a run that exited with code 0. Otherwise the model is asked again, with a note that finish is not yet allowed (up to 3 times). The run stops on finish, or after 8 decisions.
- **Scientist** and **general agent**: no tools. The scientist writes the ODE, the network, the state equation, the training, the free-run test and numbered steps. The general agent answers side questions only.
- **Executor** ([src/tools.py](src/tools.py)): the only node bound to tools. `write_file` writes a new file into `runs/<timestamp>/` and never overwrites one. `run_file` runs the file with a 300 s timeout. The executor runs an internal tool loop (at most 2 script runs per turn), then appends one turn with the real tool output verbatim.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp ../silverbox-pinn/data/SNLS80mV.mat data/   # or download SNLS80mV.mat from nonlinearbenchmark.org
```

`.env` (not committed):

```
OPEN_ROUTER_API_KEY=...        # or OPENROUTER_API_KEY; without it, init_chat_model uses MODEL's own provider
MODEL=nvidia/nemotron-3-super-120b-a12b:free
```

```bash
.venv/bin/python main.py                 # sends task5.md to the orchestrator, prints "[role] text" per turn
.venv/bin/python -m pytest -q            # offline tests: timeout, finish guard, 8-decision limit, tool ownership
.venv/bin/python report/build_report.py runs/<timestamp>   # report/solution.pdf from transcript.json (needs Chrome)
```

Each run writes `runs/<timestamp>/` with the executor's scripts, `transcript.json` (the dialogue plus run records) and `transcript.md`. Only the run used in the report is committed.

## Layout

```
src/state.py     dialogue state, reducer, decision counter
src/tools.py     write_file / run_file, RMSE parsing
src/agents.py    model setup, prompts, the four nodes, finish guard
src/graph.py     StateGraph wiring
main.py          entry point, streaming, transcript files
tests/           offline tests with a scripted fake model
report/          build_report.py, solution.pdf
```
