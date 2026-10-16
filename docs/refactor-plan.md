# Refactor plan: target layout and the steps to get there

Input: `docs/architecture-current.md` (the map of the code at `7b16615`). Spot names like A1 or C3 point to its
section 3. This file is a plan only. No code is moved by it.

## 0. Rules for the whole refactor

1. **Same behaviour.** These must stay identical:
   - the CLI commands (`agentic-ode run [TASK] [--auto-approve] [--camera]`, `agentic-ode report RUN_DIR [--out]`);
   - the MCP tool names, descriptions, input schemas and output schemas;
   - the JSON that every tool returns, including the key order, because it is copied word for word into the
     dialogue that the model reads;
   - the prompt texts and the graph nodes and edges.
2. **The same tests pass.** All 213 tests that exist today stay and pass. The refactor only adds tests
   (snapshots, unit tests for the new standard). It never deletes a test or changes what a test checks. The
   few test lines that must change (only import lines and monkeypatch targets) are listed in the step that
   changes them, and you decide on them (decision D2).
3. **Small steps.** Each step below is one commit and ends with `~/mv-venv/bin/python -m pytest -q` all green.
4. **Moves first, new code later.** Phase 1 only moves code, so a failing test there means a wrong import,
   never a logic change.

## 1. Target layout

The layout follows your proposal, with five changes the real code suggests:

- **`human.py` goes to `graph/nodes/` (not `agents/`).** The human node has no model and no prompt. Like
  `hw_approve`, it is a pause in the graph (`interrupt()`). The two share `_note()`/`_back()` (C1) and the yes/no
  rule.
- **New `agents/guards.py`.** It holds `finish_allowed()` and `accepted()`, so nodes no longer import other
  nodes (A3).
- **New `core/dialogue.py`.** It holds the text helpers that the agents, the graph nodes and the report all use
  (`render_dialogue`, `role_label`, `decision_field`, `last_orchestrator_turn`). The report then no longer
  imports `agents/` (A5), and C2 can be fixed in one place.
- **`tools/` stays, but smaller.** It keeps the executor's in-process tools (`workspace.py`, `focus.py`). The
  simulator `bench.py` becomes a device (`hardware/devices/optical_bench/simulator.py`), and the RMSE contract
  `metrics.py` goes to `core/` (A6, B5).
- **`reporting/` stays where it is.** It is its own job (a PDF from a finished run), and it is used only by the
  `report` command.

```
agentic_ode/
├── __init__.py                 same public names: Settings, build_graph, run_task, RunOutcome, __version__
├── __main__.py                 python -m agentic_ode → runtime.cli.main
├── core/                       shared basics; imports nothing else from the project except other core modules
│   ├── config.py               Settings, PROJECT_ROOT
│   ├── state.py                DialogueState, Turn, Role, SPECIALISTS, END_ROUTE, initial_state
│   ├── records.py              RunRecord, ExperimentRecord, CameraRecord
│   ├── dialogue.py             render_dialogue, role_label, decision_field, last_orchestrator_turn
│   ├── metrics.py              the FREE_RUN_*_RMSE output contract
│   └── llm.py                  make_llm, with_retry, text_of, ConfigurationError
├── agents/                     the roles that call a model, and their prompts
│   ├── prompts.py
│   ├── context.py              specialist_messages, latest_instruction
│   ├── guards.py               finish_allowed, accepted (the yes rule)
│   ├── orchestrator.py         Decision, Orchestrator
│   ├── specialists.py          scientist, general
│   └── executor.py             Executor (coding + focus tool loop; still holds the camera binding, see §6)
├── tools/                      the executor's in-process tools
│   ├── __init__.py             make_executor_tools, tool names
│   ├── workspace.py            write_file, run_file
│   └── focus.py                run_focus_search
├── graph/
│   ├── __init__.py             build_graph (so `from agentic_ode.graph import build_graph` still works)
│   ├── builder.py              build_graph: nodes and edges
│   ├── routing.py              route(), node names (HW_DECIDE, ...), _note(), _back()
│   └── nodes/
│       ├── human_turn.py       the human node
│       └── hardware_turn.py    hw_decide, hw_approve, hw_call
├── hardware/
│   ├── standard/               "Model Hardware Standard" (§2): knows no device and no MCP server
│   │   ├── instrument.py       Instrument (abstract class), ActionSpec
│   │   ├── result.py           Result: the envelope {ok, result, error, warnings}
│   │   └── registry.py         instrument name → factory
│   └── devices/
│       ├── __init__.py         the list of known instruments (the only place a new device is named)
│       ├── evcam/
│       │   ├── events.py       EVENT_DTYPE (the event format, no SDK)
│       │   ├── driver.py       EventCamera (real camera, Metavision SDK)
│       │   ├── fake.py         FakeEventCamera
│       │   ├── limits.py
│       │   ├── analysis.py
│       │   ├── store.py
│       │   └── instrument.py   EvcamInstrument: the 10 actions, on top of driver.py or fake.py
│       └── optical_bench/
│           └── simulator.py    Bench, transmission matrix, enhancement (from tools/bench.py)
├── mcp/                        everything that speaks MCP
│   ├── client.py               McpSession, evcam_session
│   ├── server_factory.py       build_server(instrument) → FastMCP
│   ├── serve.py                python -m agentic_ode.mcp.serve NAME  (any registered instrument)
│   └── evcam_server.py         the evcam server object `mcp` (tests and scripts use it)
├── runtime/                    one run, from the command line to the saved files
│   ├── cli.py
│   ├── runner.py
│   └── run_store.py            transcript.json / transcript.md
└── reporting/                  unchanged place
    ├── builder.py
    └── diagram.py
```

The dependency direction is one way only:

```
runtime  →  graph  →  agents  →  tools / core
runtime  →  mcp/client                       (the agent side never imports hardware/)
mcp      →  hardware/standard, hardware/devices
hardware/devices  →  hardware/standard  →  (nothing from the project)
reporting →  core, runtime/run_store, graph
```

## 2. The "Model Hardware Standard" (`hardware/standard/`)

Goal: one small contract that every instrument follows, so that one generic MCP server can serve any of them.

### 2.1 `ActionSpec`: one thing the instrument can do (one MCP tool)

```python
@dataclass(frozen=True)
class ActionSpec:
    name: str                    # MCP tool name, e.g. "set_bias"
    description: str             # MCP tool description: the model reads this
    kind: Literal["read", "setting", "measurement"]
    #   read:        changes nothing (camera_info, get_settings)
    #   setting:     changes the device; the agent asks the human first (set_bias, mask_pixels, ...)
    #   measurement: makes a result; a successful one can finish a task (record, spectrum, find_hot_pixels)
    handler: Callable[..., Result]   # a normal Python function with type hints, e.g.
                                     #   def set_bias(self, name: str, value: int) -> Result

    @property
    def input_schema(self) -> dict:
        """The JSON schema of the arguments, made from the handler's type hints by fastmcp's public
        Tool.from_function(handler).parameters, so it is exactly the schema the MCP server shows."""
```

Why the schema comes from type hints and is not written by hand: today fastmcp makes each schema from the tool
function's signature. I checked this in a probe (with fastmcp 4.1.0):
- a wrapper made with `functools.wraps` around a typed method gives the **same** schema;
- it also gives the **same** error for a wrong type ("Input should be a valid integer"), which
  `test_wrong_argument_type_is_refused_by_fastmcp` checks.

Hand-written schemas would lose this check and could drift from the code.

### 2.2 `Result`: one envelope for every answer

```python
@dataclass
class Result:
    ok: bool
    result: dict = field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @staticmethod
    def success(result: dict, warnings: list[str] | None = None) -> Result: ...
    @staticmethod
    def failure(error: str) -> Result: ...

    def to_wire(self) -> dict:
        """The JSON the MCP tool returns. It is TODAY's format, unchanged:
        {"ok": true, **result, "warning": "a | b"}   ("warning" only when there are warnings)
        {"ok": false, "error": "..."}"""
```

Inside the code, every instrument uses the envelope. On the wire, `to_wire()` keeps today's flat format, so the
tool results, the hardware turn text and the tests stay the same. The envelope never reaches the model.

### 2.3 `Instrument`: the abstract class every device driver implements

```python
class Instrument(ABC):
    name: str                                   # registry key, e.g. "evcam"

    @abstractmethod
    def describe(self) -> dict:
        """Static facts, no hardware access: name, model, fake or real, MCP server instructions."""

    @abstractmethod
    def actions(self) -> list[ActionSpec]:
        """All actions, in the order the MCP server lists them."""

    def close(self) -> None:
        """Release the hardware. Default: nothing to release."""

    def call(self, action: str, args: dict) -> Result:
        """Run one action. An exception becomes Result.failure("TypeName: message"), so a caller never sees a
        crash (today's _run() rule in evcam_server.py)."""
```

### 2.4 Registry

```python
# hardware/standard/registry.py
def register(name: str, factory: str) -> None   # factory as "module:function", imported only when needed,
def create(name: str) -> Instrument             # so listing instruments never imports an SDK
def names() -> list[str]

# hardware/devices/__init__.py  (the ONLY file that names the devices)
register("evcam", "agentic_ode.hardware.devices.evcam.instrument:from_env")
```

### 2.5 The generic MCP server: `mcp/server_factory.py`

```python
def build_server(instrument: Instrument, log_file: Path) -> FastMCP:
    lock = threading.Lock()                     # one tool at a time, as today
    mcp = FastMCP(instrument.name, instructions=instrument.describe()["instructions"],
                  lifespan=<calls instrument.close() at the end>)
    for spec in instrument.actions():
        mcp.add_tool(Tool.from_function(_tool_function(spec, instrument, lock, log_file),
                                        name=spec.name, description=spec.description,
                                        output_schema={"additionalProperties": True, "type": "object"}))
    return mcp
```

`_tool_function()` wraps `spec.handler` with `functools.wraps`, so fastmcp sees the handler's signature. The
wrapper does the following:
1. binds the arguments with their defaults (today's log also contains defaults, e.g. `"note": ""`);
2. runs `instrument.call()` under the lock;
3. converts the result with `to_wire()`;
4. appends the same JSONL log line as today;
5. returns the dict.

`mcp/serve.py NAME` runs `build_server(registry.create(NAME), ...)` over stdio.

### 2.6 The event camera as one device driver

- `EventCamera` (`driver.py`) and `FakeEventCamera` (`fake.py`) stay the two **backends**. They keep their
  methods and their tests.
- `EvcamInstrument` (`devices/evcam/instrument.py`) is the one **Instrument** on top of them. Its constructor
  takes a backend factory, and `from_env()` picks `FakeEventCamera` when `EVCAM_FAKE=1`, like `_camera()` today.
- The 10 tool bodies of today's `evcam_server.py` (`:96-221`) move into `EvcamInstrument` as methods that
  return `Result`. Their text, defaults and docstrings stay exactly the same. For example, the 3 warnings of
  `record()` become `Result.warnings`, and `to_wire()` joins them with `" | "` as today.
- A small `CameraBackend` Protocol (the methods both classes already have) is written down in
  `devices/evcam/instrument.py`. It makes the "same interface" of C3 visible. The duplicated bodies (C3) are not
  merged in this refactor.

```
          MCP (stdio)                     hardware/standard            hardware/devices/evcam
agent ──────────────► mcp/server_factory ──► Instrument.call() ──► EvcamInstrument ──► EventCamera ──► SDK
                      (one generic server)                         (10 actions)      └► FakeEventCamera
```

### 2.7 Adding a second device: the simulated optical bench (example, not part of this refactor)

Only two places change, and neither is in `mcp/` or `agents/`:

1. A new file `hardware/devices/optical_bench/instrument.py`:

   ```python
   class OpticalBenchInstrument(Instrument):
       name = "optical_bench"

       def __init__(self, macropixel: int = 8, seed: int = 0):
           self._bench = Bench(macropixel, seed)           # simulator.py, unchanged

       def describe(self) -> dict:
           return {"name": "optical_bench", "model": "simulated DMD + scattering medium", "fake": True,
                   "instructions": "Simulated optical bench: set a DMD pattern, take camera shots, ..."}

       def actions(self) -> list[ActionSpec]:
           return [ActionSpec("set_pattern", "Set the DMD macropixel pattern ...", "setting", self.set_pattern),
                   ActionSpec("shot", "One noisy camera image; returns the enhancement ...", "measurement",
                              self.shot)]

       def set_pattern(self, pattern: list[int]) -> Result: ...
       def shot(self, focus_size: int = 2, n_background_regions: int = 8) -> Result: ...
   ```

2. One line in `hardware/devices/__init__.py`:
   `register("optical_bench", "agentic_ode.hardware.devices.optical_bench.instrument:OpticalBenchInstrument")`.

After that, `python -m agentic_ode.mcp.serve optical_bench` serves it, and `McpSession` can open it.

**An honest limit.** Today the agent side is not device-agnostic. `agents/prompts.py` has camera text
(`CAMERA_ORCHESTRATOR`, `CAMERA_SCIENTIST`, `hardware()`), and `core/state.py` names the camera tools in
`CAMERA_SETTING_TOOLS` and `CAMERA_MEASUREMENT_TOOLS` (C6). The tests pin those names. So a second device
**served** needs no change in `mcp/`, but a second device **driven by the agents** needs Phase 3 (§4) first.

## 3. Old path → new path (every file)

"Alias" means the old file stays, with 3 lines that make the old name point to the new module:

```python
import sys
from agentic_ode.core import config as _moved
sys.modules[__name__] = _moved
```

I checked in a probe that the old name and the new name are then **the same module object**. So
`monkeypatch.setattr(old_module, ...)` in the tests still changes what the code uses, and no test needs to change
in Phase 1. The aliases are removed in the last step (decision D3).

| Old path | New path | Step | Note |
|---|---|---|---|
| `agentic_ode/__init__.py` | same | — | same public names |
| `agentic_ode/__main__.py` | same | 11 | imports `runtime.cli` |
| `main.py` | same | 11 | imports `runtime.cli` |
| `agentic_ode/config.py` | `agentic_ode/core/config.py` | 1 | alias |
| `agentic_ode/llm.py` | `agentic_ode/core/llm.py` | 1 | alias |
| `agentic_ode/state.py` | `agentic_ode/core/state.py` + `core/records.py` + `core/dialogue.py` | 2 | alias; `core/state.py` re-exports the moved names until step 19 |
| `agentic_ode/tools/metrics.py` | `agentic_ode/core/metrics.py` | 3 | alias; `tools/__init__.py` keeps re-exporting `last_rmse_line`, `parse_rmse` |
| `agentic_ode/agents/context.py` | same (`decision_field` → `core/dialogue.py`) | 4 | `context.py` re-exports `decision_field` |
| `agentic_ode/agents/orchestrator.py` | same (`finish_allowed` → `agents/guards.py`) | 5 | re-exports `finish_allowed` |
| — | `agentic_ode/agents/guards.py` | 5 | new file: `finish_allowed`, `accepted`, `YES_WORDS` |
| `agentic_ode/agents/__init__.py` | same | 7 | no longer exports `human` (only `graph.py` used it; avoids an import cycle) |
| `agentic_ode/agents/prompts.py` | same | — | text unchanged |
| `agentic_ode/agents/specialists.py` | same | — | |
| `agentic_ode/agents/executor.py` | same | — | |
| `agentic_ode/graph.py` | `agentic_ode/graph/builder.py` + `graph/__init__.py` | 6 | a module and a package cannot share a name, so `graph.py` is deleted; `graph/__init__.py` exports `build_graph` |
| — | `agentic_ode/graph/routing.py` | 6 | `route()`, node names |
| `agentic_ode/agents/human.py` | `agentic_ode/graph/nodes/human_turn.py` | 7 | alias |
| `agentic_ode/agents/hardware.py` | `agentic_ode/graph/nodes/hardware_turn.py` | 7 | alias |
| `agentic_ode/tools/__init__.py` | same | 3, 9 | imports from new places |
| `agentic_ode/tools/workspace.py` | same | — | |
| `agentic_ode/tools/focus.py` | same | 9 | imports the simulator from its new place |
| `agentic_ode/tools/bench.py` | `agentic_ode/hardware/devices/optical_bench/simulator.py` | 9 | alias |
| `agentic_ode/hardware/__init__.py` | same | — | |
| `agentic_ode/hardware/evcam/__init__.py` | same (the old package stays for the aliases) | 8 | removed in step 19 |
| `agentic_ode/hardware/evcam/connector.py` | `agentic_ode/hardware/devices/evcam/driver.py` (+ `events.py` for `EVENT_DTYPE`) | 8 | alias; `driver.py` re-exports `EVENT_DTYPE` |
| `agentic_ode/hardware/evcam/fake.py` | `agentic_ode/hardware/devices/evcam/fake.py` | 8 | alias |
| `agentic_ode/hardware/evcam/limits.py` | `agentic_ode/hardware/devices/evcam/limits.py` | 8 | alias |
| `agentic_ode/hardware/evcam/analysis.py` | `agentic_ode/hardware/devices/evcam/analysis.py` | 8 | alias |
| `agentic_ode/hardware/evcam/store.py` | `agentic_ode/hardware/devices/evcam/store.py` | 8 | alias |
| — | `agentic_ode/hardware/devices/__init__.py`, `devices/evcam/__init__.py`, `devices/optical_bench/__init__.py` | 8, 9, 13 | new |
| — | `agentic_ode/hardware/standard/{__init__,instrument,result,registry}.py` | 12 | new |
| — | `agentic_ode/hardware/devices/evcam/instrument.py` | 13 | new: tool bodies from `evcam_server.py` |
| `agentic_ode/mcp_client.py` | `agentic_ode/mcp/client.py` | 10 | alias |
| `agentic_ode/mcp_servers/__init__.py` | same | — | kept: the documented command uses this path |
| `agentic_ode/mcp_servers/evcam_server.py` | `agentic_ode/mcp/evcam_server.py` | 10 | special alias: also runnable (below) |
| — | `agentic_ode/mcp/server_factory.py`, `agentic_ode/mcp/serve.py` | 14, 15 | new |
| `agentic_ode/cli.py` | `agentic_ode/runtime/cli.py` | 11 | alias (the console script points here, see D3) |
| `agentic_ode/runner.py` | `agentic_ode/runtime/runner.py` | 11 | alias |
| `agentic_ode/transcript.py` | `agentic_ode/runtime/run_store.py` | 11 | alias |
| `agentic_ode/reporting/*` | same | 2, 4 | imports from new places |
| `scripts/*.py` | same | — | still work through the aliases; updated in step 19 |
| `tests/*` | same | — | unchanged in Phase 1; see D2 and step 19 |
| `pyproject.toml` | same | 19 | entry point `agentic_ode.runtime.cli:main` (needs `pip install -e .` once) |

The command `python -m agentic_ode.mcp_servers.evcam_server` is written in `docs/hw3-hardware.md` and the HW3
report, so its alias must also run. A plain alias does not run the new module's `if __name__ == "__main__"`
block, so this one alias is:

```python
if __name__ == "__main__":
    from agentic_ode.mcp.evcam_server import main
    main()
else:
    import sys
    from agentic_ode.mcp import evcam_server as _moved
    sys.modules[__name__] = _moved
```

## 4. Order of the steps

Each step is one commit and ends with the full suite green. "Snapshots" are the step 0 tests.

**Phase 0: safety net**

0. Snapshot tests (the step we discussed before, extended):
   - MCP tools: name, description, inputSchema **and outputSchema**;
   - graph nodes and edges;
   - **all prompt texts**: each role, with the camera on and off;
   - **golden tool results**: one fake-camera session runs `camera_info`, `get_settings`, `find_hot_pixels`,
     `mask_pixels`, `record` and `spectrum`. Only the `recording_id` and `created` values are replaced by
     fixed text; key order is kept.

   This guards everything rule 1 promises.

**Phase 1: pure moves (no logic change, no test change)**

1. `core/config.py`, `core/llm.py`, with aliases.
2. Split `state.py` into `core/state.py`, `core/records.py` and `core/dialogue.py` (`render_dialogue`,
   `role_label`). `core/state.py` re-exports the moved names. Alias at `agentic_ode/state.py`.
3. `tools/metrics.py` → `core/metrics.py`; `orchestrator.py` and `reporting/builder.py` import it from `core`.
4. `decision_field` → `core/dialogue.py`; `reporting/builder.py` imports it from `core` (A5).
5. New `agents/guards.py` with `finish_allowed` (from `orchestrator.py`) and `accepted`/`YES_WORDS` (from
   `human.py`). The old modules re-export them.
6. `graph.py` → `graph/builder.py` + `graph/__init__.py` + `graph/routing.py` (`route`, `HW_*` names).
7. `agents/human.py` → `graph/nodes/human_turn.py`; `agents/hardware.py` → `graph/nodes/hardware_turn.py`;
   `agents/__init__.py` stops exporting `human`.
8. `hardware/evcam/*` → `hardware/devices/evcam/*`. `connector.py` becomes `driver.py`, and `EVENT_DTYPE`
   moves to `events.py` (A4). Aliases go in the old package.
9. `tools/bench.py` → `hardware/devices/optical_bench/simulator.py`.
10. `mcp_client.py` → `mcp/client.py`; `mcp_servers/evcam_server.py` → `mcp/evcam_server.py` (with the special
    alias). `evcam_session()` starts `-m agentic_ode.mcp.evcam_server`.
11. `cli.py`, `runner.py` → `runtime/`; `transcript.py` → `runtime/run_store.py`; `__main__.py` and `main.py`
    import `runtime.cli`.

**Phase 2: the Model Hardware Standard (new code, same behaviour)**

12. `hardware/standard/` (`Instrument`, `ActionSpec`, `Result`, registry), with new unit tests: `to_wire()`
    output, the exception rule in `call()`, `input_schema` equal to fastmcp's, and the registry's lazy import.
    Nothing uses it yet.
13. `EvcamInstrument` in `devices/evcam/instrument.py`: the 10 tool bodies move there and return `Result`.
    `mcp/evcam_server.py` still has its 10 `@mcp.tool` functions, but each one is now one line:
    `return _run(name, args, lambda: instrument.call(...))`. Snapshots and golden results must stay green.
14. `mcp/server_factory.py`: `mcp/evcam_server.py` becomes `mcp = build_server(EvcamInstrument.from_env(), ...)`.
    Today the tests patch `evcam_server.RECORDINGS_DIR` and `LOG_FILE` and read `evcam_server._cam`. These are
    the only test lines that this step must change (decision D2):
    - `tests/test_evcam_server.py:24-28, :266`
    - `tests/test_hardware_turn.py:77-82`
15. `mcp/serve.py` and `hardware/devices/__init__.py` (registry). `evcam_session()` starts
    `-m agentic_ode.mcp.serve evcam`.

**Phase 3: agents that do not know the device (optional, decision D1)**

16. The server factory adds `meta={"mhs": {"kind": spec.kind}}` to each tool. This is the MCP `_meta` field;
    name, description and schemas are unchanged. `mcp/client.py` copies `kind` into the `StructuredTool`
    metadata. `hardware_turn.py` and `finish_allowed()` use `kind` and fall back to today's
    `CAMERA_SETTING_TOOLS`/`CAMERA_MEASUREMENT_TOOLS` for tools without it, because the test stub tools have
    none.
17. The device's prompt notes come from the server as MCP prompts. `prompts.orchestrator(settings, camera,
    notes=None)` uses `notes` when given and `CAMERA_ORCHESTRATOR` otherwise, so the pinned texts stay.

**Phase 4: clean-up (no behaviour change)**

18. Small de-duplications from the map:
    - C1 (`_note`/`_back` into `graph/routing.py`), C2;
    - C4/C5 (one constant for 20 kHz and for 64 slots; tool defaults keep their values, so the schemas do not
      change);
    - D7/D8/D11 (docstrings only);
    - A1: `agentic_ode/__init__.py` loads `graph`/`runner` lazily (PEP 562 `__getattr__`), so the MCP server
      stops loading LangGraph.
19. Test and script imports go to the new paths (import lines and monkeypatch targets only). Remove all aliases
    except `mcp_servers/evcam_server.py`. Set the `pyproject.toml` entry point and run `pip install -e .` once.

## 5. How each step is checked

- `~/mv-venv/bin/python -m pytest -q`: the old 213 tests plus the new ones, all pass.
- The snapshot and golden-result tests from step 0 pass without regenerating the snapshot files. A step may
  never rewrite a snapshot.
- After steps 10, 11 and 15, one manual smoke run:
  - `EVCAM_FAKE=1 agentic-ode run tasks/uc2_flicker.md --camera --auto-approve` must reach the same stop reason;
  - `EVCAM_FAKE=1 python -m agentic_ode.mcp_servers.evcam_server` must still start.
- `ruff check` stays clean (ruff is not installed in `~/mv-venv` yet, so this check needs `pip install ruff` first).

## 6. What this refactor does NOT change (on purpose)

These are spots from the map where a fix would change behaviour or a pinned test. They are kept and can be
planned later:

- D6, F1: the report text and what the report shows. Changing them changes the PDF.
- B2: the camera binding stays in `Executor`. Tests call `Executor(..., camera_tools=...)`,
  `ex.camera_reply()` and `ex.camera_llm`.
- F2: every graph keeps the 3 hardware nodes. The graph snapshot pins them.
- F4: `EVCAM_FAKE` stays an environment variable. The CLI must stay the same.
- D1, D2: the names `hardware`, `END_ROUTE`, `HUMAN` stay. They are in prompts, in transcripts and in the graph.
- The camera prompt texts stay word for word.

## 7. Decisions for you

- **D1. Phase 3 now or later?** (a) Later, in its own plan, after phases 0-2 and 4 *(recommended: it is the only
  phase that changes agent logic, and phases 0-2 already give the standard and the generic server)*. (b) Now,
  as steps 16-17.
- **D2. May step 14 change the 7 listed test lines** (fixture patch targets and the `_cam` check, no change to
  what is checked)? (a) Yes *(recommended: simple, visible code)*. (b) No: keep module-level names in
  `mcp/evcam_server.py` with a PEP 562 `__getattr__` trick. This needs no test edits, but the code is harder to
  read.
- **D3. Remove the aliases at the end (step 19)?** (a) Yes, and point the tests and scripts to the new paths
  *(recommended)*. (b) Keep the aliases for good: no test edits, but 17 extra files.
- **D4. Package name `agentic_ode/mcp/`.** It works: imports are absolute, so it never hides the installed `mcp`
  library. But it has the same name as that library. (a) Keep `mcp/` *(recommended, as you proposed)*.
  (b) `mcp_bridge/`.
- **D5. Step 0.** This is the snapshot step we discussed before, now with outputSchema, prompts and golden
  results. It still needs your A1/A2 and B1/B2 answers.
