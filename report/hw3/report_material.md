# HW3 — Hardware in the loop: material for the report

- **Code:** branch `hw3-evcam` at https://github.com/HenrikGharagyozyan/agentic-ode-system/tree/hw3-evcam
- **Demo video:** `report/hw3/demo_uc2_real.mp4` (3 min 11 s, run `20261015-170415`).
- **Model in all agent runs:** `nvidia/nemotron-3-super-120b-a12b:free` (OpenRouter).
- **Tests:** 213, all passing (177 of them for HW3; see the end).

---

## (a) Instrument, interface and connector

**Instrument**
- IDS **UE-39B0XCP-E-GL** event camera, with the Sony **IMX636** event sensor.
- 1280 × 720 pixels, USB3, serial 4110039066, camera firmware 3.10.255.
- **No lens:** the camera measures the total light change over time, not an image.

**Interface**
- **Prophesee OpenEB 5.2.0**, through its Python HAL bindings **`metavision_hal`**, with the **IDS uEye EVS HAL plugin 5.0.0**. The data format is EVT3.
- HAL interfaces used: `DeviceDiscovery`, `I_LL_Biases`, `I_DigitalEventMask`, `I_ROI`, `I_EventsStream`, `I_EventsStreamDecoder`, the CD event decoder, the four on-chip filter modules, `I_HW_Identification`, `I_Geometry`, `I_Monitoring`.

**The connector: `agentic_ode/hardware/evcam/connector.py`, class `EventCamera`**
- **Methods:** `open`, `close`, `info`, `get_biases`, `bias_ranges`, `set_bias`, `mask_pixels`, `clear_mask`, `set_roi`, `clear_roi`, `record`, `reapply`, plus `with EventCamera() as cam:`.
- **Commands go in:**
  - biases are set and read back
  - the 64-slot digital pixel mask
  - one ROI window
- **Measurements come back:** `record(seconds)` returns events `(x, y, p, t)`, with t in µs from 0.

**Design**
- **`open()`:**
  - Turns **off all on-chip filters** (anti-flicker, event rate controller, trail filter, digital crop), so the measurements are raw.
  - Adds **one** CD-event callback for the whole session.
  - Runs the **settle**.
- **Settle (8 s):**
  - The first stream after opening has a start-up burst (dark: peak 1.7 Mev/s, still 0.06 Mev/s after 20 s).
  - `stop()` + `start()` ends it at once (probe 07).
  - So `open()` streams 8 s **without decoding** (cheap, so Python keeps up), stops, and then runs a 0.5 s decoded **check stream** that logs the clean noise level (about 0.0009 Mev/s without the 2 defect pixels).
- **HAL loop in `record()`:**
  - `stream.start()` → `wait_next_buffer()` → `get_latest_raw_data()` → `decoder.decode()` → `stream.stop()`.
  - It stops by **camera time** (the decoder timestamp), not wall time.
  - The 0.5 s warm-up is measured from the first buffer after `start()`.
  - **Why not `EventsIterator`:** it added a new decoder callback for every recording and never removed it, so recording N had every event N times (probes 04/05).
- **Early stop:** at **30 M events** `record()` stops. It reports `last_record = {recorded_s, wall_s, stopped_early}`, and then **restarts the stream cleanly**: it reads and drops raw data for 0.5 s. Without this, an old buffer could become the start of the next recording, which then kept nothing (see UC2, run `161317`).
- **Limits (`limits.py`, checked before any hardware call):**

  | What | Limit |
  |---|---|
  | bias names | 6 |
  | bias values | our limits −100…150, **and** the camera's recommended range |
  | recording length | 0.05–5 s |
  | events per recording | 30 M |
  | mask slots | 64 |
  | ROI | 1 window, inside 1280 × 720 |

- **Findings from the probes:**
  - **Two defect pixels**, (926, 151) and (1053, 704), fire about **4.47 M events/s each**, all OFF, 4–5 per µs, always in lockstep. They give over 99.99 % of the events in the dark.
  - The **ROI window includes its end** row and column (`Window(0, 0, 640, 360)` gave 641 × 361 pixels), so `set_roi` sends w−1, h−1.
  - **Settings reset** when the device is reopened, so `state` + `reapply()` exist.
  - **Python decodes more slowly** than the camera at high rates. One example: 1.59 s of camera time took about 22 s.
  - **Known limitation, seen once:** in one recording, the events of both defect pixels came out under the address (926, 151). The cause was not found; probe 10 ruled out stale mask slots and mask coupling. Safety net: `record` warns about any pixel above 20 kHz.

**Without hardware:** `FakeEventCamera` (`fake.py`) has exactly the same public methods (a test checks this). It has 3 scenes: dark, pwm (240 Hz) and steady.

**Figures**
- [`fig/hot_pixel_map.png`](fig/hot_pixel_map.png)
- [`fig/noise_vs_bias_diff_off.png`](fig/noise_vs_bias_diff_off.png). Note that its 2026-10-14 curve was measured **before** the 8 s settle existed, so it includes the start-up burst. It shows the trend only. The UC1 point (1378 events/s at bias 0) is the clean value.

---

## (b) The MCP server: `agentic_ode/mcp_servers/evcam_server.py`

**Basics**
- FastMCP 4.1, **stdio** transport.
- **One** camera object: opened at the first tool call that needs it, kept open, and closed when the server stops.
- `EVCAM_FAKE=1` chooses the fake camera.
- A lock lets only one tool use the camera at a time, because FastMCP runs each tool in a worker thread.
- **Every tool** returns `{"ok": true, ...}` or `{"ok": false, "error": "..."}`, so the server never crashes on a bad call.
- **Every call** is logged to `runs/evcam/tool_log.jsonl`: time, tool, arguments, result, duration.

**The 10 tools**

| Tool | Arguments (defaults) | What it does |
|---|---|---|
| `camera_info` | — | serial, sensor, size, temperature, current biases, and whether it is the fake camera |
| `get_settings` | — | current biases, masked pixels, ROI, and the allowed range of each bias |
| `set_bias` | `name`, `value` | sets one of the 6 biases, checked against our limits and the camera's recommended range; returns the value read back |
| `find_hot_pixels` | `seconds=0.5`, `top=64`, `min_rate_hz=20000` | a short recording (not saved); lists the pixels faster than `min_rate_hz`, busiest first; **does not** mask them |
| `mask_pixels` | `pixels` = list of `[x, y]` | silences the pixels (at most 64); returns the newly masked pixels and the slots used |
| `clear_mask` | — | unmasks all pixels |
| `set_roi` | `x`, `y`, `width`, `height` | only pixels inside this one window send events |
| `clear_roi` | — | the whole sensor sends events again |
| `record` | `seconds=1.0`, `note=""` | records 0.05–5 s and saves the events to `runs/evcam/<id>.npz`, together with the settings (masked pixels, biases, ROI). Returns **only** `recording_id` + a summary (rate, ON fraction, active pixels, busiest pixels), **never raw events**. It adds a `warning` if it stopped early (the summary then uses the real, shorter time), if pixels above 20 kHz are still active, or if it returned 0 events much faster than real time. |
| `spectrum` | `recording_id`, `bin_us=100`, `fmin_hz=20`, `fmax_hz=2000`, `min_snr=8` | the flicker spectrum of a saved recording (it never touches the camera); see below |

**How `spectrum` works**
- **Signal:** ON − OFF events per 100 µs bin. Its FFT shows the frequencies of 20–2000 Hz.
- **snr** = amplitude ÷ the median amplitude in the band. A peak counts only if snr ≥ 8. Pure noise stays below about 4, while a real flicker reached 842.
- **`fundamental_hz`** is the lowest strong peak that explains the others as harmonics; the docstring says "report fundamental_hz as the flicker frequency". **`dominant_hz`** is the strongest peak, which can be a harmonic.
- It also returns the top peaks and the harmonics of the fundamental (each with snr), and **`recorded_with`** (the settings of that recording).
- It adds a `warning` for recordings under 1 s.

**Checked without the LLM**
- `scripts/mcp_smoke.py` on the **real** camera through MCP: 6/6 calls ok, 240 Hz, snr 114.
- `--stdio`: one session keeps the mask, while a new session per call loses it.

---

## (c) Why only the executor holds the camera tools

- **The task asks for it:** (c) says "give the MCP server to one agent in the system". So the **existing executor** holds the tools; we did not add a new role.
- **The client** is `agentic_ode/mcp_client.py`, our own small one:
  - **one persistent stdio session per run**, in a background thread with its own event loop (our graph is synchronous)
  - every MCP tool becomes a LangChain `StructuredTool`, with the server's name, description and JSON schema
  - `langchain-mcp-adapters` was **not** used, because no release works with `mcp` 2.x, which FastMCP 4 needs
- **One session per run:** `runner.py` opens the session when `--camera` is given, for the whole run (human pauses included), and closes it at the end. Settings such as the mask therefore stay between hardware turns.
- **A separate binding:**
  - `Executor.__init__` binds the coding/bench tools for its normal tool loop (`self.llm`), **and** the 10 camera tools in a **second, separate binding** of the same model (`self.camera_llm`).
  - Only the hardware turn uses `camera_llm`. The normal executor loop never sees a camera tool, so **no camera call can bypass the approval step**.
  - The orchestrator, scientist and general agent never call `bind_tools`. The scientist may only **plan** camera measurements in text.
- **Tests** (`tests/test_camera_tools.py`):

  | Test | Checks |
  |---|---|
  | `test_only_the_executor_holds_the_camera_tools` | exactly 2 `bind_tools` calls in the whole graph, both by the executor |
  | `test_camera_tools_are_a_separate_binding` | the normal loop's tools contain no camera tool |
  | `test_without_camera_nothing_changes` | HW1/HW2 runs are unchanged |
  | `test_normal_executor_turn_never_calls_a_camera_tool` | — |
  | `test_real_session_tools_go_to_the_executor_only` | with the real MCP server and the fake camera |
  | `test_run_task_with_camera_opens_and_closes_one_session` | — |
  | `test_run_task_without_camera_starts_no_server` | — |

---

## (d) The hardware turn

**Graph:** `orchestrator → hw_decide → (hw_approve) → hw_call → orchestrator`.

**The orchestrator's `Decision`**
- `Decision.next` has a new value, **`hardware`**.
- The camera paragraph of its prompt says:
  - choose `hardware` when the next step needs a camera action or measurement
  - write **one** action in `instruction`
  - read every hardware result and warning before deciding again
  - record again after a setting change
  - report `fundamental_hz`

**The three nodes**
- **`hw_decide`**: one model call with `executor.camera_llm`.
  - It is forced to make **exactly one** tool call with `tool_choice="any"`.
  - It **stores** the call in the state and never runs it.
- **`hw_approve`** (only for **setting changes**: `set_bias`, `mask_pixels`, `clear_mask`, `set_roi`, `clear_roi`):
  - `interrupt({"question": "Approve this camera call?", "kind": "hard"})`, which is the HW2 human stop, so `--auto-approve` works.
  - It has no model call and no camera call, because the node runs again when the graph resumes.
  - A refusal adds a note "did NOT run", and that **exact** call (name + arguments) is never asked again.
- **`hw_call`** has no model call either. It runs the call through the MCP session and writes `camera call: name(args)` + `result: <JSON>` **verbatim** into the dialogue, as a turn shown as "executor (hardware)".
- **Measurements** (`record`, `spectrum`, `find_hot_pixels`) and the read-only tools run without a question.

**`CameraRecord`** (kept outside the text, like HW1's `RunRecord`): `turn_index`, `tool`, `args`, `ok`, `result_line` (the exact JSON text in the turn).

**Finish guard:** finish is allowed after **any** successful measurement, that is a `record`, `spectrum` or `find_hot_pixels` with `ok: true` whose `result_line` really is in its hardware turn. A spectrum without a peak counts too ("no flicker" is a valid answer). A number the model only types into the dialogue never counts.

**Guards against bad model calls** (each one came from a real run):

| Problem | Guard |
|---|---|
| 0 tool calls | a note to the orchestrator |
| more than 1 tool call | the first runs; the others are named in a note |
| the provider rejects `tool_choice` | one retry without it, with no useless retries before it (the first forced try has no retry wrapper) |
| unknown arguments, e.g. `camera_info({"serial": ...})` | dropped before the call and named in the turn; the schemas also say `additionalProperties: false` |
| arguments that do not match the tool's schema, e.g. `mask_pixels` with `{"x", "y"}` objects (UC1) | checked with `jsonschema` in `hw_decide` **before** the approval question; a bad call becomes a note, the person is never asked, and nothing reaches the camera. A little stricter than the server: a number sent as a string (`"20"`) is refused. Added in commit `d65a58e`, **after** all runs in this report, so those runs show the old behaviour. |
| the same call failed before | it is not run again; a note gives the earlier error |
| a call the human refused before | it is not asked again |
| an unknown tool name, or a server/transport error | a note, or `ok: false`; the run never crashes |
| `record` stopped early, or defect pixels are still active, or 0 events in far less than real time | a `warning` in the result |
| `spectrum` of a recording made before a setting change | `recorded_with` shows the settings it was made with |

**Limits:** camera runs get 20 decisions and a recursion limit of 150.

---

## (e) Diagrams and figures

| File | What it shows |
|---|---|
| `architecture.png` (`.mmd`) | the whole system: HW1 roles, HW2 human stop + simulated bench, HW3 MCP client/server, connector, camera, hardware turn |
| `sequence_uc2_real.png` (`.mmd`) | UC2, run `20261015-161317`: 8 decisions, the two approvals, and where the two bugs happened |
| `sequence_uc1_real.png` (`.mmd`) | UC1, run `20261015-171638`: calibration with the camera covered |
| `sequence_uc4_real.png` (`.mmd`) | UC4, run `20261015-172112`: flickering or steady |
| `fig/spectrum_uc2.png` | the spectrum of the UC2 demo recording `20261015-170606-5aa66c`, with the fundamental and harmonics marked |
| `fig/hot_pixel_map.png` | the active pixels without a mask; the 2 defect pixels give 99.9998 % of the events |
| `fig/noise_vs_bias_diff_off.png` | dark noise vs `bias_diff_off`; read the note on the old curve |
| `fig/uc2_rate_before_after_mask.png` | the UC2 demo run: event rate before and after masking, and the 240 Hz flicker in time |

The figures are made by `scripts/make_hw3_figures.py`, and the PNG diagrams with `mmdc` (mermaid-cli 11.17.0).

---

## (f) Use cases

All runs use the real camera, unless they are marked "fake". "(approved)" means `hw_approve` asked the person, and the reply was yes.

### UC1 — calibration (camera covered), run `20261015-171638`, 5 decisions

- **Request:** "The event camera is covered, so no light reaches the sensor. Find the defect pixels and mask them, then measure the noise level of the camera and report it in events per second."
- **Tool calls:**
  1. `find_hot_pixels({"seconds": 0.5, "top": 64, "min_rate_hz": 20000})`
  2. `mask_pixels({"pixels": [{"x": 926, "y": 151}, {"x": 1053, "y": 704}]})` (approved)
  3. `mask_pixels({"pixels": [[926, 151], [1053, 704]]})` (approved)
  4. `record({"seconds": 2})`
- **Camera results:**
  1. (926, 151) and (1053, 704), 4.47 M events/s each (this call includes the open + settle, about 13 s)
  2. refused by the schema (pixels must be `[x, y]` pairs)
  3. 2 slots used
  4. 2,755 events in 2 s
- **Final answer:** **1,378 events per second** (0.0014 Mev/s), approved at the human stop. ✓
- **Problems:**
  - The approval question came **before** the arguments were checked. The person approved a call that then failed validation, and had to approve again.
    - **Fixed after the runs:** `hw_decide` now checks the arguments against the tool's JSON schema **before** the approval question. A bad call becomes a note to the orchestrator; the person is not asked, and nothing is sent to the camera.
    - **All runs in this report were made before this fix (commit `d65a58e`)**, so they still show the old behaviour (here: decision 2).
  - The noise rises slowly after a restart (probe 07), so the value depends a little on when the recording starts.

### UC2 — "how fast does it flicker?" (phone screen at minimum brightness)

**Run `20261015-160356`, 3–5 cm, 3 decisions → wrong answer**
- **Tool calls:** `record({"seconds": 2})` → `spectrum({"recording_id": "20261015-160438-7ac4b4"})`.
- **Camera results:** 30 M events, stopped early after **0.17 s**, but the summary said "2.0 s, 15 Mev/s". The spectrum had a resolution of 5.8 Hz: 480.3 Hz (snr 86.8) was just above 237.3 Hz (snr 83.4).
- **Final answer:** **480.3 Hz** ✗ (the 2nd harmonic).
- **Fixes:**
  - `record` uses the real duration and warns "stopped after X s … set an ROI, reduce the light or raise the biases" (`38b49ea`)
  - `spectrum` adds `fundamental_hz` and a warning for recordings under 1 s (`2aadb36`)
  - prompt: act on every warning, report the fundamental (`aa99d9c`)

**Run `20261015-161317`, 3–5 cm, 8 decisions → right answer, two bugs found**
- **Tool calls:**
  1. `record({"seconds": 2})`
  2. `mask_pixels({"pixels": [[926, 151], [1053, 704]]})` (approved)
  3. `record({"seconds": 2})`
  4. `get_settings({})`
  5. `set_bias({"name": "bias_diff", "value": 5})` (approved; the model asked for 3 calls, so the first ran and the rest were noted)
  6. `record({"seconds": 2})`
  7. `spectrum({"recording_id": "20261015-161656-8d1493"})`
- **Camera results:**
  1. 30 M events in 1.59 s, with the early-stop warning ✓
  2. ok
  3. **0 events in 0.02 s**
  4. ok
  5. read back 5
  6. 29.95 M events in 0.27 s, summary "2.0 s", **no** warning
  7. fundamental **240.24 Hz**, snr 273.6, warning "only 0.27 s long"
- **Final answer:** **240.24 Hz** ✓
- **Problems + fixes:**
  - Bug 1: after an early stop, an old buffer started the next recording, and nothing was kept. **Fix:** a clean stream restart after an early stop, plus a warning for 0 events in far less than real time (`9906f85`, `38b49ea`).
  - Bug 2: the early stop was detected by the event count, and the warm-up filter left fewer than 30 M. **Fix:** an early stop now means "recorded time < asked time", reported by the connector (`9906f85`, `3153707`, `38b49ea`).
  - Tests with a scripted HAL stream reproduce both bugs (`cf63f19`).

**Run `20261015-170415`, the demo video, 5 decisions → clean**
- **Tool calls:**
  1. `record({"seconds": 2})`
  2. `mask_pixels({"pixels": [[926, 151], [1053, 704]]})` (approved)
  3. `record({"seconds": 2})`
  4. `spectrum({"recording_id": "20261015-170606-5aa66c"})`
- **Camera results:**
  1. 23.1 M events (11.55 Mev/s), warning: 2 pixels above 20 kHz
  2. 2 slots used
  3. 11.5 M events (5.77 Mev/s), no warning
  4. **fundamental 240.0 Hz, snr 842.6**; resolution 0.5 Hz; harmonics 2f 480 Hz (snr 282.1), 4f 960 Hz (50.8), 5f 1200 Hz (20.5); 3f 720.5 Hz is below the threshold (snr 4.1)
- **Final answer:** **240.0 Hz** ✓
- **Note / limitation:**
  - A weak extra peak at 1002.5 Hz (snr 10.6) is **not** a harmonic of 240 Hz. Because of it, `fundamental()` found no peak that explains all the others, and fell back to the strongest peak. The answer is the same here, but one unrelated weak peak switches off the harmonic logic. **Open:** choose the fundamental that explains the **most** strong peaks.
  - In this recording, the "other pixels" rate also changed between the two recordings (10.28 → 5.77 Mev/s). Masking 2 pixels does not cause that. See `fig/uc2_rate_before_after_mask.png`.

**Other UC2 runs** (see `runs_summary.md`):
- `154851`, fake: the `camera_info({"serial": ...})` loop → fixes `b9d844d`, `f28f50e`, `8962e3d`.
- `155602`, fake: 240 Hz, but it analysed the recording made before masking → `recorded_with` + "record again after a setting change" (`38b49ea`, `aa99d9c`).
- `163237`, real, 10–15 cm: no flicker reached the sensor (strongest values snr 3.2–3.7, the noise level). It was stopped by the OpenRouter daily limit (429).

### UC4 — "flickering or steady?", run `20261015-172112`, 5 decisions

- **Request:** "Is the light in front of the event camera flickering or steady? Measure it and answer, with the measurement that shows it."
- **Light source:** phone flashlight, steady, 3–5 cm above the sensor.
- **Tool calls:**
  1. `record({"seconds": 2})`
  2. `mask_pixels({"pixels": [[926, 151], [1053, 704]]})` (approved)
  3. `record({"seconds": 2})`
  4. `spectrum({"recording_id": "20261015-172255-a1ffa4"})`
- **Camera results:**
  1. 17.87 M events, only 17 active pixels, warning: 2 pixels above 20 kHz
  2. 2 slots used
  3. **64 events** in 2 s (37 active pixels), with a normal wall time of 2.6 s, so this is not the zero-event bug
  4. no peak above the noise (`fundamental_hz: null`), `recorded_with`: 2 masked pixels
- **Final answer:** **"the light is steady"** (no flicker peak) ✓
- **The light reaches the sensor:** with the defect pixels masked, the noise is about **40 times lower** than in the dark.

  | | Light | Events per second | OFF events |
  |---|---|---|---|
  | UC1 | covered | 1,378 | 77 % |
  | UC4 | lit | 32 | 81 % |

  In the dark, the noise is mostly OFF events (77 % in UC1; the start-up burst after `open()` was also almost only OFF events). With the light on, this low-light noise is suppressed. Our only data for that are these two runs. A lens-less event camera reports only **changes**, so a steady light adds almost no events of its own.
- **Limitation:** 64 events are thin evidence for "no flicker". A longer recording would make it stronger.

### UC3 — lower the noise with the biases

The task file `tasks/uc3_bias.md` exists, but **UC3 was not run** (the OpenRouter daily limit).

---

## Tests and commits

**Tests: 213, all passing**

| Area | Tests |
|---|---|
| HW3 total | 177 |
| `test_evcam_limits` | 33 |
| `test_evcam_record` (scripted HAL) | 5 |
| `test_evcam_analysis` | 37 |
| `test_evcam_store` | 12 |
| `test_evcam_fake` | 23 |
| `test_evcam_server` | 28 |
| `test_mcp_client` | 8 |
| `test_camera_tools` | 7 |
| `test_hardware_turn` | 24 |
| HW1/HW2 (unchanged, still passing) | 36 |

**Key commits on `hw3-evcam`** (`git log main..hw3-evcam` for all of them):

| Part | Commits |
|---|---|
| (a) connector | `e0501d1` HAL record, `52fca47` 8 s settle, `9906f85` last_record + clean restart |
| analysis / fake / store | `2e270ba`, `2aadb36` (fundamental), `4dfdba5`, `3a62153` |
| (b) MCP server | `666ad46`, `e89aa88` (hot-pixel warning), `38b49ea` (real duration, warnings, recorded_with) |
| (c) client + executor | `b6eeac3` (MCP client), `f204fe7` (separate binding), `ffb0620` (one session per run), `6baadd2` (tests) |
| (d) hardware turn | `742d86e`, `6e863de`, `fd1d636`, `269c014` (finish guard), `0cfa404` / `f28f50e` (`tool_choice`), `b9d844d` (arguments, repeat guard) |
| (e)/(f) diagrams, runs, video | `78418d5`, `fcd7b24`, run transcripts `41a485d` … `f31f42b`, video `208ea52` |
