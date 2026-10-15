# HW3 camera runs on 2026-10-15

All agent runs used the same request, `tasks/uc2_flicker.md`:
"Something in front of the event camera flickers. Find out how fast it flickers and report the frequency in Hz."

- **Model:** `nvidia/nemotron-3-super-120b-a12b:free` (OpenRouter).
- **Fake camera:** `EVCAM_FAKE=1`, pwm scene, 240 Hz.
- **Real camera:** IDS UE-39B0XCP-E-GL, with a phone screen at minimum brightness in front of it.
- Every run folder is committed under `runs/`.

## Agent runs (LLM + MCP + camera)

| Run | Camera | Tool calls in order | Camera results | Final answer | What went wrong | What we fixed (commits) |
|---|---|---|---|---|---|---|
| `20261015-154851` | fake | `camera_info({"serial": "None"})` × 6 | all 6 failed: "Unexpected keyword argument serial" | none: stopped (Ctrl+C) after 7 decisions | The model added an argument that `camera_info` does not have, and repeated the same failing call. | Unknown arguments are dropped and named in the turn. The schemas say `additionalProperties: false`. The **same** failed call is not run again. Prompt: "a tool without parameters is called with {}". (`b9d844d`, `f28f50e`, `8962e3d`) |
| `20261015-155602` | fake | record(2 s) → mask_pixels (approved) → spectrum | record: warning, 2 hot pixels; mask: ok; spectrum: 240.0 Hz, snr 53.7 | **240 Hz** ✓ (4 decisions) | The spectrum was of the recording made **before** masking, but the orchestrator called it "the masked recording". | `record` saves its settings; `spectrum` returns `recorded_with`. Prompt: "after a setting change, record again before analysing". (`38b49ea`, `aa99d9c`) |
| `20261015-160356` | **real**, 3–5 cm | record(2 s) → spectrum | record: 30 M events, stopped early after 0.17 s, but the summary said "2.0 s, 15 Mev/s"; spectrum: 480.3 Hz (snr 86.8) just above 237.3 Hz (snr 83.4), resolution 5.8 Hz | **480.3 Hz** ✗ (3 decisions) | Too much light (173 Mev/s). The short recording made the 2nd harmonic the strongest peak. No warning told the agent. | `record` uses the real duration and warns "stopped after X s …". `spectrum` adds `fundamental_hz` (here 237.3 Hz) and warns for recordings under 1 s. Prompt: "act on every warning; report the fundamental". (`38b49ea`, `2aadb36`, `aa99d9c`) |
| `20261015-161317` | **real**, 3–5 cm | record(2 s) → mask_pixels (approved) → record(2 s) → get_settings → set_bias(bias_diff = 5) (approved) → record(2 s) → spectrum | record 1: 30 M events in 1.59 s, warning ✓; record 2: **0 events** in 0.02 s; record 3: 29.95 M events in 0.27 s, summary "2.0 s", **no** warning; spectrum: fundamental **240.24 Hz**, snr 273.6, warning "only 0.27 s" | **240.24 Hz** ✓ (8 decisions) | Bug 1: after an early stop, an old buffer was still waiting in the driver; the next record started from it and kept nothing. Bug 2: the early stop was checked by the event count, and the warm-up filter left fewer than 30 M events. | The connector reports `last_record` (recorded time, wall time, stopped early) and restarts the stream cleanly after an early stop. An early stop is now "recorded time < asked time". New warning for 0 events in far less than real time. (`9906f85`, `3153707`, `38b49ea`) |
| `20261015-163237` | **real**, 10–15 cm | record(2 s) → mask_pixels (approved) → record(2 s) → spectrum → record(10 s) → record(5 s) → spectrum → (clear_roi chosen) | record 1: warning, 2 hot pixels; record 2: 3.9 Mev/s; spectrum: no peak; record(10 s): refused (limit 5 s); record(5 s): 2.4 Mev/s; spectrum: no peak | none: stopped by the OpenRouter daily limit (429) after 8 decisions | At 10–15 cm no flicker reached the sensor: the strongest values had snr 3.2–3.7, the level of pure noise. Not a code bug. | Nothing yet. Next try: the phone close again, with an ROI on it, to stay under 30 M events. |

## Later runs (after all fixes)

| Run | Task | Camera | Tool calls in order | Camera results | Final answer | Notes |
|---|---|---|---|---|---|---|
| `20261015-170415` | UC2 | **real**, phone | record(2 s) → mask_pixels (approved) → record(2 s) → spectrum | 11.55 Mev/s + hot-pixel warning; 2 slots; 5.77 Mev/s; fundamental **240.0 Hz**, snr 842.6 | **240.0 Hz** ✓ (5 decisions) | Demo video run (`report/hw3/demo_uc2_real.mp4`). |
| `20261015-171638` | UC1 | **real**, covered | find_hot_pixels → mask_pixels as `{x, y}` objects (approved, refused by the schema) → mask_pixels as `[x, y]` (approved) → record(2 s) | 2 defect pixels at 4.47 M ev/s each; 2 slots; 2,755 events in 2 s | **1,378 events/s** ✓ (5 decisions) | The approval was asked before the arguments were checked. Fixed after the runs: `hw_decide` now checks the arguments against the schema first. |
| `20261015-172112` | UC4 | **real** | record(2 s) → mask_pixels (approved) → record(2 s) → spectrum | 17.87 M events (17 active pixels) + hot-pixel warning; 2 slots; 64 events in 2 s; no peak | **steady** ✓ (5 decisions) | Plausible (steady light gives almost no events), but thin evidence. |

## Scripted runs (no LLM)

| Time | Camera | What ran | Result | What went wrong | What we fixed |
|---|---|---|---|---|---|
| 14:47 | fake | `scripts/mcp_smoke.py` (in-memory MCP) | 6/6 calls ok; 240 Hz, snr 175 | — | — |
| 14:50 | **real**, phone | `scripts/mcp_smoke.py` | 6/6 calls ok; **240 Hz, snr 114** | `find_hot_pixels` returned only (926, 151), with the events of **both** hot pixels (8.67 MHz), so (1053, 704) stayed unmasked. | Probe 10 ruled out old mask slots and mask coupling, and our analysis code was proven correct. The cause was not found. Safety net: `record` warns about pixels above 20 kHz. Documented as a known limitation in `docs/hw3-hardware.md`. (`e89aa88`, `21f7638`) |
| 15:19 | fake | `scripts/mcp_smoke.py --stdio` | one session: the mask is kept; a new session per call: the mask is lost | — | (shows why the agent uses one session per run) |
