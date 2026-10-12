# Homework 2, Task 2: focusing through a simulated bench

Run: `runs/20261012-223544` (copied here: `transcript.md`, `transcript.json`, `experiments.csv`, all PNGs).
Command: `python main.py task2.md --auto-approve`. The flag `--auto-approve` answered **"yes"** to the two
hard-decision questions (the Task 1 human stop is still active; the questions are printed in the transcript).
Model: `nvidia/nemotron-3-super-120b-a12b:free` (OpenRouter). Result: `stop reason: finish`, 3 orchestrator decisions.

Task (`task2.md`), typed once; no settings were typed during the run:

> Focus the beam: form a bright spot in the centre of the camera of the simulated bench, as strong as you can,
> and report the enhancement.

## The simulated bench and the tool

- **Bench** (`agentic_ode/tools/bench.py`): DMD 64×64 mirrors, binary amplitude (0/1), grouped into square
  macropixels of side m ∈ {2, 4, 8, 16} → N = (64/m)² = 1024, 256, 64, 16 controlled macropixels.
  Scattering medium: transmission matrix T (4096 camera pixels × 4096 DMD pixels), complex circular Gaussian,
  fixed seed (2026), so every experiment sees the same medium. Camera 64×64 records E_out = |T E_in|²,
  scaled to ≈200 counts per pixel for a random half-on pattern, with Poisson shot noise and Gaussian read noise
  (σ = 2 counts): two shots of one pattern differ.
- **Regions**: focus = square of side r at the camera centre; proximal background = a ring around it
  (2 px gap, 6 px wide) split into K angular sectors. **Enhancement = mean(focus) / mean of the K sector means.**
- **Tool** `run_focus_search(macropixel, focus_size, n_background_regions, iterations, flips_per_step, seed)`
  (`agentic_ode/tools/focus.py`): random start pattern; every iteration flips `flips_per_step` random
  macropixels, takes one noisy shot, keeps the flip only if the measured enhancement grows, otherwise flips back.
  The reported `FOCUS_ENHANCEMENT` is a **fresh shot** of the kept pattern (not the lucky shot that was accepted,
  which is listed as "best measured"). Each call saves the initial and final camera image (focus and sectors
  marked), the final DMD pattern, the enhancement curve, a JSON record, and a row of `experiments.csv`.
  Largest setting (m = 2, 20 000 iterations) runs in ≈13 s.
- Only the **executor** is bound to the tool (with `write_file`/`run_file`); its turn contains every call's
  settings and `FOCUS_ENHANCEMENT` line verbatim, and the finish guard accepts a finish only after a real tool call.

## Experiments the agent ran (all chosen by the agent)

| exp | m | N | focus r | K | iterations | flips/step | seed | initial | **final (fresh shot)** | best measured | ≈ N/(2π r²) | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| focus_01 | 4 | 256 | 2 | 4 | 500 | 1 | 0 | 1.058 | **13.129** | 13.349 | 10.2 | 1.14 |
| focus_02 | 2 | 1024 | 2 | 4 | 2000 | 2 | 1 | 0.713 | **24.488** | 25.349 | 40.7 | 2.97 |
| focus_03 | 4 | 256 | 4 | 8 | 1000 | 1 | 2 | 0.777 | **4.814** | 4.999 | 2.5 | 0.69 |
| focus_04 | 8 | 64 | 2 | 2 | 500 | 1 | 3 | 0.813 | **8.667** | 8.815 | 2.5 | 0.42 |
| focus_05 | 4 | 256 | 2 | 4 | 2000 | 5 | 4 | 1.045 | **15.183** | 15.761 | 10.2 | 2.25 |

**Best: focus_02** (m = 2, N = 1024, r = 2, K = 4, 2000 iterations, 2 flips/step): enhancement **24.488**,
from ≈0.7 at the random start. Images: `focus_02_camera_initial.png`, `focus_02_camera_final.png`,
`focus_02_dmd.png`, `focus_02_enhancement.png`.

## Theory (approximate)

For binary amplitude modulation with N independently controlled segments, the expected enhancement of one
speckle grain is roughly **η ≈ N/(2π)** (large N, ideal optimisation, no noise). Here one camera pixel is about one
speckle grain, so a focus of r×r pixels (M = r² independent grains) gives roughly **η ≈ N/(2π M)**.
Treat it as an order-of-magnitude estimate.

- focus_02 reached 24.5 against ≈41: its enhancement curve is **still rising** at 2000 iterations
  (≈2 iterations per macropixel), so the search was stopped before it converged. More iterations
  (up to 20 000 are allowed and take ≈13 s) would raise it.
- The small-N runs (focus_01, focus_04, focus_05) are **above** the estimate. For small N one medium realisation
  fluctuates strongly, and N/(2π) is an asymptotic large-N result.
- Larger focus (focus_03, r = 4, M = 16) gives a much smaller enhancement, as expected from the 1/M factor.
- Developer check, not part of the agent run: m = 2, r = 1, 20 000 iterations gave 188.7 against N/(2π) ≈ 163.

Comparison with the lecture result: *(to be filled in by the student from the lecture slides).*

## Which agent did what (turns of the transcript)

| turn | role | what it did |
|---|---|---|
| 0 | user | the task text above |
| 1 | orchestrator | `next: scientist`: plan the focusing experiments |
| 2 | scientist | 5 experiments with settings and a one-line reason each; decision rule: highest fresh-shot `FOCUS_ENHANCEMENT` wins |
| 3 | orchestrator | `next: human`, kind **hard**, proposed_next executor: *"Shall I run the focusing experiments now?"* |
| 4 | human | `yes` (answered by `--auto-approve`) |
| 5 | executor | 5 calls of `run_focus_search` in one turn; table of settings + `FOCUS_ENHANCEMENT` lines; best = focus_02 |
| 6 | orchestrator | `next: human`, kind **hard**, proposed_next finish: *"Shall I report the enhancement and finish the task?"* |
| 7 | human | `yes` (answered by `--auto-approve`), finish guard passed → END, `stop reason: finish` |
