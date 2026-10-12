"""The focusing search on the simulated bench: flip macropixels, keep a flip only if the enhancement grows.

Every call saves PNGs and a JSON record into the run directory and appends one row to experiments.csv.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from agentic_ode.tools.bench import MACROPIXEL_SIZES, Bench, Regions, enhancement, make_regions

MAX_ITERATIONS = 20000
CSV_NAME = "experiments.csv"
CSV_FIELDS = [
    "experiment", "macropixel", "n_macropixels", "focus_size", "n_background_regions", "iterations",
    "flips_per_step", "seed", "initial_enhancement", "final_enhancement", "best_measured_enhancement",
    "accepted_flips", "runtime_s",
]
RESULT_KEY = "FOCUS_ENHANCEMENT"


def run_focus_search(macropixel: int = 8, focus_size: int = 2, n_background_regions: int = 8,
                     iterations: int = 1000, flips_per_step: int = 1, seed: int = 0,
                     workdir: Path | None = None) -> dict:
    """Random start pattern; each iteration flips `flips_per_step` random macropixels, takes one noisy shot,
    and keeps the flip only if the measured enhancement grows. The final enhancement is a fresh shot."""
    _check(macropixel, focus_size, n_background_regions, iterations, flips_per_step)
    t0 = time.perf_counter()
    bench = Bench(macropixel, seed)
    if flips_per_step > bench.n:
        raise ValueError(f"flips_per_step must be at most {bench.n} for macropixel {macropixel}")
    rng = np.random.default_rng(seed + 1)
    regions = make_regions(focus_size, n_background_regions)

    bench.set_pattern(rng.random(bench.n) < 0.5)
    initial_image = bench.shot()
    current = initial = enhancement(initial_image, regions)
    history = [round(current, 4)]
    accepted = 0
    for _ in range(iterations):
        idx = rng.choice(bench.n, size=flips_per_step, replace=False)
        bench.flip(idx)
        measured = enhancement(bench.shot(), regions)
        if measured > current:
            current = measured
            accepted += 1
        else:
            bench.flip(idx)  # flip back: the old pattern is kept
        history.append(round(current, 4))
    final_image = bench.shot()  # a new shot of the kept pattern, not the lucky one that was accepted
    final = enhancement(final_image, regions)

    result = {
        "settings": {"macropixel": macropixel, "focus_size": focus_size,
                     "n_background_regions": n_background_regions, "iterations": iterations,
                     "flips_per_step": flips_per_step, "seed": seed},
        "n_macropixels": bench.n,
        "initial_enhancement": round(initial, 3),
        "final_enhancement": round(final, 3),
        "best_measured_enhancement": round(current, 3),
        "accepted_flips": accepted,
        "history": history,
        "runtime_s": round(time.perf_counter() - t0, 2),
        "images": {},
    }
    if workdir is not None:
        _save(Path(workdir), result, bench, regions, initial_image, final_image)
    return result


def format_focus(result: dict) -> str:
    """The tool output as text: this exact text goes to the model and into the executor turn."""
    s = result["settings"]
    args = ", ".join(f"{k}={v}" for k, v in s.items())
    hist = result["history"]
    step = max(1, (len(hist) - 1) // 10)
    sampled = [hist[i] for i in range(0, len(hist), step)]
    return (
        f"run_focus_search({args})\n"
        f"experiment: {result.get('experiment', '-')}, controlled macropixels: {result['n_macropixels']}, "
        f"accepted flips: {result['accepted_flips']}, runtime: {result['runtime_s']} s\n"
        f"initial enhancement: {result['initial_enhancement']}\n"
        f"best measured enhancement during the search: {result['best_measured_enhancement']}\n"
        f"enhancement history (every {step} iterations): {sampled}\n"
        f"images: {', '.join(result['images'].values()) or '(not saved)'}\n"
        f"{RESULT_KEY}: {result['final_enhancement']}"
    )


def _check(macropixel, focus_size, n_background_regions, iterations, flips_per_step) -> None:
    if macropixel not in MACROPIXEL_SIZES:
        raise ValueError(f"macropixel must be one of {MACROPIXEL_SIZES}")
    if not 1 <= focus_size <= 8:
        raise ValueError("focus_size must be between 1 and 8 camera pixels")
    if not 2 <= n_background_regions <= 16:
        raise ValueError("n_background_regions must be between 2 and 16")
    if not 1 <= iterations <= MAX_ITERATIONS:
        raise ValueError(f"iterations must be between 1 and {MAX_ITERATIONS}")
    if flips_per_step < 1:
        raise ValueError("flips_per_step must be at least 1")


def _save(workdir: Path, result: dict, bench: Bench, regions: Regions,
          initial_image: np.ndarray, final_image: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(list(workdir.glob("focus_*.json"))) + 1
    name = f"focus_{n:02d}"
    result["experiment"] = name
    s = result["settings"]
    label = f"m={s['macropixel']}, r={s['focus_size']}, K={s['n_background_regions']}, it={s['iterations']}"

    def camera(img: np.ndarray, title: str, path: str) -> None:
        fig, ax = plt.subplots(figsize=(4.4, 4))
        im = ax.imshow(img, cmap="inferno")
        fig.colorbar(im, ax=ax, label="counts")
        ax.contour(regions.focus, levels=[0.5], colors="cyan", linewidths=1)
        ring = np.any(regions.sectors, axis=0)
        ax.contour(ring, levels=[0.5], colors="white", linewidths=0.6)
        cy, cx = regions.centre
        for k in range(len(regions.sectors)):
            a = -np.pi + 2 * np.pi * k / len(regions.sectors)
            ax.plot([cx + regions.inner * np.cos(a), cx + regions.outer * np.cos(a)],
                    [cy + regions.inner * np.sin(a), cy + regions.outer * np.sin(a)], color="white", lw=0.5)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(workdir / path, dpi=110, bbox_inches="tight")
        plt.close(fig)

    images = {
        "camera_initial": f"{name}_camera_initial.png",
        "camera_final": f"{name}_camera_final.png",
        "dmd_final": f"{name}_dmd.png",
        "enhancement_curve": f"{name}_enhancement.png",
    }
    camera(initial_image, f"{name} initial ({label})\nenhancement {result['initial_enhancement']}",
           images["camera_initial"])
    camera(final_image, f"{name} final ({label})\nenhancement {result['final_enhancement']}",
           images["camera_final"])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(bench.dmd_image(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{name} final DMD pattern ({label})", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(workdir / images["dmd_final"], dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(result["history"], lw=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("enhancement (kept)")
    ax.set_title(f"{name} ({label})", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(workdir / images["enhancement_curve"], dpi=110, bbox_inches="tight")
    plt.close(fig)

    result["images"] = images
    (workdir / f"{name}.json").write_text(json.dumps(result, indent=1))
    csv_path = workdir / CSV_NAME
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"experiment": name, **s, "n_macropixels": result["n_macropixels"],
                    **{k: result[k] for k in ("initial_enhancement", "final_enhancement",
                                              "best_measured_enhancement", "accepted_flips", "runtime_s")}})
