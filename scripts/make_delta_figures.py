"""Delta reconstruction figures of the 3 real recordings of run 20261016-142636 (offline; the camera is not opened).

Reads runs/evcam/<id>.npz (git-ignored, so this runs on this machine) and writes report/lumen/:
  <id>_brightness.png   brightness change of the whole sensor (each ON = +Δ, each OFF = −Δ, leaky integral,
                        tau 0.2 s): the whole recording and the first 100 ms
  <id>_delta.png        Σ polarity per 4 x 4 px bin: blue = darker, white = 0, red = brighter (linear, symmetric)
  <id>_time.png         mean event time per 4 x 4 px bin: blue = start, red = end, black = no events

The same functions and settings as the EventLab preview (agentic_ode/web/recordings.py), so the PNGs match the UI.

Run:  ~/mv-venv/bin/python scripts/make_delta_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from agentic_ode.core.config import PROJECT_ROOT  # noqa: E402
from agentic_ode.hardware.devices.evcam import analysis, limits, store  # noqa: E402
from agentic_ode.web import recordings  # noqa: E402

OUT = PROJECT_ROOT / "report" / "lumen"
RECORDINGS = {"20261016-142758-e07515": 50, "20261016-143041-19997d": 50, "20261016-143449-ed6b74": 10}  # id: Hz
BLUE, WHITE, RED = "#2166ac", "#ffffff", "#b2182b"  # the same colours as the UI tokens --dl-* / --time-*
EXTENT = (0, limits.SENSOR_W, limits.SENSOR_H, 0)  # sensor coordinates, y = 0 at the top


def brightness(rid: str, hz: int, events: np.ndarray, seconds: float) -> None:
    t_s, level = analysis.brightness_curve(events, recordings.CURVE_BIN_US, recordings.LEAK_TAU_S, seconds=seconds)
    zoom = recordings.ZOOM_US // recordings.CURVE_BIN_US
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))  # room for the caption line at the bottom
    for ax, t, v, title in ((axes[0], t_s, level, "whole recording"),
                            (axes[1], t_s[:zoom], level[:zoom], "first 100 ms (100 µs bins)")):
        ax.plot(t, v, color="#1f77b4", lw=1)
        ax.axhline(0, color="0.6", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("time in the recording (s)")
        ax.set_ylabel("ΔL (contrast steps Δ)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Brightness change (integrated events) — {rid}, LED at {hz} Hz", fontsize=11)
    fig.text(0.01, 0.005, f"Reconstructed from events: each ON = +Δ, each OFF = −Δ; leaky integration "
             f"τ = {recordings.LEAK_TAU_S} s; not an absolute brightness. Saved recording, not live.", fontsize=8)
    fig.savefig(OUT / f"{rid}_brightness.png", dpi=130)
    plt.close(fig)


def delta(rid: str, hz: int, events: np.ndarray, seconds: float) -> None:
    img = analysis.delta_image(events, limits.SENSOR_W, limits.SENSOR_H, bin_px=recordings.BIN_PX, seconds=seconds)
    nonzero = np.abs(img[img != 0])
    top = max(float(np.percentile(nonzero, recordings.DELTA_PERCENTILE)) if nonzero.size else 1.0, 1.0)
    cmap = LinearSegmentedColormap.from_list("delta", [BLUE, WHITE, RED])
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    im = ax.imshow(img, cmap=cmap, vmin=-top, vmax=top, extent=EXTENT, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Σ polarity (contrast steps Δ): blue darker, red brighter")
    ax.set_xlabel("x (sensor pixel)")
    ax.set_ylabel("y (sensor pixel)")
    ax.set_title(f"Delta image — {rid}, LED at {hz} Hz (saved recording, not live)\n"
                 f"each ON = +Δ, each OFF = −Δ per {recordings.BIN_PX}×{recordings.BIN_PX} px bin, "
                 f"{seconds:.3f} s; range {img.min()} … +{img.max()} Δ; not an absolute brightness", fontsize=9)
    fig.savefig(OUT / f"{rid}_delta.png", dpi=130)
    plt.close(fig)


def time(rid: str, hz: int, events: np.ndarray, seconds: float) -> None:
    img = analysis.time_image(events, limits.SENSOR_W, limits.SENSOR_H, bin_px=recordings.BIN_PX, seconds=seconds)
    cmap = LinearSegmentedColormap.from_list("time", [BLUE, RED])
    cmap.set_bad("black")  # NaN = no events
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    im = ax.imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=0, vmax=seconds, extent=EXTENT,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean event time (s): blue start, red end; black = no events")
    ax.set_xlabel("x (sensor pixel)")
    ax.set_ylabel("y (sensor pixel)")
    ax.set_title(f"Time-coloured event image — {rid}, LED at {hz} Hz (saved recording, not live)\n"
                 f"mean time of the events per {recordings.BIN_PX}×{recordings.BIN_PX} px bin; hot pixels left out",
                 fontsize=9)
    fig.savefig(OUT / f"{rid}_time.png", dpi=130)
    plt.close(fig)


def main() -> None:
    for rid, hz in RECORDINGS.items():
        events, meta = store.load(rid)
        seconds = float(meta["recorded_s"])
        for make in (brightness, delta, time):
            make(rid, hz, events, seconds)
        print(f"{rid}: {OUT.relative_to(PROJECT_ROOT)}/{rid}_{{brightness,delta,time}}.png")


if __name__ == "__main__":
    main()
