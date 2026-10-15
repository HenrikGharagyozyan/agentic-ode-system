"""Figures for the HW3 report, from the saved recordings in runs/evcam/ (git-ignored, so this runs on this machine).

Writes report/hw3/fig/:
  spectrum_uc2.png                  spectrum of the UC2 demo recording, fundamental and harmonics marked
  hot_pixel_map.png                 events per pixel without a mask (UC4, first recording): the 2 defect pixels
  noise_vs_bias_diff_off.png        dark noise vs bias_diff_off (docs/hw3-hardware.md table) + the UC1 measurement
  uc2_rate_before_after_mask.png    UC2 demo run: event rate before / after masking the 2 defect pixels

Run:  python scripts/make_hw3_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from agentic_ode.config import PROJECT_ROOT  # noqa: E402
from agentic_ode.hardware.evcam import analysis, limits, store  # noqa: E402

OUT = PROJECT_ROOT / "report" / "hw3" / "fig"
HOT = [(926, 151), (1053, 704)]
UC2_BEFORE, UC2_AFTER = "20261015-170511-192ba0", "20261015-170606-5aa66c"  # run 20261015-170415 (demo video)
UC4_UNMASKED = "20261015-172153-ace3ac"  # run 20261015-172112, first recording, no mask, steady light
UC1_MASKED = "20261015-171844-8b76e9"  # run 20261015-171638, covered, 2 pixels masked
# docs/hw3-hardware.md, "Noise behaviour", measured 2026-10-14 (hot pixels masked, BEFORE the 8 s settle existed)
BIAS_TABLE = {0: 0.75, 10: 0.56, 20: 0.33, 40: 0.058, 60: 0.003, 80: 0.0005}


def seconds_of(meta: dict) -> float:
    return meta.get("recorded_s", meta["seconds"])


def spectrum_figure() -> Path:
    events, meta = store.load(UC2_AFTER)
    s = analysis.spectrum(events)  # the numbers the agent saw
    # the same signal and FFT as analysis.spectrum(), to draw the whole curve
    bin_us = s["bin_us"]
    n_bins = int(events["t"].max()) // bin_us + 1
    signal = np.bincount(events["t"] // bin_us, weights=np.where(events["p"] == 1, 1.0, -1.0), minlength=n_bins)
    amp = np.abs(np.fft.rfft(signal - signal.mean()))
    freq = np.fft.rfftfreq(n_bins, d=bin_us * 1e-6)
    band = (freq >= s["fmin_hz"]) & (freq <= s["fmax_hz"])
    snr = amp[band] / s["noise_floor"]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.semilogy(freq[band], snr, lw=0.6, color="#2b5c8a")
    ax.axhline(s["min_snr"], color="#888888", ls="--", lw=1)
    ax.text(s["fmax_hz"], s["min_snr"] * 1.15, f"detection threshold snr = {s['min_snr']:g}", ha="right",
            fontsize=8, color="#555555")
    f0 = s["fundamental_hz"]
    top = max(p["snr"] for p in s["peaks"])
    ax.axvline(f0, color="#c0392b", lw=1.2)
    ax.annotate(f"fundamental {f0:g} Hz\nsnr {s['peaks'][0]['snr']:g}", (f0, top), xytext=(f0 + 60, top * 0.9),
                fontsize=9, color="#c0392b", va="top")
    weak = []
    for h in s["harmonics"]:
        if h["snr"] and h["snr"] >= s["min_snr"]:
            ax.axvline(h["hz"], color="#e67e22", lw=0.8, ls=":")
            ax.text(h["hz"], top * 1.6, f"{h['k']}f\n{h['snr']:g}", ha="center", fontsize=8, color="#e67e22")
        else:
            weak.append(f"{h['k']}f ({h['hz']:g} Hz, snr {h['snr']:g})")
    if weak:
        ax.text(0.99, 0.03, "below the threshold: " + ", ".join(weak), transform=ax.transAxes, ha="right",
                fontsize=8, color="#555555")
    ax.set_xlim(0, s["fmax_hz"])
    ax.set_ylim(0.1, top * 4)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("snr = amplitude / noise floor")
    ax.set_title(f"UC2, real camera: spectrum of recording {UC2_AFTER} "
                 f"({events.size:,} events in {s['seconds']:g} s, 2 defect pixels masked)", fontsize=10)
    return save(fig, "spectrum_uc2.png")


def hot_pixel_map() -> Path:
    events, meta = store.load(UC4_UNMASKED)
    sec = seconds_of(meta)
    counts = np.bincount(events["y"].astype(np.int64) * limits.SENSOR_W + events["x"],
                         minlength=limits.SENSOR_W * limits.SENSOR_H)
    active = np.flatnonzero(counts)  # one dot per pixel with events: single pixels are too small for an image
    xs, ys, rates = active % limits.SENSOR_W, active // limits.SENSOR_W, counts[active] / sec
    fig, ax = plt.subplots(figsize=(9, 5.6))
    sc = ax.scatter(xs, ys, c=rates, s=12 + 10 * np.log10(rates), cmap="viridis",
                    norm=matplotlib.colors.LogNorm(), edgecolors="none")
    hot_share = sum(int(counts[y * limits.SENSOR_W + x]) for x, y in HOT) / events.size
    for x, y in HOT:
        ax.add_patch(plt.Circle((x, y), 25, fill=False, color="#c0392b", lw=1.5))
        ax.annotate(f"({x}, {y})\n{counts[y * limits.SENSOR_W + x] / sec / 1e6:.2f} M events/s", (x, y),
                    xytext=(x - 330, y - 40 if y > 400 else y + 70), color="#c0392b", fontsize=9,
                    arrowprops=dict(arrowstyle="-", color="#c0392b"))
    ax.set_xlim(0, limits.SENSOR_W)
    ax.set_ylim(limits.SENSOR_H, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.set_title(f"Active pixels without a mask (UC4, steady light, {sec:g} s): {active.size} pixels with events;\n"
                 f"the 2 defect pixels give {hot_share:.4%} of all {events.size:,} events", fontsize=10)
    fig.colorbar(sc, ax=ax, label="events per second (log)", shrink=0.8)
    return save(fig, "hot_pixel_map.png")


def noise_vs_bias() -> Path:
    events, meta = store.load(UC1_MASKED)
    uc1 = events.size / seconds_of(meta) / 1e6
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogy(list(BIAS_TABLE), list(BIAS_TABLE.values()), "o-", color="#2b5c8a",
                label="2026-10-14, 0.5 s warm-up only (start-up burst included)")
    ax.semilogy([meta["settings"]["biases"]["bias_diff_off"]], [uc1], "s", color="#c0392b", ms=9,
                label=f"UC1 run, after the 8 s settle: {uc1 * 1e6:.0f} events/s")
    ax.set_xlabel("bias_diff_off (offset from the default)")
    ax.set_ylabel("dark noise (M events/s, 2 defect pixels masked)")
    ax.set_title("Dark noise vs bias_diff_off (camera covered)", fontsize=10)
    ax.legend(fontsize=8)
    ax.text(0.02, 0.03, "The 2026-10-14 curve shows the trend (a higher bias gives fewer noise events), but its\n"
            "values are too high: they were measured during the start-up burst, before the 8 s settle existed.",
            transform=ax.transAxes, fontsize=7.5, color="#555555")
    return save(fig, "noise_vs_bias_diff_off.png")


def rate_before_after() -> Path:
    fig, (bars, series) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 2]})
    names, hot_rates, other_rates, rows = ["before mask", "after mask"], [], [], []
    for rid in (UC2_BEFORE, UC2_AFTER):
        ev, meta = store.load(rid)
        sec = seconds_of(meta)
        is_hot = np.zeros(ev.size, dtype=bool)
        for x, y in HOT:
            is_hot |= (ev["x"] == x) & (ev["y"] == y)
        hot_rates.append(is_hot.sum() / sec / 1e6)
        other_rates.append((~is_hot).sum() / sec / 1e6)
        rows.append(ev[~is_hot])
    bars.bar(names, other_rates, color="#2b5c8a", label="all other pixels")
    bars.bar(names, hot_rates, bottom=other_rates, color="#c0392b", label="2 defect pixels")
    for i in range(2):
        bars.text(i, other_rates[i] + hot_rates[i], f"{other_rates[i] + hot_rates[i]:.2f}", ha="center",
                  va="bottom", fontsize=9)
    bars.set_ylabel("M events/s")
    bars.set_title("UC2 demo run: total event rate", fontsize=10)
    bars.legend(fontsize=8)
    bin_us, t_max = 250, 30_000  # 0.25 ms bins, first 30 ms
    for label, ev, color in zip(names, rows, ["#c0392b", "#2b5c8a"], strict=True):
        sel = ev["t"] < t_max
        rate = np.bincount(ev["t"][sel] // bin_us, minlength=t_max // bin_us) / (bin_us * 1e-6) / 1e6
        series.plot(np.arange(rate.size) * bin_us / 1000, rate, color=color, lw=1, label=f"{label} (other pixels)")
    series.set_xlabel("time (ms)")
    series.set_ylabel("M events/s (0.25 ms bins)")
    series.set_title("first 30 ms: the 240 Hz flicker (period 4.17 ms)", fontsize=10)
    series.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=2)
    fig.text(0.5, 0.005, f"Note: the other pixels also changed between the two recordings ({other_rates[0]:.2f} -> "
             f"{other_rates[1]:.2f} M events/s); masking 2 pixels does not cause that.", ha="center", fontsize=8,
             color="#555555")
    return save(fig, "uc2_rate_before_after_mask.png")


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for make in (spectrum_figure, hot_pixel_map, noise_vs_bias, rate_before_after):
        print("wrote", make().relative_to(PROJECT_ROOT))
