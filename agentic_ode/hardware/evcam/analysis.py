"""Analysis of recorded events: counts per pixel, hot pixels and the flicker spectrum. Pure numpy, no hardware.

Every function returns plain Python numbers, lists and dicts, so a result can be sent as JSON (for the MCP tools).
Events are the structured array from EventCamera.record(): fields x, y, p (1 = ON, 0 = OFF), t (us, from 0).
"""

from __future__ import annotations

import numpy as np

from agentic_ode.hardware.evcam import limits

PEAK_WINDOW_HZ = 5.0  # a spectrum peak must be the largest value within +-5 Hz; this drops the small side peaks
N_PEAKS = 5  # spectrum() reports the 5 strongest peaks
N_HARMONICS = 5  # spectrum() reports the harmonics 2f .. 5f of the fundamental
MIN_GOOD_SECONDS = 1.0  # spectrum() warns about shorter recordings (resolution coarser than 1 Hz)


def _check_seconds(seconds: float) -> float:
    # NaN fails every comparison, so `not 0 < seconds < inf` also rejects it
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 < seconds < float("inf"):
        raise ValueError(f"seconds must be a positive number, got {seconds!r}")
    return float(seconds)


def _pixel_counts(events: np.ndarray) -> np.ndarray:
    """Events per pixel as a flat array of SENSOR_W*SENSOR_H numbers (index = y*SENSOR_W + x)."""
    index = events["y"].astype(np.int64) * limits.SENSOR_W + events["x"]
    return np.bincount(index, minlength=limits.SENSOR_W * limits.SENSOR_H)


def _pixel_list(counts: np.ndarray, indices: np.ndarray, seconds: float) -> list[dict]:
    return [{"x": int(i % limits.SENSOR_W), "y": int(i // limits.SENSOR_W), "count": int(counts[i]),
             "rate_hz": round(counts[i] / seconds, 1)} for i in indices]


def summary(events: np.ndarray, seconds: float, top: int = 5) -> dict:
    """Event rate, share of ON events, number of pixels with events and the `top` busiest pixels."""
    seconds = _check_seconds(seconds)
    if top < 0:
        raise ValueError(f"top must be >= 0, got {top}")
    n = int(events.size)
    counts = _pixel_counts(events)
    busiest = np.argsort(counts)[::-1][:top]
    busiest = busiest[counts[busiest] > 0]  # never list a pixel without events
    return {
        "n_events": n,
        "seconds": seconds,
        "rate_mev_s": round(n / seconds / 1e6, 6),
        "on_fraction": round(float(np.mean(events["p"] == 1)), 4) if n else None,  # None: no events, no fraction
        "active_pixels": int(np.count_nonzero(counts)),
        "top_pixels": _pixel_list(counts, busiest, seconds),
    }


def hot_pixels(events: np.ndarray, seconds: float, top: int = 64, min_rate_hz: float = 20_000) -> list[dict]:
    """Pixels that fire at least min_rate_hz, busiest first, at most `top` of them.

    A real pixel cannot fire faster than about 20,000 ev/s (dead time ~73 us), so the default finds defect pixels.
    top = 64 is the number of mask slots, so the result can go straight to EventCamera.mask_pixels().
    """
    seconds = _check_seconds(seconds)
    if top < 0:
        raise ValueError(f"top must be >= 0, got {top}")
    if not min_rate_hz > 0:
        raise ValueError(f"min_rate_hz must be > 0, got {min_rate_hz}")
    counts = _pixel_counts(events)
    hot = np.flatnonzero(counts >= min_rate_hz * seconds)
    hot = hot[np.argsort(counts[hot])[::-1]][:top]  # busiest first
    return _pixel_list(counts, hot, seconds)


def fundamental(peaks_hz: list[float], resolution: float) -> float | None:
    """The lowest peak f such that every other peak is near an integer multiple k*f (k >= 1). Near means within
    (k+1)/2 * resolution: f is only known to +-resolution/2, so k*f to +-k*resolution/2, and the peak itself
    to +-resolution/2. None if no peak explains all the others (e.g. two lights that are not related)."""
    for f in sorted(peaks_hz):
        if all(round(p / f) >= 1 and abs(p - round(p / f) * f) <= (round(p / f) + 1) / 2 * resolution
               for p in peaks_hz):
            return f
    return None


def spectrum(events: np.ndarray, bin_us: int = 100, fmax_hz: float = 2000.0, fmin_hz: float = 20.0,
             min_snr: float = 8.0) -> dict:
    """Flicker spectrum of the whole sensor: the dominant (strongest) peak, the fundamental frequency, the strongest
    peaks and the harmonics of the fundamental.

    Signal: (ON events - OFF events) in each time bin of bin_us. It follows the change of the light, so a light
    that blinks at f usually gives its strongest peak at f. (The plain event count would show 2f for a symmetric
    blink.) But a harmonic can be stronger, so fundamental_hz (see fundamental()) is the flicker frequency.
    Recordings shorter than MIN_GOOD_SECONDS get a "warning" (coarse resolution).
    Frequencies below fmin_hz (slow drift of the noise) and above fmax_hz are ignored.
    snr = amplitude / noise_floor, where noise_floor is the median amplitude in the band. A peak counts only if
    snr >= min_snr: pure noise stays below ~4 (measured 3.4..4.0), the phone screen at 240 Hz reached ~90.
    No peak above min_snr -> dominant_hz is None.
    rel_amp is the amplitude relative to the dominant peak (dominant = 1.0).
    """
    if events.size == 0:
        raise ValueError("no events: nothing to analyse")
    if isinstance(bin_us, bool) or not isinstance(bin_us, int) or bin_us < 1:
        raise ValueError(f"bin_us must be an int >= 1, got {bin_us!r}")
    nyquist = 1e6 / (2 * bin_us)  # the highest frequency bins of bin_us can show
    if not 0 <= fmin_hz < fmax_hz <= nyquist:
        raise ValueError(f"need 0 <= fmin_hz < fmax_hz <= {nyquist:g} Hz (Nyquist for bin_us={bin_us}), "
                         f"got fmin_hz={fmin_hz}, fmax_hz={fmax_hz}")
    if not min_snr >= 0:
        raise ValueError(f"min_snr must be >= 0, got {min_snr}")
    t = events["t"]
    if t.min() < 0:
        raise ValueError("event times must be >= 0 (record() gives t from 0)")

    # the signal: +1 for each ON event, -1 for each OFF event, summed per bin
    n_bins = int(t.max()) // bin_us + 1
    sign = np.where(events["p"] == 1, 1.0, -1.0)
    signal = np.bincount(t // bin_us, weights=sign, minlength=n_bins)
    amp = np.abs(np.fft.rfft(signal - signal.mean()))  # minus the mean: no peak at 0 Hz
    freq = np.fft.rfftfreq(n_bins, d=bin_us * 1e-6)
    resolution = 1e6 / (n_bins * bin_us)  # = 1 / recording length
    band = np.flatnonzero((freq >= fmin_hz) & (freq <= fmax_hz))
    if band.size == 0:
        raise ValueError(f"the recording ({n_bins * bin_us / 1e6:g} s) is too short for {fmin_hz}..{fmax_hz} Hz")

    floor = float(np.median(amp[band]))

    def snr(i):  # None when the floor is 0 (almost no events): a ratio would be infinite
        return round(float(amp[i] / floor), 1) if floor > 0 else None

    # peaks: go through the band from the strongest value down; keep a value only if it is the largest
    # within +-PEAK_WINDOW_HZ (so the side peaks next to a strong peak are not counted)
    w = max(1, round(PEAK_WINDOW_HZ / resolution))
    peaks = []
    for i in band[np.argsort(amp[band])[::-1]]:
        if amp[i] == 0 or amp[i] < min_snr * floor:
            break  # too weak; all the next values are weaker still
        if amp[i] == amp[max(i - w, 0): i + w + 1].max():
            peaks.append(i)
            if len(peaks) == N_PEAKS:
                break

    seconds = n_bins * bin_us / 1e6
    result = {
        "signal": "on_minus_off",
        "bin_us": bin_us,
        "fmin_hz": fmin_hz,
        "fmax_hz": fmax_hz,
        "min_snr": min_snr,
        "seconds": seconds,
        "resolution_hz": round(resolution, 4),
        "noise_floor": round(floor, 3),
        "dominant_hz": None,
        "fundamental_hz": None,
        "peaks": [],
        "harmonics": [],
    }
    if seconds < MIN_GOOD_SECONDS:
        result["warning"] = (f"the recording is only {seconds:.2f} s long: the frequency resolution is "
                             f"{resolution:.1f} Hz; record at least {MIN_GOOD_SECONDS:g} s for a precise frequency")
    if not peaks:  # no peak stands out of the noise
        return result
    top = amp[peaks[0]]
    result["dominant_hz"] = round(float(freq[peaks[0]]), 3)
    result["peaks"] = [{"hz": round(float(freq[i]), 3), "rel_amp": round(float(amp[i] / top), 3), "snr": snr(i)}
                       for i in peaks]
    # the flicker frequency: the strongest peak can be a harmonic (2f was strongest in a 0.17 s real recording)
    f0 = fundamental([float(freq[i]) for i in peaks], resolution) or float(freq[peaks[0]])
    result["fundamental_hz"] = round(f0, 3)

    # harmonics of the fundamental: the largest value near k*f0; f0 is only known to +-resolution/2, so k*f0 to
    # +-k*resolution/2: search +-k bins
    for k in range(2, N_HARMONICS + 1):
        if k * f0 > fmax_hz:
            break
        c = round(k * f0 / resolution)
        j = max(c - k, 0) + int(np.argmax(amp[max(c - k, 0): c + k + 1]))
        result["harmonics"].append({"k": k, "hz": round(float(freq[j]), 3), "rel_amp": round(float(amp[j] / top), 3),
                                    "snr": snr(j)})
    return result
