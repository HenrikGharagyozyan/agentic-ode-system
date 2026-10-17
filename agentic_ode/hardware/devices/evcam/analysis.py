"""Analysis of recorded events: counts per pixel, hot pixels, the flicker spectrum and the delta reconstruction.
Pure numpy, no hardware.

The MCP functions (summary, hot_pixels, spectrum) return plain Python numbers, lists and dicts, so a result can be
sent as JSON. The plot functions (brightness_curve, delta_image, time_image) return numpy arrays.
Events are the structured array from EventCamera.record(): fields x, y, p (1 = ON, 0 = OFF), t (us, from 0).
"""

from __future__ import annotations

import math

import numpy as np

from agentic_ode.hardware.devices.evcam import limits

PEAK_WINDOW_HZ = 5.0  # a spectrum peak must be the largest value within +-5 Hz; this drops the small side peaks
N_PEAKS = 5  # spectrum() reports the 5 strongest peaks
N_HARMONICS = 5  # spectrum() reports the harmonics 2f .. 5f of the fundamental
MIN_GOOD_SECONDS = 1.0  # spectrum() warns about shorter recordings (resolution coarser than 1 Hz)
MAX_DIVISOR = 15  # _base_of_strongest() tests f_max / k for k = 2..15
MAX_MULTIPLE = 9  # ... and the multiples n*f for n = 2..9
MIN_MULTIPLES = 2  # ... of which at least 2 must be strong (not counting the harmonics of f_max itself)
EDGE_STEP_PART = 16  # edge_windows(): a step is the change of the curve over 1/16 period (real edges take 1-2 ms)
EDGE_WINDOW_PART = 4  # ... and each window is 1/4 period wide, centred on the edge


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


def hot_pixels(events: np.ndarray, seconds: float, top: int = limits.MAX_MASK_SLOTS,
               min_rate_hz: float = limits.MAX_PIXEL_RATE_HZ) -> list[dict]:
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


def _base_of_strongest(amp: np.ndarray, resolution: float, strongest: int, floor: float, fmin_hz: float,
                       fmax_hz: float, min_snr: float) -> tuple[float | None, list[float]]:
    """The base frequency of a square wave, found from the divisors of the strongest peak (index `strongest`).

    A sharp square wave at f has odd harmonics 3f, 5f, 7f ... of almost the same height, so the strongest peak can
    be 7f, and f itself may not be among the N_PEAKS peaks (real LED at 10 Hz, 2026-10-16: 70 Hz was reported).
    But a divisor can also fit by chance (real 50 Hz LED, 2026-10-17: 150.8 Hz was strongest, and 150.8 / 15 Hz was
    reported, because its "multiples" 50 Hz and 60 Hz were the LED itself and a sideband). So, with f_max the
    strongest peak:
    1. candidates: f_max, and f = f_max / k (k = 2..MAX_DIVISOR, only f >= fmin_hz) if the spectrum at f is strong
       (snr >= min_snr, +-1 bin) and at least MIN_MULTIPLES of n*f (n = 2..MAX_MULTIPLE, up to fmax_hz) are strong.
       Only n that are not multiples of k count: n*f = (n/k)*f_max is then a harmonic of f_max, no evidence for f.
    2. a candidate f is proven if, for every candidate g = m*f above it, the strong lines that only f explains
       (n*f with n not a multiple of m) are together at least as strong as the lines that g explains too.
    3. a candidate that is not proven, or proven ones that are not all multiples of the lowest -> unresolved.
    Returns (f0, []) with f0 the frequency of the strongest bin near the lowest proven f, (None, []) if f_max is
    the only candidate, or (None, all candidates in Hz) if unresolved.
    """
    def strong_bin(f, w=1):  # the largest bin within +-w bins of f if it is strong, else None
        c = round(f / resolution)
        lo = max(c - w, 0)
        if lo >= amp.size:
            return None
        j = lo + int(np.argmax(amp[lo: c + w + 1]))
        return j if amp[j] > 0 and amp[j] >= min_snr * floor else None

    # 1. the candidates, k -> f_max / k
    f_max = strongest * resolution
    candidates = {1: f_max}
    for k in range(2, MAX_DIVISOR + 1):
        f = f_max / k
        if f < fmin_hz:
            break  # k grows, so f only gets lower
        if strong_bin(f) is None:
            continue
        own = [n * f for n in range(2, MAX_MULTIPLE + 1) if n % k and n * f <= fmax_hz]
        if sum(strong_bin(m) is not None for m in own) >= MIN_MULTIPLES:
            candidates[k] = f
    if len(candidates) == 1:
        return None, []

    # 2. which candidates are proven; f_max has no candidate above it, so it is proven
    proven = [1]
    for k, f in candidates.items():
        if k == 1:
            continue
        # the strong lines at n*f, n >= 1: f = f_max / k is known to +-resolution/(2k), so n*f to +-n/(2k) bins
        lines = {}
        for n in range(1, int(fmax_hz / f) + 1):
            j = strong_bin(n * f, 1 + n // (2 * k))
            if j is not None:
                lines[n] = amp[j]
        for g_k in candidates:
            if g_k < k and k % g_k == 0:  # g = f_max / g_k = m*f
                m = k // g_k
                only_f = sum(a for n, a in lines.items() if n % m)
                shared = sum(a for n, a in lines.items() if n % m == 0)  # > 0: f_max is always one of them
                if only_f < shared:
                    break
        else:
            proven.append(k)

    # 3. one answer only if every candidate is proven and they all are multiples of the lowest
    lowest = max(proven)  # the largest k is the lowest f
    if len(proven) < len(candidates) or any(lowest % k for k in proven):
        return None, [round(f, 3) for f in sorted(candidates.values())]
    return strong_bin(candidates[lowest]) * resolution, []


def spectrum(events: np.ndarray, bin_us: int = 100, fmax_hz: float = 2000.0, fmin_hz: float = 20.0,
             min_snr: float = 8.0) -> dict:
    """Flicker spectrum of the whole sensor: the dominant (strongest) peak, the fundamental frequency, the strongest
    peaks and the harmonics of the fundamental.

    Signal: (ON events - OFF events) in each time bin of bin_us. It follows the change of the light, so a light
    that blinks at f usually gives its strongest peak at f. (The plain event count would show 2f for a symmetric
    blink.) But a harmonic can be stronger, so fundamental_hz (see _base_of_strongest() and fundamental()) is the
    flicker frequency. If the strongest peak fits more than one base and the spectrum does not show which one,
    fundamental_hz is None (unresolved), harmonics is empty and the "warning" names the candidates.
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
    # the flicker frequency: the strongest peak can be a harmonic (2f was strongest in a 0.17 s real recording,
    # 7f in a real 10 Hz square wave). First the divisors of the strongest peak; if none of them is a candidate,
    # the peak list, then the peak. Unresolved divisors -> no fundamental: the strongest peak may be a harmonic.
    f0, unresolved = _base_of_strongest(amp, resolution, peaks[0], floor, fmin_hz, fmax_hz, min_snr)
    if unresolved:
        text = (f"the fundamental is unresolved: the strongest peak {result['dominant_hz']:g} Hz fits more than one "
                f"base ({', '.join(f'{f:g}' for f in unresolved)} Hz) and the spectrum does not show which one")
        result["warning"] = f"{result['warning']}; {text}" if "warning" in result else text
        return result
    f0 = f0 or fundamental([float(freq[i]) for i in peaks], resolution) or float(freq[peaks[0]])
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


# ---- delta reconstruction ----
# Each event means: the log brightness of one pixel changed by one contrast step Δ (ON = +Δ, OFF = -Δ). Summing the
# events gives the brightness change. Δ is not known in absolute units, so every result is "in units of Δ".


def _normal_pixels(events: np.ndarray, seconds: float | None, width: int, height: int) -> np.ndarray:
    """True for the events of normal pixels, False for hot pixels (>= MAX_PIXEL_RATE_HZ, the hot_pixels() rule).

    Masked pixels have no events, so they never appear. This also cleans recordings made before the mask was set.
    seconds: the recording length; None = the span of the event times.
    """
    if events.size == 0:
        return np.zeros(0, dtype=bool)
    if seconds is None:
        seconds = (int(events["t"].max()) - int(events["t"].min()) + 1) / 1e6
    seconds = _check_seconds(seconds)
    index = events["y"].astype(np.int64) * width + events["x"]
    counts = np.bincount(index, minlength=width * height)
    return counts[index] < limits.MAX_PIXEL_RATE_HZ * seconds


def _check_image_args(width: int, height: int, bin_px: int) -> None:
    for name, v in (("width", width), ("height", height), ("bin_px", bin_px)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{name} must be an int >= 1, got {v!r}")


def _bin_index(events: np.ndarray, width: int, bin_px: int) -> tuple[np.ndarray, int]:
    """The image bin of each event (row by row from y = 0) and the number of bin columns."""
    cols = -(-width // bin_px)  # ceil division: a last, partial column of bins
    return (events["y"].astype(np.int64) // bin_px) * cols + events["x"].astype(np.int64) // bin_px, cols


def brightness_curve(events: np.ndarray, bin_us: int = 100, leak_tau_s: float | None = 0.2,
                     seconds: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The brightness change of the whole sensor over time, in units of Δ: (t_s, L).

    net[i] = (#ON - #OFF in bin i) / number of active pixels, so L is the mean change per active pixel (hot pixels
    dropped, see _normal_pixels). L is the leaky integral of net:  L[i] = a * L[i-1] + net[i],  a = exp(-bin / tau).
    The leak makes old steps fade with time constant leak_tau_s. Without it, L drifts away, because the ON and OFF
    thresholds are never exactly equal. leak_tau_s=None: the plain sum (a = 1).
    t_s[i] is the END of bin i (L[i] includes all events up to there). seconds: see _normal_pixels.
    """
    if events.size == 0:
        raise ValueError("no events: nothing to integrate")
    if isinstance(bin_us, bool) or not isinstance(bin_us, int) or bin_us < 1:
        raise ValueError(f"bin_us must be an int >= 1, got {bin_us!r}")
    if leak_tau_s is not None and (isinstance(leak_tau_s, bool) or not isinstance(leak_tau_s, (int, float))
                                   or not 0 < leak_tau_s < float("inf")):
        raise ValueError(f"leak_tau_s must be a positive number or None, got {leak_tau_s!r}")
    t = events["t"]
    if t.min() < 0:
        raise ValueError("event times must be >= 0 (record() gives t from 0)")

    n_bins = int(t.max()) // bin_us + 1  # from ALL events, so the time axis covers the whole recording
    keep = _normal_pixels(events, seconds, limits.SENSOR_W, limits.SENSOR_H)
    ev = events[keep]
    active = np.unique(ev["y"].astype(np.int64) * limits.SENSOR_W + ev["x"]).size
    sign = np.where(ev["p"] == 1, 1.0, -1.0)
    net = np.bincount(ev["t"] // bin_us, weights=sign, minlength=n_bins) / max(active, 1)

    a = 1.0 if leak_tau_s is None else math.exp(-bin_us * 1e-6 / leak_tau_s)
    level = np.empty(n_bins)
    value = 0.0
    for i, step in enumerate(net.tolist()):
        value = a * value + step
        level[i] = value
    t_s = np.arange(1, n_bins + 1) * bin_us / 1e6
    return t_s, level


def delta_image(events: np.ndarray, width: int, height: int, t0_us: int | None = None, t1_us: int | None = None,
                bin_px: int = 4, seconds: float | None = None) -> np.ndarray:
    """Sum of the polarity (ON = +1, OFF = -1) per bin_px x bin_px block, in units of Δ, for t0_us <= t < t1_us
    (None = no limit). Shape (ceil(height / bin_px), ceil(width / bin_px)); row 0 is y = 0 (the top of the sensor).
    > 0: the block got brighter, < 0: darker. Hot pixels are dropped over the WHOLE recording (see _normal_pixels).
    """
    _check_image_args(width, height, bin_px)
    keep = _normal_pixels(events, seconds, width, height)
    if t0_us is not None:
        keep &= events["t"] >= t0_us
    if t1_us is not None:
        keep &= events["t"] < t1_us
    return _delta_sum(events[keep], width, height, bin_px)


def _delta_sum(events: np.ndarray, width: int, height: int, bin_px: int) -> np.ndarray:
    """Sum of the polarity (ON = +1, OFF = -1) per bin_px x bin_px block of ALL the given events."""
    index, cols = _bin_index(events, width, bin_px)
    rows = -(-height // bin_px)
    sign = np.where(events["p"] == 1, 1, -1)
    return np.bincount(index, weights=sign, minlength=rows * cols).astype(np.int64).reshape(rows, cols)


def time_image(events: np.ndarray, width: int, height: int, bin_px: int = 4,
               seconds: float | None = None) -> np.ndarray:
    """The mean time (s) of the events in each bin_px x bin_px block; NaN for a block without events.
    Same shape and hot-pixel rule as delta_image(). Early blocks are small numbers, late blocks large ones."""
    _check_image_args(width, height, bin_px)
    ev = events[_normal_pixels(events, seconds, width, height)]
    index, cols = _bin_index(ev, width, bin_px)
    size = -(-height // bin_px) * cols
    counts = np.bincount(index, minlength=size)
    sums = np.bincount(index, weights=ev["t"] / 1e6, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):  # 0 / 0 -> NaN = no events
        return (sums / counts).reshape(-1, cols)


def edge_windows(events: np.ndarray, f0_hz: float | None, n_periods: int = 5, bin_us: int = 100,
                 leak_tau_s: float | None = 0.2, seconds: float | None = None,
                 curve: tuple[np.ndarray, np.ndarray] | None = None) -> list[tuple[int, int, str]]:
    """Time windows around the edges of a blinking light: [(t0_us, t1_us, kind)], kind "rise" or "fall", by time.

    A delta image over one FULL period sums to about 0 (the light goes up, then down), so the two edges are shown
    separately. In each of the first n_periods full periods of 1/f0 (from t = 0), the rising edge is the largest
    positive step of the brightness curve and the falling edge the largest negative step. A step is L[i+w] - L[i]
    with w = 1/EDGE_STEP_PART period: this finds the edge without being fooled by one noisy bin. A sharp edge gives
    the same step for several i, so the centre of the edge is the steepest single bin within that step.
    Each window is 1/EDGE_WINDOW_PART period wide and centred on its edge.
    An edge on the border of two periods can be found twice; the second, overlapping window is skipped.
    f0_hz None (no spectrum peak): [] - the caller shows the whole recording instead.
    curve: the (t_s, L) of brightness_curve() if it is already computed (it takes a moment for a large recording).
    """
    if f0_hz is None:
        return []
    if isinstance(f0_hz, bool) or not isinstance(f0_hz, (int, float)) or not 0 < f0_hz < float("inf"):
        raise ValueError(f"f0_hz must be a positive number or None, got {f0_hz!r}")
    if isinstance(n_periods, bool) or not isinstance(n_periods, int) or n_periods < 1:
        raise ValueError(f"n_periods must be an int >= 1, got {n_periods!r}")
    t_s, level = curve if curve is not None else brightness_curve(events, bin_us, leak_tau_s, seconds)
    bin_us = round(float(t_s[0]) * 1e6)  # t_s[0] is the end of the first bin = the bin width
    period = 1e6 / f0_hz / bin_us  # in bins (not a whole number)
    w = max(1, round(period / EDGE_STEP_PART))
    step = level[w:] - level[:-w]  # step[i]: the events of bins i+1 .. i+w
    one = np.diff(level, prepend=0.0)  # one[j]: the change in bin j alone
    half_us = 1e6 / f0_hz / EDGE_WINDOW_PART / 2
    end_us = t_s.size * bin_us

    windows: list[tuple[int, int, str]] = []
    last = {"rise": -1, "fall": -1}  # the end (us) of the last window of each kind
    for k in range(n_periods):
        lo, hi = round(k * period), round((k + 1) * period)
        if hi > step.size:
            break  # not a full period any more
        part = step[lo:hi]
        for i, kind, ok in ((lo + int(np.argmax(part)), "rise", part.max() > 0),
                            (lo + int(np.argmin(part)), "fall", part.min() < 0)):
            inside = one[i + 1: i + w + 1]
            j = i + 1 + int(np.argmax(inside) if kind == "rise" else np.argmin(inside))
            centre_us = (j + 0.5) * bin_us  # the middle of the steepest bin
            t0, t1 = max(0, round(centre_us - half_us)), min(end_us, round(centre_us + half_us))
            if ok and t0 >= last[kind]:
                windows.append((t0, t1, kind))
                last[kind] = t1
    return sorted(windows)


def edge_delta_images(events: np.ndarray, width: int, height: int, f0_hz: float | None, bin_px: int = 4,
                      n_periods: int = 5, seconds: float | None = None,
                      windows: list[tuple[int, int, str]] | None = None) -> dict[str, np.ndarray]:
    """{"rise": image, "fall": image}: delta_image() summed over all rise windows and over all fall windows of
    edge_windows() (or of `windows`, if already computed). Summing over several periods averages out the noise.
    Hot pixels are dropped over the whole recording, as in delta_image(). No windows: two images of zeros."""
    _check_image_args(width, height, bin_px)
    if windows is None:
        windows = edge_windows(events, f0_hz, n_periods, seconds=seconds)
    keep = _normal_pixels(events, seconds, width, height)
    t = events["t"]
    images = {}
    for kind in ("rise", "fall"):
        inside = np.zeros(events.size, dtype=bool)
        for t0, t1, k in windows:
            if k == kind:
                inside |= (t >= t0) & (t < t1)
        images[kind] = _delta_sum(events[keep & inside], width, height, bin_px)
    return images
