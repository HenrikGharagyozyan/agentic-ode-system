"""Event analysis with synthetic events (no hardware needed)."""

import math

import numpy as np
import pytest

from agentic_ode.hardware.devices.evcam.analysis import (
    _base_of_strongest,
    brightness_curve,
    delta_image,
    edge_delta_images,
    edge_windows,
    fundamental,
    hot_pixels,
    spectrum,
    summary,
    time_image,
)
from agentic_ode.hardware.devices.evcam.events import EVENT_DTYPE


def make_events(x, y, p, t):
    """A structured event array like EventCamera.record() returns, sorted by time."""
    ev = np.zeros(len(t), dtype=EVENT_DTYPE)
    ev["x"], ev["y"], ev["p"], ev["t"] = x, y, p, t
    return ev[np.argsort(ev["t"], kind="stable")]


def blinking_light(f_hz, seconds, rng, events_per_edge=20, jitter_us=200, noise=20_000, duty=0.5):
    """A light that blinks at f_hz, on for `duty` of each period: ON events at each rising edge, OFF events at each
    falling edge, each spread by jitter_us (pixels do not all react at once), plus `noise` random events."""
    period = 1e6 / f_hz
    rising = np.arange(0, seconds * 1e6 - period, period)
    t_on = np.repeat(rising, events_per_edge) + rng.normal(0, jitter_us, rising.size * events_per_edge)
    t_off = t_on + duty * period
    t_noise = rng.uniform(0, seconds * 1e6, noise)
    t = np.concatenate([t_on, t_off, t_noise])
    p = np.concatenate([np.ones(t_on.size), np.zeros(t_off.size), rng.integers(0, 2, noise)])
    t = np.clip(t, 0, seconds * 1e6 - 1).astype(np.int64)
    return make_events(rng.integers(0, 1280, t.size), rng.integers(0, 720, t.size), p, t)


def modulated_light(f_hz, fm_hz, depth, seconds, rng, events_per_edge=200, jitter_us=300, duty=1 / 6, slow=30_000,
                    noise=20_000):
    """A light at f_hz whose brightness changes at fm_hz (events per edge * (1 + depth * sin)): this gives the
    sidebands f_hz +- fm_hz. Plus `slow` events of a weak second flicker at fm_hz (ON while sin > 0, else OFF)."""
    period = 1e6 / f_hz
    rising = np.arange(0, seconds * 1e6 - period, period)
    n = np.round(events_per_edge * (1 + depth * np.sin(2 * np.pi * fm_hz * rising / 1e6))).astype(np.int64)
    t_on = np.repeat(rising, n) + rng.normal(0, jitter_us, n.sum())
    t_off = t_on + duty * period
    t_slow = rng.uniform(0, seconds * 1e6, slow)
    t_noise = rng.uniform(0, seconds * 1e6, noise)
    t = np.concatenate([t_on, t_off, t_slow, t_noise])
    p = np.concatenate([np.ones(t_on.size), np.zeros(t_off.size), np.sin(2 * np.pi * fm_hz * t_slow / 1e6) > 0,
                        rng.integers(0, 2, noise)])
    t = np.clip(t, 0, seconds * 1e6 - 1).astype(np.int64)
    return make_events(rng.integers(0, 1280, t.size), rng.integers(0, 720, t.size), p, t)


def line_spectrum(lines, resolution=0.5, fmax_hz=2000.0):
    """A hand-made amplitude spectrum without noise: 1 in every bin (the noise floor), and amp at the bin nearest to
    each frequency of lines = {hz: amp}."""
    amp = np.ones(int(fmax_hz / resolution) + 1)
    for hz, a in lines.items():
        amp[round(hz / resolution)] = a
    return amp


def base_of(amp, resolution=0.5):
    # floor 1, fmin 2 Hz, fmax 2000 Hz, min_snr 8: like spectrum(fmin_hz=2) on the hand-made spectrum
    return _base_of_strongest(amp, resolution, int(np.argmax(amp)), 1.0, 2.0, 2000.0, 8.0)


# ---- summary ----

def test_summary_counts():
    ev = make_events([10, 10, 10, 5], [20, 20, 20, 5], [1, 1, 1, 0], [0, 1, 2, 3])
    s = summary(ev, 2.0)
    assert s["n_events"] == 4 and s["seconds"] == 2.0
    assert s["rate_mev_s"] == 2e-6
    assert s["on_fraction"] == 0.75
    assert s["active_pixels"] == 2
    assert s["top_pixels"] == [{"x": 10, "y": 20, "count": 3, "rate_hz": 1.5},
                               {"x": 5, "y": 5, "count": 1, "rate_hz": 0.5}]


def test_summary_no_events():
    s = summary(make_events([], [], [], []), 1.0)
    assert s["n_events"] == 0 and s["rate_mev_s"] == 0
    assert s["on_fraction"] is None
    assert s["active_pixels"] == 0 and s["top_pixels"] == []


@pytest.mark.parametrize("seconds", [0, -1.0, math.nan, math.inf, True, "1"])
def test_summary_bad_seconds(seconds):
    with pytest.raises(ValueError, match="seconds must be a positive number"):
        summary(make_events([], [], [], []), seconds)


# ---- hot pixels ----

def test_hot_pixels_threshold_order_and_top():
    # in 1 s: pixel (1, 1) 50,000 events, pixel (2, 2) 30,000, pixel (3, 3) 100
    n = [50_000, 30_000, 100]
    x = np.repeat([1, 2, 3], n)
    ev = make_events(x, x, np.ones(x.size), np.arange(x.size))
    assert hot_pixels(ev, 1.0) == [{"x": 1, "y": 1, "count": 50_000, "rate_hz": 50_000.0},
                                   {"x": 2, "y": 2, "count": 30_000, "rate_hz": 30_000.0}]
    assert [(h["x"], h["y"]) for h in hot_pixels(ev, 1.0, top=1)] == [(1, 1)]
    assert hot_pixels(ev, 1.0, min_rate_hz=60_000) == []
    assert len(hot_pixels(ev, 1.0, min_rate_hz=50)) == 3


def test_two_hot_pixels_at_equal_rates_are_both_returned():
    # the real defect pixels: equal rates, and pixel ids y*1280+x far above the uint16 limit 65535
    n = 100_000
    x = np.repeat([1053, 926], n)
    y = np.repeat([704, 151], n)
    ev = make_events(x, y, np.zeros(2 * n), np.tile(np.arange(n), 2))
    assert ev["x"].dtype == np.uint16 and 704 * 1280 + 1053 > 65535
    hot = hot_pixels(ev, 1.0)
    assert sorted((h["x"], h["y"], h["count"]) for h in hot) == [(926, 151, n), (1053, 704, n)]
    top = summary(ev, 1.0)["top_pixels"]
    assert sorted((p["x"], p["y"], p["count"]) for p in top) == [(926, 151, n), (1053, 704, n)]


def test_hot_pixels_bad_rate():
    with pytest.raises(ValueError, match="min_rate_hz must be > 0"):
        hot_pixels(make_events([], [], [], []), 1.0, min_rate_hz=0)


# ---- spectrum ----

def test_spectrum_finds_the_blink_frequency():
    rng = np.random.default_rng(0)
    s = spectrum(blinking_light(240, 2.0, rng), bin_us=100, fmax_hz=2000)
    assert s["resolution_hz"] == 0.5  # 1 / 2 s
    assert abs(s["dominant_hz"] - 240) <= s["resolution_hz"]
    assert s["peaks"][0]["hz"] == s["dominant_hz"] and s["peaks"][0]["rel_amp"] == 1.0
    assert s["peaks"][0]["snr"] >= s["min_snr"] == 8.0
    # 50 % duty: odd harmonics (3f, 5f) are strong, even ones (2f, 4f) are almost zero
    h = {item["k"]: item for item in s["harmonics"]}
    assert sorted(h) == [2, 3, 4, 5]
    assert abs(h[3]["hz"] - 720) <= 3 * s["resolution_hz"]
    assert h[3]["rel_amp"] > 0.3 and h[2]["rel_amp"] < 0.1 and h[4]["rel_amp"] < 0.1


def test_spectrum_no_side_peaks_near_the_dominant():
    s = spectrum(blinking_light(240, 2.0, np.random.default_rng(1)))
    hz = [p["hz"] for p in s["peaks"]]
    assert all(abs(a - b) > 5 for i, a in enumerate(hz) for b in hz[i + 1:])


def test_spectrum_fmin_hides_slow_drift():
    rng = np.random.default_rng(2)
    light = blinking_light(240, 2.0, rng)
    # slow drift: 100,000 events, ON while sin(2*pi*2Hz*t) > 0, else OFF
    t = rng.integers(0, 2_000_000, 100_000)
    drift = make_events(np.zeros(t.size), np.zeros(t.size), np.sin(2 * np.pi * 2 * t / 1e6) > 0, t)
    ev = np.sort(np.concatenate([light, drift]), order="t")
    assert abs(spectrum(ev, fmin_hz=20)["dominant_hz"] - 240) <= 0.5
    assert abs(spectrum(ev, fmin_hz=0.5)["dominant_hz"] - 2) <= 0.5


# ---- fundamental frequency (first real-camera UC2 run, 2026-10-15: 480 Hz was reported instead of 240) ----

def test_fundamental_of_the_real_run_peaks():
    # the peaks of that run, strongest first; resolution 5.787 Hz (a 0.17 s recording)
    assert fundamental([480.324, 237.269, 960.648, 717.593, 1440.972], 5.787) == 237.269


@pytest.mark.parametrize("peaks, expected", [
    ([240.0], 240.0),
    ([480.0, 240.0, 720.0], 240.0),  # the lowest peak that explains all others
    ([240.0, 333.0], None),  # two lights that are not related: no fundamental
    ([], None),
])
def test_fundamental_cases(peaks, expected):
    assert fundamental(peaks, 0.5) == expected


def test_spectrum_reports_the_fundamental_when_a_harmonic_is_strongest():
    # 25 % on-time with short edges: 2f is the strongest peak, like on the real camera
    s = spectrum(blinking_light(240, 2.0, np.random.default_rng(4), duty=0.25, jitter_us=50))
    res = s["resolution_hz"]
    assert abs(s["dominant_hz"] - 480) <= res
    assert abs(s["fundamental_hz"] - 240) <= res
    h2 = s["harmonics"][0]  # harmonics are of the fundamental now
    assert h2["k"] == 2 and abs(h2["hz"] - 480) <= 2 * res


def test_spectrum_fundamental_equals_dominant_for_a_symmetric_blink():
    s = spectrum(blinking_light(240, 2.0, np.random.default_rng(5)))
    assert s["fundamental_hz"] == s["dominant_hz"] and abs(s["fundamental_hz"] - 240) <= s["resolution_hz"]


@pytest.mark.parametrize("seed", range(7, 12))
def test_spectrum_finds_the_base_of_a_square_wave(seed):
    # real LED at 10 Hz (2026-10-16): sharp edges give odd harmonics 3f, 5f, 7f ... of the same height, so the 5
    # strongest peaks may not contain 10 Hz (70 Hz was reported). The divisors of the strongest peak find it.
    # fmax_hz=150: the band holds the odd harmonics up to 15f, the largest divisor that is tested (seeds 7 and 8
    # gave 30 Hz and 90 Hz before the fix)
    s = spectrum(blinking_light(10, 2.0, np.random.default_rng(seed), events_per_edge=200, jitter_us=50),
                 fmin_hz=2, fmax_hz=150)
    assert abs(s["fundamental_hz"] - 10) <= s["resolution_hz"]


def test_spectrum_pure_sine_keeps_its_frequency():
    # ON events more likely when sin(2*pi*240*t) > 0, OFF events when < 0: one peak at 240 Hz, no harmonics,
    # so no divisor of 240 Hz is accepted
    rng = np.random.default_rng(8)
    t = rng.integers(0, 2_000_000, 200_000)
    p = rng.random(t.size) < 0.5 + 0.4 * np.sin(2 * np.pi * 240 * t / 1e6)
    s = spectrum(make_events(np.zeros(t.size), np.zeros(t.size), p, t), fmin_hz=2, fmax_hz=2000)
    assert abs(s["dominant_hz"] - 240) <= s["resolution_hz"]
    assert s["fundamental_hz"] == s["dominant_hz"]


# ---- false subharmonics (real 50 Hz LED, 2026-10-17: 150.8 Hz strongest, 10.3 Hz was reported) ----

@pytest.mark.parametrize("seed", range(4))
def test_spectrum_genuine_10hz_square_wave_in_the_wide_band(seed):
    # the band of event_export (2..2000 Hz); jitter 1 ms: the harmonics get weaker, like the real 10 Hz LED
    s = spectrum(blinking_light(10, 2.0, np.random.default_rng(seed), events_per_edge=200, jitter_us=1000),
                 fmin_hz=2, fmax_hz=2000)
    assert abs(s["fundamental_hz"] - 10) <= s["resolution_hz"]


@pytest.mark.parametrize("seed", range(4))
def test_spectrum_true_10hz_is_found_when_a_harmonic_is_strongest(seed):
    # 10.3 Hz: the strongest peak is 5f (51.5 Hz), the divisor 51.5 / 5 has its own harmonics 3f, 7f, ...
    s = spectrum(blinking_light(10.3, 2.0, np.random.default_rng(seed), events_per_edge=200, jitter_us=1000),
                 fmin_hz=2, fmax_hz=2000)
    assert abs(s["dominant_hz"] - 10.3) > 10  # the divisor path is really used
    assert abs(s["fundamental_hz"] - 10.3) <= s["resolution_hz"]


@pytest.mark.parametrize("seed", range(4))
def test_spectrum_50hz_with_a_dominant_third_harmonic(seed):
    # 1/6 on-time: 3f is the strongest peak, 2f and 4f are strong too
    s = spectrum(blinking_light(50, 2.0, np.random.default_rng(seed), events_per_edge=200, jitter_us=300, duty=1 / 6),
                 fmin_hz=2, fmax_hz=2000)
    assert abs(s["dominant_hz"] - 150) <= s["resolution_hz"]
    assert abs(s["fundamental_hz"] - 50) <= s["resolution_hz"]


@pytest.mark.parametrize("seed", range(4))
def test_spectrum_50hz_with_10hz_modulation_is_not_reported_as_10hz(seed):
    # 151 / 15 = 10.07 Hz is near the 10.3 Hz flicker, and its multiples 40, 50, 60 Hz are the 50.3 Hz light and
    # its sidebands: no evidence of its own. A near-50 Hz answer or no answer (unresolved) are both honest.
    s = spectrum(modulated_light(50.3, 10.3, 0.6, 2.0, np.random.default_rng(seed)), fmin_hz=2, fmax_hz=2000)
    res = s["resolution_hz"]
    assert abs(s["dominant_hz"] - 151) <= res
    f0 = s["fundamental_hz"]
    assert f0 is None or abs(f0 - 50.3) <= res
    if f0 is None:
        assert s["harmonics"] == [] and "unresolved" in s["warning"]
    assert len(s["peaks"]) == 5  # the peaks are still all reported


def test_base_of_strongest_needs_evidence_of_its_own():
    # 60 Hz family (3f strongest) and a weaker 12.3 Hz flicker with sidebands 60 +- 12.3 and 180 + 12.3: the divisor
    # 180 / 15 = 12 Hz has a line (12.5 Hz bin) and strong multiples, but they belong to 60 Hz or are sidebands
    family = {60: 60, 120: 90, 180: 100, 240: 80, 300: 50}
    side = {12.3: 30, 47.7: 30, 72.3: 30, 192.3: 30}
    f0, candidates = base_of(line_spectrum({**family, **side}))
    assert f0 is None and 12.0 in candidates  # unresolved: 12 Hz is not proven, and not silently dropped
    # without the sidebands, 60 Hz is the base: 60, 120, 240, 300 are not explained by 180 Hz
    assert base_of(line_spectrum(family)) == (60.0, [])


def test_base_of_strongest_finds_a_true_low_base():
    # a sharp 12 Hz square wave: odd harmonics of the same height, 7f is the strongest
    lines = {12 * n: 50 for n in range(1, 30, 2)}
    lines[84] = 60
    assert base_of(line_spectrum(lines)) == (12.0, [])


# ---- one base for all the lines (real 50 Hz LED, 2026-10-17: a 50.28 Hz carrier with 10.26 Hz sidebands) ----

def sideband_spectrum(fc, fm, resolution, n_carrier=8):
    """Like the real 50 Hz recording: the harmonics m*fc (3fc strongest), a sideband m*fc +- fm next to each of them,
    and the slow flicker fm, 2fm. The lines come every ~fm Hz, but they are not multiples of one base."""
    lines = {fm: 40, 2 * fm: 40}
    for m in range(1, n_carrier + 1):
        lines[m * fc] = 100 if m == 3 else 90
        lines[m * fc - fm] = lines[m * fc + fm] = 60
    return line_spectrum(lines, resolution)


@pytest.mark.parametrize("fc, fm, resolution", [(50.28, 10.26, 0.5131), (50.3, 10.3, 0.5), (40.4, 8.3, 1 / 2.3)])
def test_base_of_strongest_sidebands_are_not_one_base(fc, fm, resolution):
    # 3fc / 15 has a line (the flicker fm) and strong "multiples" (the sidebands), so it was reported (10.262 Hz for
    # the real values). But no single base f puts n*f within half a bin of all these lines -> unresolved, and the
    # low base is still listed, not silently dropped
    f0, candidates = base_of(sideband_spectrum(fc, fm, resolution), resolution)
    assert f0 is None
    assert any(abs(c - 3 * fc / 15) <= resolution for c in candidates)
    assert any(abs(c - fc) <= resolution for c in candidates)


@pytest.mark.parametrize("f, resolution", [(10.3, 0.5), (13.7, 1 / 1.7), (7.9, 0.4)])
def test_base_of_strongest_genuine_comb_between_bins(f, resolution):
    # the base lies between two bins, so every n*f lies between bins too, but each line is still within half a bin
    # of n*f for one common f: the base is found (5f is the strongest line)
    lines = {f * n: 50 for n in range(1, 30, 2)}
    lines[5 * f] = 60
    f0, candidates = base_of(line_spectrum(lines, resolution), resolution)
    assert candidates == [] and abs(f0 - f) <= resolution / 2


@pytest.mark.parametrize("seed", range(3))
def test_spectrum_genuine_13_7hz_in_a_1_7s_recording(seed):
    # another frequency and resolution (1 / 1.7 s): 7f is the strongest peak, the base is between bins
    s = spectrum(blinking_light(13.7, 1.7, np.random.default_rng(seed), events_per_edge=200, jitter_us=300),
                 fmin_hz=2, fmax_hz=2000)
    assert abs(s["fundamental_hz"] - 13.7) <= s["resolution_hz"]


@pytest.mark.parametrize("seed", range(3))
def test_spectrum_40hz_with_8hz_modulation_is_never_8hz(seed):
    s = spectrum(modulated_light(40.4, 8.3, 0.6, 2.3, np.random.default_rng(seed)), fmin_hz=2, fmax_hz=2000)
    f0 = s["fundamental_hz"]
    assert f0 is None or abs(f0 - 40.4) <= s["resolution_hz"]
    if f0 is None:
        assert s["harmonics"] == [] and "unresolved" in s["warning"]
    assert len(s["peaks"]) == 5 and s["peaks"][0]["hz"] == s["dominant_hz"]


def test_spectrum_two_unrelated_families_stay_unresolved():
    # 60 Hz (1/4 on) and 40 Hz (1/6 on): both have 120 Hz as their strongest harmonic. 60 and 40 Hz both explain
    # it, and 20 Hz has no line: no single fundamental
    rng = np.random.default_rng(0)
    ev = np.sort(np.concatenate([blinking_light(60, 2.0, rng, events_per_edge=200, jitter_us=50, duty=0.25),
                                 blinking_light(40, 2.0, rng, events_per_edge=200, jitter_us=50, duty=1 / 6)]),
                 order="t")
    s = spectrum(ev, fmin_hz=2, fmax_hz=2000)
    assert abs(s["dominant_hz"] - 120) <= s["resolution_hz"]
    assert s["fundamental_hz"] is None and s["harmonics"] == []
    assert s["warning"].startswith("the fundamental is unresolved")
    assert s["peaks"][0]["hz"] == s["dominant_hz"]


@pytest.mark.parametrize("seed", range(4))
def test_spectrum_sharp_10hz_in_the_wide_band_is_never_a_wrong_frequency(seed):
    # sharp edges up to 2000 Hz: the strongest peak can be 33f, above the largest divisor 15. Seed 2 gave 30 Hz
    # before; now the answer is 10 Hz or unresolved
    s = spectrum(blinking_light(10, 2.0, np.random.default_rng(seed), events_per_edge=200, jitter_us=50),
                 fmin_hz=2, fmax_hz=2000)
    assert s["fundamental_hz"] is None or abs(s["fundamental_hz"] - 10) <= s["resolution_hz"]


def test_spectrum_warns_about_short_recordings():
    short = spectrum(blinking_light(240, 0.5, np.random.default_rng(6)))
    assert short["warning"] == ("the recording is only 0.50 s long: the frequency resolution is 2.0 Hz; "
                                "record at least 1 s for a precise frequency")
    assert "warning" not in spectrum(blinking_light(240, 2.0, np.random.default_rng(6)))


@pytest.mark.parametrize("seed", range(5))
def test_spectrum_random_noise_has_no_peak(seed):
    # pure random events (no rhythm): the strongest value stays below min_snr = 8
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2_000_000, 20_000)
    s = spectrum(make_events(np.zeros(t.size), np.zeros(t.size), rng.integers(0, 2, t.size), t))
    assert s["dominant_hz"] is None and s["peaks"] == [] and s["harmonics"] == []
    assert s["fundamental_hz"] is None
    assert s["noise_floor"] > 0


def test_spectrum_min_snr_zero_reports_noise_peaks():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2_000_000, 20_000)
    s = spectrum(make_events(np.zeros(t.size), np.zeros(t.size), rng.integers(0, 2, t.size), t), min_snr=0)
    assert s["dominant_hz"] is not None and 0 < s["peaks"][0]["snr"] < 8


def test_spectrum_harmonics_stop_at_fmax():
    s = spectrum(blinking_light(240, 2.0, np.random.default_rng(3)), fmax_hz=1000)
    assert [h["k"] for h in s["harmonics"]] == [2, 3, 4]  # 5 * 240 = 1200 > 1000


@pytest.mark.parametrize("kwargs, msg", [
    ({"bin_us": 0}, "bin_us must be an int"),
    ({"bin_us": 100.0}, "bin_us must be an int"),
    ({"bin_us": 1000, "fmax_hz": 2000}, "Nyquist"),  # 1000 us bins show at most 500 Hz
    ({"fmin_hz": 300, "fmax_hz": 200}, "fmin_hz < fmax_hz"),
    ({"min_snr": -1}, "min_snr must be >= 0"),
])
def test_spectrum_bad_arguments(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        spectrum(make_events([0], [0], [1], [5000]), **kwargs)


def test_spectrum_no_events():
    with pytest.raises(ValueError, match="no events"):
        spectrum(make_events([], [], [], []))


def test_spectrum_too_short():
    # 10 ms of events: resolution 100 Hz, so nothing between 20 and 50 Hz
    with pytest.raises(ValueError, match="too short"):
        spectrum(make_events([0, 0], [0, 0], [1, 0], [0, 9_999]), fmin_hz=20, fmax_hz=50)


def test_spectrum_flat_signal_has_no_peak():
    # one ON and one OFF in the same bin: the signal is 0 everywhere
    s = spectrum(make_events([0, 0], [0, 0], [1, 0], [0, 0]), fmin_hz=0, fmax_hz=5000)
    assert s["dominant_hz"] is None and s["peaks"] == [] and s["harmonics"] == []


# ---- delta reconstruction ----

def square_wave(f_hz, seconds, n_pixels=200, on_per_edge=1, off_per_edge=1):
    """Each of n_pixels sees a light that is bright in the first half of each period: on_per_edge ON events at each
    rising edge, off_per_edge OFF events at each falling edge (10 us apart, no noise)."""
    period = 1e6 / f_hz
    rising = np.arange(0, seconds * 1e6, period)
    px = np.arange(n_pixels)
    rows = []
    for p, edges, n in ((1, rising, on_per_edge), (0, rising + period / 2, off_per_edge)):
        for k in range(n):
            t = np.repeat(edges, n_pixels) + 10 * k
            rows.append((np.tile(px, edges.size), p, t))
    x = np.concatenate([r[0] for r in rows])
    p = np.concatenate([np.full(r[0].size, r[1]) for r in rows])
    t = np.concatenate([r[2] for r in rows]).astype(np.int64)
    ok = t < seconds * 1e6
    return make_events(x[ok], np.full(ok.sum(), 300), p[ok], t[ok])


def test_brightness_curve_of_a_50_hz_square_wave():
    t_s, level = brightness_curve(square_wave(50, 1.0))
    assert t_s.size == level.size == 9901 and t_s[0] == 0.0001  # 100 us bins up to the last event; t = bin end
    full = level[:9800]  # 49 full periods: 50 Hz falls exactly on an FFT bin
    amp = np.abs(np.fft.rfft(full - full.mean()))
    assert np.fft.rfftfreq(full.size, d=100e-6)[np.argmax(amp)] == pytest.approx(50.0)
    # high in the first 10 ms of every 20 ms period, low in the second 10 ms: a step of 1 Δ per active pixel.
    # The leak removes the mean within ~0.2 s, so later periods swing around 0 (about +0.5 / -0.5).
    for k in range(49):
        high = level[k * 200 + 10: k * 200 + 100].mean()
        low = level[k * 200 + 110: k * 200 + 200].mean()
        assert 0.9 < high - low < 1.1
    assert level[:100].mean() > 0.95  # the first period starts from 0
    assert level[9600:9700].mean() == pytest.approx(0.5, abs=0.05)  # the last full period: around 0
    assert level[9700:9800].mean() == pytest.approx(-0.5, abs=0.05)


def test_brightness_curve_of_constant_light_is_flat():
    rng = np.random.default_rng(0)
    n = 20_000  # only noise events, ON and OFF equally likely
    ev = make_events(rng.integers(0, 1280, n), rng.integers(0, 720, n), rng.integers(0, 2, n),
                     rng.integers(0, 1_000_000, n))
    _, level = brightness_curve(ev)
    assert np.abs(level).max() < 0.01  # 100 times smaller than one step of the square wave


def test_the_leak_stops_the_drift_of_unequal_thresholds():
    ev = square_wave(50, 1.0, on_per_edge=2, off_per_edge=1)  # each blink: +2 Δ, -1 Δ
    _, plain = brightness_curve(ev, leak_tau_s=None)
    _, leaky = brightness_curve(ev, leak_tau_s=0.2)
    assert plain[-1] > 45  # +1 Δ per period, 50 periods: it drifts away
    assert leaky.max() < 12  # it stays near a level: about 1 / (1 - exp(-20 ms / 0.2 s)) ~ 10.5


def test_brightness_curve_drops_hot_pixels():
    ev = square_wave(50, 1.0)
    hot = make_events(np.full(30_000, 5), np.full(30_000, 5), np.ones(30_000), np.arange(30_000) * 30)  # 30 kHz
    _, clean = brightness_curve(ev)
    _, mixed = brightness_curve(np.concatenate([ev, hot]))
    assert np.allclose(clean, mixed)


@pytest.mark.parametrize("kwargs, msg", [({"bin_us": 0}, "bin_us"), ({"bin_us": 1.5}, "bin_us"),
                                         ({"leak_tau_s": 0}, "leak_tau_s"), ({"leak_tau_s": True}, "leak_tau_s")])
def test_brightness_curve_bad_arguments(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        brightness_curve(square_wave(50, 0.1), **kwargs)


def test_brightness_curve_no_events():
    with pytest.raises(ValueError, match="no events"):
        brightness_curve(make_events([], [], [], []))


def test_delta_image_is_positive_only_where_the_on_events_are():
    rng = np.random.default_rng(1)
    n = 5_000  # ON events only, in x 100..199, y 40..79
    ev = make_events(rng.integers(100, 200, n), rng.integers(40, 80, n), np.ones(n), rng.integers(0, 1_000_000, n))
    img = delta_image(ev, 1280, 720)
    assert img.shape == (180, 320) and img.sum() == n
    region = np.zeros_like(img, dtype=bool)
    region[10:20, 25:50] = True  # the region in 4 x 4 bins
    assert (img[region] > 0).all() and (img[~region] == 0).all()


def test_delta_image_signs_time_window_and_hot_pixels():
    # pixel (0, 0): ON at 10 us and 20 us; pixel (8, 4): OFF at 30 us; pixel (100, 100): 3 ON events
    ev = make_events([0, 0, 8, 100, 100, 100], [0, 0, 4, 100, 100, 100], [1, 1, 0, 1, 1, 1], [10, 20, 30, 40, 50, 60])
    # in 125 us the limit is 20 kHz * 125 us = 2.5 events: (100, 100) is hot, (0, 0) is not
    img = delta_image(ev, 1280, 720, bin_px=4, seconds=125e-6)
    assert img[0, 0] == 2 and img[1, 2] == -1 and img[25, 25] == 0
    assert delta_image(ev, 1280, 720, t0_us=15, t1_us=30, seconds=125e-6)[0, 0] == 1  # only the event at 20 us
    assert delta_image(ev, 1280, 720, bin_px=1, seconds=1.0)[100, 100] == 3  # not hot when the recording is 1 s


def test_time_image_is_the_mean_event_time_per_block():
    ev = make_events([0, 0, 40], [0, 0, 0], [1, 0, 1], [100_000, 300_000, 900_000])
    img = time_image(ev, 1280, 720, seconds=1.0)
    assert img.shape == (180, 320)
    assert img[0, 0] == pytest.approx(0.2) and img[0, 10] == pytest.approx(0.9)
    assert np.isnan(img[5, 5]) and np.count_nonzero(~np.isnan(img)) == 2


@pytest.mark.parametrize("width, height, bin_px", [(0, 720, 4), (1280, 720, 0), (1280.0, 720, 4)])
def test_image_bad_arguments(width, height, bin_px):
    with pytest.raises(ValueError, match="must be an int"):
        delta_image(make_events([], [], [], []), width, height, bin_px=bin_px)


# ---- edge windows ----

def region_square_wave(f_hz=50, seconds=0.2, seed=2):
    """A 50 Hz light that hits only the region x 100..199, y 40..79 (one event per pixel per edge, 200 us jitter),
    plus ON/OFF noise all over the sensor."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(100, 200), np.arange(40, 80))
    xs, ys = xs.ravel(), ys.ravel()
    period = 1e6 / f_hz
    x, y, p, t = [], [], [], []
    for start in np.arange(0, seconds * 1e6, period):
        for edge, pol in ((start + 4_000, 1), (start + 4_000 + period / 2, 0)):  # rise at 4 ms, fall at 14 ms
            x.append(xs), y.append(ys), p.append(np.full(xs.size, pol))
            t.append(edge + rng.integers(0, 200, xs.size))
    n = 20_000
    x.append(rng.integers(0, 1280, n)), y.append(rng.integers(0, 720, n)), p.append(rng.integers(0, 2, n))
    t.append(rng.integers(0, int(seconds * 1e6), n))
    t = np.concatenate(t)
    keep = t < seconds * 1e6
    return make_events(*(np.concatenate(a)[keep] for a in (x, y, p)), t[keep])


def test_edge_windows_find_the_rise_and_the_fall_of_each_period():
    w = edge_windows(region_square_wave(), 50.0)
    assert [k for _, _, k in w] == ["rise", "fall"] * 5  # the first 5 full periods
    for t0, t1, _ in w:
        assert t1 - t0 == 5_000  # 1/4 of 20 ms
    centres = [(t0 + t1) / 2 for t0, t1, _ in w]
    assert all(abs(c - (4_100 + 10_000 * i)) < 200 for i, c in enumerate(centres))  # edges at 4 ms, 14 ms, ...
    assert all(abs(b - a - 10_000) < 200 for a, b in zip(centres, centres[1:], strict=False))  # rise -> fall: 10 ms
    assert len(edge_windows(region_square_wave(), 50.0, n_periods=2)) == 4


def test_edge_delta_images_show_where_the_light_went_up_and_down():
    img = edge_delta_images(region_square_wave(), 1280, 720, 50.0)
    region = np.zeros((180, 320), dtype=bool)
    region[10:20, 25:50] = True
    # 16 pixels per bin, one event each per edge, 5 edges: +80 / -80 in the region; only a little noise outside
    assert (img["rise"][region] >= 70).all() and (img["fall"][region] <= -70).all()
    assert np.abs(img["rise"][~region]).max() <= 3 and np.abs(img["fall"][~region]).max() <= 3


def test_no_f0_gives_no_windows_and_empty_images():
    ev = region_square_wave()
    assert edge_windows(ev, None) == []
    img = edge_delta_images(ev, 1280, 720, None)
    assert not img["rise"].any() and not img["fall"].any()


def test_a_recording_shorter_than_one_period_gives_no_windows():
    assert edge_windows(region_square_wave(seconds=0.2), 4.0) == []  # one period of 4 Hz = 0.25 s


@pytest.mark.parametrize("kwargs, msg", [({"f0_hz": 0}, "f0_hz"), ({"f0_hz": math.inf}, "f0_hz"),
                                         ({"f0_hz": 50, "n_periods": 0}, "n_periods")])
def test_edge_windows_bad_arguments(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        edge_windows(square_wave(50, 0.1), **kwargs)
