"""Event analysis with synthetic events (no hardware needed)."""

import math

import numpy as np
import pytest

from agentic_ode.hardware.devices.evcam.analysis import (
    brightness_curve,
    delta_image,
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
