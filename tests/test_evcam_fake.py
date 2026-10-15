"""FakeEventCamera (no hardware needed)."""

import inspect

import numpy as np
import pytest

from agentic_ode.hardware.evcam import fake
from agentic_ode.hardware.evcam.analysis import hot_pixels, spectrum
from agentic_ode.hardware.evcam.connector import EVENT_DTYPE, EventCamera
from agentic_ode.hardware.evcam.fake import FakeEventCamera


def public_methods(cls):
    """{method name: [(parameter name, default, kind), ...]} of every public method, plus __enter__/__exit__."""
    names = [n for n in dir(cls) if callable(getattr(cls, n)) and (not n.startswith("_") or n in ("__enter__",
                                                                                                   "__exit__"))]
    return {n: [(p.name, p.default, p.kind) for p in inspect.signature(getattr(cls, n)).parameters.values()]
            for n in names}


def count_at(events, x, y):
    return int(np.sum((events["x"] == x) & (events["y"] == y)))


# ---- same interface as the real camera ----

def test_same_public_methods_as_the_real_camera():
    assert public_methods(FakeEventCamera) == public_methods(EventCamera)


def test_needs_open():
    cam = FakeEventCamera()
    for call in (cam.info, cam.get_biases, cam.bias_ranges, cam.clear_mask, cam.clear_roi, cam.reapply,
                 lambda: cam.record(1.0), lambda: cam.set_bias("bias_fo", 0), lambda: cam.mask_pixels([(0, 0)]),
                 lambda: cam.set_roi(0, 0, 10, 10)):
        with pytest.raises(RuntimeError, match="not open"):
            call()


@pytest.mark.parametrize("kwargs, msg", [
    ({"scene": "lamp"}, "unknown scene"),
    ({"pwm_hz": 0}, "pwm_hz must be in"),
    ({"pwm_hz": 6000}, "pwm_hz must be in"),
])
def test_bad_arguments(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        FakeEventCamera(**kwargs)


def test_info_says_simulated():
    with FakeEventCamera() as cam:
        info = cam.info()
    assert info["serial"] == "FAKE" and "simulated" in info["sensor"]
    assert (info["width"], info["height"]) == (1280, 720)


def test_record_format_and_size():
    with FakeEventCamera("pwm") as cam:
        ev = cam.record(1.0)
    assert ev.dtype == EVENT_DTYPE
    assert ev["t"].min() >= 0 and ev["t"].max() < 1_000_000 and np.all(np.diff(ev["t"]) >= 0)
    assert 60_000 < ev.size < 80_000  # 2 x 30,000 hot + ~2,000 noise + ~9,600 blink


def test_same_seed_same_events():
    a = FakeEventCamera("pwm", seed=5).open().record(0.5)
    b = FakeEventCamera("pwm", seed=5).open().record(0.5)
    c = FakeEventCamera("pwm", seed=6).open().record(0.5)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


# ---- hot pixels ----

def test_hot_pixels_are_found_and_disappear_when_masked():
    with FakeEventCamera("dark") as cam:
        found = [(h["x"], h["y"]) for h in hot_pixels(cam.record(1.0), 1.0)]
        assert sorted(found) == sorted(fake.HOT_PIXELS)
        assert cam.mask_pixels(found) == found
        ev = cam.record(1.0)
    assert hot_pixels(ev, 1.0) == []
    assert all(count_at(ev, x, y) == 0 for x, y in fake.HOT_PIXELS)


def test_clear_mask_brings_the_hot_pixels_back():
    with FakeEventCamera("dark") as cam:
        cam.mask_pixels(fake.HOT_PIXELS)
        cam.clear_mask()
        ev = cam.record(0.5)
    assert cam.state["mask"] == [] and all(count_at(ev, x, y) > 0 for x, y in fake.HOT_PIXELS)


def test_mask_slots_are_limited():
    with FakeEventCamera() as cam:
        cam.mask_pixels([(i, 0) for i in range(60)])
        with pytest.raises(ValueError, match="only 4 of 64 mask slots are free"):
            cam.mask_pixels([(i, 1) for i in range(5)])
        assert len(cam.state["mask"]) == 60  # nothing was masked


# ---- spectrum of the scenes ----

@pytest.mark.parametrize("pwm_hz", [240.0, 100.0])
def test_pwm_frequency_is_found(pwm_hz):
    with FakeEventCamera("pwm", pwm_hz=pwm_hz) as cam:
        cam.mask_pixels(fake.HOT_PIXELS)
        s = spectrum(cam.record(2.0))
    assert abs(s["dominant_hz"] - pwm_hz) <= s["resolution_hz"]
    h2 = s["harmonics"][0]
    assert h2["k"] == 2 and abs(h2["hz"] - 2 * pwm_hz) <= 2 * s["resolution_hz"] and h2["snr"] >= 8  # 40 % on: 2f


def test_pwm_is_found_with_the_hot_pixels_unmasked():
    with FakeEventCamera("pwm") as cam:
        s = spectrum(cam.record(2.0))
    assert abs(s["dominant_hz"] - 240) <= s["resolution_hz"]


@pytest.mark.parametrize("scene", ["steady", "dark"])
def test_no_peak_without_a_blinking_light(scene):
    with FakeEventCamera(scene) as cam:
        s = spectrum(cam.record(2.0))
    assert s["dominant_hz"] is None and s["peaks"] == []


def test_roi_outside_the_light_hides_the_peak():
    with FakeEventCamera("pwm") as cam:
        cam.set_roi(0, 0, 400, 200)  # the lit square is at x 540..739, y 260..459
        s = spectrum(cam.record(2.0))
    assert s["dominant_hz"] is None


# ---- biases, noise and ROI ----

def test_noise_falls_when_the_biases_go_up():
    with FakeEventCamera("dark") as cam:
        cam.mask_pixels(fake.HOT_PIXELS)
        before = cam.record(2.0)
        cam.set_bias("bias_diff_off", 40)  # only OFF noise should fall
        off_only = cam.record(2.0)
        cam.set_bias("bias_diff_on", 40)
        both = cam.record(2.0)
    on = [int(np.sum(e["p"] == 1)) for e in (before, off_only, both)]
    off = [int(np.sum(e["p"] == 0)) for e in (before, off_only, both)]
    assert off[1] < off[0] / 8  # 2 ** (40 / 10) = 16 times less
    assert 0.8 < on[1] / on[0] < 1.2
    assert on[2] < on[0] / 8


def test_steady_has_more_noise_than_dark():
    dark = FakeEventCamera("dark").open().record(1.0)
    steady = FakeEventCamera("steady").open().record(1.0)
    hot = 2 * fake.HOT_RATE_HZ
    assert steady.size - hot > 2 * (dark.size - hot)  # about 6,000 vs 2,000 noise events


def test_set_bias_reads_back_and_checks_the_range():
    with FakeEventCamera() as cam:
        assert cam.set_bias("bias_diff_off", 20) == 20
        assert cam.get_biases()["bias_diff_off"] == 20
        with pytest.raises(ValueError, match="recommended range -25..23"):
            cam.set_bias("bias_diff", 50)  # inside limits.py, outside the camera's recommended range
        with pytest.raises(ValueError, match="unknown bias"):
            cam.set_bias("bias_foo", 0)
        assert cam.bias_ranges()["bias_hpf"] == {"recommended": (0, 120), "allowed": (0, 255)}


def test_roi_keeps_only_events_inside():
    with FakeEventCamera("pwm") as cam:
        cam.set_roi(0, 0, 640, 360)
        ev = cam.record(1.0)
        cam.clear_roi()
        after = cam.record(1.0)
    assert ev.size > 0 and np.all((ev["x"] < 640) & (ev["y"] < 360))
    assert np.any((after["x"] >= 640) | (after["y"] >= 360))


# ---- close, open, reapply ----

def test_reopen_forgets_and_reapply_restores():
    cam = FakeEventCamera("dark").open()
    cam.set_bias("bias_diff_off", 20)
    cam.mask_pixels(fake.HOT_PIXELS)
    cam.set_roi(0, 0, 1280, 720)
    cam.close()
    cam.open()
    assert cam.get_biases()["bias_diff_off"] == 0  # the device forgot ...
    assert all(count_at(cam.record(0.2), x, y) > 0 for x, y in fake.HOT_PIXELS)
    assert cam.state == {"biases": {"bias_diff_off": 20}, "mask": fake.HOT_PIXELS, "roi": (0, 0, 1280, 720)}
    cam.reapply()  # ... and reapply() sets it again
    assert cam.get_biases()["bias_diff_off"] == 20
    assert all(count_at(cam.record(0.2), x, y) == 0 for x, y in fake.HOT_PIXELS)
    cam.close()
