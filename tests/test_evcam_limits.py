"""Safety limits of the event camera (no hardware needed)."""

import math

import numpy as np
import pytest

from agentic_ode.hardware.devices.evcam import limits
from agentic_ode.hardware.devices.evcam.limits import validate_bias, validate_duration, validate_pixel, validate_roi


def test_constants():
    assert len(limits.BIAS_NAMES) == 6 and "bias_diff_off" in limits.BIAS_NAMES
    assert (limits.BIAS_MIN, limits.BIAS_MAX) == (-100, 150)
    assert (limits.MIN_RECORD_S, limits.MAX_RECORD_S, limits.WARMUP_S) == (0.05, 5.0, 0.5)
    assert limits.MAX_EVENTS == 30_000_000
    assert (limits.MAX_MASK_SLOTS, limits.SENSOR_W, limits.SENSOR_H) == (64, 1280, 720)


def test_bias_ok_at_both_edges():
    assert validate_bias("bias_fo", -100) == -100
    assert validate_bias("bias_diff_off", 150) == 150


@pytest.mark.parametrize("name, value, msg", [
    ("bias_foo", 0, "unknown bias"),
    ("bias_diff_off", -101, "between -100 and 150"),
    ("bias_diff_off", 151, "between -100 and 150"),
    ("bias_diff_off", 20.0, "must be an int"),
    ("bias_diff_off", True, "must be an int"),
    ("bias_diff_off", "20", "must be an int"),
])
def test_bias_rejected(name, value, msg):
    with pytest.raises(ValueError, match=msg):
        validate_bias(name, value)


def test_duration_ok_and_returns_float():
    assert validate_duration(0.05) == 0.05
    assert validate_duration(5.0) == 5.0
    out = validate_duration(1)
    assert out == 1.0 and isinstance(out, float)


@pytest.mark.parametrize("s", [0.049, 5.01, -1.0, math.nan, math.inf])
def test_duration_out_of_range(s):
    with pytest.raises(ValueError, match="between 0.05 and 5.0"):
        validate_duration(s)


@pytest.mark.parametrize("s", [True, "1.0", None])
def test_duration_not_a_number(s):
    with pytest.raises(ValueError, match="number of seconds"):
        validate_duration(s)


def test_pixel_ok_at_corners():
    assert validate_pixel(0, 0) == (0, 0)
    assert validate_pixel(1279, 719) == (1279, 719)


@pytest.mark.parametrize("x, y", [(-1, 0), (1280, 0), (0, -1), (0, 720)])
def test_pixel_outside_sensor(x, y):
    with pytest.raises(ValueError, match="outside the sensor"):
        validate_pixel(x, y)


@pytest.mark.parametrize("x, y", [(1.0, 0), (0, True), ("1", 0)])
def test_pixel_not_int(x, y):
    with pytest.raises(ValueError, match="must be ints"):
        validate_pixel(x, y)


def test_roi_ok_at_edges():
    assert validate_roi(0, 0, 1280, 720) == (0, 0, 1280, 720)
    assert validate_roi(1279, 719, 1, 1) == (1279, 719, 1, 1)


@pytest.mark.parametrize("roi, msg", [
    ((0, 0, 0, 10), "at least 1"),
    ((0, 0, 10, -5), "at least 1"),
    ((1200, 0, 81, 10), "does not fit"),   # 1200 + 81 = 1281 > 1280
    ((0, 700, 10, 21), "does not fit"),    # 700 + 21 = 721 > 720
    ((-1, 0, 10, 10), "does not fit"),
    ((0, 0, 10.0, 10), "must be ints"),
])
def test_roi_rejected(roi, msg):
    with pytest.raises(ValueError, match=msg):
        validate_roi(*roi)


def test_numpy_ints_accepted_and_returned_as_python_int():
    x, y = np.array([1053, 704], dtype=np.uint16)
    out = validate_pixel(x, y)
    assert out == (1053, 704) and all(type(v) is int for v in out)
    assert type(validate_bias("bias_fo", np.int64(5))) is int
