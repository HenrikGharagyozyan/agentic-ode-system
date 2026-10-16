"""Safety limits for the IDS event camera. Pure Python: no hardware imports, so it can be tested anywhere."""

from __future__ import annotations

import numbers

BIAS_NAMES = ("bias_diff", "bias_diff_on", "bias_diff_off", "bias_fo", "bias_hpf", "bias_refr")
BIAS_MIN = -100
BIAS_MAX = 150

MAX_RECORD_S = 5.0
MIN_RECORD_S = 0.05
WARMUP_S = 0.5
MAX_EVENTS = 30_000_000
MAX_MASK_SLOTS = 64
MAX_PIXEL_RATE_HZ = 20_000  # ev/s; a real pixel cannot fire faster (dead time ~73 us), so faster pixels are defects
SENSOR_W = 1280
SENSOR_H = 720


def _is_int(v) -> bool:
    # numbers.Integral covers int and NumPy ints (np.uint16 from event arrays); bool is also Integral, so exclude it
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def validate_bias(name: str, value: int) -> int:
    """Return value if name is a known bias and value is an int offset in [BIAS_MIN, BIAS_MAX]."""
    if name not in BIAS_NAMES:
        raise ValueError(f"unknown bias {name!r}; valid names: {', '.join(BIAS_NAMES)}")
    if not _is_int(value):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    if not BIAS_MIN <= value <= BIAS_MAX:
        raise ValueError(f"{name} must be between {BIAS_MIN} and {BIAS_MAX}, got {value}")
    return int(value)


def validate_duration(s: float) -> float:
    """Return s as a float if it is a number of seconds in [MIN_RECORD_S, MAX_RECORD_S]."""
    if isinstance(s, bool) or not isinstance(s, (int, float)):
        raise ValueError(f"duration must be a number of seconds, got {type(s).__name__}")
    # NaN fails every comparison and inf is above the maximum, so this one check also rejects both
    if not MIN_RECORD_S <= s <= MAX_RECORD_S:
        raise ValueError(f"duration must be between {MIN_RECORD_S} and {MAX_RECORD_S} s, got {s}")
    return float(s)


def validate_pixel(x: int, y: int) -> tuple[int, int]:
    """Return (x, y) if it is a sensor pixel: 0 <= x < SENSOR_W and 0 <= y < SENSOR_H."""
    if not (_is_int(x) and _is_int(y)):
        raise ValueError(f"pixel coordinates must be ints, got x={x!r}, y={y!r}")
    if not (0 <= x < SENSOR_W and 0 <= y < SENSOR_H):
        raise ValueError(f"pixel ({x}, {y}) is outside the sensor (x 0..{SENSOR_W - 1}, y 0..{SENSOR_H - 1})")
    return int(x), int(y)


def validate_roi(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) if w, h >= 1 and the rectangle lies fully inside the sensor."""
    if not all(_is_int(v) for v in (x, y, w, h)):
        raise ValueError(f"ROI values must be ints, got x={x!r}, y={y!r}, w={w!r}, h={h!r}")
    if w < 1 or h < 1:
        raise ValueError(f"ROI width and height must be at least 1, got w={w}, h={h}")
    if x < 0 or y < 0 or x + w > SENSOR_W or y + h > SENSOR_H:
        raise ValueError(f"ROI x={x}, y={y}, w={w}, h={h} does not fit inside the {SENSOR_W}x{SENSOR_H} sensor")
    return int(x), int(y), int(w), int(h)
