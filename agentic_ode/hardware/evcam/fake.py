"""FakeEventCamera: the same public methods as EventCamera, but the events are made up. No hardware, no SDK.

Used to build and test the MCP server and the agent without the camera. Scenes:
- "dark":   noise only (plus the 2 hot pixels)
- "pwm":    a light that blinks at pwm_hz (default 240 Hz, like the phone screen) in a 200x200 square in the centre
- "steady": constant light: more noise, but no rhythm
The numbers are small on purpose (about 2,000 noise events per second, 60,000 from the hot pixels, 9,600 from the
blinking light), so tests are fast.
"""

from __future__ import annotations

import logging

import numpy as np

from agentic_ode.hardware.evcam import limits
from agentic_ode.hardware.evcam.connector import EVENT_DTYPE

log = logging.getLogger(__name__)

SCENES = ("dark", "pwm", "steady")
HOT_PIXELS = [(1053, 704), (926, 151)]  # the 2 defect pixels of the real camera
HOT_RATE_HZ = 30_000  # per hot pixel: above the 20,000 of analysis.hot_pixels() (the real ones fire ~4.5 Mev/s)
NOISE_HZ = 1_000  # ON noise events per second over the whole sensor at bias 0; the same again for OFF
NOISE_MAX_FACTOR = 8  # negative biases raise the noise at most 8 times, so the arrays stay small
STEADY_FACTOR = 3  # "steady": constant light gives 3 times the noise, but no rhythm
PWM_DUTY = 0.4  # the light is on 40 % of each period: not 50 %, so 2f also shows (50 % gives only odd harmonics)
PWM_JITTER = 0.07  # spread of the events at an edge, as a part of the period (the light does not switch at once)
PWM_EVENTS_PER_EDGE = 20
PWM_SQUARE = (540, 260, 200, 200)  # x, y, w, h of the lit area
MAX_PWM_HZ = 5_000  # = the Nyquist limit of analysis.spectrum() with its default bin of 100 us
# recommended and allowed range of each bias, as the real camera reports them (docs/hw3-hardware.md)
BIAS_RANGES = {
    "bias_diff": {"recommended": (-25, 23), "allowed": (-150, 200)},
    "bias_diff_on": {"recommended": (-85, 140), "allowed": (-150, 200)},
    "bias_diff_off": {"recommended": (-35, 190), "allowed": (-150, 200)},
    "bias_fo": {"recommended": (-35, 55), "allowed": (-150, 200)},
    "bias_hpf": {"recommended": (0, 120), "allowed": (0, 255)},
    "bias_refr": {"recommended": (-20, 235), "allowed": (-50, 255)},
}


def _make_events(x, y, p, t) -> np.ndarray:
    ev = np.zeros(len(t), dtype=EVENT_DTYPE)
    ev["x"], ev["y"], ev["p"], ev["t"] = x, y, p, t
    return ev


class FakeEventCamera:
    """Same public methods, checks and error messages as EventCamera (connector.py), with made-up events.

    Choose the scene when you create it. `seed` makes the events reproducible. Like the real camera, the fake
    device forgets all settings when it is reopened; self.state remembers them for reapply().
    """

    def __init__(self, scene: str = "dark", pwm_hz: float = 240.0, seed: int = 0):
        if scene not in SCENES:
            raise ValueError(f"unknown scene {scene!r}; valid scenes: {', '.join(SCENES)}")
        if not 0 < pwm_hz <= MAX_PWM_HZ:
            raise ValueError(f"pwm_hz must be in (0, {MAX_PWM_HZ}], got {pwm_hz}")
        self.scene, self.pwm_hz = scene, float(pwm_hz)
        self.state: dict = {"biases": {}, "mask": [], "roi": None}
        self._dev: dict | None = None  # the settings of the fake device while it is open; None = closed
        self._rng = np.random.default_rng(seed)

    # ---- open / close ----

    def open(self) -> FakeEventCamera:
        """Open the fake camera: all device settings start at their defaults, like on the real camera. Instant
        (no settle phase). Calling open() on an open camera does nothing."""
        if self._dev is not None:
            return self
        self._dev = {"biases": dict.fromkeys(limits.BIAS_NAMES, 0), "mask": [], "roi": None}
        log.info("fake camera opened: %s", self.info())
        return self

    def close(self) -> None:
        """Close the fake camera. self.state is kept for reapply()."""
        if self._dev is None:
            return
        self._dev = None
        log.info("fake camera closed")

    def __enter__(self) -> FakeEventCamera:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def _require_open(self) -> dict:
        if self._dev is None:
            raise RuntimeError("camera is not open; call open() first")
        return self._dev

    # ---- information ----

    def info(self) -> dict:
        self._require_open()
        return {
            "serial": "FAKE",
            "sensor": "IMX636 (simulated)",  # never mistaken for real data in a report
            "width": limits.SENSOR_W,
            "height": limits.SENSOR_H,
            "biases": self.get_biases(),
            "temperature_c": 35,
        }

    # ---- biases ----

    def get_biases(self) -> dict[str, int]:
        return dict(self._require_open()["biases"])

    def bias_ranges(self) -> dict[str, dict]:
        """Recommended and allowed range of each bias, as the real camera reports them."""
        self._require_open()
        return {name: dict(BIAS_RANGES[name]) for name in limits.BIAS_NAMES}

    def set_bias(self, name: str, value: int) -> int:
        """Set one bias and return the value read back."""
        value = limits.validate_bias(name, value)  # known name, int, inside our own limits
        dev = self._require_open()
        lo, hi = BIAS_RANGES[name]["recommended"]
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be in the camera's recommended range {lo}..{hi}, got {value}")
        dev["biases"][name] = value
        self.state["biases"][name] = value
        log.info("bias %s = %d", name, value)
        return value

    # ---- pixel mask ----

    def mask_pixels(self, pixels: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Mask pixels (they stop sending events), one pixel per slot. Returns the newly masked pixels.

        Pixels that are already masked are skipped. If the new pixels do not fit in the free slots,
        nothing is masked and ValueError is raised.
        """
        dev = self._require_open()
        new = []
        for x, y in pixels:
            p = limits.validate_pixel(x, y)
            if p not in self.state["mask"] and p not in new:
                new.append(p)
        free = limits.MAX_MASK_SLOTS - len(self.state["mask"])
        if len(new) > free:
            raise ValueError(f"{len(new)} new pixels but only {free} of {limits.MAX_MASK_SLOTS} mask slots are free")
        dev["mask"].extend(new)
        self.state["mask"].extend(new)
        log.info("masked %d new pixels (%d slots used)", len(new), len(self.state["mask"]))
        return new

    def clear_mask(self) -> None:
        self._require_open()["mask"] = []
        self.state["mask"] = []
        log.info("pixel mask cleared")

    # ---- region of interest ----

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        """Only pixels inside the window send events (at most 1 window)."""
        x, y, w, h = limits.validate_roi(x, y, w, h)
        self._require_open()["roi"] = (x, y, w, h)
        self.state["roi"] = (x, y, w, h)
        log.info("ROI set to x=%d y=%d w=%d h=%d", x, y, w, h)

    def clear_roi(self) -> None:
        self._require_open()["roi"] = None
        self.state["roi"] = None
        log.info("ROI cleared")

    # ---- recording ----

    def record(self, seconds: float) -> np.ndarray:
        """Make `seconds` of events for the scene. Masked pixels and pixels outside the ROI send nothing.

        Returns the same structured array as EventCamera.record(): fields x, y, p, t (t in us, from 0).
        """
        seconds = limits.validate_duration(seconds)
        dev = self._require_open()
        t_max = round(seconds * 1e6)
        parts = [self._noise(dev, t_max), self._hot_pixels(t_max)]
        if self.scene == "pwm":
            parts.append(self._pwm(t_max))
        events = np.concatenate(parts)
        events = events[self._visible(dev, events)]
        events = np.sort(events, order="t")[: limits.MAX_EVENTS]
        log.info("recorded %.2f s of fake camera time: %d events (%.3f Mev/s)",
                 seconds, events.size, events.size / seconds / 1e6)
        return events

    def _noise(self, dev: dict, t_max: int) -> np.ndarray:
        """Random events anywhere on the sensor. The ON noise halves for every +10 of bias_diff_on, the OFF noise
        for every +10 of bias_diff_off (the real camera falls in a similar way, docs/hw3-hardware.md)."""
        scene_factor = STEADY_FACTOR if self.scene == "steady" else 1
        parts = []
        for p, bias in ((1, "bias_diff_on"), (0, "bias_diff_off")):
            rate = NOISE_HZ * scene_factor * min(2.0 ** (-dev["biases"][bias] / 10), NOISE_MAX_FACTOR)
            n = self._rng.poisson(rate * t_max / 1e6)  # a random count around rate * seconds
            parts.append(_make_events(self._rng.integers(0, limits.SENSOR_W, n),
                                      self._rng.integers(0, limits.SENSOR_H, n),
                                      np.full(n, p), self._rng.integers(0, t_max, n)))
        return np.concatenate(parts)

    def _hot_pixels(self, t_max: int) -> np.ndarray:
        """HOT_RATE_HZ random events per second from each hot pixel, ON or OFF at random."""
        counts = self._rng.poisson(HOT_RATE_HZ * t_max / 1e6, len(HOT_PIXELS))
        x = np.repeat([x for x, _ in HOT_PIXELS], counts)
        y = np.repeat([y for _, y in HOT_PIXELS], counts)
        return _make_events(x, y, self._rng.integers(0, 2, x.size), self._rng.integers(0, t_max, x.size))

    def _pwm(self, t_max: int) -> np.ndarray:
        """The blinking light: at each rising edge PWM_EVENTS_PER_EDGE ON events, PWM_DUTY of a period later the
        same number of OFF events (the falling edge), spread by PWM_JITTER of a period, inside PWM_SQUARE."""
        period = 1e6 / self.pwm_hz  # us
        rising = np.arange(0, t_max, period)  # times of the rising edges
        t_on = np.repeat(rising, PWM_EVENTS_PER_EDGE) + self._rng.normal(0, PWM_JITTER * period,
                                                                          rising.size * PWM_EVENTS_PER_EDGE)
        t_off = t_on + PWM_DUTY * period
        t = np.concatenate([t_on, t_off])
        p = np.concatenate([np.ones(t_on.size), np.zeros(t_off.size)])
        keep = (t >= 0) & (t < t_max)  # the spread can push an event outside the recording
        x0, y0, w, h = PWM_SQUARE
        x = self._rng.integers(x0, x0 + w, t.size)
        y = self._rng.integers(y0, y0 + h, t.size)
        return _make_events(x[keep], y[keep], p[keep], t[keep].astype(np.int64))

    @staticmethod
    def _visible(dev: dict, events: np.ndarray) -> np.ndarray:
        """True for each event that the device lets through: not a masked pixel and inside the ROI (if any)."""
        keep = np.ones(events.size, dtype=bool)
        for x, y in dev["mask"]:
            keep &= (events["x"] != x) | (events["y"] != y)
        if dev["roi"] is not None:
            x, y, w, h = dev["roi"]
            keep &= (events["x"] >= x) & (events["x"] < x + w) & (events["y"] >= y) & (events["y"] < y + h)
        return keep

    # ---- state ----

    def reapply(self) -> None:
        """Set self.state on the fake device again (it forgets all settings when it is reopened)."""
        self._require_open()
        biases, mask, roi = dict(self.state["biases"]), list(self.state["mask"]), self.state["roi"]
        for name, value in biases.items():
            self.set_bias(name, value)
        self.state["mask"] = []  # the slots on the reopened device are empty; fill them from slot 0 again
        if mask:
            self.mask_pixels(mask)
        if roi is not None:
            self.set_roi(*roi)
        log.info("state reapplied: %s", self.state)
