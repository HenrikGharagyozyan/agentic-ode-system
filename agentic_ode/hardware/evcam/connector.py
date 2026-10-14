"""EventCamera: a small, safe connector to the IDS event camera .

The SDK (metavision_hal, metavision_core) is imported inside open(), so this module can be imported on a machine
without the camera. Facts about the device: docs/hw3-hardware.md.
"""

from __future__ import annotations

import logging

import numpy as np

from agentic_ode.hardware.evcam import limits

log = logging.getLogger(__name__)

DELTA_T_US = 10_000  # length of one EventsIterator slice in us (10 ms)
# on-chip filters that change the event stream; open() turns all of them off so measurements are raw
FILTERS = {
    "antiflicker": "get_i_antiflicker_module",
    "erc": "get_i_erc_module",
    "event_trail_filter": "get_i_event_trail_filter_module",
    "digital_crop": "get_i_digital_crop",
}


class EventCamera:
    """Open/close the camera, read and set biases, mask pixels, set an ROI and record events.

    self.state remembers what we set (biases, masked pixels, ROI). The camera forgets all settings when it is
    reopened; call reapply() after open() to set them again.
    """

    def __init__(self):
        self.state: dict = {"biases": {}, "mask": [], "roi": None}
        self._dev = None
        self._hal = None  # the metavision_hal module, set by open()
        self._events_iterator = None  # metavision_core's EventsIterator class, set by open()
        self._t_end = 0  # stream time (us) where the last recording stopped

    # ---- open / close ----

    def open(self) -> EventCamera:
        """Open the camera and turn off all on-chip filters. Calling open() on an open camera does nothing."""
        if self._dev is not None:
            return self
        import metavision_hal  # imported here: the SDK only exists where the `evcam` environment is set
        from metavision_core.event_io import EventsIterator

        dev = metavision_hal.DeviceDiscovery.open("")
        if dev is None:
            raise RuntimeError("no event camera found (run `evcam` first; close metavision_viewer)")
        self._hal, self._events_iterator, self._dev = metavision_hal, EventsIterator, dev
        self._t_end = 0  # a new device starts its stream time at 0
        try:
            self._disable_filters()
        except Exception:
            self.close()
            raise
        log.info("camera opened: %s", self.info())
        return self

    def _disable_filters(self) -> None:
        for name, getter in FILTERS.items():
            facility = getattr(self._dev, getter)()
            if facility is None:
                log.info("filter %s: not available", name)
                continue
            facility.enable(False)
            enabled = facility.is_enabled()
            log.info("filter %s: enabled=%s", name, enabled)
            if enabled:
                raise RuntimeError(f"could not disable the {name} filter; measurements would not be raw")

    def close(self) -> None:
        """Release the camera so another process can open it. self.state is kept for reapply()."""
        if self._dev is None:
            return
        self._dev = None  # dropping the last reference closes the device
        log.info("camera closed")

    def __enter__(self) -> EventCamera:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def _require_open(self):
        if self._dev is None:
            raise RuntimeError("camera is not open; call open() first")
        return self._dev

    # ---- information ----

    def info(self) -> dict:
        dev = self._require_open()
        hw, geometry = dev.get_i_hw_identification(), dev.get_i_geometry()
        return {
            "serial": hw.get_serial(),
            "sensor": hw.get_sensor_info().name,
            "width": geometry.get_width(),
            "height": geometry.get_height(),
            "biases": self.get_biases(),
            "temperature_c": dev.get_i_monitoring().get_temperature(),  # unit not documented; °C assumed
        }

    # ---- biases ----

    def get_biases(self) -> dict[str, int]:
        return dict(self._require_open().get_i_ll_biases().get_all_biases())

    def bias_ranges(self) -> dict[str, dict]:
        """Recommended and allowed range of each bias, as the camera reports them. set_bias() uses "recommended"."""
        biases = self._require_open().get_i_ll_biases()
        out = {}
        for name in limits.BIAS_NAMES:
            info = biases.get_bias_info(name)
            out[name] = {"recommended": tuple(info.get_bias_range()), "allowed": tuple(info.get_bias_allowed_range())}
        return out

    def set_bias(self, name: str, value: int) -> int:
        """Set one bias and return the value read back from the camera."""
        value = limits.validate_bias(name, value)  # known name, int, inside our own limits
        biases = self._require_open().get_i_ll_biases()
        lo, hi = biases.get_bias_info(name).get_bias_range()
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be in the camera's recommended range {lo}..{hi}, got {value}")
        biases.set(name, value)
        read_back = biases.get(name)
        if read_back != value:
            log.warning("bias %s: set %d but read back %d", name, value, read_back)
        self.state["biases"][name] = read_back
        log.info("bias %s = %d", name, read_back)
        return read_back

    # ---- pixel mask ----

    def mask_pixels(self, pixels: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Mask pixels (they stop sending events), one pixel per slot. Returns the newly masked pixels.

        Pixels that are already masked are skipped. If the new pixels do not fit in the free slots,
        nothing is masked and ValueError is raised.
        """
        masks = self._require_open().get_i_digital_event_mask().get_pixel_masks()
        new = []
        for x, y in pixels:
            p = limits.validate_pixel(x, y)
            if p not in self.state["mask"] and p not in new:
                new.append(p)
        free = limits.MAX_MASK_SLOTS - len(self.state["mask"])
        if len(new) > free:
            raise ValueError(f"{len(new)} new pixels but only {free} of {limits.MAX_MASK_SLOTS} mask slots are free")
        for x, y in new:
            slot = len(self.state["mask"])
            if not masks[slot].set_mask(x, y, True):
                raise RuntimeError(f"camera refused to mask pixel ({x}, {y}) in slot {slot}")
            self.state["mask"].append((x, y))
        log.info("masked %d new pixels (%d slots used)", len(new), len(self.state["mask"]))
        return new

    def clear_mask(self) -> None:
        masks = self._require_open().get_i_digital_event_mask().get_pixel_masks()
        for slot, (x, y) in enumerate(self.state["mask"]):
            masks[slot].set_mask(x, y, False)
        self.state["mask"] = []
        log.info("pixel mask cleared")

    # ---- region of interest ----

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        """Only pixels inside the window send events (analog ROI in the sensor, at most 1 window)."""
        x, y, w, h = limits.validate_roi(x, y, w, h)
        roi = self._require_open().get_i_roi()
        roi.set_mode(self._hal.I_ROI.Mode.ROI)
        if not roi.set_window(self._hal.I_ROI.Window(x, y, w, h)):  # verified order: x, y, width, height
            raise RuntimeError(f"camera refused ROI window x={x}, y={y}, w={w}, h={h}")
        roi.enable(True)
        self.state["roi"] = (x, y, w, h)
        log.info("ROI set to x=%d y=%d w=%d h=%d", x, y, w, h)

    def clear_roi(self) -> None:
        self._require_open().get_i_roi().enable(False)
        self.state["roi"] = None
        log.info("ROI cleared")

    # ---- recording ----

    def record(self, seconds: float) -> np.ndarray:
        """Record events: skip limits.WARMUP_S (start-up burst), then collect `seconds` of camera time.

        Returns a structured array with fields x, y, p, t (t in us, 0 = end of the warm-up).
        Stops early with a warning if more than limits.MAX_EVENTS events arrive.
        """
        seconds = limits.validate_duration(seconds)
        dev = self._require_open()
        n_skip = round(limits.WARMUP_S * 1e6 / DELTA_T_US)  # warm-up slices (50 for 0.5 s)
        n_keep = round(seconds * 1e6 / DELTA_T_US)
        it = self._events_iterator.from_device(device=dev, delta_t=DELTA_T_US)
        chunks, total, n_real, t0 = [], 0, 0, None
        for evs in it:
            slice_end = it.get_current_time()
            # the iterator's clock starts at 0, but stream time continues from the last recording:
            # slices up to there arrive at once and are always empty
            if slice_end <= self._t_end:
                continue
            n_real += 1
            if n_real <= n_skip:
                continue
            if t0 is None:
                t0 = slice_end - DELTA_T_US  # start of the first kept slice
            chunks.append(evs.copy())  # copy: the reader may reuse its buffer
            total += evs.size
            if total > limits.MAX_EVENTS:
                log.warning("more than %d events: recording stopped after %.2f s",
                            limits.MAX_EVENTS, (n_real - n_skip) * DELTA_T_US / 1e6)
                break
            if n_real >= n_skip + n_keep:
                break
        self._t_end = it.get_current_time()
        if t0 is None:
            raise RuntimeError("the event stream ended during the warm-up")
        events = np.concatenate(chunks)[: limits.MAX_EVENTS]
        events["t"] -= t0
        duration = (n_real - n_skip) * DELTA_T_US / 1e6
        log.info("recorded %.2f s: %d events (%.3f Mev/s)", duration, events.size, events.size / duration / 1e6)
        return events

    # ---- state ----

    def reapply(self) -> None:
        """Set self.state on the camera again (the camera forgets all settings when it is reopened)."""
        self._require_open()
        biases, mask, roi = dict(self.state["biases"]), list(self.state["mask"]), self.state["roi"]
        for name, value in biases.items():
            self.set_bias(name, value)
        self.state["mask"] = []  # the slots on the reopened camera are empty; fill them from slot 0 again
        if mask:
            self.mask_pixels(mask)
        if roi is not None:
            self.set_roi(*roi)
        log.info("state reapplied: %s", self.state)
