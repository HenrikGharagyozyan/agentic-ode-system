"""Record events with EventCamera and save them as .npz with metadata.

Steps: open the camera, apply the biases, find the N hottest pixels in a short calibration recording and mask them,
record --seconds of camera time, save the events and the metadata.

Run:   python scripts/evcam_capture.py --seconds 3 --mask-hot 2 --bias bias_diff_off 20 --note "lamp test" \
           2> >(grep -v InvalidVectBase >&2)
Load:  d = np.load(path); events = d["events"]; meta = json.loads(str(d["meta"]))
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from agentic_ode.hardware.evcam import limits
from agentic_ode.hardware.evcam.connector import EventCamera

CALIBRATION_S = 0.5  # short recording used only to find the hot pixels
DEFAULT_DIR = Path("data/evcam")  # *.npz files there are git-ignored


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record events with EventCamera and save them as .npz with metadata.")
    p.add_argument("--seconds", type=float, default=3.0, help="recording length in camera time (default 3, max 5)")
    p.add_argument("--mask-hot", type=int, default=2, metavar="N", help="mask the N hottest pixels (default 2, max 64)")
    p.add_argument("--bias", nargs=2, action="append", default=[], metavar=("NAME", "VALUE"),
                   help="set a bias before recording; repeatable")
    p.add_argument("--note", default="", help="free text saved in the metadata")
    p.add_argument("--out", type=Path, help="output .npz file (default data/evcam/<date>_<time>.npz)")
    args = p.parse_args(argv)
    # check every value before the camera is opened, so a typo fails at once
    try:
        args.seconds = limits.validate_duration(args.seconds)
        if not 0 <= args.mask_hot <= limits.MAX_MASK_SLOTS:
            raise ValueError(f"--mask-hot must be between 0 and {limits.MAX_MASK_SLOTS}, got {args.mask_hot}")
        args.bias = [(name, limits.validate_bias(name, int(value))) for name, value in args.bias]
    except ValueError as e:
        p.error(str(e))
    return args


def hottest_pixels(events: np.ndarray, n: int) -> list[tuple[int, int, int]]:
    """The n pixels with the most events, as (x, y, count). A pixel with no events is never chosen."""
    if n == 0 or events.size == 0:
        return []
    pix = events["y"].astype(np.int64) * limits.SENSOR_W + events["x"]
    counts = np.bincount(pix, minlength=limits.SENSOR_W * limits.SENSOR_H)
    top = np.argsort(counts)[::-1][:n]
    return [(int(i % limits.SENSOR_W), int(i // limits.SENSOR_W), int(counts[i])) for i in top if counts[i] > 0]


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = args.out or DEFAULT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.npz"
    with EventCamera() as cam:
        for name, value in args.bias:
            cam.set_bias(name, value)
        hot = hottest_pixels(cam.record(CALIBRATION_S), args.mask_hot)
        cam.mask_pixels([(x, y) for x, y, _ in hot])
        events = cam.record(args.seconds)
        meta = {
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "note": args.note,
            "command": sys.argv,
            "camera": cam.info(),  # serial, sensor, width, height, biases read back, temperature_c
            "state": cam.state,    # biases, mask, roi that we set
            "seconds": args.seconds,
            "warmup_s": limits.WARMUP_S,
            "calibration_s": CALIBRATION_S,
            "hot_pixels": [{"x": x, "y": y, "calibration_events": c} for x, y, c in hot],
            "n_events": int(events.size),
            "rate_mev_s": events.size / args.seconds / 1e6,
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, events=events, meta=np.array(json.dumps(meta)))
    print(f"saved {events.size:,} events ({meta['rate_mev_s']:.3f} Mev/s) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
