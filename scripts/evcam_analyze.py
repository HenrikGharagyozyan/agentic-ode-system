"""Analyse a saved recording (.npz from evcam_capture.py or store.save()): summary, hot pixels and spectrum.

No camera needed. Prints the results as JSON.

Run:  python scripts/evcam_analyze.py data/evcam/phone_screen_min.npz --bin-us 100 --fmin 20 --fmax 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from agentic_ode.hardware.devices.evcam import analysis


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Analyse a saved event recording (.npz).")
    p.add_argument("file", type=Path, help=".npz file with `events` and `meta`")
    p.add_argument("--bin-us", type=int, default=100, help="time bin of the spectrum in us (default 100)")
    p.add_argument("--fmin", type=float, default=20.0, help="lowest frequency in Hz (default 20)")
    p.add_argument("--fmax", type=float, default=2000.0, help="highest frequency in Hz (default 2000)")
    args = p.parse_args(argv)
    with np.load(args.file, allow_pickle=False) as d:
        events, meta = d["events"], json.loads(str(d["meta"]))
    # the recorded camera time (less than asked if the recording stopped early); not measured from the last event
    seconds = meta.get("recorded_s", meta["seconds"])
    result = {
        "file": str(args.file),
        "note": meta.get("note", ""),
        "summary": analysis.summary(events, seconds),
        "hot_pixels": analysis.hot_pixels(events, seconds),
        "spectrum": analysis.spectrum(events, args.bin_us, args.fmax, args.fmin),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
