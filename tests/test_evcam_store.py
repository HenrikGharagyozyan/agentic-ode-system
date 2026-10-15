"""Saving and loading recordings (no hardware needed)."""

import numpy as np
import pytest

from agentic_ode.config import PROJECT_ROOT
from agentic_ode.hardware.evcam import store
from agentic_ode.hardware.evcam.connector import EVENT_DTYPE


def some_events():
    ev = np.zeros(3, dtype=EVENT_DTYPE)
    ev["x"], ev["y"], ev["p"], ev["t"] = [1, 2, 3], [4, 5, 6], [1, 0, 1], [10, 20, 30]
    return ev


def test_default_root():
    assert store.DEFAULT_ROOT == PROJECT_ROOT / "runs" / "evcam"


def test_save_then_load(tmp_path):
    ev, meta = some_events(), {"note": "test", "seconds": 1.0}
    rid = store.save(ev, meta, root=tmp_path)
    assert store.ID_PATTERN.fullmatch(rid)
    assert (tmp_path / f"{rid}.npz").exists()
    loaded, loaded_meta = store.load(rid, root=tmp_path)
    assert loaded.dtype == EVENT_DTYPE and np.array_equal(loaded, ev)
    assert loaded_meta == {"note": "test", "seconds": 1.0, "recording_id": rid, "n_events": 3}
    assert meta == {"note": "test", "seconds": 1.0}  # the caller's dict is not changed


def test_two_saves_get_different_ids(tmp_path):
    assert store.save(some_events(), {}, root=tmp_path) != store.save(some_events(), {}, root=tmp_path)


def test_save_wrong_dtype(tmp_path):
    with pytest.raises(ValueError, match="EVENT_DTYPE"):
        store.save(np.zeros(3), {}, root=tmp_path)


@pytest.mark.parametrize("rid", ["../x", "20261015-143012-a3f9c1/../../x", "20261015-143012-A3F9C1",
                                 "20261015-143012-a3f9c1.npz", "", 123, None])
def test_load_bad_id(tmp_path, rid):
    with pytest.raises(ValueError, match="bad recording_id"):
        store.load(rid, root=tmp_path)


def test_load_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="no recording"):
        store.load("20261015-143012-a3f9c1", root=tmp_path)
